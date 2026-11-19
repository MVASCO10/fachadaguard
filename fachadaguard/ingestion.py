"""Processamento idempotente das mensagens IoT, independente do transporte (testável sem broker).

QoS 1 garante entrega *ao menos uma vez*: a mesma mensagem pode chegar repetida (redelivery após
queda de conexão ou reenvio do buffer do drone). A unicidade de ``event_id`` no banco torna o
processamento idempotente: duplicatas são contadas e descartadas sem efeitos colaterais.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from fachadaguard.db import insert_ignore_duplicates
from fachadaguard.domain.models import (
    Capture,
    DeadLetter,
    Facade,
    FacadeRegion,
    Inspection,
    Mission,
    MissionStatus,
    TelemetryEvent,
    utcnow,
)
from fachadaguard.pipeline import enqueue_analysis
from fachadaguard.schemas.mqtt import CaptureEventV1, MissionStatusV1, TelemetryV1, parse_message
from fachadaguard.storage.images import LocalBlobStore, ValidatedImage, capture_key

log = logging.getLogger(__name__)


@dataclass
class BatchResult:
    received: int = 0
    inserted: int = 0
    duplicates: int = 0
    invalid: int = 0
    captures: int = 0
    jobs_enqueued: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    by_kind: Counter[str] = field(default_factory=Counter)


class IngestionError(ValueError):
    pass


def _dead_letter(session: Session, topic: str, payload: bytes, error: str) -> None:
    excerpt = payload[:300].decode("utf-8", errors="replace")
    session.add(DeadLetter(topic=topic[:300], error=error[:500], payload_excerpt=excerpt))


def describe_error(exc: Exception) -> str:
    """Resumo legível e curto do motivo da rejeição (campo: mensagem), sem ecoar o payload."""
    if isinstance(exc, ValidationError):
        parts = [f"{'.'.join(map(str, e['loc'])) or 'payload'}: {e['msg']}" for e in exc.errors()[:5]]
        return f"{exc.error_count()} erro(s) de validação — " + "; ".join(parts)
    return str(exc).splitlines()[0]


def _telemetry_row(msg: TelemetryV1, payload: bytes, received_at: datetime) -> dict[str, object]:
    return {
        "event_id": msg.event_id,
        "schema_version": msg.schema_version,
        "drone_id": msg.drone_id,
        "mission_id": msg.mission_id,
        "captured_at": msg.captured_at,
        "received_at": received_at,
        "latitude": msg.latitude,
        "longitude": msg.longitude,
        "altitude_m": msg.altitude_m,
        "battery_pct": msg.battery_pct,
        "speed_mps": msg.speed_mps,
        "heading_deg": msg.heading_deg,
        "signal_quality": msg.signal_quality,
        "mission_state": msg.mission_state,
        "is_simulated": msg.simulated,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }


def resolve_region(session: Session, building_id: str, facade_id: str, region_code: str) -> FacadeRegion:
    facade = session.get(Facade, facade_id)
    if facade is None or facade.building_id != building_id:
        raise IngestionError("fachada desconhecida ou não pertence ao edifício informado")
    region = session.scalar(
        select(FacadeRegion).where(FacadeRegion.facade_id == facade_id, FacadeRegion.code == region_code)
    )
    if region is None:
        raise IngestionError(f"região {region_code} não cadastrada na fachada {facade_id}")
    return region


def get_or_create_inspection(session: Session, mission_id: str, region_id: str, at: datetime) -> Inspection:
    insp = session.scalar(
        select(Inspection).where(Inspection.mission_id == mission_id, Inspection.region_id == region_id)
    )
    if insp is None:
        insp = Inspection(mission_id=mission_id, region_id=region_id, inspected_at=at)
        session.add(insp)
        session.flush()
    return insp


def apply_capture_event(session: Session, msg: CaptureEventV1) -> tuple[bool, bool]:
    """Retorna (novo_metadado, job_enfileirado). Duplicatas retornam (False, False)."""
    mission = session.get(Mission, msg.mission_id)
    if mission is None:
        raise IngestionError("missão desconhecida")
    if mission.building_id != msg.building_id:
        raise IngestionError("edifício do evento difere do edifício da missão")
    region = resolve_region(session, msg.building_id, msg.facade_id, msg.region_code)
    cap = session.scalar(select(Capture).where(Capture.event_id == msg.event_id))
    if cap is not None and cap.metadata_received:
        return False, False
    insp = get_or_create_inspection(session, mission.id, region.id, msg.captured_at)
    if cap is None:
        cap = Capture(event_id=msg.event_id, mission_id=mission.id, drone_id=msg.drone_id, captured_at=msg.captured_at)
        session.add(cap)
    cap.inspection_id, cap.region_id = insp.id, region.id
    cap.captured_at = msg.captured_at
    cap.latitude, cap.longitude, cap.altitude_m = msg.latitude, msg.longitude, msg.altitude_m
    cap.heading_deg, cap.gimbal_pitch_deg = msg.heading_deg, msg.gimbal_pitch_deg
    cap.expected_sha256 = msg.image_sha256
    cap.is_synthetic = msg.simulated
    cap.metadata_received = True
    cap.raw_metadata = msg.model_dump(mode="json")
    session.flush()
    if cap.blob_received and cap.sha256 != cap.expected_sha256:
        raise IngestionError("SHA-256 do arquivo recebido difere do informado no evento")
    return True, enqueue_analysis(session, cap) is not None


def apply_mission_status(session: Session, msg: MissionStatusV1) -> bool:
    mission = session.get(Mission, msg.mission_id)
    if mission is None:
        raise IngestionError("missão desconhecida")
    target = MissionStatus(msg.status)
    if mission.status == target:
        return False
    mission.status = target
    if target == MissionStatus.IN_PROGRESS and mission.started_at is None:
        mission.started_at = msg.captured_at
    if target in (MissionStatus.COMPLETED, MissionStatus.ABORTED):
        mission.ended_at = msg.captured_at
    return True


def process_batch(session: Session, messages: list[tuple[str, bytes]], prefix: str) -> BatchResult:
    """Processa um lote. Telemetria vai em INSERT em massa com ON CONFLICT DO NOTHING."""
    res = BatchResult(received=len(messages))
    now = utcnow()
    telemetry: dict[str, dict[str, object]] = {}
    for topic, payload in messages:
        try:
            msg = parse_message(topic, payload, prefix)
        except (ValidationError, ValueError) as exc:
            res.invalid += 1
            _dead_letter(session, topic, payload, describe_error(exc))
            continue
        res.by_kind[msg.kind] += 1
        if isinstance(msg, TelemetryV1):
            if msg.event_id in telemetry:
                res.duplicates += 1
                continue
            telemetry[msg.event_id] = _telemetry_row(msg, payload, now)
            res.latencies_ms.append((now - msg.captured_at).total_seconds() * 1000)
            continue
        try:
            with session.begin_nested():
                if isinstance(msg, CaptureEventV1):
                    new, enq = apply_capture_event(session, msg)
                    res.captures += int(new)
                    res.inserted += int(new)
                    res.duplicates += int(not new)
                    res.jobs_enqueued += int(enq)
                else:
                    changed = apply_mission_status(session, msg)
                    res.inserted += int(changed)
                    res.duplicates += int(not changed)
        except IngestionError as exc:
            res.invalid += 1
            _dead_letter(session, topic, payload, str(exc))
    inserted = insert_ignore_duplicates(session, TelemetryEvent.__table__, list(telemetry.values()))
    res.inserted += inserted
    res.duplicates += len(telemetry) - inserted
    return res


def attach_blob(
    session: Session,
    store: LocalBlobStore,
    event_id: str,
    mission_id: str,
    captured_at: datetime,
    data: bytes,
    info: ValidatedImage,
    drone_id: str = "unknown",
    source: str = "http",
) -> tuple[Capture, bool, bool]:
    """Associa o binário (HTTP) à captura de mesmo event_id. Retorna (captura, novo, job_enfileirado)."""
    if session.get(Mission, mission_id) is None:
        raise IngestionError("missão desconhecida")
    cap = session.scalar(select(Capture).where(Capture.event_id == event_id))
    if cap is not None and cap.blob_received:
        if cap.sha256 != info.sha256:
            raise IngestionError("event_id já possui imagem diferente (conflito)")
        return cap, False, False
    if cap is None:
        cap = Capture(
            event_id=event_id, mission_id=mission_id, drone_id=drone_id, captured_at=captured_at, source=source
        )
        session.add(cap)
        session.flush()
    elif cap.mission_id != mission_id:
        raise IngestionError("event_id pertence a outra missão")
    if cap.expected_sha256 and cap.expected_sha256 != info.sha256:
        raise IngestionError("SHA-256 do arquivo difere do informado no evento MQTT")
    key = capture_key(cap.id, info.extension)
    store.put(key, data)
    cap.storage_key, cap.sha256 = key, info.sha256
    cap.content_type, cap.size_bytes = info.content_type, info.size_bytes
    cap.width, cap.height = info.width, info.height
    cap.blob_received = True
    session.flush()
    return cap, True, enqueue_analysis(session, cap) is not None
