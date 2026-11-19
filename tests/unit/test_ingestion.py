"""Testes mínimos 2 e 3 — idempotência/redelivery e associação da captura (metadados MQTT + binário HTTP)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select

from fachadaguard.demo_seed import seed_demo
from fachadaguard.domain.models import Capture, DeadLetter, Inspection, Mission, ModelRun, TelemetryEvent
from fachadaguard.ingestion import IngestionError, attach_blob, process_batch
from fachadaguard.storage.images import LocalBlobStore, validate_image_upload
from fachadaguard.vision.dataset import DEMO_REGIONS, demo_region_image
from tests.conftest import PREFIX, png_bytes


def tel(i: int, mission: str = "msn-t") -> tuple[str, bytes]:
    return (
        f"{PREFIX}/drones/drn-t/telemetry",
        json.dumps(
            {
                "event_id": f"evt-{i:08d}",
                "schema_version": "1.0",
                "drone_id": "drn-t",
                "mission_id": mission,
                "captured_at": f"2026-05-12T13:{i // 60 % 60:02d}:{i % 60:02d}Z",
                "latitude": -30.05,
                "longitude": -51.26,
                "altitude_m": 10,
                "battery_pct": 90,
                "speed_mps": 2,
                "heading_deg": 10,
                "signal_quality": 80,
                "mission_state": "transit",
                "simulated": True,
            }
        ).encode(),
    )


def count(session, model) -> int:
    return session.scalar(select(func.count()).select_from(model))


def test_redelivery_does_not_duplicate(session):
    batch = [tel(i) for i in range(50)]
    r1 = process_batch(session, batch, PREFIX)
    assert (r1.inserted, r1.duplicates) == (50, 0)
    # reentrega completa (QoS 1) + duplicatas dentro do mesmo lote
    r2 = process_batch(session, batch + batch[:5], PREFIX)
    assert r2.inserted == 0
    assert r2.duplicates == 55
    assert count(session, TelemetryEvent) == 50


def test_invalid_payload_goes_to_dead_letter(session):
    r = process_batch(
        session,
        [
            (f"{PREFIX}/drones/drn-t/telemetry", b'{"event_id": 1}'),
            (f"{PREFIX}/drones/drn-t/telemetry", b"not json"),
            tel(1),
        ],
        PREFIX,
    )
    assert (r.invalid, r.inserted) == (2, 1)
    assert count(session, DeadLetter) == 2
    assert all(len(d.payload_excerpt) <= 500 for d in session.scalars(select(DeadLetter)))


def _mission(session) -> str:
    seed_demo(session)
    session.add(
        Mission(id="msn-t", building_id="bld-demo-atlantico", drone_id="drn-t", name="Missão teste", is_simulated=True)
    )
    session.flush()
    return "msn-t"


def _capture_event(data: bytes, eid: str = "cap-00000001", region: str = "N-01") -> tuple[str, bytes]:
    return (
        f"{PREFIX}/missions/msn-t/captures",
        json.dumps(
            {
                "event_id": eid,
                "schema_version": "1.0",
                "drone_id": "drn-t",
                "mission_id": "msn-t",
                "captured_at": "2026-05-12T13:05:00Z",
                "building_id": "bld-demo-atlantico",
                "facade_id": "fac-demo-norte",
                "region_code": region,
                "latitude": -30.0497,
                "longitude": -51.2603,
                "altitude_m": 8,
                "heading_deg": 180,
                "image_sha256": hashlib.sha256(data).hexdigest(),
                "image_size_bytes": len(data),
                "content_type": "image/png",
                "simulated": True,
            }
        ).encode(),
    )


@pytest.mark.parametrize("order", ["metadata_first", "blob_first"])
def test_capture_association_in_any_order(session, settings, order):
    _mission(session)
    store = LocalBlobStore(settings.data_dir / "store")
    data = png_bytes(demo_region_image(DEMO_REGIONS[0], "A").image)
    info = validate_image_upload(data, "image/png", "x.png", 10 * 2**20)
    at = datetime(2026, 5, 12, 13, 5, tzinfo=UTC)
    if order == "metadata_first":
        r = process_batch(session, [_capture_event(data)], PREFIX)
        assert r.captures == 1 and r.jobs_enqueued == 0  # ainda sem binário
        _, new, enq = attach_blob(session, store, "cap-00000001", "msn-t", at, data, info, "drn-t")
        assert new and enq
    else:
        _, new, enq = attach_blob(session, store, "cap-00000001", "msn-t", at, data, info, "drn-t")
        assert new and not enq  # ainda sem metadados
        r = process_batch(session, [_capture_event(data)], PREFIX)
        assert r.jobs_enqueued == 1
    cap = session.scalar(select(Capture))
    insp = session.get(Inspection, cap.inspection_id)
    assert insp.region_id == "reg-demo-n-01" and insp.mission_id == "msn-t"
    assert cap.sha256 == cap.expected_sha256 and store.exists(cap.storage_key)
    assert count(session, ModelRun) == 1
    # redelivery do evento e reenvio do upload não criam novos registros nem jobs
    r = process_batch(session, [_capture_event(data)], PREFIX)
    assert r.duplicates == 1
    _, new, enq = attach_blob(session, store, "cap-00000001", "msn-t", at, data, info, "drn-t")
    assert not new and not enq
    assert count(session, Capture) == 1 and count(session, ModelRun) == 1


def test_capture_hash_mismatch_rejected(session, settings):
    _mission(session)
    store = LocalBlobStore(settings.data_dir / "store")
    good = png_bytes(demo_region_image(DEMO_REGIONS[0], "A").image)
    other = png_bytes(demo_region_image(DEMO_REGIONS[1], "A").image)
    process_batch(session, [_capture_event(good)], PREFIX)
    info = validate_image_upload(other, "image/png", "x.png", 10 * 2**20)
    with pytest.raises(IngestionError, match="SHA-256"):
        attach_blob(session, store, "cap-00000001", "msn-t", datetime.now(UTC), other, info)


def test_capture_for_unknown_region_or_mission_is_dead_lettered(session):
    _mission(session)
    data = b"\x89PNG fake"
    r = process_batch(session, [_capture_event(data, region="Z-99")], PREFIX)
    assert r.invalid == 1
    assert "não cadastrada" in session.scalar(select(DeadLetter.error))
