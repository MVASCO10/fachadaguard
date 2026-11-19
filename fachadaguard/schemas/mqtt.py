"""Contratos versionados das mensagens MQTT (schema_version "1.0").

Tópicos (prefixo configurável, padrão ``fachadaguard/v1``):

* ``{prefix}/drones/{drone_id}/telemetry``      → :class:`TelemetryV1` (QoS 1)
* ``{prefix}/missions/{mission_id}/captures``   → :class:`CaptureEventV1` (QoS 1)
* ``{prefix}/missions/{mission_id}/status``     → :class:`MissionStatusV1` (QoS 1)

Imagens NÃO trafegam no MQTT: o evento de captura leva apenas metadados e o SHA-256 do arquivo,
que é enviado por HTTP (``POST /api/v1/captures/{event_id}/image``) com o mesmo ``event_id``.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.0"
ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_\-]{1,63}$"
Identifier = Annotated[str, Field(pattern=ID_PATTERN)]
MAX_PAYLOAD_BYTES = 16 * 1024

TOPIC_RE = re.compile(
    r"^(?P<prefix>.+)/(?:drones/(?P<drone_id>[^/]+)/(?P<dkind>telemetry)"
    r"|missions/(?P<mission_id>[^/]+)/(?P<mkind>captures|status))$"
)


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: Annotated[str, Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")]
    schema_version: Literal["1.0"]
    drone_id: Identifier
    mission_id: Identifier
    captured_at: datetime
    simulated: bool = False

    @field_validator("captured_at")
    @classmethod
    def _utc_and_plausible(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("captured_at deve conter timezone (use UTC, sufixo Z)")
        value = value.astimezone(UTC)
        if value > datetime.now(UTC) + timedelta(days=400) or value.year < 2000:
            raise ValueError("captured_at fora do intervalo plausível")
        return value


class TelemetryV1(_Base):
    kind: Literal["telemetry"] = "telemetry"
    latitude: Annotated[float, Field(ge=-90, le=90)]
    longitude: Annotated[float, Field(ge=-180, le=180)]
    altitude_m: Annotated[float, Field(ge=-10, le=500)]
    battery_pct: Annotated[float, Field(ge=0, le=100)]
    speed_mps: Annotated[float, Field(ge=0, le=40)]
    heading_deg: Annotated[float, Field(ge=0, lt=360)]
    signal_quality: Annotated[float, Field(ge=0, le=100, description="Qualidade do enlace, 0–100")]
    mission_state: Literal["takeoff", "transit", "inspecting", "returning", "landed", "hover"]


class CaptureEventV1(_Base):
    kind: Literal["capture"] = "capture"
    building_id: Identifier
    facade_id: Identifier
    region_code: Annotated[str, Field(pattern=r"^[A-Za-z0-9_\-]{1,40}$")]
    latitude: Annotated[float, Field(ge=-90, le=90)]
    longitude: Annotated[float, Field(ge=-180, le=180)]
    altitude_m: Annotated[float, Field(ge=-10, le=500)]
    heading_deg: Annotated[float, Field(ge=0, lt=360)]
    gimbal_pitch_deg: Annotated[float, Field(ge=-90, le=30)] = 0.0
    image_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    image_size_bytes: Annotated[int, Field(gt=0)]
    content_type: Literal["image/png", "image/jpeg"]


class MissionStatusV1(_Base):
    kind: Literal["mission_status"] = "mission_status"
    status: Literal["in_progress", "completed", "aborted"]


class TopicInfo(BaseModel):
    kind: Literal["telemetry", "captures", "status"]
    drone_id: str | None = None
    mission_id: str | None = None


def parse_topic(topic: str, prefix: str) -> TopicInfo:
    match = TOPIC_RE.match(topic)
    if not match or match.group("prefix") != prefix:
        raise ValueError(f"tópico não reconhecido: {topic[:120]}")
    if match.group("dkind"):
        return TopicInfo(kind="telemetry", drone_id=match.group("drone_id"))
    return TopicInfo(kind=match.group("mkind"), mission_id=match.group("mission_id"))  # type: ignore[arg-type]


def telemetry_topic(prefix: str, drone_id: str) -> str:
    return f"{prefix}/drones/{drone_id}/telemetry"


def captures_topic(prefix: str, mission_id: str) -> str:
    return f"{prefix}/missions/{mission_id}/captures"


def status_topic(prefix: str, mission_id: str) -> str:
    return f"{prefix}/missions/{mission_id}/status"


Message = TelemetryV1 | CaptureEventV1 | MissionStatusV1


def parse_message(topic: str, payload: bytes, prefix: str) -> Message:
    """Valida tópico + payload e verifica coerência entre IDs do tópico e do corpo."""
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError(f"payload excede {MAX_PAYLOAD_BYTES} bytes")
    info = parse_topic(topic, prefix)
    model: type[Message]
    if info.kind == "telemetry":
        model = TelemetryV1
    elif info.kind == "captures":
        model = CaptureEventV1
    else:
        model = MissionStatusV1
    msg = model.model_validate_json(payload)
    if info.drone_id is not None and info.drone_id != msg.drone_id:
        raise ValueError("drone_id do tópico difere do payload")
    if info.mission_id is not None and info.mission_id != msg.mission_id:
        raise ValueError("mission_id do tópico difere do payload")
    return msg
