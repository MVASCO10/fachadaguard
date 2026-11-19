"""Teste mínimo 1 — validação dos esquemas MQTT versionados."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from fachadaguard.schemas.mqtt import CaptureEventV1, TelemetryV1, parse_message, parse_topic

PREFIX = "fachadaguard/v1"


def telemetry(**over):
    base = {
        "event_id": "evt-00000001",
        "schema_version": "1.0",
        "drone_id": "drn-1",
        "mission_id": "msn-1",
        "captured_at": "2026-05-12T13:00:00Z",
        "latitude": -30.05,
        "longitude": -51.26,
        "altitude_m": 12.5,
        "battery_pct": 97.2,
        "speed_mps": 3.1,
        "heading_deg": 180.0,
        "signal_quality": 88.0,
        "mission_state": "transit",
    }
    base.update(over)
    return base


def test_valid_telemetry_parses_and_normalizes_utc():
    msg = parse_message(
        f"{PREFIX}/drones/drn-1/telemetry",
        json.dumps(telemetry(captured_at="2026-05-12T10:00:00-03:00")).encode(),
        PREFIX,
    )
    assert isinstance(msg, TelemetryV1)
    assert msg.captured_at.utcoffset().total_seconds() == 0
    assert msg.captured_at.hour == 13


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "2.0"),
        ("battery_pct", 120),
        ("latitude", 95),
        ("mission_state", "flying"),
        ("captured_at", "2026-05-12T13:00:00"),  # sem timezone
        ("event_id", "../../etc"),
    ],
)
def test_invalid_telemetry_rejected(field, value):
    with pytest.raises(ValidationError):
        TelemetryV1.model_validate(telemetry(**{field: value}))


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        TelemetryV1.model_validate(telemetry(image_base64="AAAA"))


def test_topic_payload_id_mismatch_rejected():
    with pytest.raises(ValueError, match="drone_id"):
        parse_message(f"{PREFIX}/drones/outro-drone/telemetry", json.dumps(telemetry()).encode(), PREFIX)


def test_unknown_topic_and_prefix_rejected():
    with pytest.raises(ValueError):
        parse_topic("outro/v1/drones/x/telemetry", PREFIX)
    with pytest.raises(ValueError):
        parse_topic(f"{PREFIX}/drones/x/video", PREFIX)


def test_oversized_payload_rejected():
    with pytest.raises(ValueError, match="excede"):
        parse_message(f"{PREFIX}/drones/drn-1/telemetry", b"{" + b" " * 20000 + b"}", PREFIX)


def test_capture_requires_hash_and_allowed_type():
    base = {
        "event_id": "cap-00000001",
        "schema_version": "1.0",
        "drone_id": "drn-1",
        "mission_id": "msn-1",
        "captured_at": "2026-05-12T13:00:00Z",
        "building_id": "bld-1",
        "facade_id": "fac-1",
        "region_code": "N-01",
        "latitude": -30.05,
        "longitude": -51.26,
        "altitude_m": 8,
        "heading_deg": 180,
        "image_sha256": "a" * 64,
        "image_size_bytes": 1000,
        "content_type": "image/png",
    }
    assert CaptureEventV1.model_validate(base).region_code == "N-01"
    with pytest.raises(ValidationError):
        CaptureEventV1.model_validate({**base, "content_type": "image/gif"})
    with pytest.raises(ValidationError):
        CaptureEventV1.model_validate({**base, "image_sha256": "xyz"})
