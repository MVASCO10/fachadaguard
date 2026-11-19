"""Smoke test contra a stack Docker em execução (``make up``). Executar com: ``make test-integration``.

Publica telemetria real no Mosquitto (credenciais do drone, de .env), reenvia as mesmas mensagens
(redelivery), envia um payload inválido e verifica persistência sem duplicatas e dead-letter.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx
import paho.mqtt.client as mqtt
import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
API = "http://localhost:8000"


def load_env() -> dict[str, str]:
    values: dict[str, str] = {}
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip()
    values.update({k: v for k, v in os.environ.items() if k.startswith(("FG_", "MQTT_"))})
    return values


@pytest.fixture(scope="module")
def cfg() -> dict[str, str]:
    try:
        httpx.get(f"{API}/health", timeout=3).raise_for_status()
    except httpx.HTTPError:
        pytest.skip("stack não está em execução (rode `make up`)")
    return load_env()


def headers(cfg: dict[str, str]) -> dict[str, str]:
    return {"X-API-Token": cfg.get("FG_API_TOKEN", "")}


def test_ready(cfg):
    r = httpx.get(f"{API}/ready", timeout=10).json()
    assert r["status"] == "ready" and r["checks"]["database"] == "ok"


def test_mqtt_to_database_with_redelivery(cfg):
    mission = f"msn-smoke-{uuid.uuid4().hex[:8]}"
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"smoke-{mission}", protocol=mqtt.MQTTv5)
    client.username_pw_set("fg_drone", cfg["MQTT_DRONE_PASSWORD"])
    client.connect("localhost", 1883)
    client.loop_start()
    now = datetime.now(UTC).isoformat()
    msgs = [
        json.dumps(
            {
                "event_id": f"{mission}-{i:04d}",
                "schema_version": "1.0",
                "drone_id": "drn-smoke",
                "mission_id": mission,
                "captured_at": now,
                "latitude": -30.05,
                "longitude": -51.26,
                "altitude_m": 10,
                "battery_pct": 90,
                "speed_mps": 1,
                "heading_deg": 0,
                "signal_quality": 80,
                "mission_state": "hover",
                "simulated": True,
            }
        ).encode()
        for i in range(30)
    ]
    topic = "fachadaguard/v1/drones/drn-smoke/telemetry"
    for _ in range(2):  # envio + reenvio integral (simula redelivery)
        for m in msgs:
            client.publish(topic, m, qos=1).wait_for_publish(10)
    client.publish(topic, b"payload invalido", qos=1).wait_for_publish(10)
    client.loop_stop()
    client.disconnect()
    time.sleep(3)
    r = httpx.post(f"{API}/api/v1/analytics/export", headers=headers(cfg), timeout=120)
    r.raise_for_status()
    rows = httpx.get(f"{API}/api/v1/analytics/telemetry-by-mission", timeout=60).json()
    mine = [row for row in rows if row["mission_id"] == mission]
    assert mine and mine[0]["eventos"] == 30, "60 mensagens (30 únicas) devem resultar em 30 eventos"
    dead = httpx.get(f"{API}/api/v1/ingestion/status", timeout=10).json()["dead_letters"]
    assert any("json" in d["error"].lower() for d in dead)


def test_upload_rejects_non_image(cfg):
    r = httpx.post(
        f"{API}/api/v1/captures/smoke-evt-0001/image",
        headers=headers(cfg),
        files={"file": ("x.png", b"not an image", "image/png")},
        data={"mission_id": "msn-demo-a", "captured_at": "2026-05-12T13:00:00Z"},
        timeout=10,
    )
    assert r.status_code == 422


def test_write_requires_token(cfg):
    assert httpx.post(f"{API}/api/v1/demo/seed", timeout=10).status_code == 401
