"""Simulador de drone de inspeção (SIMULADO — não controla hardware real).

Modos:
  mission  – voo sobre o edifício fictício publicando telemetria MQTT e capturas (HTTP + MQTT);
             opcionalmente com perda de enlace, buffer local (store-and-forward) e reenvio.
  load     – carga acelerada de telemetria (ex.: 100 mil eventos) para o benchmark Big Data.

Exemplos:
  python -m simulator.drone mission --epoch A
  python -m simulator.drone mission --epoch B --outage-at 0.35 --outage-seconds 40
  python -m simulator.drone load --events 100000 --drones 10 --duplicate-rate 0.02
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sqlite3
import ssl
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import paho.mqtt.client as mqtt

from fachadaguard.config import Settings, get_settings
from fachadaguard.demo_seed import region_center
from fachadaguard.schemas.mqtt import (
    SCHEMA_VERSION,
    CaptureEventV1,
    MissionStatusV1,
    TelemetryV1,
    captures_topic,
    status_topic,
    telemetry_topic,
)
from fachadaguard.vision.dataset import DEMO_BUILDING, DEMO_EPOCHS, DEMO_REGIONS, demo_region_image

log = logging.getLogger("fachadaguard.simulator")
NS = uuid.UUID("6f1f6a52-2b8e-4c3e-9a51-0d6f3c1c7a10")
DRONE_ID = "drn-sim-01"
STEP_S = 0.5  # período lógico da telemetria (2 Hz)
M_PER_DEG = 111_320.0
MAX_OUTSTANDING = 5_000


def event_id(*parts: object) -> str:
    return uuid.uuid5(NS, "|".join(map(str, parts))).hex


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------------------- buffer local


class StoreAndForwardBuffer:
    """Fila persistente em SQLite no "drone": sobrevive a reinício do processo."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS outbox (seq INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, "
            "topic TEXT, payload BLOB, meta TEXT)"
        )
        self.db.commit()

    def push(self, kind: str, topic: str, payload: bytes, meta: dict[str, Any] | None = None) -> None:
        self.db.execute(
            "INSERT INTO outbox(kind, topic, payload, meta) VALUES (?,?,?,?)",
            (kind, topic, payload, json.dumps(meta or {})),
        )
        self.db.commit()

    def __len__(self) -> int:
        return int(self.db.execute("SELECT count(*) FROM outbox").fetchone()[0])

    def drain(self) -> Iterator[tuple[int, str, str, bytes, dict[str, Any]]]:
        rows = self.db.execute(
            "SELECT seq, kind, topic, payload, meta FROM outbox ORDER BY CASE kind WHEN 'upload' THEN 0 ELSE 1 END, seq"
        ).fetchall()
        for seq, kind, topic, payload, meta in rows:
            yield seq, kind, topic, payload, json.loads(meta)

    def ack(self, seq: int) -> None:
        self.db.execute("DELETE FROM outbox WHERE seq = ?", (seq,))
        self.db.commit()


# ----------------------------------------------------------------------------- enlaces


class MqttLink:
    def __init__(self, s: Settings, client_id: str, max_inflight: int = 100) -> None:
        self.s = s
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, protocol=mqtt.MQTTv5)
        self.client.username_pw_set(s.mqtt_username, s.mqtt_password)
        if s.mqtt_tls:
            self.client.tls_set(ca_certs=s.mqtt_ca_file, tls_version=ssl.PROTOCOL_TLS_CLIENT)
        self.client.max_inflight_messages_set(max_inflight)
        self.connected = threading.Event()
        self.acked = 0
        self.published = 0
        self._lock = threading.Lock()
        self.client.on_connect = lambda c, u, f, rc, p: self.connected.set() if not rc.is_failure else None
        self.client.on_disconnect = lambda c, u, f, rc, p: self.connected.clear()
        self.client.on_publish = self._on_publish

    def _on_publish(self, *_: Any) -> None:
        with self._lock:
            self.acked += 1

    def connect(self) -> None:
        self.client.connect(self.s.mqtt_host, self.s.mqtt_port, keepalive=30)
        self.client.loop_start()
        if not self.connected.wait(10):
            raise RuntimeError("não foi possível conectar ao broker MQTT (verifique host/credenciais)")

    def publish(self, topic: str, payload: bytes) -> mqtt.MQTTMessageInfo:
        """Publica com backpressure: o paho indexa mensagens pendentes pelo packet id (16 bits); mais de
        65.535 pendentes colidem e são perdidas. Limitamos as não confirmadas a MAX_OUTSTANDING."""
        if self.s.mqtt_qos > 0:
            deadline = time.monotonic() + 60
            while self.published - self.acked >= MAX_OUTSTANDING and time.monotonic() < deadline:
                time.sleep(0.002)
        self.published += 1
        return self.client.publish(topic, payload, qos=self.s.mqtt_qos)

    def drop(self) -> None:
        self.client.disconnect()
        self.client.loop_stop()
        self.connected.clear()

    def close(self, wait_for: int | None = None, timeout: float = 120) -> None:
        deadline = time.monotonic() + timeout
        while wait_for is not None and self.acked < wait_for and time.monotonic() < deadline:
            time.sleep(0.05)
        self.client.disconnect()
        self.client.loop_stop()


class Api:
    def __init__(self, s: Settings) -> None:
        headers = {"X-API-Token": s.api_token} if s.api_token else {}
        self.http = httpx.Client(base_url=s.api_base_url, headers=headers, timeout=30)

    def post(self, path: str, **kw: Any) -> dict[str, Any]:
        r = self.http.post(path, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"POST {path} → {r.status_code}: {r.text[:300]}")
        return r.json()


# ----------------------------------------------------------------------------- plano de voo


@dataclass
class State:
    t: datetime
    lat: float
    lon: float
    alt: float
    speed: float
    heading: float
    state: str
    region: str | None = None
    capture: bool = False


def _offset(lat: float, lon: float, north_m: float, east_m: float) -> tuple[float, float]:
    return lat + north_m / M_PER_DEG, lon + east_m / (M_PER_DEG * math.cos(math.radians(lat)))


def _dist_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    dn = (b[0] - a[0]) * M_PER_DEG
    de = (b[1] - a[1]) * M_PER_DEG * math.cos(math.radians(a[0]))
    return math.hypot(dn, de)


def flight_plan(epoch: str, rng: np.random.Generator) -> list[State]:
    """Decola, visita cada região (standoff de 8 m), paira para capturar e retorna."""
    t = datetime.fromisoformat(DEMO_EPOCHS[epoch].replace("Z", "+00:00"))
    home = _offset(DEMO_BUILDING["latitude"], DEMO_BUILDING["longitude"], -35, -20)
    states: list[State] = []
    pos = (home[0], home[1], 0.0)

    def move_to(target: tuple[float, float, float], label: str, speed: float = 3.0) -> None:
        nonlocal pos, t
        d = math.hypot(_dist_m(pos[:2], target[:2]), target[2] - pos[2])
        n = max(1, int(d / (speed * STEP_S)))
        dn = (target[0] - pos[0]) * M_PER_DEG
        de = (target[1] - pos[1]) * M_PER_DEG * math.cos(math.radians(pos[0]))
        heading = (math.degrees(math.atan2(de, dn)) + 360) % 360 if (dn or de) else 0.0
        start = pos
        for k in range(1, n + 1):
            f = k / n
            t += timedelta(seconds=STEP_S)
            jitter = rng.normal(0, 0.15, 3)
            states.append(
                State(
                    t,
                    start[0] + (target[0] - start[0]) * f + jitter[0] / M_PER_DEG,
                    start[1] + (target[1] - start[1]) * f + jitter[1] / M_PER_DEG,
                    max(0.0, start[2] + (target[2] - start[2]) * f + jitter[2]),
                    float(abs(speed + rng.normal(0, 0.2))),
                    round(heading % 360, 1),
                    label,
                )
            )
        pos = target

    move_to((home[0], home[1], 12.0), "takeoff", 2.0)
    for reg in DEMO_REGIONS:
        lat, lon, alt = region_center(reg["code"])
        if reg["facade_id"] == "fac-demo-norte":
            stand = (*_offset(lat, lon, 8, 0), alt)
            face = 180.0
        else:
            stand = (*_offset(lat, lon, 0, 8), alt)
            face = 270.0
        move_to(stand, "transit")
        for k in range(12):  # 6 s pairando
            t += timedelta(seconds=STEP_S)
            states.append(
                State(
                    t,
                    stand[0] + rng.normal(0, 0.1) / M_PER_DEG,
                    stand[1] + rng.normal(0, 0.1) / M_PER_DEG,
                    stand[2] + rng.normal(0, 0.05),
                    float(abs(rng.normal(0, 0.05))),
                    round((face + rng.normal(0, 1.0)) % 360, 1),
                    "inspecting",
                    reg["code"],
                    k == 6,
                )
            )
    move_to((home[0], home[1], 12.0), "returning")
    move_to((home[0], home[1], 0.0), "returning", 1.5)
    t += timedelta(seconds=STEP_S)
    states.append(State(t, home[0], home[1], 0.0, 0.0, states[-1].heading, "landed"))
    return states


# ----------------------------------------------------------------------------- modo missão


def run_mission(
    s: Settings,
    epoch: str,
    seed: int,
    speed: float,
    outage_at: float | None,
    outage_seconds: float,
    resend_last: int,
    buffer_path: Path,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + (0 if epoch == "A" else 1))
    mission_id = f"msn-demo-{epoch.lower()}"
    api = Api(s)
    api.post("/api/v1/demo/seed")
    label = "maio/2026" if epoch == "A" else "novembro/2026"
    api.post(
        "/api/v1/missions",
        json={
            "id": mission_id,
            "building_id": DEMO_BUILDING["id"],
            "drone_id": DRONE_ID,
            "name": f"Inspeção semestral {label} (SIMULADA)",
            "planned_for": DEMO_EPOCHS[epoch],
            "is_simulated": True,
            "notes": "Missão gerada pelo simulador; dados sintéticos.",
        },
    )
    link = MqttLink(s, client_id=f"{DRONE_ID}-{epoch}-{uuid.uuid4().hex[:6]}")
    link.connect()
    buf = StoreAndForwardBuffer(buffer_path)
    plan = flight_plan(epoch, rng)
    outage_range = None
    if outage_at is not None:
        start = int(len(plan) * outage_at)
        outage_range = range(start, start + int(outage_seconds / STEP_S))
    counters = {
        "telemetry": 0,
        "captures": 0,
        "uploads": 0,
        "buffered": 0,
        "resent_duplicates": 0,
        "published_total": 0,
    }
    sent_log: list[tuple[str, bytes]] = []
    battery = float(rng.uniform(96, 99))
    p = s.mqtt_topic_prefix

    def status(st: str, at: datetime) -> None:
        msg = MissionStatusV1(
            event_id=event_id(mission_id, "status", st),
            schema_version=SCHEMA_VERSION,
            drone_id=DRONE_ID,
            mission_id=mission_id,
            captured_at=at,
            simulated=True,
            status=st,
        )
        deliver("mqtt", status_topic(p, mission_id), msg.model_dump_json().encode())

    def deliver(kind: str, topic: str, payload: bytes, meta: dict[str, Any] | None = None) -> None:
        if not link.connected.is_set():
            buf.push(kind, topic, payload, meta)
            counters["buffered"] += 1
            return
        if kind == "upload":
            assert meta is not None
            upload(payload, meta)
            return
        link.publish(topic, payload)
        counters["published_total"] += 1
        sent_log.append((topic, payload))

    def upload(data: bytes, meta: dict[str, Any]) -> None:
        api.post(
            f"/api/v1/captures/{meta['event_id']}/image",
            files={"file": (f"{meta['event_id']}.png", data, "image/png")},
            data={"mission_id": mission_id, "captured_at": meta["captured_at"], "drone_id": DRONE_ID},
        )
        counters["uploads"] += 1

    status("in_progress", plan[0].t)
    was_down = False
    pre_outage_len = 0
    for i, st in enumerate(plan):
        in_outage = outage_range is not None and i in outage_range
        if in_outage and link.connected.is_set():
            log.warning("simulando perda de enlace (passo %d)", i)
            link.drop()
            was_down = True
            pre_outage_len = len(sent_log)
        elif not in_outage and was_down:
            link.connect()
            log.warning("enlace restabelecido; reenviando %d itens do buffer", len(buf))
            for seq, kind, topic, payload, meta in list(buf.drain()):
                deliver(kind, topic, payload, meta)
                buf.ack(seq)
            # reenvio de mensagens já entregues (ACK perdido) → devem ser descartadas pelo ingestor
            for topic, payload in sent_log[max(0, pre_outage_len - resend_last) : pre_outage_len]:
                link.publish(topic, payload)
                counters["resent_duplicates"] += 1
            was_down = False
        battery -= 0.035 * STEP_S * (1.6 if st.state in ("takeoff", "transit") else 1.0) + rng.normal(0, 0.004)
        signal_q = 0.0 if in_outage else float(np.clip(92 - 0.25 * st.alt + rng.normal(0, 3), 5, 100))
        tel = TelemetryV1(
            event_id=event_id(mission_id, "tel", i),
            schema_version=SCHEMA_VERSION,
            drone_id=DRONE_ID,
            mission_id=mission_id,
            captured_at=st.t,
            simulated=True,
            latitude=st.lat,
            longitude=st.lon,
            altitude_m=round(st.alt, 2),
            battery_pct=round(max(battery, 0), 2),
            speed_mps=round(st.speed, 2),
            heading_deg=st.heading % 360,
            signal_quality=round(signal_q, 1),
            mission_state="hover" if st.state == "inspecting" and not st.capture else st.state,
        )
        deliver("mqtt", telemetry_topic(p, DRONE_ID), tel.model_dump_json().encode())
        counters["telemetry"] += 1
        if st.capture and st.region:
            region = next(r for r in DEMO_REGIONS if r["code"] == st.region)
            ok, png = cv2.imencode(".png", demo_region_image(region, epoch).image)
            assert ok
            data = png.tobytes()
            eid = event_id(mission_id, "cap", st.region)
            cap = CaptureEventV1(
                event_id=eid,
                schema_version=SCHEMA_VERSION,
                drone_id=DRONE_ID,
                mission_id=mission_id,
                captured_at=st.t,
                simulated=True,
                building_id=DEMO_BUILDING["id"],
                facade_id=region["facade_id"],
                region_code=st.region,
                latitude=st.lat,
                longitude=st.lon,
                altitude_m=round(st.alt, 2),
                heading_deg=st.heading % 360,
                gimbal_pitch_deg=0.0,
                image_sha256=hashlib.sha256(data).hexdigest(),
                image_size_bytes=len(data),
                content_type="image/png",
            )
            deliver("upload", "", data, {"event_id": eid, "captured_at": iso(st.t)})  # binário por HTTP
            deliver("mqtt", captures_topic(p, mission_id), cap.model_dump_json().encode())  # metadados por MQTT
            counters["captures"] += 1
        if speed > 0:
            time.sleep(STEP_S / speed)
    if was_down:
        link.connect()
        for seq, kind, topic, payload, meta in list(buf.drain()):
            deliver(kind, topic, payload, meta)
            buf.ack(seq)
    status("completed", plan[-1].t)
    link.close(wait_for=counters["published_total"] + counters["resent_duplicates"])
    return {
        "mission_id": mission_id,
        "epoch": epoch,
        "steps": len(plan),
        **counters,
        "buffer_remaining": len(buf),
        "simulated": True,
    }


# ----------------------------------------------------------------------------- modo carga


def run_load(
    s: Settings, events: int, drones: int, duplicate_rate: float, seed: int, tag: str, max_inflight: int = 2000
) -> dict[str, Any]:
    """Publica ``events`` telemetrias únicas (+ duplicatas) o mais rápido possível."""
    rng = np.random.default_rng(seed)
    link = MqttLink(s, client_id=f"fg-load-{tag}", max_inflight=max_inflight)
    link.connect()
    base = datetime(2026, 10, 1, 12, tzinfo=UTC)
    per_drone = events // drones
    p = s.mqtt_topic_prefix
    dup_pool: list[tuple[str, bytes]] = []
    sent = dups = 0
    t0 = time.perf_counter()
    first_captured = None
    for d in range(drones):
        drone = f"drn-load-{d:02d}"
        lat, lon = _offset(
            DEMO_BUILDING["latitude"], DEMO_BUILDING["longitude"], rng.normal(0, 200), rng.normal(0, 200)
        )
        battery, alt = 99.0, 20.0
        for k in range(per_drone if d < drones - 1 else events - per_drone * (drones - 1)):
            mission = f"msn-load-{tag}-{d:02d}-{k // 2000:03d}"
            if k % 2000 == 0:
                battery, alt = float(rng.uniform(92, 100)), 20.0
            day = (d * 3 + k // 2000) % 30
            captured = base + timedelta(days=day, seconds=(k % 2000) * STEP_S)
            first_captured = first_captured or captured
            lat += rng.normal(0, 1.5) / M_PER_DEG
            lon += rng.normal(0, 1.5) / M_PER_DEG
            alt = float(np.clip(alt + rng.normal(0, 0.4), 2, 120))
            battery = max(5.0, battery - 0.02 + rng.normal(0, 0.003))
            msg = TelemetryV1(
                event_id=event_id("load", tag, d, k),
                schema_version=SCHEMA_VERSION,
                drone_id=drone,
                mission_id=mission,
                captured_at=captured,
                simulated=True,
                latitude=lat,
                longitude=lon,
                altitude_m=round(alt, 2),
                battery_pct=round(battery, 2),
                speed_mps=round(abs(rng.normal(3, 1)), 2),
                heading_deg=float(rng.uniform(0, 359.9)),
                signal_quality=round(float(np.clip(rng.normal(78, 12), 0, 100)), 1),
                mission_state="inspecting",
            )
            topic, payload = telemetry_topic(p, drone), msg.model_dump_json().encode()
            link.publish(topic, payload)
            sent += 1
            if rng.random() < duplicate_rate:
                dup_pool.append((topic, payload))
            if len(dup_pool) >= 50:
                for t_, p_ in dup_pool:
                    link.publish(t_, p_)
                    dups += 1
                dup_pool.clear()
    for t_, p_ in dup_pool:
        link.publish(t_, p_)
        dups += 1
    publish_s = time.perf_counter() - t0
    link.close(wait_for=sent + dups, timeout=600)
    acked_s = time.perf_counter() - t0
    return {
        "unique_events": sent,
        "duplicates_published": dups,
        "drones": drones,
        "tag": tag,
        "publish_seconds": round(publish_s, 3),
        "broker_ack_seconds": round(acked_s, 3),
        "publish_rate_eps": round((sent + dups) / acked_s, 1),
        "acked": link.acked,
        "simulated": True,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Simulador de drone FachadaGuard (dados SIMULADOS)")
    sub = ap.add_subparsers(dest="mode", required=True)
    m = sub.add_parser("mission", help="voo de inspeção no edifício fictício")
    m.add_argument("--epoch", choices=sorted(DEMO_EPOCHS), default="A")
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--speed", type=float, default=40.0, help="segundos lógicos por segundo real (0 = máximo)")
    m.add_argument("--outage-at", type=float, default=None, help="fração do voo em que o enlace cai (0–1)")
    m.add_argument("--outage-seconds", type=float, default=30.0, help="duração lógica da queda")
    m.add_argument("--resend-last", type=int, default=10, help="mensagens já entregues a reenviar após reconectar")
    m.add_argument("--buffer", type=Path, default=None)
    ld = sub.add_parser("load", help="carga acelerada de telemetria")
    ld.add_argument("--events", type=int, default=100_000)
    ld.add_argument("--drones", type=int, default=10)
    ld.add_argument("--duplicate-rate", type=float, default=0.02)
    ld.add_argument("--seed", type=int, default=7)
    ld.add_argument("--tag", default=datetime.now(UTC).strftime("%Y%m%d%H%M%S"))
    args = ap.parse_args()
    s = get_settings()
    if args.mode == "mission":
        buffer = args.buffer or (s.data_dir / "sim_buffer" / f"{DRONE_ID}.sqlite")
        out = run_mission(
            s, args.epoch, args.seed, args.speed, args.outage_at, args.outage_seconds, args.resend_last, buffer
        )
    else:
        out = run_load(s, args.events, args.drones, args.duplicate_rate, args.seed, args.tag)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
