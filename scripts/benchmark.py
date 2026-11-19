"""Benchmark Big Data reproduzível: carga de telemetria via MQTT → Postgres → Parquet → DuckDB.

Executar dentro da stack (precisa de broker, banco e ingestor):
    docker compose run --rm --entrypoint python simulator scripts/benchmark.py --events 100000

Mede (sem inventar): eventos publicados e únicos, duplicatas injetadas e descartadas pelo ingestor,
tempo e vazão de ingestão até o último evento persistir, latência de ponta a ponta de uma sonda em
tempo real publicada durante a carga, duração da exportação para Parquet, tamanho do lake e tempo de
consultas agregadas no DuckDB. Grava artifacts/benchmark/benchmark_<tag>.json, latest.json e history.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fachadaguard.analytics import lake  # noqa: E402
from fachadaguard.config import get_settings  # noqa: E402
from fachadaguard.db import get_engine, init_engine, session_scope  # noqa: E402
from fachadaguard.domain.models import ServiceHeartbeat, TelemetryEvent  # noqa: E402
from fachadaguard.schemas.mqtt import SCHEMA_VERSION, TelemetryV1, telemetry_topic  # noqa: E402
from simulator.drone import MqttLink, event_id, run_load  # noqa: E402


def ingestor_stats() -> dict[str, int]:
    with session_scope() as s:
        hb = s.get(ServiceHeartbeat, "mqtt_ingestor")
        return dict(hb.stats) if hb else {}


def count_like(prefix: str) -> int:
    with session_scope() as s:
        return int(
            s.scalar(
                select(func.count()).select_from(TelemetryEvent).where(TelemetryEvent.mission_id.like(f"{prefix}%"))
            )
            or 0
        )


def probe(settings, tag: str, n: int, stop: threading.Event) -> None:
    """Sonda em tempo real (captured_at = agora) publicada durante a carga para medir latência."""
    link = MqttLink(settings, client_id=f"fg-probe-{tag}")
    link.connect()
    for i in range(n):
        if stop.is_set():
            break
        msg = TelemetryV1(
            event_id=event_id("probe", tag, i),
            schema_version=SCHEMA_VERSION,
            drone_id="drn-probe",
            mission_id=f"msn-probe-{tag}",
            captured_at=datetime.now(UTC),
            simulated=True,
            latitude=-30.05,
            longitude=-51.26,
            altitude_m=10,
            battery_pct=90,
            speed_mps=1,
            heading_deg=0,
            signal_quality=90,
            mission_state="hover",
        )
        link.publish(telemetry_topic(settings.mqtt_topic_prefix, "drn-probe"), msg.model_dump_json().encode())
        time.sleep(0.05)
    link.close(wait_for=n, timeout=60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=100_000)
    ap.add_argument("--drones", type=int, default=10)
    ap.add_argument("--duplicate-rate", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args()
    s = get_settings()
    init_engine(s.database_url)
    tag = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    out_dir = s.artifacts_dir / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    before = ingestor_stats()
    print(f"[{tag}] publicando {a.events} eventos únicos (+~{a.duplicate_rate:.0%} duplicatas)...", flush=True)
    stop = threading.Event()
    probe_thread = threading.Thread(target=probe, args=(s, tag, 200, stop), daemon=True)
    t0 = time.perf_counter()
    probe_thread.start()
    pub = run_load(s, a.events, a.drones, a.duplicate_rate, a.seed, tag)
    t_published = time.perf_counter() - t0
    prefix = f"msn-load-{tag}"
    persisted = 0
    while time.perf_counter() - t0 < a.timeout:
        persisted = count_like(prefix)
        if persisted >= pub["unique_events"]:
            break
        time.sleep(0.5)
    t_ingested = time.perf_counter() - t0
    stop.set()
    probe_thread.join(timeout=30)
    time.sleep(6)  # próximo heartbeat do ingestor com contadores atualizados
    after = ingestor_stats()
    with session_scope() as sess:
        lat_rows = sess.execute(
            select(TelemetryEvent.captured_at, TelemetryEvent.received_at).where(
                TelemetryEvent.mission_id == f"msn-probe-{tag}"
            )
        ).all()
    lat_ms = sorted((r - c).total_seconds() * 1000 for c, r in lat_rows)
    print("exportando para Parquet...", flush=True)
    exp = lake.export_all(get_engine(), s.lake_dir)
    queries = {}
    for name, fn in [
        ("telemetry_by_mission", lake.telemetry_by_mission),
        ("telemetry_distribution", lake.telemetry_distribution),
        ("events_per_day", lake.events_per_day),
        ("anomalies_by_facade", lake.anomalies_by_facade),
    ]:
        q0 = time.perf_counter()
        rows = fn(s.lake_dir)
        queries[name] = {"rows": len(rows), "seconds": round(time.perf_counter() - q0, 3)}
    load_rows = [r for r in lake.telemetry_by_mission(s.lake_dir) if r["mission_id"].startswith(prefix)]
    result = {
        "tag": tag,
        "generated_at": datetime.now(UTC).isoformat(),
        "synthetic": True,
        "config": {
            "events": a.events,
            "drones": a.drones,
            "duplicate_rate": a.duplicate_rate,
            "seed": a.seed,
            "qos": s.mqtt_qos,
        },
        "publish": pub,
        "ingestion": {
            "persisted_unique": persisted,
            "complete": persisted >= pub["unique_events"],
            "seconds_until_last_persisted": round(t_ingested, 3),
            "seconds_publish_phase": round(t_published, 3),
            "throughput_eps": round(persisted / t_ingested, 1) if t_ingested else None,
            "duplicates_discarded_by_ingestor": int(after.get("duplicates", 0)) - int(before.get("duplicates", 0)),
            "invalid_messages": int(after.get("invalid", 0)) - int(before.get("invalid", 0)),
            "batches": int(after.get("batches", 0)) - int(before.get("batches", 0)),
            "db_retries": int(after.get("db_retries", 0)) - int(before.get("db_retries", 0)),
        },
        "latency_probe_ms": {
            "n": len(lat_ms),
            "p50": round(statistics.median(lat_ms), 1) if lat_ms else None,
            "p95": round(lat_ms[int(0.95 * (len(lat_ms) - 1))], 1) if lat_ms else None,
            "max": round(lat_ms[-1], 1) if lat_ms else None,
            "note": "captured_at → received_at (relógios do mesmo host Docker), sob carga simultânea",
        },
        "lake_export": exp,
        "lake": lake.lake_stats(s.lake_dir),
        "duckdb_queries": queries,
        "lake_check": {"load_missions": len(load_rows), "load_events_in_lake": sum(r["eventos"] for r in load_rows)},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "host_note": "executado em contêineres Docker locais",
        },
    }
    (out_dir / f"benchmark_{tag}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    (out_dir / "latest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    hist = out_dir / "history.csv"
    new = not hist.exists()
    with hist.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(
                [
                    "tag",
                    "events",
                    "persisted",
                    "ingest_s",
                    "throughput_eps",
                    "dups_published",
                    "dups_discarded",
                    "lat_p50_ms",
                    "lat_p95_ms",
                    "export_s",
                    "lake_mb",
                ]
            )
        w.writerow(
            [
                tag,
                a.events,
                persisted,
                result["ingestion"]["seconds_until_last_persisted"],
                result["ingestion"]["throughput_eps"],
                pub["duplicates_published"],
                result["ingestion"]["duplicates_discarded_by_ingestor"],
                result["latency_probe_ms"]["p50"],
                result["latency_probe_ms"]["p95"],
                exp["seconds"],
                round(result["lake"]["bytes"] / 1e6, 2),
            ]
        )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("publish", "ingestion", "latency_probe_ms", "lake_export", "duckdb_queries", "lake_check")
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not result["ingestion"]["complete"]:
        sys.exit("ingestão incompleta dentro do tempo limite")


if __name__ == "__main__":
    main()
