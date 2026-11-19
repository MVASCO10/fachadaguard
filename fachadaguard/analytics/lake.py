"""Caminho analítico: exportação incremental Postgres → Parquet particionado + consultas DuckDB.

Layout (particionamento Hive):
    lake/telemetry/dt=YYYY-MM-DD/mission_id=<id>/part-<seq>.parquet   (incremental por id)
    lake/findings/building_id=<id>/findings.parquet                    (snapshot regravado)
    lake/_state.json                                                    (marca d'água da exportação)

O banco operacional atende consultas recentes/transacionais; o lake guarda o histórico em formato
colunar comprimido para agregações sobre milhões de eventos sem carregar o Postgres.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sqlalchemy import Engine, select

from fachadaguard.domain.models import (
    Capture,
    Facade,
    FacadeRegion,
    Finding,
    Inspection,
    TelemetryEvent,
)

TELEMETRY_COLUMNS = [
    "id",
    "event_id",
    "drone_id",
    "mission_id",
    "captured_at",
    "received_at",
    "latitude",
    "longitude",
    "altitude_m",
    "battery_pct",
    "speed_mps",
    "heading_deg",
    "signal_quality",
    "mission_state",
    "is_simulated",
]


def _state(lake: Path) -> dict[str, Any]:
    path = lake / "_state.json"
    return json.loads(path.read_text()) if path.exists() else {"telemetry_last_id": 0, "exports": 0}


def export_telemetry(engine: Engine, lake: Path, chunk: int = 200_000) -> dict[str, Any]:
    """Exporta apenas eventos novos (id > marca d'água). Idempotente entre execuções."""
    lake.mkdir(parents=True, exist_ok=True)
    state = _state(lake)
    last_id = int(state["telemetry_last_id"])
    t0 = time.perf_counter()
    exported = 0
    cols = [getattr(TelemetryEvent, c) for c in TELEMETRY_COLUMNS]
    with engine.connect() as conn:
        while True:
            rows = conn.execute(
                select(*cols).where(TelemetryEvent.id > last_id).order_by(TelemetryEvent.id).limit(chunk)
            ).all()
            if not rows:
                break
            table = pa.Table.from_pylist([dict(zip(TELEMETRY_COLUMNS, r, strict=True)) for r in rows])
            dt = pc.strftime(table["captured_at"], format="%Y-%m-%d")
            table = table.append_column("dt", dt)
            pq.write_to_dataset(
                table,
                root_path=str(lake / "telemetry"),
                partition_cols=["dt", "mission_id"],
                basename_template=f"part-{uuid.uuid4().hex[:8]}-{{i}}.parquet",
                compression="zstd",
            )
            exported += len(rows)
            last_id = int(rows[-1][0])
    state.update(telemetry_last_id=last_id, exports=int(state["exports"]) + 1)
    (lake / "_state.json").write_text(json.dumps(state, indent=2))
    return {"exported_rows": exported, "last_id": last_id, "seconds": round(time.perf_counter() - t0, 3)}


def export_findings(engine: Engine, lake: Path) -> dict[str, Any]:
    stmt = (
        select(
            Finding.id.label("finding_id"),
            Finding.candidate_category,
            Finding.severity_level,
            Finding.severity_points,
            Finding.area_ratio,
            Finding.peak_score,
            Finding.growth_ratio,
            Finding.review_status,
            Inspection.inspected_at,
            Inspection.mission_id,
            FacadeRegion.code.label("region_code"),
            Facade.id.label("facade_id"),
            Facade.name.label("facade_name"),
            Facade.building_id,
            Capture.is_synthetic,
        )
        .join(Inspection, Inspection.id == Finding.inspection_id)
        .join(FacadeRegion, FacadeRegion.id == Finding.region_id)
        .join(Facade, Facade.id == FacadeRegion.facade_id)
        .join(Capture, Capture.id == Finding.capture_id)
    )
    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(stmt)]
    out = lake / "findings"
    if out.exists():
        for f in out.rglob("*.parquet"):
            f.unlink()
    if rows:
        pq.write_to_dataset(
            pa.Table.from_pylist(rows),
            root_path=str(out),
            partition_cols=["building_id"],
            basename_template="findings-{i}.parquet",
        )
    return {"exported_findings": len(rows)}


def export_all(engine: Engine, lake: Path) -> dict[str, Any]:
    return {**export_telemetry(engine, lake), **export_findings(engine, lake)}


# ----------------------------------------------------------------------------- consultas DuckDB


def _connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=":memory:")


def _has(lake: Path, dataset: str) -> bool:
    return any((lake / dataset).rglob("*.parquet")) if (lake / dataset).exists() else False


def _glob(lake: Path, dataset: str) -> str:
    return str(lake / dataset / "**" / "*.parquet").replace("'", "''")


def query(lake: Path, dataset: str, sql_body: str) -> list[dict[str, Any]]:
    """Executa SQL sobre a view ``t`` (dataset Parquet). Corpo SQL é fixo no código, nunca do usuário."""
    if not _has(lake, dataset):
        return []
    con = _connect()
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{_glob(lake, dataset)}', hive_partitioning=true)")
    cur = con.execute(sql_body)
    names = [d[0] for d in cur.description or []]
    return [dict(zip(names, row, strict=True)) for row in cur.fetchall()]


def telemetry_by_mission(lake: Path) -> list[dict[str, Any]]:
    return query(
        lake,
        "telemetry",
        """
        SELECT mission_id, count(*) AS eventos, min(captured_at) AS inicio, max(captured_at) AS fim,
               round(min(battery_pct), 1) AS bateria_min, round(avg(battery_pct), 1) AS bateria_media,
               round(quantile_cont(signal_quality, 0.05), 1) AS sinal_p05,
               round(avg(signal_quality), 1) AS sinal_medio, round(max(altitude_m), 1) AS altitude_max,
               round(median(epoch_ms(received_at) - epoch_ms(captured_at)), 1) AS latencia_mediana_ms
        FROM t GROUP BY mission_id ORDER BY inicio""",
    )


def telemetry_distribution(lake: Path) -> list[dict[str, Any]]:
    return query(
        lake,
        "telemetry",
        """
        SELECT mission_id, floor(battery_pct / 10) * 10 AS faixa_bateria,
               floor(signal_quality / 10) * 10 AS faixa_sinal, count(*) AS eventos
        FROM t GROUP BY ALL ORDER BY ALL""",
    )


def events_per_day(lake: Path) -> list[dict[str, Any]]:
    return query(
        lake,
        "telemetry",
        """
        SELECT dt, count(*) AS eventos, count(DISTINCT mission_id) AS missoes,
               count(DISTINCT drone_id) AS drones FROM t GROUP BY dt ORDER BY dt""",
    )


def anomalies_by_facade(lake: Path) -> list[dict[str, Any]]:
    return query(
        lake,
        "findings",
        """
        SELECT facade_name, region_code, CAST(inspected_at AS DATE) AS data_inspecao,
               count(*) AS achados, round(sum(area_ratio) * 100, 2) AS area_total_pct,
               round(avg(severity_points), 1) AS severidade_media,
               count(*) FILTER (WHERE severity_level IN ('alta', 'critica')) AS achados_alta_ou_critica
        FROM t GROUP BY ALL ORDER BY facade_name, region_code, data_inspecao""",
    )


def findings_by_category(lake: Path) -> list[dict[str, Any]]:
    return query(
        lake,
        "findings",
        """
        SELECT candidate_category AS categoria, severity_level AS severidade, count(*) AS achados
        FROM t GROUP BY ALL ORDER BY ALL""",
    )


def lake_stats(lake: Path) -> dict[str, Any]:
    files = list(lake.rglob("*.parquet")) if lake.exists() else []
    return {"parquet_files": len(files), "bytes": sum(f.stat().st_size for f in files), **_state(lake)}
