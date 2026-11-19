"""Testes mínimos 9, 10 e 11 — fluxo principal em processo (API + ingestão + worker + lake + relatório).

Usa SQLite e armazenamento temporário; a mesma lógica roda em PostgreSQL na stack Docker
(ver tests/e2e para o smoke test contra os contêineres).
"""

from __future__ import annotations

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from fachadaguard.analytics import lake
from fachadaguard.db import get_engine, session_scope
from fachadaguard.ingestion import process_batch
from fachadaguard.reporting.genai import validate_summary
from fachadaguard.storage.images import LocalBlobStore
from fachadaguard.vision.dataset import DEMO_REGIONS, demo_region_image
from services.ai_worker.main import process_one
from tests.conftest import PREFIX, png_bytes

H = {"X-API-Token": "test-token"}


@pytest.fixture()
def api(settings):
    return TestClient(create_app(settings))


def _capture_msg(mission: str, region: dict, epoch: str, data: bytes) -> tuple[str, bytes]:
    return (
        f"{PREFIX}/missions/{mission}/captures",
        json.dumps(
            {
                "event_id": f"cap-{mission}-{region['code']}",
                "schema_version": "1.0",
                "drone_id": "drn-t",
                "mission_id": mission,
                "captured_at": "2026-05-12T13:05:00Z" if epoch == "A" else "2026-11-10T13:05:00Z",
                "building_id": "bld-demo-atlantico",
                "facade_id": region["facade_id"],
                "region_code": region["code"],
                "latitude": -30.05,
                "longitude": -51.26,
                "altitude_m": 8,
                "heading_deg": 180,
                "image_sha256": hashlib.sha256(data).hexdigest(),
                "image_size_bytes": len(data),
                "content_type": "image/png",
                "simulated": True,
            }
        ).encode(),
    )


def _fly(api, settings, mission: str, epoch: str, regions: list[dict]) -> None:
    r = api.post(
        "/api/v1/missions",
        headers=H,
        json={
            "id": mission,
            "building_id": "bld-demo-atlantico",
            "drone_id": "drn-t",
            "name": f"Missão {epoch}",
            "is_simulated": True,
        },
    )
    assert r.status_code == 201, r.text
    tel = [
        (
            f"{PREFIX}/drones/drn-t/telemetry",
            json.dumps(
                {
                    "event_id": f"tel-{mission}-{i:05d}",
                    "schema_version": "1.0",
                    "drone_id": "drn-t",
                    "mission_id": mission,
                    "captured_at": f"2026-{'05-12' if epoch == 'A' else '11-10'}T13:{i // 60:02d}:{i % 60:02d}Z",
                    "latitude": -30.05,
                    "longitude": -51.26,
                    "altitude_m": 10 + i % 7,
                    "battery_pct": 98 - i * 0.05,
                    "speed_mps": 2.5,
                    "heading_deg": 90,
                    "signal_quality": 70 + i % 20,
                    "mission_state": "inspecting",
                    "simulated": True,
                }
            ).encode(),
        )
        for i in range(120)
    ]
    with session_scope() as s:
        process_batch(s, tel, PREFIX)
    for region in regions:
        data = png_bytes(demo_region_image(region, epoch).image)
        eid = f"cap-{mission}-{region['code']}"
        up = api.post(
            f"/api/v1/captures/{eid}/image",
            headers=H,
            files={"file": ("x.png", data, "image/png")},
            data={"mission_id": mission, "captured_at": "2026-05-12T13:05:00Z", "drone_id": "drn-t"},
        )
        assert up.status_code == 200, up.text
        with session_scope() as s:
            res = process_batch(s, [_capture_msg(mission, region, epoch, data)], PREFIX)
            assert res.jobs_enqueued == 1


def test_end_to_end_flow(api, settings, model):
    assert api.get("/health").json()["status"] == "ok"
    assert api.post("/api/v1/demo/seed").status_code == 401  # escrita exige token
    assert api.post("/api/v1/demo/seed", headers=H).json()["created"]["regions"] == 6
    regions = [DEMO_REGIONS[0], DEMO_REGIONS[3]]  # N-01 (fissura cresce) e L-01 (desplacamento novo)
    store = LocalBlobStore(settings.data_dir / "store")

    _fly(api, settings, "msn-a", "A", regions)
    while process_one(model, store):
        pass
    _fly(api, settings, "msn-b", "B", regions)
    while process_one(model, store):
        pass

    runs = api.get("/api/v1/analyses").json()
    assert len(runs) == 4 and all(r["status"] == "succeeded" for r in runs)
    assert all(r["model_version"] == model.version for r in runs)
    detail = api.get("/api/v1/missions/msn-b").json()
    assert len(detail["inspections"]) == 2 and detail["findings"]

    # comparação automática gerada pelo worker para a inspeção B, com alinhamento validado
    cmps = api.get("/api/v1/comparisons").json()
    assert len(cmps) == 2 and all(c["status"] == "ok" for c in cmps)
    l01 = [f for f in api.get("/api/v1/findings", params={"mission_id": "msn-b"}).json() if f["region_code"] == "L-01"]
    assert l01, "desplacamento novo em L-01 deveria gerar achado"
    top = max(l01, key=lambda f: f["severity_points"])
    full = api.get(f"/api/v1/findings/{top['id']}").json()
    assert full["severity_explanation"]["has_history"] is True
    assert {c["fator"] for c in full["severity_explanation"]["contributions"]} >= {"crescimento_temporal"}

    # revisão humana com histórico
    r = api.post(
        f"/api/v1/findings/{top['id']}/review",
        headers=H,
        json={"status": "encaminhado_inspecao_tecnica", "note": "verificar in loco"},
    )
    assert r.status_code == 200 and r.json()["reviews"][0]["previous_status"] == "pendente"
    assert (
        api.get("/api/v1/findings", params={"review_status": "encaminhado_inspecao_tecnica"}).json()[0]["id"]
        == top["id"]
    )

    # comparação manual: incompatível sem force é recusada; mesma região é aceita
    insp = api.get("/api/v1/inspections").json()
    a = next(i for i in insp if i["region_code"] == "N-01" and i["mission_id"] == "msn-a")
    b = next(i for i in insp if i["region_code"] == "N-01" and i["mission_id"] == "msn-b")
    other = next(i for i in insp if i["region_code"] == "L-01" and i["mission_id"] == "msn-b")
    assert (
        api.post(
            "/api/v1/comparisons", headers=H, json={"before_inspection_id": a["id"], "after_inspection_id": other["id"]}
        ).status_code
        == 422
    )
    ok = api.post(
        "/api/v1/comparisons", headers=H, json={"before_inspection_id": a["id"], "after_inspection_id": b["id"]}
    ).json()
    assert ok["status"] == "ok" and ok["interval_days"] > 150
    assert api.get(ok["overlay_url"]).status_code == 200

    # arquivos: path traversal bloqueado
    assert api.get("/api/v1/files/captures/../../test.db").status_code in (400, 404)

    # Big Data: export incremental para Parquet e consulta DuckDB (teste mínimo 9)
    first = lake.export_all(get_engine(), settings.lake_dir)
    assert first["exported_rows"] == 240 and first["exported_findings"] > 0
    assert lake.export_telemetry(get_engine(), settings.lake_dir)["exported_rows"] == 0  # incremental
    by_mission = {r["mission_id"]: r for r in api.get("/api/v1/analytics/telemetry-by-mission").json()}
    assert by_mission["msn-a"]["eventos"] == 120 and by_mission["msn-b"]["bateria_min"] == pytest.approx(
        92.05, abs=0.06
    )
    assert sorted(p.name for p in (settings.lake_dir / "telemetry").iterdir()) == ["dt=2026-05-12", "dt=2026-11-10"]
    assert api.get("/api/v1/analytics/anomalies-by-facade").json()

    # relatório preliminar (teste mínimo 10)
    rep = api.post("/api/v1/missions/msn-b/reports", headers=H).json()
    html = api.get(rep["url"]).text
    assert "Relatório preliminar" in html and "Não é laudo" in html and "Dados sintéticos" in html
    assert rep["summary"]["achados_total"] == len(detail["findings"])
    assert rep["generator"] == "template"

    m = api.get("/api/v1/metrics/summary").json()
    assert m["telemetry_events"] == 240 and m["captures_analyzed"] == 4


def test_genai_output_validation_rejects_invented_numbers_and_terms():
    facts = {"achados_total": 5, "alertas": 2, "regioes_inspecionadas": 6}
    assert validate_summary("Foram identificadas 5 regiões candidatas e 2 alertas em 6 regiões.", facts) == []
    assert any("números" in p for p in validate_summary("Foram 7 fissuras.", facts))
    assert any("proibido" in p for p in validate_summary("O laudo indica 5 achados.", facts))


def test_manual_upload_validation(api):
    api.post("/api/v1/demo/seed", headers=H)
    api.post(
        "/api/v1/missions",
        headers=H,
        json={"id": "msn-m", "building_id": "bld-demo-atlantico", "drone_id": "drn-t", "name": "Manual"},
    )
    bad = api.post(
        "/api/v1/missions/msn-m/images",
        headers=H,
        files={"file": ("x.png", b"oi", "image/png")},
        data={"region_id": "reg-demo-n-01"},
    )
    assert bad.status_code == 422
    good = api.post(
        "/api/v1/missions/msn-m/images",
        headers=H,
        files={"file": ("x.png", png_bytes(demo_region_image(DEMO_REGIONS[1], "A").image), "image/png")},
        data={"region_id": "reg-demo-n-01"},
    )
    assert good.status_code == 201 and good.json()["analysis"]["status"] == "queued"
