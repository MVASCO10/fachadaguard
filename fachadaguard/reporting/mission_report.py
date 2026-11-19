"""Relatório PRELIMINAR por missão: HTML autocontido, gerado de forma determinística por template."""

from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from fachadaguard import DISCLAIMER, __version__
from fachadaguard.config import Settings
from fachadaguard.domain.models import (
    Alert,
    Building,
    Capture,
    Comparison,
    Facade,
    FacadeRegion,
    Finding,
    Inspection,
    JobStatus,
    Mission,
    ModelRun,
    Report,
    TelemetryEvent,
)
from fachadaguard.reporting.genai import generate_summary
from fachadaguard.storage.images import LocalBlobStore

TEMPLATES = Path(__file__).parent / "templates"
REVIEW_LABEL = {
    "pendente": "Pendente",
    "confirmado": "Confirmado",
    "descartado": "Descartado",
    "encaminhado_inspecao_tecnica": "Encaminhado para inspeção técnica",
}
LEVEL_ORDER = {"critica": 0, "alta": 1, "media": 2, "baixa": 3}
CATEGORY_LABEL = {
    "fissura": "Fissura (candidata)",
    "umidade_infiltracao": "Umidade/infiltração (candidata)",
    "desplacamento_aparente": "Desplacamento aparente (candidato)",
    "indeterminado": "Indeterminado",
}


def _thumb(store: LocalBlobStore, key: str | None, width: int = 440) -> str | None:
    if not key or not store.exists(key):
        return None
    img = store.read_image(key)
    scale = width / img.shape[1]
    img = cv2.resize(img, (width, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


def _crop(store: LocalBlobStore, key: str | None, f: Finding, pad: int = 24) -> str | None:
    if not key or not store.exists(key):
        return None
    img = store.read_image(key)
    h, w = img.shape[:2]
    x0, y0 = max(0, f.bbox_x - pad), max(0, f.bbox_y - pad)
    x1, y1 = min(w, f.bbox_x + f.bbox_w + pad), min(h, f.bbox_y + f.bbox_h + pad)
    crop = np.ascontiguousarray(img[y0:y1, x0:x1])
    cv2.rectangle(
        crop, (f.bbox_x - x0, f.bbox_y - y0), (f.bbox_x - x0 + f.bbox_w, f.bbox_y - y0 + f.bbox_h), (0, 0, 255), 2
    )
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


def telemetry_summary(session: Session, mission_id: str) -> dict[str, Any]:
    T = TelemetryEvent
    row = session.execute(
        select(
            func.count(),
            func.min(T.captured_at),
            func.max(T.captured_at),
            func.max(T.battery_pct),
            func.min(T.battery_pct),
            func.min(T.signal_quality),
            func.avg(T.signal_quality),
            func.min(T.altitude_m),
            func.max(T.altitude_m),
            func.min(T.latitude),
            func.max(T.latitude),
            func.min(T.longitude),
            func.max(T.longitude),
        ).where(T.mission_id == mission_id)
    ).one()
    n, t0, t1, b_max, b_min, s_min, s_avg, a_min, a_max, lat0, lat1, lon0, lon1 = row
    if not n:
        return {"events": 0}
    return {
        "events": int(n),
        "start": t0,
        "end": t1,
        "duration_min": round((t1 - t0).total_seconds() / 60, 1) if t0 and t1 else None,
        "battery_start_max": round(b_max, 1),
        "battery_min": round(b_min, 1),
        "signal_min": round(s_min, 1),
        "signal_avg": round(float(s_avg), 1),
        "altitude_min": round(a_min, 1),
        "altitude_max": round(a_max, 1),
        "bbox": [round(lat0, 6), round(lon0, 6), round(lat1, 6), round(lon1, 6)],
    }


def collect_facts(session: Session, store: LocalBlobStore, mission_id: str, images: bool = True) -> dict[str, Any]:
    mission = session.get(Mission, mission_id)
    if mission is None:
        raise LookupError("missão não encontrada")
    building = session.get(Building, mission.building_id)
    assert building is not None
    n_regions = (
        session.scalar(select(func.count(FacadeRegion.id)).join(Facade).where(Facade.building_id == building.id)) or 0
    )
    inspections = session.scalars(
        select(Inspection).where(Inspection.mission_id == mission_id).order_by(Inspection.inspected_at)
    ).all()
    items: list[dict[str, Any]] = []
    all_findings: list[Finding] = []
    for insp in inspections:
        region = session.get(FacadeRegion, insp.region_id)
        assert region is not None
        facade = session.get(Facade, region.facade_id)
        cap = session.scalar(select(Capture).where(Capture.inspection_id == insp.id).order_by(Capture.captured_at))
        run = (
            session.scalar(
                select(ModelRun)
                .where(ModelRun.capture_id == cap.id, ModelRun.status == JobStatus.SUCCEEDED)
                .order_by(ModelRun.finished_at.desc())
            )
            if cap
            else None
        )
        findings = session.scalars(select(Finding).where(Finding.model_run_id == run.id)).all() if run else []
        findings = sorted(findings, key=lambda f: (LEVEL_ORDER.get(f.severity_level, 9), -f.severity_points))
        all_findings.extend(findings)
        cmp = session.scalar(
            select(Comparison).where(Comparison.inspection_after_id == insp.id).order_by(Comparison.created_at.desc())
        )
        items.append(
            {
                "inspection": insp,
                "region": region,
                "facade": facade,
                "capture": cap,
                "run": run,
                "findings": [
                    {
                        "f": f,
                        "crop": _crop(store, cap.storage_key if cap else None, f) if images else None,
                        "label": CATEGORY_LABEL.get(f.candidate_category, f.candidate_category),
                    }
                    for f in findings
                ],
                "overlay": _thumb(store, run.overlay_key if run else None) if images else None,
                "comparison": cmp,
                "comparison_overlay": _thumb(store, cmp.overlay_key if cmp else None) if images else None,
            }
        )
    alerts = session.scalars(select(Alert).where(Alert.mission_id == mission_id)).all()
    levels = {lvl: sum(1 for f in all_findings if f.severity_level == lvl) for lvl in LEVEL_ORDER}
    cats = {c: sum(1 for f in all_findings if f.candidate_category == c) for c in CATEGORY_LABEL}
    reviews = {
        s: sum(1 for f in all_findings if f.review_status == s)
        for s in ("pendente", "confirmado", "descartado", "encaminhado_inspecao_tecnica")
    }
    analyzed = sum(1 for i in items if i["run"])
    return {
        "mission": mission,
        "building": building,
        "items": items,
        "alerts": alerts,
        "telemetry": telemetry_summary(session, mission_id),
        "coverage": {
            "regions_inspected": len(inspections),
            "regions_total": int(n_regions),
            "captures_analyzed": analyzed,
        },
        "levels": levels,
        "categories": cats,
        "reviews": reviews,
        "n_findings": len(all_findings),
        "model_versions": sorted({i["run"].model_version for i in items if i["run"]}),
        "synthetic": bool(building.is_synthetic or mission.is_simulated),
    }


def structured_facts(facts: dict[str, Any]) -> dict[str, Any]:
    """Subconjunto serializável usado pelo resumo e pelo adaptador generativo."""
    return {
        "missao": facts["mission"].name,
        "edificio": facts["building"].name,
        "dados_sinteticos": facts["synthetic"],
        "regioes_inspecionadas": facts["coverage"]["regions_inspected"],
        "regioes_cadastradas": facts["coverage"]["regions_total"],
        "capturas_analisadas": facts["coverage"]["captures_analyzed"],
        "achados_total": facts["n_findings"],
        "achados_por_severidade": facts["levels"],
        "achados_por_categoria_candidata": facts["categories"],
        "alertas": len(facts["alerts"]),
        "revisao_humana": facts["reviews"],
        "telemetria_eventos": facts["telemetry"].get("events", 0),
        "bateria_minima_pct": facts["telemetry"].get("battery_min"),
    }


def template_summary(facts: dict[str, Any]) -> str:
    s = structured_facts(facts)
    lv = s["achados_por_severidade"]
    high = lv["critica"] + lv["alta"]
    parts = [
        f"A missão inspecionou {s['regioes_inspecionadas']} de {s['regioes_cadastradas']} regiões cadastradas "
        f"e analisou {s['capturas_analisadas']} capturas.",
        f"O detector de anomalia apontou {s['achados_total']} regiões candidatas, das quais {high} com "
        f"severidade preliminar alta ou crítica, gerando {s['alertas']} alertas para revisão.",
        f"Revisão humana: {s['revisao_humana']['pendente']} pendentes, {s['revisao_humana']['confirmado']} "
        f"confirmados, {s['revisao_humana']['descartado']} descartados e "
        f"{s['revisao_humana']['encaminhado_inspecao_tecnica']} encaminhados para inspeção técnica.",
    ]
    if s["dados_sinteticos"]:
        parts.append("Os dados desta missão são SINTÉTICOS e servem apenas para demonstrar o funcionamento.")
    return " ".join(parts)


def render_html(facts: dict[str, Any], summary: str, summary_source: str, generated_at: datetime) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    env.filters["pct"] = lambda v, d=1: "—" if v is None else f"{v * 100:.{d}f}%"
    env.filters["dt"] = lambda v: "—" if v is None else v.astimezone(UTC).strftime("%d/%m/%Y %H:%M UTC")
    return env.get_template("mission_report.html").render(
        **facts,
        summary=summary,
        summary_source=summary_source,
        generated_at=generated_at,
        disclaimer=DISCLAIMER,
        version=__version__,
        category_label=CATEGORY_LABEL,
        review_label=REVIEW_LABEL,
    )


def generate_report(session: Session, store: LocalBlobStore, settings: Settings, mission_id: str) -> Report:
    facts = collect_facts(session, store, mission_id)
    summary = template_summary(facts)
    source = "template determinístico"
    generator = "template"
    if settings.genai_enabled:
        text, reason = generate_summary(settings, structured_facts(facts))
        if text:
            summary, source, generator = text, reason, "template+genai"
        else:
            source = f"template determinístico ({reason})"
    now = datetime.now(UTC)
    html = render_html(facts, summary, source, now)
    data = html.encode("utf-8")
    report = Report(
        mission_id=mission_id,
        generator=generator,
        storage_key="pending",
        sha256=hashlib.sha256(data).hexdigest(),
        summary={**structured_facts(facts), "resumo": summary, "fonte_resumo": source},
    )
    session.add(report)
    session.flush()
    report.storage_key = f"reports/{report.id.lower()}.html"
    store.put(report.storage_key, data)
    return report
