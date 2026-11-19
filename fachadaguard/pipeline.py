"""Orquestração: captura → análise de IA → comparação temporal → severidade → achados/alertas."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from fachadaguard.domain.models import (
    Alert,
    Capture,
    Comparison,
    Finding,
    Inspection,
    JobStatus,
    ModelRun,
    utcnow,
)
from fachadaguard.domain.severity import Growth, assess, explain
from fachadaguard.storage.images import LocalBlobStore
from fachadaguard.vision.anomaly import MODEL_NAME, PatchAnomalyModel, heatmap_overlay
from fachadaguard.vision.categorize import RULES_VERSION, categorize_components
from fachadaguard.vision.compare import change_overlay, compare
from fachadaguard.vision.quality import image_quality

log = logging.getLogger(__name__)
NEW_FINDING_OVERLAP = 0.10


def enqueue_analysis(session: Session, capture: Capture, force: bool = False) -> ModelRun | None:
    """Cria job ``queued`` se a captura está completa (metadados + binário) e não há job ativo."""
    if not (capture.metadata_received and capture.blob_received):
        return None
    active = session.scalar(
        select(ModelRun).where(
            ModelRun.capture_id == capture.id,
            ModelRun.status.in_([JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.SUCCEEDED]),
        )
    )
    if active is not None and not (force and active.status in (JobStatus.SUCCEEDED,)):
        return None
    run = ModelRun(capture_id=capture.id, status=JobStatus.QUEUED)
    session.add(run)
    session.flush()
    return run


def claim_next_job(session: Session) -> ModelRun | None:
    """Reserva o job mais antigo. Em PostgreSQL usa FOR UPDATE SKIP LOCKED (vários workers seguros)."""
    stmt = select(ModelRun).where(ModelRun.status == JobStatus.QUEUED).order_by(ModelRun.queued_at).limit(1)
    if session.get_bind().dialect.name == "postgresql":
        stmt = stmt.with_for_update(skip_locked=True)
    run = session.scalar(stmt)
    if run is None:
        return None
    run.status = JobStatus.RUNNING
    run.started_at = utcnow()
    run.attempts += 1
    session.flush()
    return run


def previous_inspection(session: Session, inspection: Inspection) -> tuple[Inspection, ModelRun] | None:
    """Inspeção anterior da mesma região que já possui análise concluída."""
    stmt = (
        select(Inspection, ModelRun)
        .join(Capture, Capture.inspection_id == Inspection.id)
        .join(ModelRun, ModelRun.capture_id == Capture.id)
        .where(
            Inspection.region_id == inspection.region_id,
            Inspection.inspected_at < inspection.inspected_at,
            ModelRun.status == JobStatus.SUCCEEDED,
        )
        .order_by(Inspection.inspected_at.desc(), ModelRun.finished_at.desc())
        .limit(1)
    )
    row = session.execute(stmt).first()
    return (row[0], row[1]) if row else None


def _run_keys(run_id: str) -> dict[str, str]:
    rid = run_id.lower()
    return {
        "index": f"runs/{rid}_index.npy",
        "mask": f"runs/{rid}_mask.png",
        "heat": f"runs/{rid}_heat.png",
        "overlay": f"runs/{rid}_overlay.png",
    }


def save_comparison(
    session: Session,
    store: LocalBlobStore,
    before: Inspection,
    after: Inspection,
    img_before: np.ndarray,
    img_after: np.ndarray,
    run_before: ModelRun | None,
    run_after: ModelRun | None,
    extra_warnings: list[str] | None = None,
) -> tuple[Comparison, Any]:
    result = compare(img_before, img_after)
    cmp = Comparison(
        inspection_before_id=before.id,
        inspection_after_id=after.id,
        status=result.status,
        compatible=before.region_id == after.region_id,
        interval_days=round((after.inspected_at - before.inspected_at).total_seconds() / 86400, 2),
        alignment=result.alignment_dict(),
        ssim=result.ssim,
        changed_area_ratio=result.changed_area_ratio,
        score_before=run_before.image_score if run_before else None,
        score_after=run_after.image_score if run_after else None,
        anomalous_area_before=run_before.anomalous_area_ratio if run_before else None,
        anomalous_area_after=run_after.anomalous_area_ratio if run_after else None,
        warnings=[*(extra_warnings or []), *result.warnings],
    )
    session.add(cmp)
    session.flush()
    if result.status == "ok" and result.change_map is not None:
        assert result.change_mask is not None and result.aligned_before is not None
        cmp.overlay_key = f"comparisons/{cmp.id.lower()}_overlay.png"
        cmp.aligned_key = f"comparisons/{cmp.id.lower()}_aligned.png"
        store.put_image(cmp.overlay_key, change_overlay(img_after, result.change_map, result.change_mask))
        store.put_image(cmp.aligned_key, result.aligned_before)
    return cmp, result


def analyze_run(session: Session, run: ModelRun, model: PatchAnomalyModel, store: LocalBlobStore) -> None:
    t0 = time.perf_counter()
    capture = session.get(Capture, run.capture_id)
    assert capture is not None and capture.storage_key
    image = store.read_image(capture.storage_key)
    quality = image_quality(image)
    result = model.analyze(image)
    categories = categorize_components(result.maps, result.components)

    keys = _run_keys(run.id)
    buf = np.ascontiguousarray(result.index_map.astype(np.float16))
    store.put(keys["index"], buf.tobytes())
    store.put_image(keys["mask"], result.mask * 255)
    heat = cv2.applyColorMap((np.clip(result.index_map / 2, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    store.put_image(keys["heat"], heat)
    store.put_image(keys["overlay"], heatmap_overlay(image, result.index_map, result.mask))

    run.model_name = MODEL_NAME
    run.model_version = model.version
    run.params = {**model.metadata.get("params", {}), "categorization": RULES_VERSION}
    run.reference_set_hash = model.metadata.get("reference_set_hash")
    run.image_score = round(result.image_index, 4)
    run.threshold = 1.0
    run.anomalous_area_ratio = round(result.anomalous_area_ratio, 5)
    run.quality = quality
    run.heatmap_key, run.overlay_key = keys["heat"], keys["overlay"]

    # ---- histórico: alinhar a inspeção anterior e medir crescimento por componente
    inspection = session.get(Inspection, capture.inspection_id) if capture.inspection_id else None
    prev = previous_inspection(session, inspection) if inspection else None
    warped_prev_mask: np.ndarray | None = None
    history_note = ""
    if inspection and prev:
        prev_insp, prev_run = prev
        prev_cap = session.get(Capture, prev_run.capture_id)
        assert prev_cap is not None and prev_cap.storage_key
        prev_img = store.read_image(prev_cap.storage_key)
        _, cres = save_comparison(session, store, prev_insp, inspection, prev_img, image, prev_run, run)
        if cres.status == "ok":
            prev_mask = cv2.imdecode(np.frombuffer(store.get(_run_keys(prev_run.id)["mask"]), np.uint8), 0)
            h, w = image.shape[:2]
            warped_prev_mask = cv2.warpPerspective(
                (prev_mask > 0).astype(np.uint8), cres.alignment.homography, (w, h), flags=cv2.INTER_NEAREST
            )
        else:
            history_note = "alinhamento com a inspeção anterior falhou; crescimento não avaliado"

    h, w = image.shape[:2]
    for comp, cat in zip(result.components, categories, strict=True):
        if warped_prev_mask is not None:
            near = cv2.dilate(comp["mask"].astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
            prev_area = float(warped_prev_mask[near].sum())
            if prev_area < NEW_FINDING_OVERLAP * comp["area_px"]:
                growth = Growth(has_history=True, is_new=True)
            else:
                growth = Growth(has_history=True, growth_ratio=round((comp["area_px"] - prev_area) / prev_area, 3))
        else:
            growth = Growth(has_history=False)
        sev = assess(
            comp["peak_index"], comp["area_ratio"], growth, quality["quality_score"], cat.category, cat.confidence
        )
        explanation = sev.as_dict()
        if history_note:
            explanation["notes"].append(history_note)
        x, y, bw, bh = comp["bbox"]
        finding = Finding(
            model_run_id=run.id,
            capture_id=capture.id,
            inspection_id=capture.inspection_id,
            region_id=capture.region_id,
            bbox_x=x,
            bbox_y=y,
            bbox_w=bw,
            bbox_h=bh,
            area_px=comp["area_px"],
            area_ratio=round(comp["area_ratio"], 5),
            peak_score=round(comp["peak_index"], 4),
            score_ratio=round(comp["peak_index"], 4),
            candidate_category=cat.category,
            category_scores={"scores": cat.scores, "measures": cat.measures},
            category_confidence=cat.confidence,
            category_explanation=cat.explanation,
            growth_ratio=growth.growth_ratio if not growth.is_new else None,
            severity_points=sev.points,
            severity_level=sev.level,
            severity_explanation={
                **explanation,
                "is_new": growth.is_new,
                "has_history": growth.has_history,
                "summary": explain(sev),
            },
        )
        session.add(finding)
        session.flush()
        if sev.should_alert:
            session.add(
                Alert(finding_id=finding.id, mission_id=capture.mission_id, level=sev.level, reason=explain(sev))
            )
    run.status = JobStatus.SUCCEEDED
    run.finished_at = utcnow()
    run.duration_ms = round((time.perf_counter() - t0) * 1000, 1)
    run.error = None


def mark_failed(session: Session, run_id: str, exc: BaseException) -> None:
    run = session.get(ModelRun, run_id)
    if run is None:
        return
    run.status = JobStatus.FAILED
    run.finished_at = utcnow()
    run.error = f"{type(exc).__name__}: {str(exc)[:300]}"  # sanitizado: sem traceback nem caminhos completos


def interval_days(a: datetime, b: datetime) -> float:
    return round(abs((b - a).total_seconds()) / 86400, 2)
