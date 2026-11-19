"""Serialização explícita das entidades para JSON (evita expor campos internos como caminhos de disco)."""

from __future__ import annotations

from typing import Any

from fachadaguard.domain.models import (
    Alert,
    Building,
    Capture,
    Comparison,
    Facade,
    FacadeRegion,
    Finding,
    FindingReview,
    Inspection,
    Mission,
    ModelRun,
    Report,
)


def file_url(key: str | None) -> str | None:
    return f"/api/v1/files/{key}" if key else None


def building(b: Building, facades: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": b.id,
        "name": b.name,
        "address": b.address,
        "latitude": b.latitude,
        "longitude": b.longitude,
        "is_synthetic": b.is_synthetic,
        "created_at": b.created_at,
    }
    if facades:
        out["facades"] = [facade(f, regions=True) for f in b.facades]
    return out


def facade(f: Facade, regions: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": f.id,
        "building_id": f.building_id,
        "name": f.name,
        "orientation": f.orientation,
        "cladding": f.cladding,
    }
    if regions:
        out["regions"] = [region(r) for r in sorted(f.regions, key=lambda r: r.code)]
    return out


def region(r: FacadeRegion) -> dict[str, Any]:
    return {
        "id": r.id,
        "facade_id": r.facade_id,
        "code": r.code,
        "description": r.description,
        "center_latitude": r.center_latitude,
        "center_longitude": r.center_longitude,
        "center_altitude_m": r.center_altitude_m,
    }


def mission(m: Mission) -> dict[str, Any]:
    return {
        "id": m.id,
        "building_id": m.building_id,
        "drone_id": m.drone_id,
        "name": m.name,
        "status": m.status,
        "is_simulated": m.is_simulated,
        "planned_for": m.planned_for,
        "started_at": m.started_at,
        "ended_at": m.ended_at,
        "notes": m.notes,
        "created_at": m.created_at,
    }


def inspection(i: Inspection) -> dict[str, Any]:
    return {"id": i.id, "mission_id": i.mission_id, "region_id": i.region_id, "inspected_at": i.inspected_at}


def capture(c: Capture) -> dict[str, Any]:
    return {
        "id": c.id,
        "event_id": c.event_id,
        "mission_id": c.mission_id,
        "inspection_id": c.inspection_id,
        "region_id": c.region_id,
        "drone_id": c.drone_id,
        "captured_at": c.captured_at,
        "latitude": c.latitude,
        "longitude": c.longitude,
        "altitude_m": c.altitude_m,
        "heading_deg": c.heading_deg,
        "gimbal_pitch_deg": c.gimbal_pitch_deg,
        "sha256": c.sha256,
        "width": c.width,
        "height": c.height,
        "size_bytes": c.size_bytes,
        "metadata_received": c.metadata_received,
        "blob_received": c.blob_received,
        "is_synthetic": c.is_synthetic,
        "source": c.source,
        "image_url": file_url(c.storage_key),
    }


def run(r: ModelRun) -> dict[str, Any]:
    return {
        "id": r.id,
        "capture_id": r.capture_id,
        "status": r.status,
        "attempts": r.attempts,
        "model_name": r.model_name,
        "model_version": r.model_version,
        "params": r.params,
        "reference_set_hash": r.reference_set_hash,
        "image_index": r.image_score,
        "threshold_index": r.threshold,
        "anomalous_area_ratio": r.anomalous_area_ratio,
        "quality": r.quality,
        "heatmap_url": file_url(r.heatmap_key),
        "overlay_url": file_url(r.overlay_key),
        "duration_ms": r.duration_ms,
        "error": r.error,
        "queued_at": r.queued_at,
        "started_at": r.started_at,
        "finished_at": r.finished_at,
        "n_findings": len(r.findings) if r.status == "succeeded" else None,
    }


def review(v: FindingReview) -> dict[str, Any]:
    return {
        "id": v.id,
        "previous_status": v.previous_status,
        "status": v.status,
        "note": v.note,
        "reviewer": v.reviewer,
        "created_at": v.created_at,
    }


def finding(f: Finding, detail: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": f.id,
        "model_run_id": f.model_run_id,
        "capture_id": f.capture_id,
        "inspection_id": f.inspection_id,
        "region_id": f.region_id,
        "bbox": {"x": f.bbox_x, "y": f.bbox_y, "w": f.bbox_w, "h": f.bbox_h},
        "area_px": f.area_px,
        "area_ratio": f.area_ratio,
        "peak_index": f.peak_score,
        "candidate_category": f.candidate_category,
        "category_confidence": f.category_confidence,
        "growth_ratio": f.growth_ratio,
        "severity_points": f.severity_points,
        "severity_level": f.severity_level,
        "review_status": f.review_status,
        "created_at": f.created_at,
        "severity_summary": f.severity_explanation.get("summary"),
    }
    if detail:
        out.update(
            category_scores=f.category_scores,
            category_explanation=f.category_explanation,
            severity_explanation=f.severity_explanation,
            reviews=[review(v) for v in f.reviews],
        )
    return out


def alert(a: Alert) -> dict[str, Any]:
    return {
        "id": a.id,
        "finding_id": a.finding_id,
        "mission_id": a.mission_id,
        "level": a.level,
        "reason": a.reason,
        "status": a.status,
        "created_at": a.created_at,
    }


def comparison(c: Comparison) -> dict[str, Any]:
    return {
        "id": c.id,
        "inspection_before_id": c.inspection_before_id,
        "inspection_after_id": c.inspection_after_id,
        "status": c.status,
        "compatible": c.compatible,
        "interval_days": c.interval_days,
        "alignment": c.alignment,
        "ssim": c.ssim,
        "changed_area_ratio": c.changed_area_ratio,
        "index_before": c.score_before,
        "index_after": c.score_after,
        "anomalous_area_before": c.anomalous_area_before,
        "anomalous_area_after": c.anomalous_area_after,
        "warnings": c.warnings,
        "overlay_url": file_url(c.overlay_key),
        "aligned_before_url": file_url(c.aligned_key),
        "created_at": c.created_at,
    }


def report(r: Report) -> dict[str, Any]:
    return {
        "id": r.id,
        "mission_id": r.mission_id,
        "generator": r.generator,
        "sha256": r.sha256,
        "summary": r.summary,
        "generated_at": r.generated_at,
        "url": f"/api/v1/reports/{r.id}/html",
    }
