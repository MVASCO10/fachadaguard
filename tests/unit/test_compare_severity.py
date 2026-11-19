"""Testes mínimos 7 e 8 — comparação temporal (incl. falha de alinhamento) e severidade explicável."""

from __future__ import annotations

import numpy as np
import pytest

from fachadaguard.domain.severity import Growth, assess, explain, level_for
from fachadaguard.vision.compare import compare
from fachadaguard.vision.synthetic import Condition, DefectSpec, FacadeStyle, render

STYLE = FacadeStyle(seed=42, windows=True)
SPEC = DefectSpec("crack", 7, 0.45, 0.5, 1.1)


def test_aligned_pair_detects_growth_where_defect_grew():
    before = render(STYLE, [(SPEC, 0.4)], Condition(seed=1))
    after = render(STYLE, [(SPEC, 1.0)], Condition(dx=10, dy=-6, rotation_deg=1.2, gain=0.92, seed=2))
    res = compare(before.image, after.image)
    assert res.status == "ok" and res.alignment.method == "orb_homography"
    assert res.alignment.metrics["inliers"] >= 30
    grown = (after.mask > 0) & ~(np.roll(before.mask, 0) > 0)
    assert res.change_map[grown].mean() > 3 * res.change_map[res.valid_mask & ~grown].mean()


def test_identical_images_have_no_change():
    img = render(STYLE).image
    res = compare(img, img)
    assert res.status == "ok" and res.ssim > 0.99 and res.changed_area_ratio < 0.001


def test_incompatible_images_fail_safely():
    a = render(STYLE).image
    b = render(FacadeStyle(seed=999, palette="terracota", panel_w=120, panel_h=60)).image
    res = compare(a, b)
    assert res.status == "alignment_failed"
    assert res.change_map is None and res.ssim is None
    assert "suspensa" in res.warnings[0]


def test_uniform_image_fails_alignment():
    flat = np.full((480, 640, 3), 128, np.uint8)
    res = compare(flat, flat.copy())
    assert res.status == "alignment_failed"


def test_severity_components_and_levels():
    low = assess(1.1, 0.002, Growth(has_history=False), 0.9, "indeterminado", "baixa")
    high = assess(3.2, 0.05, Growth(has_history=True, growth_ratio=1.5), 0.95, "desplacamento_aparente", "alta")
    assert low.level == "baixa" and not low.should_alert
    assert high.level in ("alta", "critica") and high.should_alert
    names = [c["fator"] for c in high.contributions]
    assert names == [
        "intensidade_anomalia",
        "area_relativa",
        "crescimento_temporal",
        "categoria_candidata",
        "qualidade_evidencia",
    ]
    raw = sum(c["pontos"] for c in high.contributions if c["pontos"] is not None)
    assert high.points == pytest.approx(raw * (0.6 + 0.4 * 0.95), abs=0.11)
    assert "Principais fatores" in explain(high)


def test_low_quality_caps_level_and_explains():
    r = assess(3.5, 0.08, Growth(has_history=True, is_new=True), 0.2, "desplacamento_aparente", "alta")
    assert r.level == "media" and not r.should_alert
    assert any("baixa qualidade" in n for n in r.notes)


def test_new_finding_gets_growth_points_and_history_note():
    new = assess(2.0, 0.01, Growth(has_history=True, is_new=True), 1.0, "fissura", "media")
    none = assess(2.0, 0.01, Growth(has_history=False), 1.0, "fissura", "media")
    assert new.points > none.points
    assert any("Sem histórico" in n for n in none.notes)


@pytest.mark.parametrize("pts,level", [(0, "baixa"), (29.9, "baixa"), (30, "media"), (55, "alta"), (75, "critica")])
def test_level_boundaries(pts, level):
    assert level_for(pts) == level
