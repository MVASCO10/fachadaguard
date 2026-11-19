"""Testes mínimos 5 e 6 — treino/inferência do baseline e mapa de calor/máscara."""

from __future__ import annotations

import numpy as np
import pytest

from fachadaguard.vision.anomaly import PatchAnomalyModel, heatmap_overlay
from fachadaguard.vision.categorize import categorize, categorize_components
from fachadaguard.vision.features import N_FEATURES, extract_patch_features
from fachadaguard.vision.quality import image_quality
from fachadaguard.vision.synthetic import DefectSpec, FacadeStyle, render


def test_feature_shapes_and_finiteness():
    img = render(FacadeStyle(seed=3)).image
    feats, ys, xs = extract_patch_features(img, 32, 16)
    assert feats.shape == (len(ys), len(xs), N_FEATURES)
    assert np.isfinite(feats).all()
    assert np.allclose(feats[..., 6:15].sum(-1), 1, atol=1e-3)  # histograma de orientação normalizado


def test_model_metadata_is_versioned(model):
    md = model.metadata
    assert md["model_version"].startswith("knn-")
    assert md["reference_set_hash"] == "test" and md["n_calibration_images"] > 0
    assert md["params"]["seed"] == 42 and model.threshold > model.cal_median


def test_training_is_deterministic(model):
    from fachadaguard.vision.dataset import reference_samples

    refs = [s.rendered.image for s in reference_samples(45)]
    again = PatchAnomalyModel(model.config).fit(refs, reference_hash="test")
    assert again.version == model.version
    assert again.threshold == pytest.approx(model.threshold)


def test_normal_image_mostly_below_threshold(model):
    res = model.analyze(render(FacadeStyle(seed=777, palette="cinza_claro", windows=False)).image)
    assert res.index_map.shape == (480, 640)
    assert res.anomalous_area_ratio < 0.02


@pytest.mark.parametrize("kind", ["crack", "spall"])
def test_defect_is_localized(model, kind):
    r = render(FacadeStyle(seed=778, palette="bege", windows=False), [(DefectSpec(kind, 11, 0.5, 0.5, 1.2), 1.0)])
    res = model.analyze(r.image)
    gt = r.mask > 0
    assert res.image_index > 1.0
    assert res.index_map[gt].mean() > 1.5 * res.index_map[~gt].mean()
    assert res.components, "esperava ao menos uma região candidata"
    top = res.components[0]["mask"]
    assert (top & gt).sum() > 0, "região mais anômala deve sobrepor o defeito"


def test_inference_is_deterministic(model):
    img = render(FacadeStyle(seed=5), [(DefectSpec("crack", 2, 0.5, 0.5), 1.0)]).image
    a, b = model.analyze(img), model.analyze(img)
    assert np.array_equal(a.index_map, b.index_map) and np.array_equal(a.mask, b.mask)


def test_overlay_and_mask_properties(model):
    img = render(FacadeStyle(seed=6), [(DefectSpec("spall", 3, 0.5, 0.5, 1.3), 1.0)]).image
    res = model.analyze(img)
    ov = heatmap_overlay(img, res.index_map, res.mask)
    assert ov.shape == img.shape and ov.dtype == np.uint8
    assert set(np.unique(res.mask)) <= {0, 1}
    assert all(c["area_px"] >= model.config.min_component_px for c in res.components)


def test_categorization_is_explained_and_bounded(model):
    img = render(FacadeStyle(seed=9, windows=False), [(DefectSpec("spall", 4, 0.5, 0.5, 1.3), 1.0)]).image
    res = model.analyze(img)
    cats = categorize_components(res.maps, res.components)
    assert cats and all(0 <= v <= 1 for c in cats for v in c.scores.values())
    assert cats[0].category == "desplacamento_aparente"
    assert "Não é diagnóstico" in cats[0].explanation or cats[0].category == "indeterminado"


def test_ambiguous_measures_yield_indeterminate():
    r = categorize({"line_frac": 0.0, "darkness": 0.0, "texture_ratio": 1.0, "chroma_drop": 0.0, "elongation": 1.0})
    assert r.category == "indeterminado" and r.confidence == "baixa"


def test_quality_penalizes_blur():
    import cv2

    img = render(FacadeStyle(seed=10)).image
    sharp = image_quality(img)["quality_score"]
    blurred = image_quality(cv2.GaussianBlur(img, (0, 0), 4))["quality_score"]
    assert blurred < sharp
