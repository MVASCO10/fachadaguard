"""Teste mínimo 4 — geração reproduzível do dataset sintético."""

from __future__ import annotations

import hashlib

import numpy as np

from fachadaguard.vision.dataset import (
    DEMO_REGIONS,
    demo_region_image,
    labeled_samples,
    reference_samples,
    verify_dataset,
    write_dataset,
)
from fachadaguard.vision.synthetic import DefectSpec, FacadeStyle, render


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()


def test_same_seed_same_pixels():
    a = labeled_samples("dev", 6)
    b = labeled_samples("dev", 6)
    assert [digest(s.rendered.image) for s in a] == [digest(s.rendered.image) for s in b]
    assert [digest(s.rendered.mask) for s in a] == [digest(s.rendered.mask) for s in b]


def test_splits_are_disjoint():
    ref = {digest(s.rendered.image) for s in reference_samples(10)}
    dev = {digest(s.rendered.image) for s in labeled_samples("dev", 10)}
    test = {digest(s.rendered.image) for s in labeled_samples("test", 10)}
    assert not (ref & dev) and not (dev & test) and not (ref & test)


def test_reference_has_no_defects_and_masks_match_metadata():
    assert all(not s.rendered.mask.any() for s in reference_samples(10))
    for s in labeled_samples("test", 12):
        kinds = {d["kind"] for d in s.meta["defects"]}
        labels = set(np.unique(s.rendered.mask)) - {0}
        expected = {{"crack": 1, "stain": 2, "spall": 3}[k] for k in kinds}
        assert labels <= expected  # nenhum rótulo sem defeito correspondente
        assert bool(labels) == bool(kinds)  # imagens com defeito têm máscara não vazia


def test_defect_growth_is_monotonic():
    style = FacadeStyle(seed=1)
    spec = DefectSpec("stain", 5, 0.5, 0.5)
    small = render(style, [(spec, 0.3)]).mask.astype(bool).sum()
    large = render(style, [(spec, 1.0)]).mask.astype(bool).sum()
    assert large > small > 0


def test_demo_scenario_has_expected_story():
    n01 = DEMO_REGIONS[0]
    assert demo_region_image(n01, "B").mask.sum() > demo_region_image(n01, "A").mask.sum()
    l01 = DEMO_REGIONS[3]
    assert 3 not in demo_region_image(l01, "A").mask and 3 in demo_region_image(l01, "B").mask
    assert not demo_region_image(DEMO_REGIONS[1], "B").mask.any()


def test_write_and_verify_manifest(tmp_path):
    manifest = write_dataset(tmp_path, n_reference=6, n_dev=3, n_test=3)
    assert manifest["synthetic"] is True and len(manifest["reference_set_hash"]) == 64
    assert verify_dataset(tmp_path) == []
    first = next(k for k in manifest["files"] if k.startswith("test/"))
    (tmp_path / first).write_bytes(b"corrompido")
    assert verify_dataset(tmp_path) == [f"checksum divergente: {first}"]
