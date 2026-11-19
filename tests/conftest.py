from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from fachadaguard.config import Settings
from fachadaguard.db import create_schema, init_engine, session_scope
from fachadaguard.vision.anomaly import AnomalyConfig, PatchAnomalyModel
from fachadaguard.vision.dataset import reference_samples

PREFIX = "fachadaguard/v1"


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        api_token="test-token",
        environment="test",
    )
    init_engine(s.database_url)
    create_schema()
    return s


@pytest.fixture()
def session(settings: Settings):
    with session_scope() as s:
        yield s


@pytest.fixture(scope="session")
def model() -> PatchAnomalyModel:
    """Modelo reduzido (45 referências) para testes rápidos e determinísticos."""
    refs = [s.rendered.image for s in reference_samples(45)]
    return PatchAnomalyModel(AnomalyConfig(knn_bank_size=12_000, seed=42)).fit(refs, reference_hash="test")


def png_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()
