"""Registro de modelos: carrega o modelo atual ou treina de forma determinística a partir das referências."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import cv2

from fachadaguard.vision.anomaly import AnomalyConfig, PatchAnomalyModel
from fachadaguard.vision.dataset import reference_samples

log = logging.getLogger(__name__)


def train_reference_model(models_dir: Path, config: AnomalyConfig | None = None) -> PatchAnomalyModel:
    samples = reference_samples()
    images = [s.rendered.image for s in samples]
    digests = sorted(hashlib.sha256(cv2.imencode(".png", im)[1].tobytes()).hexdigest() for im in images)
    ref_hash = hashlib.sha256("".join(digests).encode()).hexdigest()
    model = PatchAnomalyModel(config or AnomalyConfig()).fit(images, reference_hash=ref_hash)
    model.save(models_dir)
    log.info("modelo treinado: %s (referência %s)", model.version, ref_hash[:12])
    return model


def get_or_train(models_dir: Path) -> PatchAnomalyModel:
    """Reutiliza o modelo salvo somente se foi treinado com a configuração atual do código."""
    if (models_dir / "current.json").exists():
        try:
            model = PatchAnomalyModel.load_current(models_dir)
            if model.config == AnomalyConfig():
                return model
            log.info("configuração do modelo salvo difere da atual; retreinando")
        except (OSError, KeyError, ValueError) as exc:  # pragma: no cover - arquivo corrompido
            log.warning("modelo atual ilegível (%s); retreinando", exc)
    return train_reference_model(models_dir)
