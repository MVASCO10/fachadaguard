"""Detector de anomalia visual one-class por patches, treinado só com referências NORMAIS.

Fluxo: patches → 31 características → padronização → detector one-class:

* ``knn`` (padrão): distância ao vizinho mais próximo num banco de patches normais (ideia do
  PatchCore, aqui com características clássicas). Lida bem com normalidade multimodal
  (painel, junta, vidro de janela);
* ``iforest`` (IsolationForest) e ``ocsvm`` (OneClassSVM): alternativas para comparação.
O limiar é o quantil ``threshold_quantile`` dos escores em referências normais de CALIBRAÇÃO
(não usadas no ajuste). O "índice de anomalia" é ``(s - mediana_cal) / (limiar - mediana_cal)``:
vale 1,0 no limiar. NÃO é uma probabilidade calibrada.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from fachadaguard.vision.features import FEATURE_NAMES, PixelMaps, extract_patch_features, pixel_maps

MODEL_NAME = "patch-handcrafted-oneclass"


@dataclass(frozen=True)
class AnomalyConfig:
    patch: int = 32
    stride: int = 16
    detector: str = "knn"  # knn | iforest | ocsvm
    knn_bank_size: int = 30_000
    knn_k: int = 1
    n_estimators: int = 300
    max_samples: int = 2048
    ocsvm_nu: float = 0.01
    ocsvm_gamma: str = "scale"
    max_train_patches: int = 60_000
    calibration_fraction: float = 0.2
    threshold_quantile: float = 0.99  # escolhido no split dev (relatório técnico, §IA)
    smooth_sigma: float = 6.0
    min_component_px: int = 500
    seed: int = 42


@dataclass
class AnomalyResult:
    index_map: np.ndarray  # float32 (h, w); 1.0 = limiar
    mask: np.ndarray  # uint8 0/1
    components: list[dict[str, Any]]
    image_index: float
    anomalous_area_ratio: float
    maps: PixelMaps


@dataclass
class PatchAnomalyModel:
    config: AnomalyConfig = field(default_factory=AnomalyConfig)
    scaler: StandardScaler | None = None
    detector: Any = None
    cal_median: float = 0.0
    threshold: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "untrained"))

    def _raw(self, feats: np.ndarray) -> np.ndarray:
        assert self.scaler is not None and self.detector is not None, "modelo não treinado"
        x = self.scaler.transform(feats.reshape(-1, feats.shape[-1]))
        if isinstance(self.detector, NearestNeighbors):
            dist, _ = self.detector.kneighbors(x)
            return dist.mean(axis=1)
        return -self.detector.score_samples(x)  # maior = mais anômalo

    def fit(self, images: list[np.ndarray], reference_hash: str = "") -> PatchAnomalyModel:
        if len(images) < 5:
            raise ValueError("são necessárias ao menos 5 imagens normais de referência")
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        order = rng.permutation(len(images))
        n_cal = max(1, int(round(len(images) * cfg.calibration_fraction)))
        cal_idx, fit_idx = order[:n_cal], order[n_cal:]
        fit_feats = np.concatenate(
            [
                extract_patch_features(images[i], cfg.patch, cfg.stride)[0].reshape(-1, len(FEATURE_NAMES))
                for i in fit_idx
            ]
        )
        if len(fit_feats) > cfg.max_train_patches:
            fit_feats = fit_feats[rng.choice(len(fit_feats), cfg.max_train_patches, replace=False)]
        self.scaler = StandardScaler().fit(fit_feats)
        x = self.scaler.transform(fit_feats)
        if cfg.detector == "knn":
            bank = x[rng.choice(len(x), min(len(x), cfg.knn_bank_size), replace=False)]
            self.detector = NearestNeighbors(n_neighbors=cfg.knn_k).fit(bank)
        elif cfg.detector == "iforest":
            self.detector = IsolationForest(
                n_estimators=cfg.n_estimators, max_samples=cfg.max_samples, random_state=cfg.seed, n_jobs=1
            ).fit(x)
        elif cfg.detector == "ocsvm":
            sub = x[rng.choice(len(x), min(len(x), 8000), replace=False)]
            self.detector = OneClassSVM(nu=cfg.ocsvm_nu, gamma=cfg.ocsvm_gamma).fit(sub)
        else:
            raise ValueError(f"detector desconhecido: {cfg.detector}")
        cal_scores = np.concatenate(
            [self._raw(extract_patch_features(images[i], cfg.patch, cfg.stride)[0]) for i in cal_idx]
        )
        self.cal_median = float(np.median(cal_scores))
        self.threshold = float(np.quantile(cal_scores, cfg.threshold_quantile))
        if self.threshold <= self.cal_median:
            self.threshold = self.cal_median + 1e-6
        params = json.dumps(asdict(cfg), sort_keys=True)
        digest = hashlib.sha256(f"{params}|{reference_hash}".encode()).hexdigest()[:10]
        self.metadata = {
            "model_name": MODEL_NAME,
            "model_version": f"{cfg.detector}-{digest}",
            "params": asdict(cfg),
            "reference_set_hash": reference_hash,
            "n_reference_images": len(images),
            "n_fit_images": len(fit_idx),
            "n_calibration_images": len(cal_idx),
            "n_fit_patches": int(len(fit_feats)),
            "n_calibration_patches": int(len(cal_scores)),
            "calibration_median": self.cal_median,
            "threshold_raw": self.threshold,
            "feature_names": FEATURE_NAMES,
            "trained_at": datetime.now(UTC).isoformat(),
        }
        return self

    def to_index(self, raw: np.ndarray) -> np.ndarray:
        return (raw - self.cal_median) / (self.threshold - self.cal_median)

    def patch_index_grid(self, image: np.ndarray, maps: PixelMaps | None = None) -> tuple[np.ndarray, ...]:
        feats, ys, xs = extract_patch_features(image, self.config.patch, self.config.stride, maps)
        idx = self.to_index(self._raw(feats)).reshape(len(ys), len(xs)).astype(np.float32)
        return idx, ys, xs

    def analyze(self, image: np.ndarray) -> AnomalyResult:
        cfg = self.config
        h, w = image.shape[:2]
        maps = pixel_maps(image)
        grid, ys, xs = self.patch_index_grid(image, maps)
        acc = np.zeros((h, w), np.float32)
        cnt = np.zeros((h, w), np.float32)
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                acc[y : y + cfg.patch, x : x + cfg.patch] += grid[i, j]
                cnt[y : y + cfg.patch, x : x + cfg.patch] += 1
        index_map = acc / np.maximum(cnt, 1)
        index_map[cnt == 0] = float(np.median(grid))
        if cfg.smooth_sigma > 0:
            index_map = cv2.GaussianBlur(index_map, (0, 0), cfg.smooth_sigma)
        mask = (index_map >= 1.0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        components: list[dict[str, Any]] = []
        clean = np.zeros_like(mask)
        for k in range(1, n):
            x, y, bw, bh, area = (int(v) for v in stats[k])
            if area < cfg.min_component_px:
                continue
            comp = labels == k
            clean[comp] = 1
            components.append(
                {
                    "label": k,
                    "bbox": (x, y, bw, bh),
                    "area_px": area,
                    "area_ratio": area / float(h * w),
                    "peak_index": float(index_map[comp].max()),
                    "mean_index": float(index_map[comp].mean()),
                    "mask": comp,
                }
            )
        components.sort(key=lambda c: c["peak_index"], reverse=True)
        return AnomalyResult(
            index_map=index_map,
            mask=clean,
            components=components,
            image_index=float(index_map.max()),
            anomalous_area_ratio=float(clean.mean()),
            maps=maps,
        )

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.version}.joblib"
        joblib.dump(
            {
                "config": asdict(self.config),
                "scaler": self.scaler,
                "detector": self.detector,
                "cal_median": self.cal_median,
                "threshold": self.threshold,
                "metadata": self.metadata,
            },
            path,
        )
        (directory / f"{self.version}.json").write_text(json.dumps(self.metadata, indent=2, ensure_ascii=False))
        (directory / "current.json").write_text(json.dumps({"model_version": self.version}, indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> PatchAnomalyModel:
        blob = joblib.load(path)  # arquivo gerado localmente pelo próprio projeto
        return cls(
            config=AnomalyConfig(**blob["config"]),
            scaler=blob["scaler"],
            detector=blob["detector"],
            cal_median=blob["cal_median"],
            threshold=blob["threshold"],
            metadata=blob["metadata"],
        )

    @classmethod
    def load_current(cls, directory: Path) -> PatchAnomalyModel:
        current = json.loads((directory / "current.json").read_text())
        return cls.load(directory / f"{current['model_version']}.joblib")


def heatmap_overlay(image: np.ndarray, index_map: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Sobreposição colorida: índice 0 → azul, ≥2 → vermelho; contorno branco na máscara."""
    norm = np.clip(index_map / 2.0, 0, 1)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    alpha = (0.15 + 0.55 * norm)[..., None]
    out = (image.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha).astype(np.uint8)
    if mask is not None and mask.any():
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), 2)
    return out
