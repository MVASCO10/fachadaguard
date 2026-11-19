"""Categorização HEURÍSTICA de regiões anômalas — separada da detecção.

A detecção (``anomaly.py``) responde "esta região difere do padrão normal?". Esta etapa propõe uma
**categoria candidata** a partir de medidas interpretáveis da região. Não é um classificador
treinado, não mede profundidade nem causa, e não substitui a avaliação de um profissional.

Medidas (todas relativas à própria imagem):
* ``line_frac``     – fração da região coberta por traços escuros finos e não retilíneos
                       (black-hat, descontando juntas retas horizontais/verticais);
* ``darkness``      – escurecimento médio (ΔL*, unidades Lab) em relação à mediana da imagem;
* ``texture_ratio`` – energia do Laplaciano na região ÷ mediana da imagem (rugosidade);
* ``chroma_drop``   – queda de croma (apenas informativa; não discriminou no split dev);
* ``elongation``    – razão entre eixos principais da região.

As constantes foram ajustadas apenas no split ``dev`` sintético (ver docs/metodologia.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from fachadaguard.domain.models import Category
from fachadaguard.vision.features import PixelMaps

RULES_VERSION = "heuristica-1.0"


@dataclass(frozen=True)
class CategoryResult:
    category: str
    confidence: str  # baixa | media | alta
    scores: dict[str, float]
    measures: dict[str, float]
    explanation: str


def _sig(x: float, center: float, width: float) -> float:
    return float(1.0 / (1.0 + np.exp(-(x - center) / width)))


def curvy_dark_lines(maps: PixelMaps, thresh: float = 8.0) -> np.ndarray:
    thin = (maps.blackhat > thresh).astype(np.uint8)
    straight = cv2.morphologyEx(thin, cv2.MORPH_OPEN, np.ones((1, 31), np.uint8)) | cv2.morphologyEx(
        thin, cv2.MORPH_OPEN, np.ones((31, 1), np.uint8)
    )
    straight = cv2.dilate(straight, np.ones((5, 5), np.uint8))
    return (thin & (1 - straight)).astype(bool)


def region_measures(maps: PixelMaps, region: np.ndarray, curvy: np.ndarray | None = None) -> dict[str, float]:
    region = region.astype(bool)
    curvy = curvy_dark_lines(maps) if curvy is None else curvy
    lab = maps.lab
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    med_chroma = float(np.median(chroma))
    lap_ref = float(np.median(maps.laplacian)) + 1e-3
    ys, xs = np.nonzero(region)
    elong = 1.0
    if len(xs) > 5:
        cov = np.cov(np.stack([xs, ys]).astype(np.float64))
        ev = np.sort(np.linalg.eigvalsh(cov))
        elong = float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
    return {
        "line_frac": float(curvy[region].mean()),
        "darkness": float(-maps.l_rel[region].mean()),
        "texture_ratio": float(maps.laplacian[region].mean() / lap_ref),
        "chroma_drop": float(med_chroma - chroma[region].mean()),
        "elongation": elong,
    }


def categorize(measures: dict[str, float]) -> CategoryResult:
    m = measures
    crack = _sig(m["line_frac"], 0.08, 0.02) * _sig(m["elongation"], 1.6, 0.2)
    stain = (
        _sig(m["darkness"], 6.0, 1.5)
        * (1 - _sig(m["texture_ratio"], 2.4, 0.3))
        * (1 - _sig(m["line_frac"], 0.06, 0.015))
    )
    spall = (
        _sig(m["texture_ratio"], 4.8, 0.5) * (1 - _sig(m["elongation"], 1.7, 0.15)) * _sig(m["line_frac"], 0.08, 0.03)
    )
    scores = {Category.FISSURA.value: crack, Category.UMIDADE.value: stain, Category.DESPLACAMENTO.value: spall}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, top), (_, second) = ranked[0], ranked[1]
    margin = top - second
    if top < 0.45 or margin < 0.10:
        cat, conf = Category.INDETERMINADO.value, "baixa"
    else:
        cat = best
        conf = "alta" if (top >= 0.75 and margin >= 0.35) else "media" if margin >= 0.2 else "baixa"
    reasons = {
        Category.FISSURA.value: f"traços escuros finos e irregulares cobrem {m['line_frac']:.1%} da região",
        Category.UMIDADE.value: f"escurecimento difuso de {m['darkness']:.1f} ΔL* com textura pouco alterada "
        f"(×{m['texture_ratio']:.1f})",
        Category.DESPLACAMENTO.value: f"rugosidade ×{m['texture_ratio']:.1f} em região compacta "
        f"(elongação {m['elongation']:.1f}), compatível com material exposto",
    }
    if cat == Category.INDETERMINADO.value:
        explanation = (
            "Nenhuma regra heurística se destacou com margem suficiente "
            f"(maior escore {top:.2f}, margem {margin:.2f}). A região difere do padrão normal, mas a "
            "natureza da alteração precisa ser avaliada por um profissional."
        )
    else:
        explanation = (
            f"Categoria candidata por regra heurística ({RULES_VERSION}): {reasons[cat]}. "
            f"Escore heurístico {top:.2f}, margem {margin:.2f} sobre a 2ª opção. "
            "Não é diagnóstico; exige confirmação visual por profissional habilitado."
        )
    return CategoryResult(
        cat, conf, {k: round(v, 4) for k, v in scores.items()}, {k: round(v, 4) for k, v in m.items()}, explanation
    )


def majority_label(gt_mask: np.ndarray, region: np.ndarray) -> int:
    vals = gt_mask[region.astype(bool)]
    vals = vals[vals > 0]
    if vals.size < max(20, 0.03 * region.sum()):
        return 0
    return int(np.bincount(vals).argmax())


def categorize_components(maps: PixelMaps, components: list[dict[str, Any]]) -> list[CategoryResult]:
    curvy = curvy_dark_lines(maps)
    return [categorize(region_measures(maps, c["mask"], curvy)) for c in components]
