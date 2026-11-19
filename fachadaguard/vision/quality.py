"""Qualidade da evidência fotográfica (nitidez, exposição, contraste) em escala 0–1."""

from __future__ import annotations

import cv2
import numpy as np


def image_quality(image: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clipped = float(((gray <= 3) | (gray >= 252)).mean())
    contrast = float(gray.std())
    sharp_s = float(np.clip((lap_var - 15) / (120 - 15), 0, 1))
    expo_s = float(np.clip(1 - clipped / 0.15, 0, 1))
    contrast_s = float(np.clip((contrast - 8) / (35 - 8), 0, 1))
    score = 0.5 * sharp_s + 0.3 * expo_s + 0.2 * contrast_s
    return {
        "laplacian_var": round(lap_var, 2),
        "clipped_frac": round(clipped, 4),
        "contrast_std": round(contrast, 2),
        "sharpness_score": round(sharp_s, 3),
        "exposure_score": round(expo_s, 3),
        "contrast_score": round(contrast_s, 3),
        "quality_score": round(score, 3),
    }
