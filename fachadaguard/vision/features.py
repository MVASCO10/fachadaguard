"""Extração de características reproduzíveis por patch (sem rede neural, executa em CPU).

Para cada patch quadrado (padrão 32 px, passo 16 px) calculamos 31 características:

* cor Lab relativa à mediana da imagem (média e desvio por canal)            → 6
* histograma de orientação de gradiente ponderado por magnitude (tipo HOG)   → 9
* magnitude de gradiente relativa (média e desvio)                           → 2
* histograma LBP uniforme (P=8, R=1)                                         → 10
* resposta black-hat (estruturas escuras e finas): média e fração forte      → 2
* densidade de bordas Canny e energia do Laplaciano relativa                 → 2

Valores relativos à própria imagem tornam o modelo menos sensível à cor do revestimento e à
iluminação global. Todas as médias por patch usam imagens integrais (custo O(pixels)).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

FEATURE_NAMES = (
    [f"lab_rel_mean_{c}" for c in "Lab"]
    + [f"lab_rel_std_{c}" for c in "Lab"]
    + [f"hog_bin_{i}" for i in range(9)]
    + ["grad_rel_mean", "grad_rel_std"]
    + [f"lbp_{i}" for i in range(10)]
    + ["blackhat_mean", "blackhat_strong_frac", "edge_density", "laplacian_rel"]
)
N_FEATURES = len(FEATURE_NAMES)


@dataclass(frozen=True)
class PixelMaps:
    """Mapas por pixel reutilizados pela categorização heurística."""

    lab: np.ndarray  # float32, L em 0–100
    l_rel: np.ndarray
    grad_mag: np.ndarray
    blackhat: np.ndarray
    laplacian: np.ndarray
    edges: np.ndarray
    median_lab: np.ndarray


def pixel_maps(image_bgr: np.ndarray) -> PixelMaps:
    lab = cv2.cvtColor(image_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    med = np.median(lab.reshape(-1, 3), axis=0).astype(np.float32)
    L = lab[..., 0]
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, kernel)
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(L, (3, 3), 0), cv2.CV_32F))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    v = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(0, 0.66 * v)), int(min(255, 1.33 * v))) > 0
    return PixelMaps(lab, L - med[0], mag, blackhat, lap, edges, med)


def patch_origins(h: int, w: int, patch: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    ys = np.arange(0, h - patch + 1, stride)
    xs = np.arange(0, w - patch + 1, stride)
    return ys, xs


def _integral(stack: np.ndarray) -> np.ndarray:
    s = np.zeros((stack.shape[0] + 1, stack.shape[1] + 1, stack.shape[2]), np.float64)
    s[1:, 1:] = stack.astype(np.float64).cumsum(0).cumsum(1)
    return s


def _box_means(stack: np.ndarray, ys: np.ndarray, xs: np.ndarray, patch: int) -> np.ndarray:
    s = _integral(stack)
    y0, x0 = np.meshgrid(ys, xs, indexing="ij")
    y1, x1 = y0 + patch, x0 + patch
    total = s[y1, x1] - s[y0, x1] - s[y1, x0] + s[y0, x0]
    return (total / (patch * patch)).astype(np.float32)  # (ny, nx, C)


def extract_patch_features(
    image_bgr: np.ndarray, patch: int = 32, stride: int = 16, maps: PixelMaps | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retorna (features[ny, nx, F], ys, xs)."""
    m = maps or pixel_maps(image_bgr)
    h, w = image_bgr.shape[:2]
    ys, xs = patch_origins(h, w, patch, stride)
    rel = m.lab - m.median_lab
    mag_ref = float(np.mean(m.grad_mag)) + 1e-6
    lap_ref = float(np.median(m.laplacian)) + 1e-3
    L = m.lab[..., 0]
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    theta = (np.arctan2(gy, gx) % np.pi) / np.pi
    bins = np.minimum((theta * 9).astype(np.int32), 8)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform").astype(np.int32)
    mag_rel = m.grad_mag / mag_ref
    channels = [rel, rel * rel]
    channels.append(np.stack([m.grad_mag * (bins == k) for k in range(9)], -1))
    channels.append(np.stack([mag_rel, mag_rel * mag_rel], -1))
    channels.append(np.stack([(lbp == k).astype(np.float32) for k in range(10)], -1))
    channels.append(
        np.stack(
            [
                m.blackhat,
                (m.blackhat > 6.0).astype(np.float32),
                m.edges.astype(np.float32),
                m.laplacian / lap_ref,
            ],
            -1,
        )
    )
    means = _box_means(np.concatenate(channels, -1), ys, xs, patch)
    lab_mean = means[..., 0:3]
    lab_std = np.sqrt(np.maximum(means[..., 3:6] - lab_mean**2, 0))
    hog = means[..., 6:15]
    hog = hog / (hog.sum(-1, keepdims=True) + 1e-6)
    g_mean = means[..., 15:16]
    g_std = np.sqrt(np.maximum(means[..., 16:17] - g_mean**2, 0))
    rest = means[..., 17:]
    feats = np.concatenate([lab_mean, lab_std, hog, g_mean, g_std, rest], -1)
    assert feats.shape[-1] == N_FEATURES
    return feats, ys, xs
