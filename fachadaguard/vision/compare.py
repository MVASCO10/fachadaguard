"""Comparação temporal entre duas capturas da MESMA região de fachada.

1. Registro: ORB + correspondência com teste de razão + homografia RANSAC, validada por número e
   proporção de inliers, erro de reprojeção, sanidade geométrica e sobreposição. Se falhar, tenta
   ECC (modelo afim). Se ambos falharem, NÃO há comparação automática (fallback seguro).
2. Normalização fotométrica: casamento de histograma do "antes" alinhado com o "depois".
3. Mudança: 1 − SSIM local (janela gaussiana), restrita à área válida após o alinhamento.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity

MIN_INLIERS = 30
MIN_INLIER_RATIO = 0.25
MAX_REPROJ_PX = 3.0
MIN_OVERLAP = 0.70
MIN_ECC_CORR = 0.80
CHANGE_THRESHOLD = 0.45


@dataclass
class AlignmentResult:
    ok: bool
    method: str
    homography: np.ndarray | None
    metrics: dict[str, float] = field(default_factory=dict)
    reason: str = ""


@dataclass
class ComparisonResult:
    status: str  # ok | alignment_failed
    alignment: AlignmentResult
    ssim: float | None = None
    changed_area_ratio: float | None = None
    change_map: np.ndarray | None = None
    change_mask: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    aligned_before: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)

    def alignment_dict(self) -> dict[str, Any]:
        return {
            "ok": self.alignment.ok,
            "method": self.alignment.method,
            "reason": self.alignment.reason,
            **{k: round(float(v), 4) for k, v in self.alignment.metrics.items()},
        }


def _gray(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def _overlap(H: np.ndarray, shape: tuple[int, int]) -> float:
    h, w = shape
    ones = np.ones((h, w), np.uint8)
    return float(cv2.warpPerspective(ones, H, (w, h)).mean())


def _sane(H: np.ndarray) -> bool:
    det = float(np.linalg.det(H[:2, :2]))
    return 0.7 < det < 1.4 and abs(H[2, 0]) < 2e-3 and abs(H[2, 1]) < 2e-3


def align_orb(before: np.ndarray, after: np.ndarray, n_features: int = 3000) -> AlignmentResult:
    g0, g1 = _gray(before), _gray(after)
    orb = cv2.ORB_create(  # type: ignore[attr-defined]
        nfeatures=n_features, fastThreshold=10
    )
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(g1, None)
    if d0 is None or d1 is None or len(k0) < MIN_INLIERS or len(k1) < MIN_INLIERS:
        return AlignmentResult(
            False, "orb", None, {"keypoints": float(min(len(k0), len(k1)))}, "poucos pontos de interesse"
        )
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d0, d1, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < MIN_INLIERS:
        return AlignmentResult(False, "orb", None, {"good_matches": float(len(good))}, "correspondências insuficientes")
    src = np.array([k0[m.queryIdx].pt for m in good], dtype=np.float32).reshape(-1, 1, 2)
    dst = np.array([k1[m.trainIdx].pt for m in good], dtype=np.float32).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None:
        return AlignmentResult(False, "orb", None, {"good_matches": float(len(good))}, "homografia não estimada")
    inl = inl.ravel().astype(bool)
    proj = cv2.perspectiveTransform(src[inl], H)
    reproj = float(np.median(np.linalg.norm(proj - dst[inl], axis=2))) if inl.any() else 99.0
    metrics = {
        "good_matches": float(len(good)),
        "inliers": float(inl.sum()),
        "inlier_ratio": float(inl.mean()),
        "reproj_median_px": reproj,
        "overlap": _overlap(H, g0.shape),
    }
    checks = [
        (metrics["inliers"] >= MIN_INLIERS, "poucos inliers"),
        (metrics["inlier_ratio"] >= MIN_INLIER_RATIO, "proporção de inliers baixa"),
        (reproj <= MAX_REPROJ_PX, "erro de reprojeção alto"),
        (_sane(H), "homografia geometricamente implausível"),
        (metrics["overlap"] >= MIN_OVERLAP, "sobreposição insuficiente"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    return AlignmentResult(not failed, "orb_homography", H if not failed else None, metrics, "; ".join(failed))


def align_ecc(before: np.ndarray, after: np.ndarray) -> AlignmentResult:
    g0 = cv2.GaussianBlur(_gray(before), (5, 5), 0).astype(np.float32) / 255
    g1 = cv2.GaussianBlur(_gray(after), (5, 5), 0).astype(np.float32) / 255
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        corr, warp = cv2.findTransformECC(
            g1, g0, warp, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5), None, 5
        )
    except cv2.error:
        return AlignmentResult(False, "ecc_affine", None, {}, "ECC não convergiu")
    # warp mapeia coordenadas do "depois" no "antes"; invertemos para levar "antes" → "depois"
    H = np.linalg.inv(np.vstack([warp, [0, 0, 1]]).astype(np.float64))
    metrics = {"ecc_correlation": float(corr), "overlap": _overlap(H, g0.shape)}
    failed = []
    if corr < MIN_ECC_CORR:
        failed.append("correlação ECC baixa")
    if not _sane(H):
        failed.append("transformação implausível")
    if metrics["overlap"] < MIN_OVERLAP:
        failed.append("sobreposição insuficiente")
    return AlignmentResult(not failed, "ecc_affine", H if not failed else None, metrics, "; ".join(failed))


def align(before: np.ndarray, after: np.ndarray) -> AlignmentResult:
    res = align_orb(before, after)
    if res.ok:
        return res
    ecc = align_ecc(before, after)
    if ecc.ok:
        ecc.reason = f"ORB rejeitado ({res.reason}); ECC aceito"
        ecc.metrics.update({f"orb_{k}": v for k, v in res.metrics.items()})
        return ecc
    return AlignmentResult(False, "none", None, {**res.metrics, **ecc.metrics}, f"ORB: {res.reason}; ECC: {ecc.reason}")


def compare(before: np.ndarray, after: np.ndarray) -> ComparisonResult:
    if before.shape != after.shape:
        before = cv2.resize(before, (after.shape[1], after.shape[0]), interpolation=cv2.INTER_AREA)
    al = align(before, after)
    if not al.ok or al.homography is None:
        return ComparisonResult(
            "alignment_failed",
            al,
            warnings=[
                "Alinhamento não validado: comparação automática suspensa para evitar falsa mudança. "
                f"Motivo: {al.reason}. Revise manualmente ou recapture a região."
            ],
        )
    h, w = after.shape[:2]
    warped = cv2.warpPerspective(before, al.homography, (w, h))
    valid = cv2.erode(
        cv2.warpPerspective(np.ones((h, w), np.uint8), al.homography, (w, h)), np.ones((15, 15), np.uint8)
    ).astype(bool)
    matched = match_histograms(warped, after, channel_axis=-1).astype(np.uint8)
    g0 = cv2.cvtColor(matched, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    _, ssim_map = structural_similarity(g0, g1, data_range=255, gaussian_weights=True, sigma=2.0, full=True)
    change = np.clip(1.0 - ssim_map, 0, 1).astype(np.float32)
    change = cv2.GaussianBlur(change, (0, 0), 3)
    change[~valid] = 0
    cmask = ((change >= CHANGE_THRESHOLD) & valid).astype(np.uint8)
    cmask = cv2.morphologyEx(cmask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    warnings = []
    if al.method != "orb_homography":
        warnings.append(f"Alinhamento por {al.method} (fallback); revisar bordas da sobreposição.")
    return ComparisonResult(
        status="ok",
        alignment=al,
        ssim=float(ssim_map[valid].mean()),
        changed_area_ratio=float(cmask[valid].mean()),
        change_map=change,
        change_mask=cmask,
        valid_mask=valid,
        aligned_before=warped,
        warnings=warnings,
    )


def change_overlay(after: np.ndarray, change: np.ndarray, mask: np.ndarray) -> np.ndarray:
    norm = np.clip(change / 0.8, 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = (0.65 * norm)[..., None]
    out = (after.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
    return out
