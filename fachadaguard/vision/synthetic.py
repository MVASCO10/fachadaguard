"""Gerador procedural e determinístico de fachadas SINTÉTICAS com defeitos rotulados.

Finalidade: validar o funcionamento técnico do pipeline (ingestão → IA → comparação) sem depender de
imagens reais. As imagens NÃO representam a variabilidade de fachadas reais; métricas obtidas com elas
não estimam desempenho em campo.

Rótulos das máscaras (uint8): 0 = fundo, 1 = fissura, 2 = mancha de umidade, 3 = desplacamento.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

GENERATOR_VERSION = "synthfacade-1.2"
LABELS = {0: "fundo", 1: "fissura", 2: "umidade_infiltracao", 3: "desplacamento_aparente"}
LABEL_OF_KIND = {"crack": 1, "stain": 2, "spall": 3}
DefectKind = Literal["crack", "stain", "spall"]

PALETTES_BGR: dict[str, tuple[int, int, int]] = {
    "bege": (170, 196, 214),
    "cinza_claro": (196, 198, 200),
    "terracota": (112, 140, 188),
    "azul_acm": (186, 160, 128),
    "branco_gelo": (222, 226, 228),
    "grafite": (118, 116, 114),
}


@dataclass(frozen=True)
class FacadeStyle:
    seed: int
    width: int = 640
    height: int = 480
    palette: str = "bege"
    panel_w: int = 96
    panel_h: int = 72
    windows: bool = True


@dataclass(frozen=True)
class DefectSpec:
    kind: DefectKind
    seed: int
    cx: float  # centro relativo (0–1)
    cy: float
    size: float = 1.0  # escala relativa


@dataclass(frozen=True)
class Condition:
    gain: float = 1.0
    gamma: float = 1.0
    shadow: float = 0.0  # 0 = sem sombra; 0.4 = sombra forte projetada
    shadow_seed: int = 0
    blur_sigma: float = 0.0
    noise_sigma: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    rotation_deg: float = 0.0
    scale: float = 1.0
    perspective: float = 0.0
    seed: int = 0


@dataclass
class Rendered:
    image: np.ndarray  # BGR uint8
    mask: np.ndarray  # uint8 rótulos
    homography: np.ndarray = field(default_factory=lambda: np.eye(3))


def random_style(rng: np.random.Generator, width: int = 640, height: int = 480) -> FacadeStyle:
    return FacadeStyle(
        seed=int(rng.integers(0, 2**31 - 1)),
        width=width,
        height=height,
        palette=str(rng.choice(list(PALETTES_BGR))),
        panel_w=int(rng.integers(70, 130)),
        panel_h=int(rng.integers(52, 100)),
        windows=bool(rng.random() < 0.6),
    )


def _low_freq_noise(rng: np.random.Generator, h: int, w: int, cells: int, amp: float) -> np.ndarray:
    small = rng.normal(0, 1, (max(2, h // cells), max(2, w // cells))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * amp


def render_base(style: FacadeStyle) -> np.ndarray:
    """Fachada normal: painéis com juntas, textura fina, variação por painel, janelas e gradiente de luz."""
    rng = np.random.default_rng(style.seed)
    h, w = style.height, style.width
    base = np.array(PALETTES_BGR[style.palette], np.float32)
    img = np.empty((h, w, 3), np.float32)
    ox, oy = int(rng.integers(0, style.panel_w)), int(rng.integers(0, style.panel_h))
    xs = list(range(-ox, w, style.panel_w))
    ys = list(range(-oy, h, style.panel_h))
    for y0 in ys:
        for x0 in xs:
            jitter = 1 + rng.normal(0, 0.025)
            img[max(0, y0) : y0 + style.panel_h, max(0, x0) : x0 + style.panel_w] = base * jitter
    img += rng.normal(0, 3.0, (h, w, 1)).astype(np.float32)
    img += _low_freq_noise(rng, h, w, 40, 4.0)[..., None]
    joint = base * 0.62
    for x0 in xs:
        if 0 <= x0 < w:
            cv2.line(img, (x0, 0), (x0, h - 1), joint.tolist(), 2)
    for y0 in ys:
        if 0 <= y0 < h:
            cv2.line(img, (0, y0), (w - 1, y0), joint.tolist(), 2)
    if style.windows:
        row = int(rng.integers(0, max(1, len(ys) - 1)))
        y0 = max(4, ys[row] + 8)
        y1 = min(h - 4, y0 + style.panel_h - 16)
        step = style.panel_w * int(rng.integers(1, 3))
        for x0 in range(xs[0] + 10, w, step):
            x1 = min(w - 4, x0 + style.panel_w - 20)
            if x1 - x0 < 20 or y1 - y0 < 20 or x0 < 4:
                continue
            glass = np.linspace(0.75, 1.15, y1 - y0, dtype=np.float32)[:, None, None]
            img[y0:y1, x0:x1] = np.array([78, 64, 52], np.float32) * glass
            cv2.rectangle(img, (x0, y0), (x1, y1), (150, 150, 152), 3)
    angle = rng.uniform(0, 2 * np.pi)
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = ((gx / w - 0.5) * np.cos(angle) + (gy / h - 0.5) * np.sin(angle)) * rng.uniform(0.06, 0.14)
    img *= (1 + grad)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def _crack_path(spec: DefectSpec, w: int, h: int) -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(spec.seed)
    n = int(140 * spec.size)
    angle = rng.uniform(0, 2 * np.pi)
    p = np.array([spec.cx * w, spec.cy * h], np.float64)
    pts = [p.copy()]
    for _ in range(n):
        angle += rng.normal(0, 0.28)
        p = p + 3.0 * np.array([np.cos(angle), np.sin(angle)])
        pts.append(p.copy())
    main = np.array(pts)
    branches = []
    for _ in range(int(rng.integers(0, 3))):
        start = main[int(rng.integers(n // 4, n))].copy()
        a = angle + rng.choice([-1, 1]) * rng.uniform(0.6, 1.2)
        bp = [start.copy()]
        for _ in range(int(rng.integers(20, 50))):
            a += rng.normal(0, 0.3)
            start = start + 2.5 * np.array([np.cos(a), np.sin(a)])
            bp.append(start.copy())
        branches.append(np.array(bp))
    return main, branches


def add_crack(img: np.ndarray, mask: np.ndarray, spec: DefectSpec, stage: float) -> None:
    h, w = mask.shape
    main, branches = _crack_path(spec, w, h)
    layer = np.zeros((h, w), np.float32)
    k = max(2, int(len(main) * stage))
    width = 2 if spec.size >= 1 else 1
    cv2.polylines(layer, [main[:k].astype(np.int32)], False, 1.0, width, cv2.LINE_AA)
    if stage > 0.5:
        for br in branches:
            kb = max(2, int(len(br) * (stage - 0.5) * 2))
            cv2.polylines(layer, [br[:kb].astype(np.int32)], False, 1.0, 1, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.6)
    img[:] = np.clip(img.astype(np.float32) * (1 - 0.62 * layer[..., None]), 0, 255).astype(np.uint8)
    lab = cv2.dilate((layer > 0.25).astype(np.uint8), np.ones((3, 3), np.uint8))
    mask[lab > 0] = LABEL_OF_KIND["crack"]


def add_stain(img: np.ndarray, mask: np.ndarray, spec: DefectSpec, stage: float) -> None:
    h, w = mask.shape
    rng = np.random.default_rng(spec.seed)
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    alpha = np.zeros((h, w), np.float32)
    scale = spec.size * (0.45 + 0.55 * stage)
    cx, cy = spec.cx * w, spec.cy * h
    for _ in range(7):
        bx = cx + rng.normal(0, 22) * scale
        by = cy + abs(rng.normal(0, 30)) * scale  # escorre para baixo
        sx, sy = rng.uniform(16, 34) * scale, rng.uniform(24, 52) * scale
        alpha += np.exp(-(((gx - bx) / sx) ** 2 + ((gy - by) / sy) ** 2))
    alpha *= 1 + _low_freq_noise(rng, h, w, 12, 0.25)
    alpha = np.clip(alpha, 0, 1) ** 0.85
    tint = np.array(rng.choice([[60, 82, 92], [70, 96, 84]]), np.float32)  # marrom / esverdeado
    f = img.astype(np.float32)
    f = f * (1 - 0.42 * alpha[..., None]) + tint * 0.18 * alpha[..., None]
    img[:] = np.clip(f, 0, 255).astype(np.uint8)
    mask[(alpha > 0.35) & (mask == 0)] = LABEL_OF_KIND["stain"]


def add_spall(img: np.ndarray, mask: np.ndarray, spec: DefectSpec, stage: float) -> None:
    h, w = mask.shape
    rng = np.random.default_rng(spec.seed)
    r = 34 * spec.size * (0.5 + 0.5 * stage)
    angles = np.sort(rng.uniform(0, 2 * np.pi, 11))
    radii = r * rng.uniform(0.6, 1.25, 11)
    poly = np.stack([spec.cx * w + radii * np.cos(angles), spec.cy * h + radii * np.sin(angles)], 1)
    poly_i = poly.astype(np.int32)
    region = np.zeros((h, w), np.uint8)
    cv2.fillPoly(region, [poly_i], 1)
    substrate = np.array([118, 122, 128], np.float32) + rng.normal(0, 16, (h, w, 1)).astype(np.float32)
    speck = (rng.random((h, w)) < 0.03)[..., None] * rng.normal(0, 40, (h, w, 1)).astype(np.float32)
    substrate = substrate + speck
    f = img.astype(np.float32)
    shadow = cv2.warpAffine(region, np.array([[1, 0, 3], [0, 1, 3]], dtype=np.float32), (w, h)) & (1 - region)
    f[shadow > 0] *= 0.55
    f[region > 0] = substrate[region > 0]
    edge = cv2.morphologyEx(region, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    f[edge > 0] *= 0.6
    img[:] = np.clip(f, 0, 255).astype(np.uint8)
    mask[region > 0] = LABEL_OF_KIND["spall"]


RENDERERS = {"crack": add_crack, "stain": add_stain, "spall": add_spall}


def _homography(cond: Condition, w: int, h: int) -> np.ndarray:
    rng = np.random.default_rng(cond.seed)
    c = np.array([w / 2, h / 2])
    a = np.deg2rad(cond.rotation_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * cond.scale
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = (src - c) @ rot.T + c + np.array([cond.dx, cond.dy])
    if cond.perspective:
        dst = dst + rng.uniform(-1, 1, (4, 2)) * cond.perspective * min(w, h)
    return cv2.getPerspectiveTransform(src, dst.astype(np.float32))


def apply_condition(img: np.ndarray, mask: np.ndarray, cond: Condition) -> Rendered:
    """Variações de aquisição: iluminação, sombra projetada, desfoque, ruído e ponto de vista."""
    h, w = mask.shape
    H = _homography(cond, w, h)
    out = cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_REFLECT)
    m = cv2.warpPerspective(mask, H, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    f = out.astype(np.float32) / 255.0
    f = np.power(np.clip(f * cond.gain, 0, 1), cond.gamma)
    if cond.shadow > 0:
        rng = np.random.default_rng(cond.shadow_seed)
        gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
        a = rng.uniform(0, np.pi)
        d = (gx - w * rng.uniform(0.3, 0.7)) * np.cos(a) + (gy - h * rng.uniform(0.3, 0.7)) * np.sin(a)
        sh = 1 / (1 + np.exp(-d / 6.0))
        f *= (1 - cond.shadow * sh)[..., None]
    out = np.clip(f * 255, 0, 255).astype(np.uint8)
    if cond.blur_sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), cond.blur_sigma)
    if cond.noise_sigma > 0:
        rng = np.random.default_rng(cond.seed + 7)
        out = np.clip(out + rng.normal(0, cond.noise_sigma, out.shape), 0, 255).astype(np.uint8)
    return Rendered(out, m, H)


def render(
    style: FacadeStyle,
    defects: list[tuple[DefectSpec, float]] | None = None,
    condition: Condition | None = None,
) -> Rendered:
    img = render_base(style)
    mask = np.zeros(img.shape[:2], np.uint8)
    for spec, stage in defects or []:
        if stage > 0:
            RENDERERS[spec.kind](img, mask, spec, stage)
    if condition is None:
        return Rendered(img, mask)
    return apply_condition(img, mask, condition)


def random_condition(rng: np.random.Generator, strength: float = 1.0) -> Condition:
    return Condition(
        gain=float(rng.uniform(1 - 0.12 * strength, 1 + 0.12 * strength)),
        gamma=float(rng.uniform(1 - 0.1 * strength, 1 + 0.1 * strength)),
        blur_sigma=float(rng.uniform(0, 0.6 * strength)),
        noise_sigma=float(rng.uniform(0, 2.5 * strength)),
        dx=float(rng.uniform(-8, 8) * strength),
        dy=float(rng.uniform(-8, 8) * strength),
        rotation_deg=float(rng.uniform(-1.5, 1.5) * strength),
        scale=float(rng.uniform(1 - 0.02 * strength, 1 + 0.02 * strength)),
        seed=int(rng.integers(0, 2**31 - 1)),
    )


def random_defects(rng: np.random.Generator, n: int) -> list[DefectSpec]:
    kinds: list[DefectKind] = ["crack", "stain", "spall"]
    return [
        DefectSpec(
            kind=kinds[int(rng.integers(0, 3))],
            seed=int(rng.integers(0, 2**31 - 1)),
            cx=float(rng.uniform(0.2, 0.8)),
            cy=float(rng.uniform(0.2, 0.75)),
            size=float(rng.uniform(0.8, 1.3)),
        )
        for _ in range(n)
    ]
