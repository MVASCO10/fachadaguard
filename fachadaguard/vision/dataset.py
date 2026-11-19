"""Conjuntos sintéticos determinísticos: referência (normal), dev, teste, pares temporais e demo.

Separação de responsabilidades para evitar vazamento:
* ``reference`` → ajuste + calibração do detector (somente fachadas normais);
* ``dev``       → ajuste das constantes das heurísticas de categoria e da severidade;
* ``test``      → métricas reportadas (nunca usado para ajustar nada);
* ``demo``      → cenário do edifício fictício usado pelo simulador de drone.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fachadaguard.vision.synthetic import (
    GENERATOR_VERSION,
    Condition,
    DefectSpec,
    FacadeStyle,
    Rendered,
    random_condition,
    random_defects,
    random_style,
    render,
)

SPLIT_SEEDS = {"reference": 1001, "dev": 2002, "test": 3003, "pairs": 4004}
CONDITION_GROUPS = ("nominal", "iluminacao", "desfoque_ruido")


@dataclass(frozen=True)
class Sample:
    name: str
    split: str
    rendered: Rendered
    meta: dict[str, Any]


def _group_condition(rng: np.random.Generator, group: str) -> Condition:
    base = random_condition(rng, strength=0.6)
    if group == "iluminacao":
        return Condition(
            **{
                **asdict(base),
                "shadow": float(rng.uniform(0.25, 0.45)),
                "shadow_seed": int(rng.integers(0, 2**31 - 1)),
                "gain": float(rng.choice([rng.uniform(0.7, 0.8), rng.uniform(1.15, 1.25)])),
            }
        )
    if group == "desfoque_ruido":
        return Condition(
            **{**asdict(base), "blur_sigma": float(rng.uniform(1.2, 2.2)), "noise_sigma": float(rng.uniform(4, 8))}
        )
    return base


def reference_samples(n: int = 90) -> list[Sample]:
    """Fachadas normais cobrindo as mesmas famílias de condição de aquisição (sem defeitos)."""
    rng = np.random.default_rng(SPLIT_SEEDS["reference"])
    out = []
    for i in range(n):
        style = random_style(rng)
        group = CONDITION_GROUPS[i % 3]
        cond = _group_condition(rng, group)
        out.append(
            Sample(
                f"ref_{i:03d}",
                "reference",
                render(style, None, cond),
                {"style": asdict(style), "defects": [], "condition_group": group},
            )
        )
    return out


def labeled_samples(split: str, n: int, normal_fraction: float = 0.33) -> list[Sample]:
    rng = np.random.default_rng(SPLIT_SEEDS[split])
    out = []
    for i in range(n):
        style = random_style(rng)
        group = CONDITION_GROUPS[i % 3]
        is_normal = rng.random() < normal_fraction
        defects = [] if is_normal else random_defects(rng, int(rng.integers(1, 3)))
        cond = _group_condition(rng, group)
        stages = [(d, float(rng.uniform(0.6, 1.0))) for d in defects]
        out.append(
            Sample(
                f"{split}_{i:03d}",
                split,
                render(style, stages, cond),
                {
                    "style": asdict(style),
                    "defects": [asdict(d) for d in defects],
                    "condition_group": group,
                    "condition": asdict(cond),
                },
            )
        )
    return out


def temporal_pairs(n: int = 30) -> list[dict[str, Any]]:
    """Pares (antes, depois). Tipos: evolução com deslocamento pequeno/médio/grande e pares incompatíveis."""
    rng = np.random.default_rng(SPLIT_SEEDS["pairs"])
    kinds = ["deslocamento_pequeno", "deslocamento_medio", "deslocamento_grande", "incompativel"]
    pairs = []
    for i in range(n):
        kind = kinds[i % 4]
        style = random_style(rng)
        defects = random_defects(rng, 1)
        shift = {"deslocamento_pequeno": 0.4, "deslocamento_medio": 1.2, "deslocamento_grande": 3.5}.get(kind, 1.0)
        c0 = random_condition(rng, 0.3)
        c1 = Condition(
            **{
                **asdict(random_condition(rng, shift)),
                "dx": float(rng.uniform(-6, 6) * shift),
                "dy": float(rng.uniform(-6, 6) * shift),
                "rotation_deg": float(rng.uniform(-1.5, 1.5) * shift),
            }
        )
        before = render(style, [(d, 0.45) for d in defects], c0)
        after_style = random_style(rng) if kind == "incompativel" else style
        after = render(after_style, [(d, 1.0) for d in defects], c1)
        pairs.append(
            {
                "name": f"pair_{i:03d}",
                "kind": kind,
                "before": before,
                "after": after,
                "defects": [asdict(d) for d in defects],
            }
        )
    return pairs


# ----------------------------------------------------------------------------- cenário de demonstração

DEMO_BUILDING = {
    "id": "bld-demo-atlantico",
    "name": "Shopping Atlântico (FICTÍCIO — dados sintéticos)",
    "address": "Endereço fictício para demonstração",
    "latitude": -30.0500,
    "longitude": -51.2600,
    "is_synthetic": True,
}
DEMO_FACADES = [
    {"id": "fac-demo-norte", "name": "Fachada Norte", "orientation": "N", "cladding": "Placas cerâmicas bege"},
    {"id": "fac-demo-leste", "name": "Fachada Leste", "orientation": "L", "cladding": "Painéis ACM azul"},
]
DEMO_EPOCHS = {"A": "2026-05-12T13:00:00Z", "B": "2026-11-10T13:00:00Z"}

_VIEW_B = Condition(gain=0.93, gamma=1.03, dx=11, dy=-7, rotation_deg=1.4, scale=1.02, noise_sigma=1.0, seed=77)
_VIEW_A = Condition(gain=1.0, dx=0, dy=0, noise_sigma=1.0, seed=11)

DEMO_REGIONS: list[dict[str, Any]] = [
    {
        "facade_id": "fac-demo-norte",
        "code": "N-01",
        "description": "Pavimentos 1–2, vão esquerdo",
        "style": FacadeStyle(seed=501, palette="bege", panel_w=96, panel_h=72, windows=True),
        "defects": [(DefectSpec("crack", 9101, 0.42, 0.45, 1.1), {"A": 0.45, "B": 1.0})],
        "story": "fissura que se prolonga entre as inspeções",
    },
    {
        "facade_id": "fac-demo-norte",
        "code": "N-02",
        "description": "Pavimentos 1–2, vão central",
        "style": FacadeStyle(seed=502, palette="bege", panel_w=96, panel_h=72, windows=False),
        "defects": [],
        "story": "região de controle sem alteração",
    },
    {
        "facade_id": "fac-demo-norte",
        "code": "N-03",
        "description": "Pavimentos 3–4, vão direito",
        "style": FacadeStyle(seed=503, palette="bege", panel_w=96, panel_h=72, windows=True),
        "defects": [(DefectSpec("stain", 9303, 0.55, 0.35, 1.1), {"A": 0.35, "B": 1.0})],
        "story": "mancha compatível com umidade que se expande",
    },
    {
        "facade_id": "fac-demo-leste",
        "code": "L-01",
        "description": "Pavimento 2, junto à marquise",
        "style": FacadeStyle(seed=601, palette="azul_acm", panel_w=110, panel_h=84, windows=False),
        "defects": [
            (DefectSpec("spall", 9601, 0.6, 0.55, 1.2), {"A": 0.0, "B": 1.0}),
            (DefectSpec("crack", 9602, 0.45, 0.5, 0.7), {"A": 0.3, "B": 0.8}),
        ],
        "story": "desplacamento aparente novo próximo a fissura existente",
    },
    {
        "facade_id": "fac-demo-leste",
        "code": "L-02",
        "description": "Pavimento 3, vão central",
        "style": FacadeStyle(seed=602, palette="azul_acm", panel_w=110, panel_h=84, windows=True),
        "defects": [(DefectSpec("crack", 9701, 0.3, 0.6, 0.9), {"A": 1.0, "B": 1.0})],
        "story": "fissura estável (sem evolução)",
    },
    {
        "facade_id": "fac-demo-leste",
        "code": "L-03",
        "description": "Pavimento 4, canto superior",
        "style": FacadeStyle(seed=603, palette="azul_acm", panel_w=110, panel_h=84, windows=True),
        "defects": [],
        "story": "sem defeito; inspeção B com sombra forte (teste de falso positivo)",
        "extra_condition_B": {"shadow": 0.4, "shadow_seed": 5},
    },
]


def demo_region_image(region: dict[str, Any], epoch: str) -> Rendered:
    cond = _VIEW_A if epoch == "A" else _VIEW_B
    if epoch == "B" and region.get("extra_condition_B"):
        cond = Condition(**{**asdict(cond), **region["extra_condition_B"]})
    defects = [(spec, stages[epoch]) for spec, stages in region["defects"]]
    return render(region["style"], defects, cond)


# ----------------------------------------------------------------------------- persistência


def _write_png(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", image)
    assert ok
    data = buf.tobytes()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def write_dataset(root: Path, n_reference: int = 90, n_dev: int = 60, n_test: int = 120) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "generator_version": GENERATOR_VERSION,
        "split_seeds": SPLIT_SEEDS,
        "synthetic": True,
        "files": {},
        "splits": {},
    }
    splits = {
        "reference": reference_samples(n_reference),
        "dev": labeled_samples("dev", n_dev),
        "test": labeled_samples("test", n_test),
    }
    for split, samples in splits.items():
        entries = []
        for s in samples:
            img_rel = f"{split}/{s.name}.png"
            manifest["files"][img_rel] = _write_png(root / img_rel, s.rendered.image)
            entry = {"name": s.name, "image": img_rel, **s.meta}
            if split != "reference":
                mask_rel = f"{split}/{s.name}_mask.png"
                manifest["files"][mask_rel] = _write_png(root / mask_rel, s.rendered.mask)
                entry["mask"] = mask_rel
            entries.append(entry)
        manifest["splits"][split] = entries
    demo_entries = []
    for region in DEMO_REGIONS:
        for epoch in DEMO_EPOCHS:
            r = demo_region_image(region, epoch)
            img_rel = f"demo/{region['code']}_{epoch}.png"
            mask_rel = f"demo/{region['code']}_{epoch}_mask.png"
            manifest["files"][img_rel] = _write_png(root / img_rel, r.image)
            manifest["files"][mask_rel] = _write_png(root / mask_rel, r.mask)
            demo_entries.append(
                {
                    "region": region["code"],
                    "facade_id": region["facade_id"],
                    "epoch": epoch,
                    "image": img_rel,
                    "mask": mask_rel,
                    "story": region["story"],
                }
            )
    manifest["splits"]["demo"] = demo_entries
    manifest["reference_set_hash"] = reference_set_hash(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def reference_set_hash(manifest: dict[str, Any]) -> str:
    refs = sorted(v for k, v in manifest["files"].items() if k.startswith("reference/"))
    return hashlib.sha256("".join(refs).encode()).hexdigest()


def verify_dataset(root: Path) -> list[str]:
    """Confere checksums do manifesto. Retorna lista de problemas (vazia = íntegro)."""
    manifest = json.loads((root / "manifest.json").read_text())
    problems = []
    for rel, digest in manifest["files"].items():
        path = root / rel
        if not path.exists():
            problems.append(f"ausente: {rel}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(f"checksum divergente: {rel}")
    return problems


def load_images(root: Path, split: str) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    manifest = json.loads((root / "manifest.json").read_text())
    entries = manifest["splits"][split]
    images = [cv2.imread(str(root / e["image"]), cv2.IMREAD_COLOR) for e in entries]
    return images, entries
