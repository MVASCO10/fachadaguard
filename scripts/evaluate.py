"""Avaliação reproduzível no split de TESTE sintético (nunca usado para ajustar nada).

Produz em artifacts/evaluation/:
  metrics.json              métricas de detecção, localização, categorização, comparação e ablações
  confusion_matrix.png      categorias candidatas × rótulo sintético das regiões detectadas
  roc_image.png             curvas ROC no nível de imagem (kNN, IsolationForest, OneClassSVM)
  examples_*.png            exemplos de acerto, falso positivo e falso negativo (mapas de calor)
  comparison_example.png    alinhamento + mapa de mudança de um par temporal
  fp_fn_summary.md          resumo dos falsos positivos/negativos por condição

ATENÇÃO: dados SINTÉTICOS. As métricas validam o funcionamento técnico e expõem modos de falha;
não estimam desempenho em fachadas reais.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fachadaguard.vision.anomaly import AnomalyConfig, PatchAnomalyModel, heatmap_overlay  # noqa: E402
from fachadaguard.vision.categorize import categorize_components, majority_label  # noqa: E402
from fachadaguard.vision.compare import change_overlay, compare  # noqa: E402
from fachadaguard.vision.dataset import (  # noqa: E402
    CONDITION_GROUPS,
    SPLIT_SEEDS,
    labeled_samples,
    reference_samples,
    temporal_pairs,
)
from fachadaguard.vision.synthetic import GENERATOR_VERSION, LABELS  # noqa: E402

OUT = ROOT / "artifacts" / "evaluation"
CAT_OF_LABEL = {0: "fundo (FP)", 1: "fissura", 2: "umidade_infiltracao", 3: "desplacamento_aparente"}
CATS = ["fissura", "umidade_infiltracao", "desplacamento_aparente", "indeterminado"]
HIT_OVERLAP = 0.10  # defeito "detectado" se ≥10% de seus pixels caem na máscara prevista


def r4(x: float) -> float:
    return round(float(x), 4)


def evaluate_detector(model: PatchAnomalyModel, samples: list[Any], keep: bool = False) -> dict[str, Any]:
    y_img, s_img, y_px, s_px = [], [], [], []
    per_group: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"y": [], "s": []})
    results = []
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    for s in samples:
        res = model.analyze(s.rendered.image)
        gt = s.rendered.mask > 0
        y_img.append(int(gt.any()))
        s_img.append(res.image_index)
        idx = rng.choice(gt.size, 4000, replace=False)
        y_px.append(gt.ravel()[idx])
        s_px.append(res.index_map.ravel()[idx])
        g = per_group[s.meta["condition_group"]]
        g["y"].append(int(gt.any()))
        g["s"].append(res.image_index)
        if keep:
            results.append(res)
    secs = time.perf_counter() - t0
    yi, si = np.array(y_img), np.array(s_img)
    ypx, spx = np.concatenate(y_px), np.concatenate(s_px)
    pred = si >= 1.0
    out = {
        "image_auroc": r4(roc_auc_score(yi, si)),
        "image_ap": r4(average_precision_score(yi, si)),
        "pixel_auroc": r4(roc_auc_score(ypx, spx)),
        "image_tpr_at_threshold": r4(pred[yi == 1].mean()),
        "image_fpr_at_threshold": r4(pred[yi == 0].mean()),
        "seconds_per_image": r4(secs / len(samples)),
        "by_condition": {
            k: {"n": len(v["y"]), "image_auroc": r4(roc_auc_score(v["y"], v["s"])) if len(set(v["y"])) > 1 else None}
            for k, v in per_group.items()
        },
        "_roc": (yi, si),
    }
    if keep:
        out["_results"] = results
    return out


def region_level(samples: list[Any], results: list[Any]) -> dict[str, Any]:
    hits: dict[int, list[int]] = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
    hits_group: dict[str, dict[int, list[int]]] = defaultdict(lambda: {1: [0, 0], 2: [0, 0], 3: [0, 0]})
    comp_total = comp_tp = 0
    confusion: Counter[tuple[str, str]] = Counter()
    fp_rows, fn_rows = [], []
    for s, res in zip(samples, results, strict=True):
        gt = s.rendered.mask
        group = s.meta["condition_group"]
        for lab in (1, 2, 3):
            g = gt == lab
            if g.sum() < 150:
                continue
            hit = (res.mask.astype(bool) & g).sum() >= HIT_OVERLAP * g.sum()
            hits[lab][1] += 1
            hits[lab][0] += int(hit)
            hits_group[group][lab][1] += 1
            hits_group[group][lab][0] += int(hit)
            if not hit:
                fn_rows.append((s.name, LABELS[lab], group, int(g.sum())))
        cats = categorize_components(res.maps, res.components)
        for comp, cat in zip(res.components, cats, strict=True):
            comp_total += 1
            lab = majority_label(gt, comp["mask"])
            comp_tp += int(lab > 0)
            confusion[(CAT_OF_LABEL[lab], cat.category)] += 1
            if lab == 0:
                fp_rows.append((s.name, group, round(comp["peak_index"], 2), comp["area_px"], cat.category))
    rows = [CAT_OF_LABEL[k] for k in (1, 2, 3, 0)]
    matrix = [[confusion[(r, c)] for c in CATS] for r in rows]
    correct = sum(confusion[(CAT_OF_LABEL[k], CAT_OF_LABEL[k])] for k in (1, 2, 3))
    on_true = sum(sum(m) for m in matrix[:3])
    return {
        "defect_recall": {
            LABELS[k]: {"detected": v[0], "total": v[1], "recall": r4(v[0] / v[1]) if v[1] else None}
            for k, v in hits.items()
        },
        "defect_recall_by_condition": {
            g: {LABELS[k]: f"{v[0]}/{v[1]}" for k, v in d.items()} for g, d in sorted(hits_group.items())
        },
        "region_precision": r4(comp_tp / comp_total) if comp_total else None,
        "regions_predicted": comp_total,
        "categorization": {
            "rows_true_label": rows,
            "cols_candidate": CATS,
            "matrix": matrix,
            "accuracy_on_true_defects_incl_indeterminado": r4(correct / on_true) if on_true else None,
            "indeterminado_rate_on_true_defects": r4(sum(m[3] for m in matrix[:3]) / on_true) if on_true else None,
        },
        "_fp": fp_rows,
        "_fn": fn_rows,
    }


def temporal(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    change_ratio = []
    example = None
    for p in pairs:
        res = compare(p["before"].image, p["after"].image)
        by_kind[p["kind"]][res.status] += 1
        if res.status == "ok" and p["kind"] != "incompativel":
            grown = (p["after"].mask > 0) & res.valid_mask
            rest = res.valid_mask & ~cv2.dilate((p["after"].mask > 0).astype(np.uint8), np.ones((21, 21))).astype(bool)
            if grown.sum() > 50:
                change_ratio.append(float(res.change_map[grown].mean() / (res.change_map[rest].mean() + 1e-6)))
            if example is None and p["kind"] == "deslocamento_medio":
                example = (p, res)
    incompat_ok = by_kind["incompativel"]["ok"]
    return {
        "pairs": len(pairs),
        "status_by_kind": {k: dict(v) for k, v in by_kind.items()},
        "false_alignment_on_incompatible": incompat_ok,
        "change_contrast_defect_vs_background_median": r4(np.median(change_ratio)) if change_ratio else None,
        "_example": example,
    }


def save_examples(samples: list[Any], results: list[Any]) -> list[str]:
    files = []
    picks: dict[str, list[int]] = {"acertos": [], "falsos_positivos": [], "falsos_negativos": []}
    for i, (s, r) in enumerate(zip(samples, results, strict=True)):
        gt = s.rendered.mask > 0
        if gt.any() and r.components and (r.mask.astype(bool) & gt).any():
            picks["acertos"].append(i)
        elif not gt.any() and r.components:
            picks["falsos_positivos"].append(i)
        elif gt.any() and not (r.mask.astype(bool) & gt).any():
            picks["falsos_negativos"].append(i)
    for name, idx in picks.items():
        idx = idx[:3]
        if not idx:
            continue
        fig, axes = plt.subplots(len(idx), 3, figsize=(11, 2.9 * len(idx)), squeeze=False)
        for row, i in enumerate(idx):
            s, r = samples[i], results[i]
            axes[row][0].imshow(cv2.cvtColor(s.rendered.image, cv2.COLOR_BGR2RGB))
            axes[row][0].set_title(f"{s.name} · {s.meta['condition_group']}", fontsize=9)
            axes[row][1].imshow(s.rendered.mask, cmap="viridis", vmin=0, vmax=3)
            axes[row][1].set_title("máscara sintética (verdade)", fontsize=9)
            axes[row][2].imshow(cv2.cvtColor(heatmap_overlay(s.rendered.image, r.index_map, r.mask), cv2.COLOR_BGR2RGB))
            axes[row][2].set_title(f"índice máx. {r.image_index:.2f} · {len(r.components)} regiões", fontsize=9)
            for ax in axes[row]:
                ax.axis("off")
        fig.suptitle(f"Exemplos — {name.replace('_', ' ')} (dados sintéticos)", fontsize=11)
        fig.tight_layout()
        path = OUT / f"examples_{name}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        files.append(path.name)
    return files


def plot_confusion(cat: dict[str, Any]) -> None:
    m = np.array(cat["matrix"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(cat["cols_candidate"])), [c.replace("_", "\n") for c in cat["cols_candidate"]], fontsize=8)
    ax.set_yticks(range(len(cat["rows_true_label"])), [c.replace("_", " ") for c in cat["rows_true_label"]], fontsize=8)
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > m.max() / 2 else "black")
    ax.set_xlabel("categoria candidata (heurística)")
    ax.set_ylabel("rótulo sintético da região detectada")
    ax.set_title("Categorização das regiões detectadas — teste sintético", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], aucs: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for name, (y, s) in curves.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{name} (AUROC {aucs[name]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("taxa de falsos positivos")
    ax.set_ylabel("taxa de verdadeiros positivos")
    ax.set_title("ROC nível de imagem — teste sintético", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "roc_image.png", dpi=120)
    plt.close(fig)


def plot_comparison(example: tuple[dict[str, Any], Any] | None) -> None:
    if example is None:
        return
    p, res = example
    imgs = [
        p["before"].image,
        p["after"].image,
        res.aligned_before,
        change_overlay(p["after"].image, res.change_map, res.change_mask),
    ]
    titles = ["antes", "depois (novo ponto de vista)", "antes alinhado ao depois", "mudança (1 − SSIM)"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.1))
    for ax, im, t in zip(axes, imgs, titles, strict=True):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"Comparação temporal ({res.alignment.method}, {int(res.alignment.metrics.get('inliers', 0))} inliers)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "comparison_example.png", dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=120)
    ap.add_argument("--quick", action="store_true", help="execução reduzida (CI)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    n_test = 30 if args.quick else args.n_test
    t0 = time.perf_counter()
    refs = reference_samples(45 if args.quick else 90)
    ref_imgs = [s.rendered.image for s in refs]
    test = labeled_samples("test", n_test)
    base_cfg = AnomalyConfig()
    print(f"treinando kNN com {len(ref_imgs)} referências normais...", flush=True)
    knn = PatchAnomalyModel(base_cfg).fit(ref_imgs, reference_hash="evaluation")
    main_eval = evaluate_detector(knn, test, keep=True)
    results = main_eval.pop("_results")
    region = region_level(test, results)

    detectors = {"knn": main_eval}
    if not args.quick:
        for det in ("iforest", "ocsvm"):
            print(f"comparando detector {det}...", flush=True)
            m = PatchAnomalyModel(replace(base_cfg, detector=det)).fit(ref_imgs, reference_hash="evaluation")
            detectors[det] = evaluate_detector(m, test)
        print("ablação: referências sem o grupo 'iluminacao'...", flush=True)
        no_light = [s.rendered.image for s in refs if s.meta["condition_group"] != "iluminacao"]
        m_nl = PatchAnomalyModel(base_cfg).fit(no_light, reference_hash="ablation")
        abl = evaluate_detector(m_nl, test, keep=True)
        abl_region = region_level(test, abl.pop("_results"))
        ablation = {
            "description": "Mesmo detector kNN, sem imagens de referência com sombra/iluminação extrema",
            "n_reference": len(no_light),
            "image_auroc": abl["image_auroc"],
            "image_fpr_at_threshold": abl["image_fpr_at_threshold"],
            "by_condition": abl["by_condition"],
            "defect_recall": abl_region["defect_recall"],
            "region_precision": abl_region["region_precision"],
        }
    else:
        ablation = None
    print("comparação temporal...", flush=True)
    pairs = temporal(temporal_pairs(16 if args.quick else 40))
    example = pairs.pop("_example")

    plot_roc({k: v["_roc"] for k, v in detectors.items()}, {k: v["image_auroc"] for k, v in detectors.items()})
    for v in detectors.values():
        v.pop("_roc")
    plot_confusion(region["categorization"])
    plot_comparison(example)
    examples = save_examples(test, results)

    fp, fn = region.pop("_fp"), region.pop("_fn")
    lines = [
        "# Falsos positivos e falsos negativos — teste sintético",
        "",
        f"Gerado em {datetime.now(UTC).isoformat()} por `scripts/evaluate.py`. Dados SINTÉTICOS.",
        "",
        f"## Falsos negativos (defeito com < {HIT_OVERLAP:.0%} de sobreposição): {len(fn)}",
        "",
        "| imagem | tipo | condição | pixels do defeito |",
        "|---|---|---|---|",
    ]
    lines += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in fn]
    lines += [
        "",
        f"## Falsos positivos (região sem defeito sintético): {len(fp)}",
        "",
        "| imagem | condição | índice de pico | área (px) | categoria candidata |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in fp]
    fn_by = Counter((b, c) for _, b, c, _ in fn)
    fp_by = Counter(b for _, b, *_ in fp)
    lines += [
        "",
        "## Agregado",
        "",
        "Falsos negativos por (tipo, condição): " + ", ".join(f"{k[0]}/{k[1]}={v}" for k, v in sorted(fn_by.items())),
        "",
        "Falsos positivos por condição: " + ", ".join(f"{k}={v}" for k, v in sorted(fp_by.items())),
    ]
    (OUT / "fp_fn_summary.md").write_text("\n".join(lines) + "\n")

    metrics = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "type": "SINTÉTICO",
            "generator": GENERATOR_VERSION,
            "split_seeds": SPLIT_SEEDS,
            "n_reference": len(ref_imgs),
            "n_test": len(test),
            "n_test_with_defect": int(sum(bool(s.rendered.mask.any()) for s in test)),
            "condition_groups": list(CONDITION_GROUPS),
            "warning": "Métricas sobre dados sintéticos não estimam desempenho em fachadas reais.",
        },
        "model": {k: v for k, v in knn.metadata.items() if k not in ("feature_names", "trained_at")},
        "operating_point": f"índice ≥ 1,00 (quantil {base_cfg.threshold_quantile:.3f} das referências de calibração)",
        "detection": detectors,
        "localization_and_categorization": region,
        "ablation_reference_without_illumination": ablation,
        "temporal_comparison": pairs,
        "artifacts": ["confusion_matrix.png", "roc_image.png", "comparison_example.png", "fp_fn_summary.md", *examples],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "config": asdict(base_cfg),
        },
        "duration_seconds": round(time.perf_counter() - t0, 1),
        "quick_mode": args.quick,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
    print(
        json.dumps(
            {
                "detection": {
                    k: {kk: vv for kk, vv in v.items() if kk != "by_condition"} for k, v in detectors.items()
                },
                "defect_recall": region["defect_recall"],
                "region_precision": region["region_precision"],
                "temporal": pairs["status_by_kind"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"artefatos em {OUT}")


if __name__ == "__main__":
    main()
