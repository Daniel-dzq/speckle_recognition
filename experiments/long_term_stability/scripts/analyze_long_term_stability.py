#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze and plot JPEG folders: disturbance_sensitivity, long_term_stability,
power_common_mode — using analysis pipeline (dataset discovery, preprocessing,
extract_features, intra/inter & temporal NCC metrics, plots consistent with analysis/).

Outputs: figures/new_datasets_analysis/*.png|.pdf|.svg

Run from repo root:
    python scripts/analyze_new_datasets.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

RELEASE_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = EXP_ROOT / "data"
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.caching.cache import FeatureCache  # noqa: E402
from analysis.experiments._features import CaptureFeature, extract_features  # noqa: E402
from analysis.io.dataset import DatasetLayout, discover_captures  # noqa: E402
from analysis.metrics.basic import normalized_cross_correlation  # noqa: E402
from analysis.metrics.group import intra_inter_ratio, within_class_similarity  # noqa: E402
from analysis.metrics.stability import temporal_stability_score  # noqa: E402
from analysis.plotting.style import (  # noqa: E402
    PALETTE,
    apply_style,
    add_panel_label,
    save_figure,
    DOUBLE_COL_W,
)
from analysis.preprocessing.pipeline import PreprocessConfig  # noqa: E402

OUT_DIR = EXP_ROOT / "outputs" / "figures"
CACHE_ROOT = EXP_ROOT / "outputs" / ".analysis_cache"


def fiber_color(fname: str) -> str:
    m = re.match(r"Fiber(\d+)", fname, re.I)
    if m:
        return PALETTE[(int(m.group(1)) - 1) % len(PALETTE)]
    return PALETTE[0]


def preprocess_config() -> PreprocessConfig:
    return PreprocessConfig(
        grayscale=True,
        center_crop_size=400,
        resize=112,
        normalize="minmax",
        frame_strategy="middle",
        n_frames=1,
        aggregate="mean",
    )


def load_features(data_root: Path, bucket: str, log: logging.Logger) -> Tuple[List[CaptureFeature], DatasetLayout]:
    layout = DatasetLayout.from_config(
        {
            "root": str(data_root),
            "layout": "flat_fiber_repeat",
        },
        base_dir=None,
    )
    caps = discover_captures(layout)
    log.info("%s: %d captures", data_root.name, len(caps))
    cache = FeatureCache(root=CACHE_ROOT, bucket=bucket, enabled=True)
    feats = extract_features(caps, preprocess_config(), cache=cache, logger=log)
    log.info("%s: %d feature rows", data_root.name, len(feats))
    return feats, layout


def fig_disturbance(feats: List[CaptureFeature], log: logging.Logger) -> None:
    mats, labels = _stack_xy(feats)
    intra_inter = intra_inter_ratio(mats, labels)
    wsim = within_class_similarity(mats, labels)

    fibers = sorted(wsim.keys(), key=_fiber_sort)
    ys = [wsim[f] for f in fibers]
    colors = [fiber_color(f) for f in fibers]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, DOUBLE_COL_W / 2.3), layout="none")
    ax = axes[0]
    x = np.arange(len(fibers))
    ax.bar(x, ys, color=colors, edgecolor="white", linewidth=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("Fiber", "F") for f in fibers], rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("Mean within-fiber NCC")
    ax.set_title("Speckle consistency (disturbance sensitivity)")
    add_panel_label(ax, "(a)")

    ax2 = axes[1]
    ax2.axis("off")
    lines = [
        "Pooled fiber-discriminability (same metric set as length experiment):",
        f"  Intra-class L2 distance: {intra_inter['intra']:.4f}",
        f"  Inter-class L2 distance: {intra_inter['inter']:.4f}",
        f"  Inter/intra distance ratio: {intra_inter['ratio']:.4f}",
        "",
        "Higher mean within-fiber NCC ⇒ speckles more similar across repeated",
        "captures under disturbance; larger inter/intra distance ratio ⇒ better separability.",
    ]
    ax2.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=8, family="monospace")

    fig.suptitle("disturbance_sensitivity", fontsize=10, fontweight="bold")
    fig.subplots_adjust(bottom=0.22, top=0.88)
    save_figure(fig, OUT_DIR / "disturbance_sensitivity_analysis", formats=("png", "pdf"))
    plt.close(fig)
    log.info("Wrote disturbance_sensitivity_analysis")


def _fiber_sort(name: str) -> Tuple[int, str]:
    m = re.match(r"Fiber(\d+)$", name, re.I)
    return (int(m.group(1)), name) if m else (9999, name)


def _parse_power(folder: str) -> int | None:
    m = re.match(r"^P(\d+)$", folder, re.I)
    return int(m.group(1)) if m else None


def fig_long_term(feats: List[CaptureFeature], log: logging.Logger) -> None:
    by_fiber: Dict[str, List[CaptureFeature]] = defaultdict(list)
    for cf in feats:
        by_fiber[cf.capture.fiber].append(cf)
    for fib in by_fiber:
        by_fiber[fib].sort(key=lambda c: c.capture.repeat if c.capture.repeat is not None else 0)

    fibers = sorted(by_fiber.keys(), key=_fiber_sort)
    consec = []
    vs_first = []
    for f in fibers:
        seq = [c.vector for c in by_fiber[f]]
        st = temporal_stability_score(seq)
        consec.append(st["consecutive_ncc"])
        vs_first.append(st["vs_first_ncc"])

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, DOUBLE_COL_W / 2.35), layout="none")
    x = np.arange(len(fibers))

    ax = axes[0]
    w = 0.35
    adj_color = PALETTE[0]
    drift_color = PALETTE[1]
    ax.bar(x - w / 2, consec, width=w, label="Mean adjacent-frame NCC", color=adj_color,
           alpha=0.92, edgecolor="white", linewidth=0.3)
    ax.bar(x + w / 2, vs_first, width=w, label="Mean NCC vs first acquisition", color=drift_color,
           alpha=0.88, edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("Fiber", "F") for f in fibers], rotation=55, ha="right", fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("NCC")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=2, frameon=True)
    add_panel_label(ax, "(a)")

    example = "Fiber1" if "Fiber1" in by_fiber else fibers[0]
    seq_ex = [c.vector for c in by_fiber[example]]
    ncc_tf = []
    reps = []
    for i in range(1, len(seq_ex)):
        ncc_tf.append(normalized_cross_correlation(seq_ex[0], seq_ex[i]))
        ri = by_fiber[example][i].capture.repeat
        reps.append(int(ri) if ri is not None else i + 1)

    ax2 = axes[1]
    if ncc_tf:
        ax2.plot(reps, ncc_tf, "o-", color=fiber_color(example), lw=1.0, markersize=3)
    ax2.set_xlabel(f"Acquisition index (#) — {example}")
    ax2.set_ylabel("NCC vs first frame")
    ax2.set_ylim(0.3, 1.05)
    add_panel_label(ax2, "(b)")

    fig.suptitle("Long-term temporal stability (repeat captures)", fontsize=10, fontweight="bold")
    fig.subplots_adjust(bottom=0.32, top=0.88)
    save_figure(fig, OUT_DIR / "long_term_stability_analysis", formats=("png", "pdf"))
    plt.close(fig)
    log.info("Wrote long_term_stability_analysis")


def fig_power(feats: List[CaptureFeature], log: logging.Logger) -> None:
    by_pg: Dict[str, List[CaptureFeature]] = defaultdict(list)
    for cf in feats:
        lg = cf.capture.length_group or "?"
        by_pg[lg].append(cf)

    powers_sorted = sorted(by_pg.keys(), key=lambda k: (_parse_power(k) or 1e9, k))

    ratios, intras, inters = [], [], []
    for pg in powers_sorted:
        sub = by_pg[pg]
        mats, labels = _stack_xy(sub)
        d = intra_inter_ratio(mats, labels)
        ratios.append(d["ratio"])
        intras.append(d["intra"])
        inters.append(d["inter"])

    xs = [(_parse_power(p) if _parse_power(p) is not None else i) for i, p in enumerate(powers_sorted)]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, DOUBLE_COL_W / 2.35), layout="none")

    ax = axes[0]
    ax.plot(xs, ratios, "o-", color=PALETTE[0], lw=1.2)
    ax.set_xlabel(r"Pump level tag (extracted from folder e.g. P10 $\rightarrow$ 10)")
    ax.set_ylabel("Inter / intra (L2 distance ratio)")
    add_panel_label(ax, "(a)")

    ax2 = axes[1]
    ax2.plot(xs, intras, "s-", label="Intra-class", color=PALETTE[1], lw=1.0, markersize=3)
    ax2.plot(xs, inters, "^-", label="Inter-class", color=PALETTE[3], lw=1.0, markersize=3)
    ax2.set_xlabel("Pump level tag (numeric)")
    ax2.set_ylabel("Mean pairwise L2 distance")
    ax2.legend(fontsize=6)
    add_panel_label(ax2, "(b)")

    fig.suptitle("power_common_mode", fontsize=10, fontweight="bold")
    fig.subplots_adjust(top=0.88)
    save_figure(fig, OUT_DIR / "power_common_mode_analysis", formats=("png", "pdf"))
    plt.close(fig)
    log.info("Wrote power_common_mode_analysis")


def _stack_xy(feats: Sequence[CaptureFeature]) -> Tuple[np.ndarray, List[str]]:
    mats = np.stack([f.vector.astype(np.float64) for f in feats], axis=0)
    labels = [f.capture.fiber for f in feats]
    return mats, labels


def collect_metrics_summary(
    fd: List[CaptureFeature],
    fl: List[CaptureFeature],
    fp: List[CaptureFeature],
) -> dict:
    """Serializable metrics for downstream reporting."""
    md, lab = _stack_xy(fd)
    dist_ii = intra_inter_ratio(md, lab)
    dist_ncc = within_class_similarity(md, lab)

    by_fiber: Dict[str, List[CaptureFeature]] = defaultdict(list)
    for cf in fl:
        by_fiber[cf.capture.fiber].append(cf)
    for fib in by_fiber:
        by_fiber[fib].sort(key=lambda c: c.capture.repeat if c.capture.repeat is not None else 0)
    fibers_lt = sorted(by_fiber.keys(), key=_fiber_sort)
    lt_rows = []
    for f in fibers_lt:
        seq = [c.vector for c in by_fiber[f]]
        st = temporal_stability_score(seq)
        lt_rows.append(
            {
                "fiber": f,
                "consecutive_ncc": float(st["consecutive_ncc"]),
                "vs_first_ncc": float(st["vs_first_ncc"]),
                "n_samples": len(seq),
            }
        )

    by_pg: Dict[str, List[CaptureFeature]] = defaultdict(list)
    for cf in fp:
        lg = cf.capture.length_group or "?"
        by_pg[lg].append(cf)
    powers_sorted = sorted(by_pg.keys(), key=lambda k: (_parse_power(k) or 1e9, k))
    power_rows = []
    for pg in powers_sorted:
        sub = by_pg[pg]
        m2, l2 = _stack_xy(sub)
        d = intra_inter_ratio(m2, l2)
        power_rows.append(
            {
                "folder": pg,
                "power_index": _parse_power(pg),
                "inter_intra_ratio": float(d["ratio"]) if d["ratio"] == d["ratio"] else None,
                "intra_l2": float(d["intra"]) if d["intra"] == d["intra"] else None,
                "inter_l2": float(d["inter"]) if d["inter"] == d["inter"] else None,
                "n_images": len(sub),
            }
        )

    return {
        "disturbance_sensitivity": {
            "n_captures": len(fd),
            "pooled_intra_l2": float(dist_ii["intra"]) if dist_ii["intra"] == dist_ii["intra"] else None,
            "pooled_inter_l2": float(dist_ii["inter"]) if dist_ii["inter"] == dist_ii["inter"] else None,
            "pooled_inter_intra_ratio": float(dist_ii["ratio"]) if dist_ii["ratio"] == dist_ii["ratio"] else None,
            "within_fiber_mean_ncc": {k: float(v) for k, v in sorted(dist_ncc.items(), key=lambda x: _fiber_sort(x[0]))},
        },
        "long_term_stability": {
            "n_captures": len(fl),
            "per_fiber": lt_rows,
        },
        "power_common_mode": {
            "n_captures": len(fp),
            "per_power_setting": power_rows,
        },
    }


def write_metrics_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    apply_style()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("long_term_stability")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Data  -> %s", DATA_DIR)
    log.info("Output -> %s", OUT_DIR)

    fl, _ = load_features(DATA_DIR, "long_term_feats", log)
    fig_long_term(fl, log)

    summary = collect_metrics_summary([], fl, [])
    write_metrics_json(
        {"long_term_stability": summary["long_term_stability"]},
        EXP_ROOT / "outputs" / "metrics_summary.json",
    )
    log.info("Wrote %s", EXP_ROOT / "outputs" / "metrics_summary.json")

    print("Done. Figures under:", OUT_DIR)


if __name__ == "__main__":
    main()
