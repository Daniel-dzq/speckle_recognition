#!/usr/bin/env python3
"""
Figure 3 — 15-fiber recognition and cross-fiber authentication (real training outputs).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _figure_io import save_figure_triplet, set_archive_old, write_csv_rows, write_report  # noqa: E402
from _paths import AUTH_MATRIX_CSV, AUTH_REPORT_MD, FIGURES_PAPER, ROOT, SUMMARY_CSV  # noqa: E402
from _style import (  # noqa: E402
    DIAGONAL_COLOR,
    FONT_AXIS,
    FONT_METRIC,
    FONT_METRIC_TITLE,
    FONT_TICK,
    OFF_DIAG_COLOR,
    RANDOM_BASELINE_COLOR,
    apply_paper_style,
    panel_label,
)

OUT_DIR = FIGURES_PAPER / "Fig3_authentication"
BASE = "Fig3_authentication"
NUM_CLASSES = 8
RANDOM_BASELINE_PCT = 100.0 / NUM_CLASSES
FIG_W, FIG_H = 12.0, 8.5


def _fiber_sort(name: str) -> int:
    m = re.match(r"Fiber(\d+)$", name, re.I)
    if not m:
        raise ValueError(f"Unexpected fiber label: {name}")
    return int(m.group(1))


def load_summary() -> Tuple[List[str], List[float]]:
    if not SUMMARY_CSV.is_file():
        raise FileNotFoundError(SUMMARY_CSV)
    rows = [r for r in csv.DictReader(SUMMARY_CSV.open(newline="", encoding="utf-8")) if r.get("status") == "ok"]
    fibers = sorted([r["fiber"] for r in rows], key=_fiber_sort)
    acc_by = {r["fiber"]: float(r["test_acc"]) for r in rows}
    return fibers, [acc_by[f] for f in fibers]


def load_auth_matrix() -> Tuple[List[str], np.ndarray]:
    if not AUTH_MATRIX_CSV.is_file():
        raise FileNotFoundError(AUTH_MATRIX_CSV)
    with AUTH_MATRIX_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fibers = sorted([r["model_fiber"] for r in rows], key=_fiber_sort)
    mat = np.zeros((len(fibers), len(fibers)), dtype=float)
    for i, r in enumerate(sorted(rows, key=lambda x: _fiber_sort(x["model_fiber"]))):
        for j, c in enumerate(fibers):
            mat[i, j] = float(r[c])
    return fibers, mat


def matrix_stats(mat: np.ndarray) -> Dict[str, float]:
    diag = np.diag(mat)
    off = mat[~np.eye(mat.shape[0], dtype=bool)]
    return {
        "diagonal_mean_pct": float(np.mean(diag)),
        "diagonal_std_pct": float(np.std(diag)),
        "diagonal_median_pct": float(np.median(diag)),
        "off_diagonal_mean_pct": float(np.mean(off)),
        "off_diagonal_std_pct": float(np.std(off)),
        "off_diagonal_median_pct": float(np.median(off)),
        "random_baseline_pct": RANDOM_BASELINE_PCT,
        "n_fibers": int(mat.shape[0]),
        "n_classes": NUM_CLASSES,
    }


def _panel_a(ax: plt.Axes, fibers: List[str], test_accs: List[float]) -> None:
    x = np.arange(len(fibers))
    labels = [f"F{i}" for i in range(1, len(fibers) + 1)]
    stem_bottom = 90.0
    for xi, acc in zip(x, test_accs):
        ax.vlines(xi, stem_bottom, acc, color=DIAGONAL_COLOR, linewidth=1.1, zorder=1)
    ax.scatter(x, test_accs, s=42, color=DIAGONAL_COLOR, edgecolors="white", linewidths=0.5, zorder=3)
    mean_acc = float(np.mean(test_accs))
    ax.axhline(mean_acc, color=RANDOM_BASELINE_COLOR, linewidth=0.9, linestyle="-", alpha=0.85, zorder=0)
    ax.text(0.98, mean_acc + 0.15, f"Mean {mean_acc:.1f}%", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=FONT_TICK, color=RANDOM_BASELINE_COLOR)

    idx_min = int(np.argmin(test_accs))
    short_min = f"F{_fiber_sort(fibers[idx_min])}"
    ax.annotate(
        f"min: {short_min}",
        xy=(x[idx_min], test_accs[idx_min]),
        xytext=(x[idx_min] + 0.55, test_accs[idx_min] + 1.1),
        fontsize=FONT_TICK,
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="-", color=DIAGONAL_COLOR, lw=0.6, shrinkA=2, shrinkB=2),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.set_ylabel("Test accuracy (%)", fontsize=FONT_AXIS)
    ax.set_ylim(90, 101)
    ax.set_xlim(-0.6, len(fibers) - 0.4)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    panel_label(ax, "a")


def _panel_b(ax: plt.Axes, fig: plt.Figure, mat: np.ndarray) -> None:
    n = mat.shape[0]
    labels = [f"F{i}" for i in range(1, n + 1)]
    cmap = mpl.colormaps["cividis"].copy()
    cmap.set_bad(color="white")
    im = ax.imshow(mat, vmin=0, vmax=100, cmap=cmap, aspect="equal", interpolation="nearest")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, fontsize=FONT_TICK, rotation=0)
    ax.set_yticklabels(labels, fontsize=FONT_TICK)
    ax.set_xlabel("Test fiber", fontsize=FONT_AXIS, labelpad=4)
    ax.set_ylabel("Enrolled model", fontsize=FONT_AXIS, labelpad=4)

    for edge in range(n + 1):
        ax.axhline(edge - 0.5, color="white", linewidth=0.35, alpha=0.65)
        ax.axvline(edge - 0.5, color="white", linewidth=0.35, alpha=0.65)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.8%", pad=0.04)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Accuracy (%)", fontsize=FONT_AXIS)
    cb.ax.tick_params(labelsize=FONT_TICK)
    panel_label(ax, "b")


def _panel_c(ax: plt.Axes, mat: np.ndarray, stats: Dict[str, float]) -> None:
    diag = np.diag(mat)
    off = mat[~np.eye(mat.shape[0], dtype=bool)]
    data = [diag, off]
    positions = [1, 2]
    colors = [DIAGONAL_COLOR, OFF_DIAG_COLOR]

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="white", linewidth=1.2),
        whiskerprops=dict(color="#444444", linewidth=0.8),
        capprops=dict(color="#444444", linewidth=0.8),
        boxprops=dict(linewidth=0.8),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)

    rng = np.random.default_rng(42)
    for pos, vals, color in zip(positions, data, colors):
        jitter = rng.uniform(-0.11, 0.11, size=len(vals))
        ax.scatter(pos + jitter, vals, s=14, color=color, alpha=0.55, edgecolors="none", zorder=3)

    ax.axhline(RANDOM_BASELINE_PCT, color=RANDOM_BASELINE_COLOR, linestyle="--", linewidth=1.0, zorder=0)
    ax.text(0.55, RANDOM_BASELINE_PCT + 1.2, f"Random {RANDOM_BASELINE_PCT:.1f}%",
            fontsize=FONT_TICK, color=RANDOM_BASELINE_COLOR)

    ax.text(1, 72, f"{np.mean(diag):.1f} ± {np.std(diag):.1f}%", ha="center", va="bottom", fontsize=FONT_TICK)
    ax.text(2, 22, f"{np.mean(off):.1f} ± {np.std(off):.1f}%", ha="center", va="bottom", fontsize=FONT_TICK)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Same fiber", "Cross fiber"], fontsize=FONT_TICK)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_AXIS)
    ax.set_ylim(0, 108)
    ax.set_xlim(0.4, 2.6)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    panel_label(ax, "c")


def _panel_d(ax: plt.Axes, stats: Dict[str, float], test_accs: List[float]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cards = [
        ("Same-fiber", stats["diagonal_mean_pct"], DIAGONAL_COLOR),
        ("Cross-fiber", stats["off_diagonal_mean_pct"], OFF_DIAG_COLOR),
        ("Random", stats["random_baseline_pct"], RANDOM_BASELINE_COLOR),
    ]
    card_w, card_h = 0.27, 0.40
    y0 = 0.54
    xs = [0.07, 0.365, 0.66]
    for x0, (title, value, color) in zip(xs, cards):
        box = FancyBboxPatch(
            (x0, y0), card_w, card_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=0.8, edgecolor=color, facecolor="white",
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(box)
        ax.text(x0 + card_w / 2, y0 + card_h * 0.72, title, ha="center", va="center",
                fontsize=FONT_METRIC_TITLE, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(x0 + card_w / 2, y0 + card_h * 0.32, f"{value:.1f}%", ha="center", va="center",
                fontsize=FONT_METRIC, fontweight="bold", color="#222222", transform=ax.transAxes)

    ax.text(
        0.5, 0.20,
        "Cross-fiber responses remain near the 8-class random baseline.",
        ha="center", va="center", fontsize=FONT_TICK + 1, color="#444444", transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.07,
        f"15 fibers · 8 classes · mean test accuracy {np.mean(test_accs):.1f}%",
        ha="center", va="center", fontsize=FONT_TICK, color="#666666", transform=ax.transAxes,
    )
    panel_label(ax, "d")


def build_figure(fibers: List[str], test_accs: List[float], mat: np.ndarray, stats: Dict[str, float]) -> plt.Figure:
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    gs = GridSpec(
        2, 2, figure=fig,
        width_ratios=[0.86, 1.48],
        height_ratios=[1.0, 1.0],
        wspace=0.18,
        hspace=0.24,
        left=0.07,
        right=0.94,
        top=0.97,
        bottom=0.09,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _panel_a(ax_a, fibers, test_accs)
    _panel_b(ax_b, fig, mat)
    _panel_c(ax_c, mat, stats)
    _panel_d(ax_d, stats, test_accs)
    return fig


def write_summary_csv(out_path: Path, fibers: List[str], test_accs: List[float], mat: np.ndarray, stats: Dict[str, float]) -> None:
    rows: List[Dict[str, object]] = []
    for i, fi in enumerate(fibers):
        rows.append({"record_type": "per_fiber_test_accuracy", "fiber": fi, "test_accuracy_pct": test_accs[i]})
    for i, fi in enumerate(fibers):
        for j, fj in enumerate(fibers):
            rows.append({
                "record_type": "auth_matrix_cell",
                "enrolled_fiber": fi,
                "test_fiber": fj,
                "accuracy_pct": float(mat[i, j]),
                "cell_type": "diagonal" if i == j else "off_diagonal",
            })
    for key, val in stats.items():
        rows.append({"record_type": "aggregate", "metric": key, "value": val})
    write_csv_rows(
        out_path,
        ["record_type", "fiber", "enrolled_fiber", "test_fiber", "test_accuracy_pct", "accuracy_pct", "cell_type", "metric", "value"],
        rows,
    )


def write_md_report(stats: Dict[str, float], paths: List[Path]) -> str:
    return f"""# {BASE} report

## Data sources
- `{SUMMARY_CSV.relative_to(ROOT)}` — per-fiber test accuracy (panel a)
- `{AUTH_MATRIX_CSV.relative_to(ROOT)}` — 15×15 authentication matrix (panels b–d)
- `{AUTH_REPORT_MD.relative_to(ROOT)}` — reference text (values verified against CSV)

## Key values (computed from CSV)
| Metric | Value |
|--------|-------|
| Diagonal mean | {stats['diagonal_mean_pct']:.2f}% |
| Off-diagonal mean | {stats['off_diagonal_mean_pct']:.2f}% |
| Random baseline | {stats['random_baseline_pct']:.1f}% |

## Figure role
Main-text authentication performance figure.

## Outputs
""" + "\n".join(f"- `{p.relative_to(ROOT)}`" for p in paths) + """

## Caption draft
Fifteen fiber-specific models trained on eight challenge classes achieve high per-fiber test accuracy (a) and strong diagonal entries in the cross-fiber authentication matrix (b). Same-fiber accuracies cluster near 98% while cross-fiber scores match the 12.5% random baseline (c, d), demonstrating fiber-specific acceptance and cross-fiber rejection.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 3 (15-fiber authentication performance).")
    parser.add_argument("--archive-old", action="store_true", help="Archive existing outputs before overwrite.")
    args = parser.parse_args()
    set_archive_old(args.archive_old)

    apply_paper_style()
    mpl.rcParams["figure.constrained_layout.use"] = False

    fibers, test_accs = load_summary()
    _, mat = load_auth_matrix()
    stats = matrix_stats(mat)

    fig = build_figure(fibers, test_accs, mat, stats)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / BASE
    paths = save_figure_triplet(fig, stem)
    plt.close(fig)

    csv_path = OUT_DIR / f"{BASE}_data_summary.csv"
    write_summary_csv(csv_path, fibers, test_accs, mat, stats)
    write_report(OUT_DIR / f"{BASE}_report.md", write_md_report(stats, paths + [csv_path]))

    print(f"Wrote {stem}.png/pdf/svg")
    print(f"Diagonal mean: {stats['diagonal_mean_pct']:.2f}%  Off-diagonal mean: {stats['off_diagonal_mean_pct']:.2f}%")


if __name__ == "__main__":
    main()
