#!/usr/bin/env python3
"""Figure 6 — Disturbance sensitivity (supplementary summary from metrics JSON)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _figure_io import save_figure_triplet, set_archive_old, write_csv_rows, write_report  # noqa: E402
from _paths import DS_METRICS, DS_METRICS_EXP, FIGURES_PAPER, ROOT  # noqa: E402
from _style import (  # noqa: E402
    FONT_AXIS,
    FONT_METRIC,
    FONT_METRIC_TITLE,
    FONT_TICK,
    INTRA_COLOR,
    INTER_COLOR,
    RATIO_COLOR,
    apply_paper_style,
    panel_label,
)

OUT_DIR = FIGURES_PAPER / "Fig6_disturbance"
BASE = "Fig6_disturbance"


def _fiber_sort(name: str) -> int:
    m = re.match(r"Fiber(\d+)$", name, re.I)
    return int(m.group(1)) if m else 9999


def load_metrics() -> Dict[str, Any]:
    for path in (DS_METRICS, DS_METRICS_EXP):
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            block = data.get("disturbance_sensitivity")
            if block:
                return block
    raise FileNotFoundError("No disturbance_sensitivity metrics found.")


def _metric_cards(ax: plt.Axes, block: Dict[str, Any]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    cards = [
        ("Intra L2", float(block["pooled_intra_l2"]), INTRA_COLOR),
        ("Inter L2", float(block["pooled_inter_l2"]), INTER_COLOR),
        ("Inter/intra", float(block["pooled_inter_intra_ratio"]), RATIO_COLOR),
    ]
    card_w, card_h = 0.28, 0.55
    xs = [0.08, 0.36, 0.64]
    y0 = 0.22
    for x0, (title, value, color) in zip(xs, cards):
        box = FancyBboxPatch(
            (x0, y0), card_w, card_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=0.8, edgecolor=color, facecolor="white",
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(box)
        ax.text(x0 + card_w / 2, y0 + card_h * 0.68, title, ha="center", va="center",
                fontsize=FONT_METRIC_TITLE, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(x0 + card_w / 2, y0 + card_h * 0.28, f"{value:.2f}", ha="center", va="center",
                fontsize=FONT_METRIC, fontweight="bold", color="#222222", transform=ax.transAxes)
    ax.text(0.5, 0.88, f"n = {block.get('n_captures')} captures", ha="center", fontsize=FONT_AXIS, transform=ax.transAxes)
    panel_label(ax, "b")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 6 (disturbance sensitivity summary).")
    parser.add_argument("--full-analysis", action="store_true")
    parser.add_argument("--archive-old", action="store_true")
    args = parser.parse_args()
    set_archive_old(args.archive_old)

    if args.full_analysis:
        script = ROOT / "experiments" / "disturbance_sensitivity" / "scripts" / "analyze_disturbance_sensitivity.py"
        import runpy
        runpy.run_path(str(script), run_name="__main__")

    apply_paper_style()
    mpl.rcParams["figure.constrained_layout.use"] = False
    block = load_metrics()
    wsim: Dict[str, float] = block.get("within_fiber_mean_ncc") or {}
    fibers = sorted(wsim.keys(), key=_fiber_sort)
    values = [float(wsim[f]) for f in fibers]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), facecolor="white", gridspec_kw={"width_ratios": [1.45, 1.0]})

    ax = axes[0]
    x = np.arange(len(fibers))
    ax.bar(x, values, color=INTRA_COLOR, edgecolor="white", linewidth=0.3, width=0.72)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("Fiber", "F") for f in fibers], rotation=55, ha="right", fontsize=FONT_TICK)
    ax.set_ylabel("Mean within-fiber NCC", fontsize=FONT_AXIS)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    panel_label(ax, "a")

    _metric_cards(axes[1], block)
    fig.subplots_adjust(wspace=0.28, bottom=0.28, left=0.07, right=0.98, top=0.90)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / BASE
    paths = save_figure_triplet(fig, stem)
    plt.close(fig)

    rows = [{"fiber": f, "within_fiber_mean_ncc": wsim[f]} for f in fibers]
    rows.append({
        "fiber": "POOLED",
        "pooled_intra_l2": block.get("pooled_intra_l2"),
        "pooled_inter_l2": block.get("pooled_inter_l2"),
        "pooled_inter_intra_ratio": block.get("pooled_inter_intra_ratio"),
        "n_captures": block.get("n_captures"),
    })
    csv_path = OUT_DIR / f"{BASE}_data_summary.csv"
    write_csv_rows(
        csv_path,
        ["fiber", "within_fiber_mean_ncc", "pooled_intra_l2", "pooled_inter_l2", "pooled_inter_intra_ratio", "n_captures"],
        rows,
    )

    src = DS_METRICS if DS_METRICS.is_file() else DS_METRICS_EXP
    write_report(OUT_DIR / f"{BASE}_report.md", f"""# {BASE} report

## Status
**Supplementary only** — repeat captures per fiber; no graded disturbance-level sweep in data.

## Data source
- `{src.relative_to(ROOT)}`

## Missing for main-text figure
- Accuracy vs disturbance level
- Confidence degradation vs level

## Outputs
""" + "\n".join(f"- `{p.relative_to(ROOT)}`" for p in paths + [csv_path]))
    print(f"Wrote {stem.name}.* (supplementary)")


if __name__ == "__main__":
    main()
