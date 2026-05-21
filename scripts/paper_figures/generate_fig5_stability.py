#!/usr/bin/env python3
"""Figure 5 — Long-term stability (supplementary summary from metrics JSON)."""
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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _figure_io import save_figure_triplet, set_archive_old, write_csv_rows, write_report  # noqa: E402
from _paths import FIGURES_PAPER, LT_METRICS, LT_METRICS_EXP, ROOT  # noqa: E402
from _style import FONT_AXIS, FONT_TICK, INTRA_COLOR, INTER_COLOR, apply_paper_style, panel_label  # noqa: E402

OUT_DIR = FIGURES_PAPER / "Fig5_stability"
BASE = "Fig5_stability"


def _fiber_sort(name: str) -> int:
    m = re.match(r"Fiber(\d+)$", name, re.I)
    return int(m.group(1)) if m else 9999


def load_metrics() -> Dict[str, Any]:
    for path in (LT_METRICS, LT_METRICS_EXP):
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            block = data.get("long_term_stability")
            if block:
                return block
    raise FileNotFoundError("No long_term_stability metrics found.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 5 (long-term stability summary).")
    parser.add_argument("--full-analysis", action="store_true")
    parser.add_argument("--archive-old", action="store_true")
    args = parser.parse_args()
    set_archive_old(args.archive_old)

    if args.full_analysis:
        script = ROOT / "experiments" / "long_term_stability" / "scripts" / "analyze_long_term_stability.py"
        import runpy
        runpy.run_path(str(script), run_name="__main__")

    apply_paper_style()
    mpl.rcParams["figure.constrained_layout.use"] = False
    block = load_metrics()
    per_fiber: List[Dict[str, Any]] = sorted(block["per_fiber"], key=lambda r: _fiber_sort(r["fiber"]))
    fibers = [r["fiber"] for r in per_fiber]
    consec = [float(r["consecutive_ncc"]) for r in per_fiber]
    vs_first = [float(r["vs_first_ncc"]) for r in per_fiber]

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.2), facecolor="white")
    x = np.arange(len(fibers))
    w = 0.34
    ax.bar(x - w / 2, consec, width=w, label="Adjacent NCC", color=INTRA_COLOR, edgecolor="white", linewidth=0.25, alpha=0.9)
    ax.bar(x + w / 2, vs_first, width=w, label="NCC vs first", color=INTER_COLOR, edgecolor="white", linewidth=0.25, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("Fiber", "F") for f in fibers], rotation=55, ha="right", fontsize=FONT_TICK)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("NCC", fontsize=FONT_AXIS)
    ax.legend(
        fontsize=FONT_TICK,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    panel_label(ax, "a")
    fig.subplots_adjust(bottom=0.28, left=0.08, right=0.98, top=0.82)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / BASE
    paths = save_figure_triplet(fig, stem)
    plt.close(fig)

    csv_path = OUT_DIR / f"{BASE}_data_summary.csv"
    write_csv_rows(csv_path, ["fiber", "consecutive_ncc", "vs_first_ncc", "n_samples"], per_fiber)

    src = LT_METRICS if LT_METRICS.is_file() else LT_METRICS_EXP
    write_report(OUT_DIR / f"{BASE}_report.md", f"""# {BASE} report

## Status
**Supplementary only** — aggregated per-fiber NCC; no per-time-index accuracy series in summary JSON.

## Data source
- `{src.relative_to(ROOT)}` (n_captures={block.get('n_captures')})

## Missing for main-text figure
- Classifier accuracy vs time
- Per-acquisition drift curves (run `analyze_long_term_stability.py` on JPEG data for experiment figures)

## Outputs
""" + "\n".join(f"- `{p.relative_to(ROOT)}`" for p in paths + [csv_path]))
    print(f"Wrote {stem.name}.* (supplementary)")


if __name__ == "__main__":
    main()
