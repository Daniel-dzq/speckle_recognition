#!/usr/bin/env python3
"""
Fig. 3 — Fiber length optimization (verified `length_optimization_green` only).
Layout: 2×2 panels, no twin y-axes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR.parent))

from paper_figures.io_utils import archive_existing_outputs  # noqa: E402
from paper_figures.style import (  # noqa: E402
    COL_GREEN,
    COL_RED,
    COL_BLUE,
    COL_ORANGE,
    COL_MAROON,
    DOUBLE_COL_W,
    FONT_SIZE_LEGEND,
    apply_style,
    figure_root,
    panel_label,
    save_figure_bundle,
    write_table_csv,
)

CSV_PATH = (
    RELEASE_ROOT
    / "experiments/length_optimization/outputs/length_optimization_green/tables/per_length_summary.csv"
)
OPTIMAL_PATH = (
    RELEASE_ROOT / "experiments/length_optimization/outputs/length_optimization_green/optimal_length.json"
)
ALLOWED_MM = {80, 90, 110, 130, 160}
SELECTED_CM = 9.0
COL_VLINE = "#999999"


def main() -> None:
    apply_style()
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Missing canonical length data: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    required = {
        "length_mm", "green_loss_dB_mean", "red_loss_dB_mean",
        "intra_distance_mean", "inter_distance", "inter_intra_ratio",
        "entropy_bits_mean", "entropy_bits_std",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"per_length_summary.csv missing columns: {sorted(missing)}")

    lengths = set(int(x) for x in df["length_mm"].dropna().unique())
    if not lengths.issubset(ALLOWED_MM):
        bad = lengths - ALLOWED_MM
        raise ValueError(
            f"Unexpected length_mm values {bad}. Refusing to plot mixed-era lengths "
            f"(expected subset of {sorted(ALLOWED_MM)} mm)."
        )

    df = df.sort_values("length_mm").reset_index(drop=True)
    xs = (df["length_mm"].astype(float) / 10.0).to_numpy()
    green = df["green_loss_dB_mean"].astype(float).to_numpy()
    green_e = df["green_loss_dB_std"].astype(float).to_numpy() if "green_loss_dB_std" in df else np.zeros(len(df))
    red = df["red_loss_dB_mean"].astype(float).to_numpy()
    red_e = df["red_loss_dB_std"].astype(float).to_numpy() if "red_loss_dB_std" in df else np.zeros(len(df))
    intra = df["intra_distance_mean"].astype(float).to_numpy()
    inter = df["inter_distance"].astype(float).to_numpy()
    ratio = df["inter_intra_ratio"].astype(float).to_numpy()
    ent = df["entropy_bits_mean"].astype(float).to_numpy()
    ent_e = df["entropy_bits_std"].astype(float).to_numpy()

    fig, axes = plt.subplots(
        2, 2,
        figsize=(DOUBLE_COL_W, 5.5),
        layout="constrained",
    )
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # (a) Transmission loss
    mg = np.isfinite(green)
    mr = np.isfinite(red)
    ax_a.errorbar(
        xs[mg], green[mg], yerr=green_e[mg], marker="o", ms=4, lw=1.1, capsize=2.5,
        color=COL_GREEN, label="Green ~520 nm",
    )
    ax_a.errorbar(
        xs[mr], red[mr], yerr=red_e[mr], marker="s", ms=4, lw=1.1, capsize=2.5,
        color=COL_RED, label="Red ~650 nm",
    )
    ax_a.axvline(SELECTED_CM, color=COL_VLINE, ls="--", lw=1.0, zorder=0)
    ax_a.set_xlabel("Total fiber length (cm)")
    ax_a.set_ylabel("Transmission loss (dB)")
    ax_a.legend(
        frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, fontsize=FONT_SIZE_LEGEND,
    )
    ax_a.text(
        0.98, 0.94, "Selected: 9 cm", transform=ax_a.transAxes,
        ha="right", va="top", fontsize=FONT_SIZE_LEGEND, color="#444444",
    )
    panel_label(ax_a, "a")

    # (b) Mean L2 distance — intra / inter only
    if len(xs) > 1:
        bar_w = 0.32 * float(np.min(np.diff(xs)))
    else:
        bar_w = 0.35
    for i, x in enumerate(xs):
        ax_b.bar(
            x - bar_w / 2, intra[i], width=bar_w, color=COL_BLUE,
            edgecolor="white", linewidth=0.4,
        )
        ax_b.bar(
            x + bar_w / 2, inter[i], width=bar_w, color=COL_ORANGE,
            edgecolor="white", linewidth=0.4,
        )
    h_intra = mpatches.Patch(facecolor=COL_BLUE, edgecolor="white", linewidth=0.4, label="Intra-class")
    h_inter = mpatches.Patch(facecolor=COL_ORANGE, edgecolor="white", linewidth=0.4, label="Inter-class")
    ax_b.legend(
        handles=[h_intra, h_inter], frameon=False, loc="lower center",
        bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=FONT_SIZE_LEGEND,
    )
    ax_b.set_xlabel("Total fiber length (cm)")
    ax_b.set_ylabel(r"Mean $L_2$ distance (ROI)")
    panel_label(ax_b, "b")

    # (c) Inter/intra ratio
    ax_c.plot(xs, ratio, "D-", color=COL_MAROON, ms=5, lw=1.15)
    ax_c.axvline(SELECTED_CM, color=COL_VLINE, ls="--", lw=1.0, zorder=0)
    ax_c.set_xlabel("Total fiber length (cm)")
    ax_c.set_ylabel("Inter/intra ratio")
    imax = int(np.nanargmax(ratio))
    if np.isfinite(ratio[imax]) and abs(xs[imax] - SELECTED_CM) < 0.25:
        ax_c.text(
            0.97, 0.96, "Maximum at 9 cm", transform=ax_c.transAxes,
            ha="right", va="top", fontsize=FONT_SIZE_LEGEND, color="#444444",
        )
    panel_label(ax_c, "c")

    # (d) Shannon entropy
    lo, hi = ent - ent_e, ent + ent_e
    ax_d.fill_between(xs, lo, hi, alpha=0.22, color=COL_GREEN)
    ax_d.plot(xs, ent, "o-", color=COL_GREEN, lw=1.15, ms=4.5)
    ax_d.axvline(SELECTED_CM, color=COL_VLINE, ls="--", lw=1.0, zorder=0)
    ax_d.set_xlabel("Total fiber length (cm)")
    ax_d.set_ylabel("Shannon entropy (bits)")
    panel_label(ax_d, "d")

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_xticks(xs)

    out_dir = RELEASE_ROOT / "experiments/length_optimization/outputs/fig3"
    base = "Fig3_length_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_existing_outputs(out_dir, base)

    plot_df = df.copy()
    plot_df["total_length_cm"] = plot_df["length_mm"] / 10.0
    plot_df["length_meaning"] = "total_fiber_length_cm"
    plot_df["is_selected_optimal"] = plot_df["length_mm"] == 90
    write_table_csv(plot_df, out_dir / f"{base}_data.csv")

    extra = {
        "input_csv": str(CSV_PATH.resolve()),
        "input_optimal_json": str(OPTIMAL_PATH.resolve()) if OPTIMAL_PATH.is_file() else None,
        "length_mm_seen": sorted(lengths),
        "length_meaning": "total_fiber_length_cm",
        "confirmed_by_PI": True,
        "optimal_total_fiber_length_cm": 9,
        "legacy_length_datasets_excluded": True,
    }
    paths = save_figure_bundle(fig, out_dir, base, "plot_fig3_length_optimization.py", extra_meta=extra)
    plt.close(fig)
    print("Wrote", paths)


if __name__ == "__main__":
    main()
