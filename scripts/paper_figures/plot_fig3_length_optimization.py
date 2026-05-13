#!/usr/bin/env python3
"""
Fig. 3 — Fiber length optimization (verified `length_optimization_green` only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR.parent))

from paper_figures.style import (  # noqa: E402
    COL_GREEN,
    COL_RED,
    COL_BLUE,
    COL_ORANGE,
    COL_MAROON,
    apply_style,
    figure_root,
    panel_label,
    save_figure_bundle,
    write_table_csv,
)

CSV_PATH = REPO_ROOT / "results" / "length_optimization_green" / "tables" / "per_length_summary.csv"
OPTIMAL_PATH = REPO_ROOT / "results" / "length_optimization_green" / "optimal_length.json"
ALLOWED_MM = {80, 90, 110, 130, 160}  # 8–16 cm campaign; flags 5/30/45 cm style mistakes


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
            f"(expected subset of {sorted(ALLOWED_MM)} mm). Audit old 5/30/45 cm data separately."
        )

    df = df.sort_values("length_mm")
    xs = df["length_mm"].astype(float) / 10.0
    col_vline = "#999999"
    green = df["green_loss_dB_mean"].astype(float)
    green_e = df.get("green_loss_dB_std", pd.Series([0.0] * len(df)))
    red = df["red_loss_dB_mean"].astype(float)
    red_e = df.get("red_loss_dB_std", pd.Series([0.0] * len(df)))
    intra = df["intra_distance_mean"].astype(float)
    inter = df["inter_distance"].astype(float)
    ratio = df["inter_intra_ratio"].astype(float)
    ent = df["entropy_bits_mean"].astype(float)
    ent_e = df["entropy_bits_std"].astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    # (a) loss
    ax = axes[0]
    ax.errorbar(xs, green, yerr=green_e, marker="o", lw=1.2, capsize=2.5, color=COL_GREEN, label="Green ~520 nm")
    ax.errorbar(xs, red, yerr=red_e, marker="s", lw=1.2, capsize=2.5, color=COL_RED, label="Red ~650 nm")
    ax.axvline(9.0, color=col_vline, ls=":", lw=1.0)
    ax.set_xlabel("Total fiber length (cm)")
    ax.set_ylabel("Transmission loss (dB)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.18), ncol=2)
    panel_label(ax, "a")
    # (b) distances + ratio
    axb = axes[1]
    w = 0.35
    axb.bar(xs - w / 2, intra, width=w, label="Intra-class distance", color=COL_BLUE, edgecolor="white", lw=0.35)
    axb.bar(xs + w / 2, inter, width=w, label="Inter-class distance", color=COL_ORANGE, edgecolor="white", lw=0.35)
    axb2 = axb.twinx()
    axb2.plot(xs, ratio, "D--", color=COL_MAROON, ms=3.5, lw=1.1, label="Inter/intra ratio")
    axb2.set_ylabel("Inter/intra distance ratio")
    axb.set_xlabel("Total fiber length (cm)")
    axb.set_ylabel(r"Mean $L_2$ distance (ROI)")
    axb.axvline(9.0, color=col_vline, ls=":", lw=1.0)
    h1, l1 = axb.get_legend_handles_labels()
    h2, l2 = axb2.get_legend_handles_labels()
    axb.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.28), ncol=2, fontsize=6)
    panel_label(axb, "b")
    axb.spines["top"].set_visible(False)
    axb2.spines["top"].set_visible(False)
    # (c) entropy
    axc = axes[2]
    lo = ent - ent_e
    hi = ent + ent_e
    axc.fill_between(xs, lo, hi, alpha=0.22, color=COL_GREEN)
    axc.plot(xs, ent, "o-", color=COL_GREEN, lw=1.2, ms=4)
    axc.axvline(9.0, color=col_vline, ls=":", lw=1.0)
    axc.set_xlabel("Total fiber length (cm)")
    axc.set_ylabel("Shannon entropy (bits)")
    panel_label(axc, "c")
    fig.subplots_adjust(top=0.82, bottom=0.18, wspace=0.4)

    out_dir = figure_root(REPO_ROOT) / "Fig3_length_optimization"
    base = "Fig3_length_optimization"
    plot_df = df.copy()
    plot_df["total_length_cm"] = plot_df["length_mm"] / 10.0
    write_table_csv(plot_df, out_dir / f"{base}_data.csv")

    extra = {
        "input_csv": str(CSV_PATH.resolve()),
        "input_optimal_json": str(OPTIMAL_PATH.resolve()) if OPTIMAL_PATH.is_file() else None,
        "length_mm_seen": sorted(lengths),
    }
    paths = save_figure_bundle(fig, out_dir, base, "plot_fig3_length_optimization.py", extra_meta=extra)
    plt.close(fig)
    print("Wrote", paths)


if __name__ == "__main__":
    main()
