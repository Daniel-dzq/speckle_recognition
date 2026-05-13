#!/usr/bin/env python3
"""
Fig. 6 — Common-mode suppression.

Bar (a): recomputed raw green intensity CV over all `power_common_mode/**/P90/*.JPG` pooled;
         second bar: manuscript η = G/R CV (summary) with conflict note in metadata.
Bar (b): illustrative reinstall NCC improvement (summary statistics until dedicated export exists).
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

from paper_figures.io_utils import archive_existing_outputs  # noqa: E402
from paper_figures.style import (
    COL_GREEN,
    COL_BLUE,
    COL_ORANGE,
    apply_style,
    figure_root,
    panel_label,
    save_figure_bundle,
    write_table_csv,
)

POWER_ROOT = REPO_ROOT / "power_common_mode"
ETA_CV_MANUSCRIPT = 4.3
NCC_BASE = 0.72
IMPROVE = 0.28


def pooled_green_cv_p90() -> tuple[float, int, float]:
    from PIL import Image

    gvals, etas = [], []
    if not POWER_ROOT.is_dir():
        raise FileNotFoundError(f"Missing {POWER_ROOT}")
    for jpg in sorted(POWER_ROOT.rglob("P90/*.JPG")):
        im = np.array(Image.open(jpg))
        if im.ndim != 3:
            continue
        g = float(im[:, :, 1].mean())
        r = float(im[:, :, 0].mean())
        gvals.append(g)
        etas.append(g / (r + 1e-6))
    if len(gvals) < 2:
        raise RuntimeError("Not enough P90 images for CV")
    g = np.array(gvals)
    e = np.array(etas)
    return float(g.std(ddof=0) / g.mean() * 100.0), int(g.size), float(e.std(ddof=0) / e.mean() * 100.0)


def main() -> None:
    apply_style()
    cv_g, n_im, cv_eta_diag = pooled_green_cv_p90()

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
    ax = axes[0]
    vals = np.array([cv_g, ETA_CV_MANUSCRIPT])
    ax.bar([0, 1], vals, width=0.45, color=[COL_GREEN, COL_BLUE], edgecolor="white", lw=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Raw green\nintensity", "eta = G/R"], fontsize=7)
    ax.set_ylabel("Coefficient of variation (%)")
    ax.annotate(f"≈ {vals[0]/vals[1]:.1f}× lower", xy=(0.5, max(vals) * 0.65), ha="center", fontsize=7)
    panel_label(ax, "a")
    ax.text(0.02, 0.05, f"P90 pooled N={n_im}; η bar = summary {ETA_CV_MANUSCRIPT}%", transform=ax.transAxes, fontsize=6)

    axb = axes[1]
    ncc_eta = NCC_BASE * (1.0 + IMPROVE)
    axb.bar([0, 1], [NCC_BASE, ncc_eta], width=0.42, color=[COL_ORANGE, COL_BLUE], edgecolor="white", lw=0.5)
    axb.set_xticks([0, 1])
    axb.set_xticklabels(["Raw intensity\nfeature", "eta = G/R\nratio feature"], fontsize=7)
    axb.set_ylabel("Intra-class correlation (illustrative)")
    axb.set_ylim(0, min(1.05, ncc_eta + 0.08))
    axb.annotate(f"+{IMPROVE*100:.0f}% vs raw", xy=(1, ncc_eta + 0.02), ha="center", fontsize=7)
    panel_label(axb, "b")
    axb.text(
        0.02, 0.06,
        "Summary only until reinstall NCC export is added.",
        transform=axb.transAxes,
        fontsize=6,
    )

    fig.subplots_adjust(bottom=0.22, top=0.88, wspace=0.35)

    out_dir = figure_root(REPO_ROOT) / "Fig6_common_mode_suppression"
    base = "Fig6_common_mode_suppression"
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_existing_outputs(out_dir, base)

    df = pd.DataFrame([
        {"panel": "a", "series": "raw_green_CV_pct", "value": cv_g, "n_images": n_im, "verification": "computed"},
        {"panel": "a", "series": "eta_CV_pct_bar", "value": ETA_CV_MANUSCRIPT, "n_images": "", "verification": "manuscript_summary"},
        {"panel": "a", "series": "eta_CV_diagnostic_scalar", "value": cv_eta_diag, "n_images": n_im, "verification": "diagnostic_only"},
        {"panel": "b", "series": "ncc_raw", "value": NCC_BASE, "n_images": "", "verification": "summary"},
        {"panel": "b", "series": "ncc_eta", "value": ncc_eta, "n_images": "", "verification": "summary"},
    ])
    write_table_csv(df, out_dir / f"{base}_data.csv")
    paths = save_figure_bundle(
        fig,
        out_dir,
        base,
        "plot_fig6_common_mode.py",
        extra_meta={
            "power_common_mode_root": str(POWER_ROOT.resolve()),
            "eta_cv_diagnostic_pct": cv_eta_diag,
            "conflict_note": "η bar uses manuscript 4.3%; simple mean G/R pool CV differs (see CSV).",
            "manuscript_ready": False,
            "reason": (
                "G/R ratio not fully recomputed from verified paired red/green raw images yet."
            ),
            "paired_data_search_note": (
                "Continue searching under power_common_mode/ for time-aligned red/green pairs; "
                "do not treat this figure as final or silently replace the plot until pairs are verified."
            ),
        },
    )
    plt.close(fig)
    print("Wrote", paths)


if __name__ == "__main__":
    main()
