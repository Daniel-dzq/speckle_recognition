#!/usr/bin/env python3
"""
Fig. 6 — Common-mode suppression (draft): clean bar panels only; caveats in meta/README.
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
    DOUBLE_COL_W,
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

FIG6_WARNING = (
    "G/R ratio values are not final until paired red/green raw data are verified."
)


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

    fig, axes = plt.subplots(
        1, 2,
        figsize=(DOUBLE_COL_W * 0.92, 2.75),
        layout="constrained",
    )
    ax, axb = axes

    vals = np.array([cv_g, ETA_CV_MANUSCRIPT])
    ax.bar([0, 1], vals, width=0.48, color=[COL_GREEN, COL_BLUE], edgecolor="white", lw=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Raw green\nintensity", "G/R ratio"], fontsize=8)
    ax.set_ylabel("Coefficient of variation (%)")
    panel_label(ax, "a")

    ncc_eta = NCC_BASE * (1.0 + IMPROVE)
    axb.bar([0, 1], [NCC_BASE, ncc_eta], width=0.45, color=[COL_ORANGE, COL_BLUE], edgecolor="white", lw=0.5)
    axb.set_xticks([0, 1])
    axb.set_xticklabels(["Raw intensity\nfeature", "G/R ratio\nfeature"], fontsize=8)
    axb.set_ylabel("Intra-class correlation")
    axb.set_ylim(0, min(1.05, ncc_eta + 0.1))
    panel_label(axb, "b")

    out_dir = figure_root(REPO_ROOT) / "Fig6_common_mode_suppression"
    base = "Fig6_common_mode_suppression"
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_existing_outputs(out_dir, base)

    readme = out_dir / "README.md"
    readme.write_text(
        "# Fig6_common_mode_suppression\n\n"
        "**Draft figure** (`data_status: draft`, `manuscript_ready: false`).\n\n"
        f"- Warning: {FIG6_WARNING}\n"
        "- G/R bar and panel (b) values may be manuscript-derived or illustrative; see `Fig6_common_mode_suppression_meta.json` and `*_data.csv`.\n",
        encoding="utf-8",
    )

    df = pd.DataFrame([
        {"panel": "a", "series": "raw_green_CV_pct", "value": cv_g, "n_images": n_im, "verification": "computed"},
        {"panel": "a", "series": "gr_ratio_CV_pct_bar", "value": ETA_CV_MANUSCRIPT, "n_images": "", "verification": "manuscript_summary"},
        {"panel": "a", "series": "gr_ratio_CV_diagnostic_pool_pct", "value": cv_eta_diag, "n_images": n_im, "verification": "diagnostic_only"},
        {"panel": "b", "series": "ncc_raw", "value": NCC_BASE, "n_images": "", "verification": "summary"},
        {"panel": "b", "series": "ncc_gr_ratio", "value": ncc_eta, "n_images": "", "verification": "summary"},
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
            "p90_image_count": n_im,
            "manuscript_ready": False,
            "data_status": "draft",
            "warning": FIG6_WARNING,
            "gr_ratio_bar_source": "manuscript_summary_4.3_pct",
            "panel_b_note": "Values are illustrative/summary until reinstall NCC export is verified.",
        },
    )
    plt.close(fig)
    print("Wrote", paths)


if __name__ == "__main__":
    main()
