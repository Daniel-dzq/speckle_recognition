#!/usr/bin/env python3
"""
Fig. 7 — Cross-fiber authentication matrix only (main text).

Letter confusion, ROC, and confidence histogram are excluded from this bundle until
scores are verified; see metadata.
"""
from __future__ import annotations

import json
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
    apply_style,
    figure_root,
    save_figure_bundle,
    write_table_csv,
)

AUTH_JSON = REPO_ROOT / "results" / "fiber_auth" / "auth_matrix.json"

SAFE_CLAIM = (
    "The 5×5 matrix supports fiber/device specificity and cross-fiber rejection behavior."
)
UNSAFE_CLAIM = (
    "Do not claim final ROC/EER or threshold authentication from this figure."
)


def _remove_stale_outputs(out_dir: Path, base: str) -> None:
    """Drop legacy auxiliary exports not part of the current main figure."""
    stale_names = (
        f"{base}_roc_curve.csv",
        f"{base}_letter_confusion.csv",
        f"{base}_fiber_matrix.csv",
    )
    for name in stale_names:
        p = out_dir / name
        if p.is_file():
            p.unlink()


def main() -> None:
    apply_style()
    if not AUTH_JSON.is_file():
        raise FileNotFoundError(AUTH_JSON)

    auth = json.loads(AUTH_JSON.read_text(encoding="utf-8"))
    matrix = auth.get("matrix") or {}
    fibers = sorted(matrix.keys(), key=lambda s: int(s.replace("Fiber", "")))
    mat = np.array([[matrix[fi].get(fj, np.nan) for fj in fibers] for fi in fibers], dtype=float)
    labels = [f"F{i + 1}" for i in range(len(fibers))]

    fig, ax = plt.subplots(
        figsize=(4.2, 3.9),
        layout="constrained",
    )
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=100, aspect="equal")
    ax.set_xticks(np.arange(len(fibers)))
    ax.set_yticks(np.arange(len(fibers)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Test fiber", fontsize=9)
    ax.set_ylabel("Enrolled fiber", fontsize=9)
    ax.set_title("Cross-fiber authentication matrix", fontsize=9, pad=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Accuracy (%)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isfinite(v):
                continue
            use_light = v > 55
            ax.text(
                j, i, f"{v:.0f}", ha="center", va="center",
                color="white" if use_light else "black", fontsize=8,
            )

    out_dir = figure_root(REPO_ROOT) / "Fig7_authentication"
    base = "Fig7_authentication"
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_existing_outputs(out_dir, base)
    _remove_stale_outputs(out_dir, base)

    rows = []
    for i, fi in enumerate(fibers):
        for j, fj in enumerate(fibers):
            rows.append({
                "enrolled_fiber": fi,
                "test_fiber": fj,
                "accuracy_pct": mat[i, j],
            })
    data_df = pd.DataFrame(rows)
    diag = np.diag(mat)
    off = mat[np.triu_indices_from(mat, k=1)]
    off_low = mat[np.tril_indices_from(mat, k=-1)]
    off_vals = np.concatenate([off[np.isfinite(off)], off_low[np.isfinite(off_low)]])
    off_mean = float(np.nanmean(off_vals)) if off_vals.size else None

    write_table_csv(data_df, out_dir / f"{base}_data.csv")

    paths = save_figure_bundle(
        fig,
        out_dir,
        base,
        "plot_fig7_authentication.py",
        extra_meta={
            "auth_matrix_json": str(AUTH_JSON.resolve()),
            "manuscript_ready": False,
            "safe_claim": SAFE_CLAIM,
            "unsafe_claim": UNSAFE_CLAIM,
            "matrix_shape": list(mat.shape),
            "diagonal_mean_pct": float(np.nanmean(diag)),
            "off_diagonal_mean_pct": off_mean,
            "n_fibers": len(fibers),
            "excluded_from_main_figure": [
                "letter_confusion_26_class",
                "roc_from_unified_predictions_confidence",
                "confidence_score_histogram",
            ],
            "note": (
                "ROC, letter CM, and histogram omitted until score definition is verified; "
                "this bundle is fiber matrix only."
            ),
        },
    )
    plt.close(fig)
    print("Wrote", paths)


if __name__ == "__main__":
    main()