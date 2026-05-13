#!/usr/bin/env python3
"""
Fig. 7 — Authentication performance (fiber matrix + letter protocol on unified export).

Letter random baseline: 1/26 ≈ 3.85%. Fiber matrix uses `auth_matrix.json` percentages.
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
    COL_BLUE,
    COL_GRAY,
    COL_RED,
    apply_style,
    figure_root,
    panel_label,
    save_figure_bundle,
    write_table_csv,
)

AUTH_JSON = REPO_ROOT / "results" / "fiber_auth" / "auth_matrix.json"
UNIFIED_CSV = REPO_ROOT / "results" / "unified" / "test_predictions.csv"
N_CLASSES = 26
CHANCE = 1.0 / N_CLASSES


def main() -> None:
    apply_style()
    if not AUTH_JSON.is_file():
        raise FileNotFoundError(AUTH_JSON)
    if not UNIFIED_CSV.is_file():
        raise FileNotFoundError(UNIFIED_CSV)

    from sklearn.metrics import auc, confusion_matrix, roc_curve
    from sklearn.preprocessing import LabelEncoder

    auth = json.loads(AUTH_JSON.read_text(encoding="utf-8"))
    matrix = auth.get("matrix") or {}
    fibers = sorted(matrix.keys(), key=lambda s: int(s.replace("Fiber", "")))
    mat = np.array([[matrix[fi].get(fj, np.nan) for fj in fibers] for fi in fibers], dtype=float)

    df = pd.read_csv(UNIFIED_CSV)
    needed = {"true_label", "pred_label", "confidence"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{UNIFIED_CSV} missing {needed - set(df.columns)}")

    y_true_bin = (df["true_label"] == df["pred_label"]).astype(int).values
    scores = df["confidence"].astype(float).values

    le = LabelEncoder()
    le.fit(sorted(df["true_label"].unique()))
    y = le.transform(df["true_label"])
    pred = le.transform(df["pred_label"])
    cm = confusion_matrix(y, pred, labels=np.arange(len(le.classes_)))

    fpr, tpr, _ = roc_curve(y_true_bin, scores)
    roc_auc = auc(fpr, tpr)

    out_dir = figure_root(REPO_ROOT) / "Fig7_authentication"
    base = "Fig7_authentication"
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_existing_outputs(out_dir, base)

    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(out_dir / f"{base}_roc_curve.csv", index=False)
    pd.DataFrame(cm, index=[f"true_{c}" for c in le.classes_], columns=list(le.classes_)).to_csv(
        out_dir / f"{base}_letter_confusion.csv"
    )
    pd.DataFrame(mat, index=fibers, columns=fibers).to_csv(out_dir / f"{base}_fiber_matrix.csv")

    ok = scores[y_true_bin == 1]
    bad = scores[y_true_bin == 0]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))
    ax = axes[0, 0]
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(fibers)))
    ax.set_yticks(np.arange(len(fibers)))
    ax.set_xticklabels([f.replace("Fiber", "F") for f in fibers], fontsize=7)
    ax.set_yticklabels([f.replace("Fiber", "F") for f in fibers], fontsize=7)
    ax.set_xlabel("Predicted fiber (%)")
    ax.set_ylabel("Enrolled fiber (%)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", color="black", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    panel_label(ax, "a")

    axb = axes[0, 1]
    im2 = axb.imshow(cm, cmap="Blues")
    tick_idx = np.arange(len(le.classes_))
    axb.set_xlabel("Predicted letter")
    axb.set_ylabel("True letter")
    axb.set_xticks(tick_idx)
    axb.set_yticks(tick_idx)
    axb.set_xticklabels(le.classes_, fontsize=5, rotation=90)
    axb.set_yticklabels(le.classes_, fontsize=5)
    fig.colorbar(im2, ax=axb, fraction=0.046, pad=0.04)
    panel_label(axb, "b")

    axc = axes[1, 0]
    axc.plot(fpr, tpr, color=COL_BLUE, lw=1.5, label=f"ROC (AUC = {roc_auc:.3f})")
    axc.plot([0, 1], [0, 1], "--", color=COL_GRAY, lw=0.8)
    axc.set_xlabel("False positive rate")
    axc.set_ylabel("True positive rate")
    axc.legend(frameon=False, loc="lower right", fontsize=7)
    axc.set_title("Correct vs incorrect (score = reported confidence)")
    panel_label(axc, "c")

    axd = axes[1, 1]
    axd.hist(ok, bins=28, alpha=0.65, color=COL_BLUE, label="Correct", density=True)
    axd.hist(bad, bins=28, alpha=0.55, color=COL_RED, label="Incorrect", density=True)
    axd.set_xlabel("Model confidence")
    axd.set_ylabel("Density")
    axd.legend(frameon=False, fontsize=7)
    axd.text(
        0.02,
        0.97,
        f"Letter random baseline = 1/{N_CLASSES} ({100 * CHANCE:.2f}%)",
        transform=axd.transAxes,
        fontsize=6,
        va="top",
    )
    panel_label(axd, "d")

    fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.08, hspace=0.38, wspace=0.35)

    summary = pd.DataFrame(
        [
            {"metric": "roc_auc", "value": roc_auc},
            {"metric": "n_unified_rows", "value": len(df)},
            {"metric": "chance_letter", "value": CHANCE},
        ]
    )
    write_table_csv(summary, out_dir / f"{base}_data.csv")

    paths = save_figure_bundle(
        fig,
        out_dir,
        base,
        "plot_fig7_authentication.py",
        extra_meta={
            "auth_matrix_json": str(AUTH_JSON.resolve()),
            "unified_predictions_csv": str(UNIFIED_CSV.resolve()),
            "auxiliary_csv": [
                str((out_dir / f"{base}_roc_curve.csv").resolve()),
                str((out_dir / f"{base}_letter_confusion.csv").resolve()),
                str((out_dir / f"{base}_fiber_matrix.csv").resolve()),
            ],
            "note": "7-day stability time series not in repository.",
        },
    )
    plt.close(fig)
    print("Wrote", paths)


if __name__ == "__main__":
    main()
