#!/usr/bin/env python3
"""
Fig. 5 — Dual-channel characteristics using verified `metrics_summary.json`
(from `python scripts/analyze_new_datasets.py`) plus representative video frames.
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

from paper_figures.style import (
    COL_GREEN,
    COL_RED,
    COL_BLUE,
    COL_ORANGE,
    apply_style,
    figure_root,
    panel_label,
    save_figure_bundle,
    write_table_csv,
)

METRICS = REPO_ROOT / "figures" / "new_datasets_analysis" / "metrics_summary.json"
VIDEO_GREEN = REPO_ROOT / "videocapture" / "Green" / "Fiber1" / "A.avi"
VIDEO_DUAL = REPO_ROOT / "videocapture" / "GreenAndRed" / "Fiber1" / "A.avi"


def _video_mid_gray(path: Path, channel: str):
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("OpenCV (cv2) required for video panels") from e
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = max(0, n // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Empty frame {path}")
    if channel == "green":
        return frame[:, :, 1].astype(np.float64)
    if channel == "red":
        return frame[:, :, 2].astype(np.float64)
    raise ValueError(channel)


def _radial(gray: np.ndarray):
    h, w = gray.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_max = int(min(cx, cy, w // 2, h // 2))
    prof = np.zeros(r_max)
    g = gray / (np.nanmax(gray) + 1e-9)
    for i in range(r_max):
        m = (r >= i) & (r < i + 1)
        prof[i] = g[m].mean() if np.any(m) else np.nan
    return np.arange(r_max), prof / (np.nanmax(prof) + 1e-9)


def main() -> None:
    apply_style()
    if not METRICS.is_file():
        raise FileNotFoundError(
            f"Missing {METRICS}; run: python scripts/analyze_new_datasets.py"
        )
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    lt = m.get("long_term_stability", {})
    per = lt.get("per_fiber") or []
    if len(per) < 2:
        raise ValueError("long_term_stability.per_fiber too small in metrics_summary.json")

    df_lt = pd.DataFrame(per)
    df_lt["metric"] = "consecutive_ncc"

    # Disturbance: within-fiber mean NCC bar chart
    ds = m.get("disturbance_sensitivity", {})
    wfn = ds.get("within_fiber_mean_ncc") or {}
    if not wfn:
        raise ValueError("disturbance_sensitivity.within_fiber_mean_ncc missing")
    df_ds = pd.DataFrame([{"fiber": k, "within_fiber_mean_ncc": v} for k, v in sorted(wfn.items())])

    fig = plt.figure(figsize=(7.2, 2.8))
    gs = fig.add_gridspec(1, 3, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    fibers = df_lt["fiber"].tolist()
    x = np.arange(len(fibers))
    ax1.bar(x, df_lt["consecutive_ncc"], color=COL_BLUE, edgecolor="white", lw=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f.replace("Fiber", "F") for f in fibers], rotation=60, ha="right", fontsize=6)
    ax1.set_ylabel("Adjacent-frame NCC")
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Fiber (long-term stability)")
    panel_label(ax1, "a")

    ax2 = fig.add_subplot(gs[0, 1])
    x2 = np.arange(len(df_ds))
    ax2.bar(x2, df_ds["within_fiber_mean_ncc"], color=COL_ORANGE, edgecolor="white", lw=0.3)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([f.replace("Fiber", "F") for f in df_ds["fiber"]], rotation=60, ha="right", fontsize=6)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Mean within-fiber NCC")
    ax2.set_xlabel("Fiber (disturbance)")
    panel_label(ax2, "b")

    ax3 = fig.add_subplot(gs[0, 2])
    g_gray = _video_mid_gray(VIDEO_GREEN, "green")
    r_gray = _video_mid_gray(VIDEO_DUAL, "red")
    h, w = min(g_gray.shape[0], r_gray.shape[0]), min(g_gray.shape[1], r_gray.shape[1])
    g_gray, r_gray = g_gray[:h, :w], r_gray[:h, :w]
    r1, p1 = _radial(g_gray)
    r2, p2 = _radial(r_gray)
    ax3.plot(r2, p2, color=COL_RED, lw=1.2, label="Red (R ch.)")
    ax3.plot(r1, p1, color=COL_GREEN, lw=1.2, label="Green (G ch.)")
    ax3.set_xlabel("Radius (px)")
    ax3.set_ylabel("Norm. radial mean intensity")
    ax3.legend(frameon=False, loc="upper right", fontsize=6)
    panel_label(ax3, "c")

    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.28)

    out_dir = figure_root(REPO_ROOT) / "Fig5_dual_channel"
    base = "Fig5_dual_channel"
    plot_csv = pd.concat(
        [
            df_lt.assign(panel="temporal_stability"),
            df_ds.assign(panel="disturbance"),
        ],
        ignore_index=True,
    )
    write_table_csv(plot_csv, out_dir / f"{base}_data.csv")

    paths = save_figure_bundle(
        fig,
        out_dir,
        base,
        "plot_fig5_dual_channel.py",
        extra_meta={
            "metrics_summary": str(METRICS.resolve()),
            "VIDEO_GREEN": str(VIDEO_GREEN.resolve()),
            "VIDEO_DUAL": str(VIDEO_DUAL.resolve()),
            "note": "Panel (c) uses representative middle-frame proxies from videocapture.",
        },
    )
    plt.close(fig)
    print("Wrote", paths)


if __name__ == "__main__":
    main()
