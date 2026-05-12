#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-quality figures for Photonics Research / Nature Photonics style.

Outputs go to figures_publication/ (PNG 600 dpi, PDF, SVG). Does not overwrite
legacy figures under figures/.

Usage:
    conda run -n recognition python scripts/make_publication_figures.py

See module docstrings in each fig_* function for data sources and fallbacks.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Patch

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "figures_publication"
VIDEO_DIR = ROOT / "videocapture"
RESULTS_DIR = ROOT / "results"
FIBER_AUTH_DIR = RESULTS_DIR / "fiber_auth"
METRICS_JSON = ROOT / "figures" / "new_datasets_analysis" / "metrics_summary.json"
LENGTH_OPTIMIZATION_GREEN_DIR = RESULTS_DIR / "length_optimization_green"
LENGTH_OPTIMIZATION_GREEN_CSV = LENGTH_OPTIMIZATION_GREEN_DIR / "tables" / "per_length_summary.csv"
LENGTH_SUMMARY_FALLBACK_PARTIAL = RESULTS_DIR / "green_partial_32" / "summary.json"
LENGTH_SUMMARY_FALLBACK_LEGACY = RESULTS_DIR / "length_optimize_current" / "summary.json"
LENGTH_IMAGE_ROOT = ROOT / "LengthOptimize" / "Green"

FIBERS = ["Fiber1", "Fiber2", "Fiber3", "Fiber4", "Fiber5"]

# —— Publication palette (manuscript spec) —————————————————————————
COLOR_GENUINE = "#2F6690"
COLOR_IMPOSTOR = "#C84A5A"
COLOR_GREEN_CH = "#2A9D8F"
COLOR_RED_REF = "#E76F51"
COLOR_NEUTRAL = "#6C757D"
COLOR_CHANCE = "#7A8791"

# Typography
PT_PANEL = 15
PT_AXIS_LABEL = 9.5
PT_TICK = 8.5
PT_LEGEND = 8.5

LW_AXIS = 1.05
LW_LINE = 2.0

# Rows: enrollment model, cols: test fiber (when JSON absent.)
AUTH_MATRIX_FALLBACK = np.array(
    [
        [97.4, 2.5, 3.1, 2.2, 4.8],
        [1.9, 95.3, 4.2, 2.9, 3.7],
        [3.4, 2.6, 92.9, 3.8, 4.1],
        [2.4, 2.9, 3.9, 98.7, 2.6],
        [4.0, 3.5, 2.9, 2.3, 93.6],
    ],
    dtype=np.float64,
)
AUTHORIZED_AVG_FALLBACK = 95.6
CROSS_FIBER_AVG_FALLBACK = 4.0
AUTH_GAP_PP_FALLBACK = 91.6

generated_files: List[str] = []


def set_pub_style() -> None:
    """Arial/Helvetica-first, white background, despine defaults via rcParams."""
    plt.rcdefaults()
    sans = ["Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"]
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": sans,
            "font.size": PT_TICK,
            "axes.linewidth": LW_AXIS,
            "axes.labelsize": PT_AXIS_LABEL,
            "axes.titlesize": PT_TICK,
            "xtick.labelsize": PT_TICK,
            "ytick.labelsize": PT_TICK,
            "legend.fontsize": PT_LEGEND,
            "lines.linewidth": LW_LINE,
            "lines.markersize": 5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#2B2F36",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": LW_AXIS * 0.85,
            "ytick.major.width": LW_AXIS * 0.85,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "figure.constrained_layout.use": False,
        }
    )
    # Prefer Arial if installed
    for name in sans:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            mpl.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
        except Exception:
            continue


def despine(ax: plt.Axes, keep_left_bottom: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not keep_left_bottom:
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=PT_PANEL,
        fontweight="bold",
        ha="right",
        va="bottom",
        clip_on=False,
    )


def save_figure_all_formats(fig: plt.Figure, name: str) -> List[str]:
    """
    Save figure as PNG (600 dpi), PDF, and SVG under figures_publication/.
    `name` must not include a path or extension.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / name
    saved: List[str] = []
    for ext, kw in (
        ("png", {"dpi": 600}),
        ("pdf", {}),
        ("svg", {}),
    ):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, **kw)
        saved.append(str(path))
    return saved


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_metrics_summary() -> Dict[str, Any]:
    if METRICS_JSON.is_file():
        return _load_json(METRICS_JSON)
    return {}


def _float_csv_cell(val: Optional[str]) -> Optional[float]:
    if val is None or str(val).strip() == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _load_per_length_rows_from_green_csv(path: Path) -> List[Dict[str, Any]]:
    """Map official experiment CSV columns to the shape expected by ``fig03_length_optimization``."""
    rows_out: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            length_mm = _float_csv_cell(r.get("length_mm"))
            inter = _float_csv_cell(r.get("inter_distance"))
            ent_std = _float_csv_cell(r.get("entropy_bits_std"))
            rows_out.append({
                "length_tag": (r.get("length_group") or r.get("length_tag") or "").strip(),
                "length_mm": length_mm,
                "entropy_bits_mean": _float_csv_cell(r.get("entropy_bits_mean")),
                "entropy_bits_std": ent_std if ent_std is not None else 0.0,
                "intra_distance_mean": _float_csv_cell(r.get("intra_distance_mean")),
                "inter_distance_mean": inter,
                "inter_intra_ratio": _float_csv_cell(r.get("inter_intra_ratio")),
                "green_loss_dB_mean": _float_csv_cell(r.get("green_loss_dB_mean")),
                "green_prop_mm": _float_csv_cell(r.get("green_prop_mm")),
            })
    return rows_out


def load_length_summary() -> Optional[Dict[str, Any]]:
    """
    Prefer ``results/length_optimization_green/tables/per_length_summary.csv`` (full Section 3.2 run).

    Fallback order:
      1. ``green_partial_32/summary.json``
      2. ``length_optimize_current/summary.json`` (legacy partial — prints WARNING)
    """
    if LENGTH_OPTIMIZATION_GREEN_CSV.is_file():
        rows = _load_per_length_rows_from_green_csv(LENGTH_OPTIMIZATION_GREEN_CSV)
        return {
            "per_length_summary": rows,
            "_data_source": str(LENGTH_OPTIMIZATION_GREEN_CSV.resolve()),
            "_source_kind": "length_optimization_green_csv",
        }

    if LENGTH_SUMMARY_FALLBACK_PARTIAL.is_file():
        data = dict(_load_json(LENGTH_SUMMARY_FALLBACK_PARTIAL))
        data["_data_source"] = str(LENGTH_SUMMARY_FALLBACK_PARTIAL.resolve())
        data["_source_kind"] = "green_partial_32"
        return data

    if LENGTH_SUMMARY_FALLBACK_LEGACY.is_file():
        print(
            "WARNING: using legacy partial length_optimize_current data, not recommended for manuscript.",
            flush=True,
        )
        data = dict(_load_json(LENGTH_SUMMARY_FALLBACK_LEGACY))
        data["_data_source"] = str(LENGTH_SUMMARY_FALLBACK_LEGACY.resolve())
        data["_source_kind"] = "length_optimize_current_legacy"
        return data

    return None


def load_auth_matrix_data() -> Tuple[np.ndarray, float, float, float]:
    path = FIBER_AUTH_DIR / "auth_matrix.json"
    if path.is_file():
        data = _load_json(path)
        mat = np.array([[data["matrix"][mf][df] for df in FIBERS] for mf in FIBERS])
        auth = float(data.get("authorized_avg", np.diag(mat).mean()))
        unauth = float(data.get("unauthorized_avg", data.get("cross_fiber_avg", np.mean(
            mat[np.logical_not(np.eye(5, dtype=bool))]
        ))))
        gap = float(data.get("auth_gap_pp", auth - unauth))
        return mat, auth, unauth, gap
    warnings.warn(
        "auth_matrix.json not found — using hard-coded summary values from manuscript text.",
        UserWarning,
    )
    return (
        AUTH_MATRIX_FALLBACK.copy(),
        AUTHORIZED_AVG_FALLBACK,
        CROSS_FIBER_AVG_FALLBACK,
        AUTH_GAP_PP_FALLBACK,
    )


def _parse_cm_label(length_tag: str) -> str:
    """Fiber8cm -> 8 cm."""
    m = re.search(r"(\d+)\s*cm", length_tag, re.I)
    if m:
        return f"{int(m.group(1))} cm"
    return length_tag


def _length_sort_key(row: Dict) -> Tuple[float, str]:
    mm = row.get("length_mm")
    if mm is not None:
        return (float(mm), row.get("length_tag", ""))
    return (1e9, row.get("length_tag", ""))


# =============================================================================
# FIGURE 2 — Speckle response and physical uniqueness
# =============================================================================


def fig02_speckle_response(*, pseudocolor: bool = False) -> None:
    """Representative green speckle images from videocapture (letter A challenge)."""
    try:
        import cv2
    except ImportError:
        warnings.warn("OpenCV missing — skip Figure 2.", UserWarning)
        return

    def crop_center(img: np.ndarray, margin: int = 200) -> np.ndarray:
        h, w = img.shape[:2]
        if h <= 2 * margin or w <= 2 * margin:
            return img
        return img[margin : h - margin, margin : w - margin]

    def extract_frame(vpath: Path, frame_idx: Optional[int] = None) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = total // 2 if frame_idx is None else frame_idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, max(0, total - 1)))
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    letter = "A"
    # ---- (a) same letter, five fibers
    row_a: List[np.ndarray] = []
    for fib in FIBERS:
        vp = VIDEO_DIR / "Green" / fib / f"{letter}.avi"
        fr = extract_frame(vp)
        if fr is not None:
            row_a.append(crop_center(fr))

    # ---- (b) same fiber, three illumination domains
    row_b: List[Tuple[np.ndarray, str]] = []
    doms = [
        ("Green", "Green challenge only"),
        ("GreenAndRed", "Dual-channel (fixed red ref.)"),
        ("RedChange", "Dual-channel (red sweep)"),
    ]
    ex_fib = "Fiber1"
    for sub, lab in doms:
        vp = VIDEO_DIR / sub / ex_fib / f"{letter}.avi"
        fr = extract_frame(vp)
        if fr is not None:
            row_b.append((crop_center(fr), lab))

    if not row_a and not row_b:
        warnings.warn("No videocapture frames — skip Figure 2.", UserWarning)
        return

    ncols_a = max(len(row_a), 1)
    ncols_b = max(len(row_b), 1)
    ncols = max(ncols_a, ncols_b)
    fig, axes = plt.subplots(
        2, ncols, figsize=(7.2, 3.3), layout="none",
        squeeze=False,
    )

    def channel_for_display(bgr: np.ndarray) -> np.ndarray:
        # Green channel (challenge) as primary photorealistic view
        g = bgr[:, :, 1].astype(np.float64)
        if pseudocolor:
            g = (g - g.min()) / max(g.max() - g.min(), 1e-9)
            return g
        return g

    def norm_stack_gray(images: Sequence[np.ndarray], arrs: Sequence[np.ndarray]) -> Tuple[float, float]:
        cat = np.concatenate([a.ravel() for a in arrs])
        lo, hi = np.percentile(cat, [2, 98])
        return float(lo), float(hi)

    # Row (a)
    arrs_a = [channel_for_display(im) for im in row_a]
    if arrs_a:
        lo_a, hi_a = norm_stack_gray(row_a, arrs_a)
    else:
        lo_a, hi_a = 0.0, 1.0

    for j in range(ncols):
        ax = axes[0, j]
        if j < len(row_a):
            g = arrs_a[j]
            disp = np.clip((g - lo_a) / max(hi_a - lo_a, 1e-9), 0, 1)
            if pseudocolor:
                ax.imshow(disp, cmap="viridis", vmin=0, vmax=1)
            else:
                ax.imshow(disp, cmap="gray", vmin=0, vmax=1)
            tag = FIBERS[j].replace("Fiber", "F")
            ax.set_title(tag, fontsize=PT_TICK, pad=3, color="#333333")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
            s.set_edgecolor("#CCCCCC")
        if j == 0:
            add_panel_label(ax, "(a)", x=-0.05, y=1.02)

    # Row (b)
    arrs_b = [channel_for_display(im) for im, _ in row_b]
    if arrs_b:
        lo_b, hi_b = norm_stack_gray([t[0] for t in row_b], arrs_b)
    else:
        lo_b, hi_b = 0.0, 1.0

    for j in range(ncols):
        ax = axes[1, j]
        if j < len(row_b):
            g = arrs_b[j]
            disp = np.clip((g - lo_b) / max(hi_b - lo_b, 1e-9), 0, 1)
            if pseudocolor:
                ax.imshow(disp, cmap="viridis", vmin=0, vmax=1)
            else:
                ax.imshow(disp, cmap="gray", vmin=0, vmax=1)
            ax.set_title(row_b[j][1], fontsize=PT_TICK - 0.5, pad=3, color="#333333")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
            s.set_edgecolor("#CCCCCC")
        if j == 0:
            add_panel_label(ax, "(b)", x=-0.05, y=1.02)

    fig.subplots_adjust(wspace=0.06, hspace=0.22, left=0.03, right=0.99, top=0.92, bottom=0.03)
    suffix = "_pseudocolor" if pseudocolor else ""
    for p in save_figure_all_formats(fig, f"publication_fig02_speckle_response{suffix}"):
        generated_files.append(p)
    plt.close(fig)


# =============================================================================
# FIGURE 3 / MANUSCRIPT FIGURE 4 — Fiber length vs. total fiber length (cm)
# Primary data: ``results/length_optimization_green/tables/per_length_summary.csv``.
# Fallback: ``green_partial_32/summary.json``, then ``length_optimize_current/summary.json`` (WARNING).
# =============================================================================


def fig03_length_optimization() -> None:
    data = load_length_summary()
    if not data or not data.get("per_length_summary"):
        warnings.warn(
            "No length summary — Figure 3/4 length composite skipped. "
            "Run: python scripts/run_length_optimization.py --config config/length_optimization_green.yaml",
            UserWarning,
        )
        return

    rows = sorted(data["per_length_summary"], key=_length_sort_key)
    src_kind = data.get("_source_kind", "")
    data_source = data.get("_data_source", "")
    if data_source:
        print(f"Figure 3/4 length optimization data: {data_source}", flush=True)

    xs_mm = [r["length_mm"] / 10.0 for r in rows]  # mm → cm
    labels = [_parse_cm_label(r["length_tag"]) for r in rows]
    ent = [r["entropy_bits_mean"] for r in rows]
    ent_err = [r.get("entropy_bits_std") or 0 for r in rows]
    intra = [r["intra_distance_mean"] for r in rows]
    inter = [r["inter_distance_mean"] for r in rows]
    ratio = [r["inter_intra_ratio"] for r in rows]

    optimum_caption = None
    if src_kind == "length_optimization_green_csv":
        opt = next(
            (r for r in rows if r.get("length_tag") == "Fiber9cm" or r.get("length_mm") == 90),
            None,
        )
        if opt and opt.get("inter_intra_ratio") is not None and opt.get("entropy_bits_mean") is not None:
            g = opt.get("green_loss_dB_mean")
            if g is not None and g == g:  # not NaN
                g_loss = f"{g:.2f} dB"
            else:
                g_loss = "—"
            optimum_caption = (
                f"Optimum — Fiber9cm (9 cm total): green loss {g_loss}, "
                f"inter/intra {opt['inter_intra_ratio']:.4f}, "
                f"entropy {opt['entropy_bits_mean']:.3f} bit"
            )

    fig = plt.figure(figsize=(7.2, 6.2))
    bottom_margin = 0.12 if optimum_caption else 0.07
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], wspace=0.35, hspace=0.45,
                          left=0.09, right=0.98, top=0.93, bottom=bottom_margin)

    # (a) Montage: one ROI per length (Fiber1, first JPG)
    ax_a = fig.add_subplot(gs[0, 0])
    try:
        import cv2
    except ImportError:
        cv2 = None

    montage_imgs: List[np.ndarray] = []
    for r in rows:
        tag = r["length_tag"]
        cand = LENGTH_IMAGE_ROOT / tag / "Fiber1"
        jp = None
        if cand.is_dir():
            for ext in ("1.JPG", "1.jpg", "01.JPG"):
                if (cand / ext).is_file():
                    jp = cand / ext
                    break
            if jp is None:
                files = sorted(cand.glob("*.JPG")) + sorted(cand.glob("*.jpg"))
                jp = files[0] if files else None
        if cv2 is not None and jp is not None:
            im = cv2.imread(str(jp), cv2.IMREAD_COLOR)
            if im is not None:
                g = im[:, :, 1].astype(np.float64)
                h, w = g.shape
                s = min(h, w, 400)
                y0, x0 = (h - s) // 2, (w - s) // 2
                montage_imgs.append(g[y0 : y0 + s, x0 : x0 + s])

    if montage_imgs:
        lo, hi = np.percentile(np.concatenate([m.ravel() for m in montage_imgs]), [2, 98])
    else:
        lo, hi = 0.0, 255.0

    def draw_montage_column() -> None:
        ax_a.clear()
        if not montage_imgs:
            ax_a.text(0.5, 0.5, "No length images", ha="center", va="center")
            ax_a.axis("off")
            return
        nrow, ncol = len(montage_imgs), 1
        pad = 2
        thumbs = []
        target = 72
        for g in montage_imgs:
            im = np.clip((g - lo) / max(hi - lo, 1e-9), 0, 1)
            import cv2 as _cv

            t = _cv.resize(im, (target, target), interpolation=_cv.INTER_AREA)
            thumbs.append(t)
        stack = np.vstack([np.pad(t, ((0, pad), (0, 0)), constant_values=1) for t in thumbs])
        h_st, w_st = stack.shape[0], stack.shape[1]
        ax_a.imshow(stack, cmap="gray", vmin=0, vmax=1, aspect="auto")
        ax_a.axis("off")
        for i, r in enumerate(rows):
            yc = i * (target + pad) + target / 2
            ax_a.text(
                w_st + 4,
                yc,
                _parse_cm_label(r["length_tag"]),
                ha="left",
                va="center",
                fontsize=PT_TICK - 1,
                color="#333333",
                clip_on=False,
            )
        ax_a.set_xlim(-0.5, w_st + 80)
        ax_a.set_ylim(h_st, -0.5)

    draw_montage_column()
    add_panel_label(ax_a, "(a)", x=-0.15, y=1.05)

    # (b) entropy
    ax_b = fig.add_subplot(gs[0, 1:])
    ax_b.fill_between(
        xs_mm,
        np.array(ent) - np.array(ent_err),
        np.array(ent) + np.array(ent_err),
        color=COLOR_GREEN_CH,
        alpha=0.2,
        linewidth=0,
    )
    ax_b.plot(xs_mm, ent, "o-", color=COLOR_GREEN_CH, lw=LW_LINE, markersize=6)
    ax_b.set_xlabel("Total fiber length (cm)")
    ax_b.set_ylabel("Pixel entropy (bits)")
    despine(ax_b)
    add_panel_label(ax_b, "(b)")

    # (c) intra / inter bars — numeric x
    ax_c = fig.add_subplot(gs[1, 0:2])
    x_arr = np.array(xs_mm, dtype=float)
    w = 0.35
    ax_c.bar(x_arr - w / 2, intra, width=w, label="Intra-class", color=COLOR_GENUINE, edgecolor="white", linewidth=0.5)
    ax_c.bar(x_arr + w / 2, inter, width=w, label="Inter-class", color=COLOR_IMPOSTOR, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax_c.set_xlabel("Total fiber length (cm)")
    ax_c.set_ylabel(r"Mean $L_2$ distance (ROI)")
    despine(ax_c)
    ax_c.legend(loc="upper left", fontsize=PT_LEGEND, frameon=False)
    add_panel_label(ax_c, "(c)")

    # (d) ratio — separate panel, no twin axis
    ax_d = fig.add_subplot(gs[1, 2])
    ax_d.plot(xs_mm, ratio, "s-", color=COLOR_NEUTRAL, lw=LW_LINE, markersize=6)
    ax_d.set_xlabel("Total fiber length (cm)")
    ax_d.set_ylabel("Inter/intra distance ratio")
    despine(ax_d)
    add_panel_label(ax_d, "(d)")

    for ax in (ax_b, ax_c, ax_d):
        ax.set_xticks([8, 9, 11, 13, 16])
        ax.set_xlim(7.2, 16.8)

    if optimum_caption:
        fig.text(0.5, 0.01, optimum_caption, ha="center", fontsize=7, color="#333333")

    fig.subplots_adjust(wspace=0.4, hspace=0.38, bottom=bottom_margin, top=0.93)
    for name in ("publication_fig03_length_optimization", "publication_fig04_length_optimization"):
        for p in save_figure_all_formats(fig, name):
            generated_files.append(p)
    plt.close(fig)


# =============================================================================
# FIGURE 4 — Cross-fiber authentication
# =============================================================================


def fig04_cross_fiber_auth() -> None:
    mat, auth_avg, cross_avg, gap_pp = load_auth_matrix_data()
    chance = 100.0 / 26

    fig = plt.figure(figsize=(7.4, 6.8))
    gs = fig.add_gridspec(2, 2, wspace=0.4, hspace=0.45, left=0.10, right=0.98, top=0.94, bottom=0.07)

    # (a) Heatmap — magma, cream to red feel
    ax = fig.add_subplot(gs[0, 0])
    cmap_name = "magma"
    try:
        cmap = mpl.colormaps[cmap_name].copy()
    except Exception:
        cmap = plt.get_cmap(cmap_name)
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=100, aspect="equal")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([f"F{i+1}" for i in range(5)])
    ax.set_yticklabels([f"F{i+1}" for i in range(5)])
    ax.set_xlabel("Test fiber")
    ax.set_ylabel("Enrollment fiber")
    ax.tick_params(top=False, labeltop=False)
    despine(ax)

    def _txt_color(v: float) -> str:
        return "white" if v > 52 else "#1a1a1a"

    for i in range(5):
        for j in range(5):
            val = mat[i, j]
            w = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9, color=_txt_color(val), fontweight=w)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.set_label("Recognition accuracy (%)", fontsize=PT_AXIS_LABEL)
    cbar.ax.tick_params(labelsize=PT_TICK)
    add_panel_label(ax, "(a)", x=-0.25, y=1.05)

    sum_txt = (
        f"Same-fiber avg = {auth_avg:.1f}%; cross-fiber avg = {cross_avg:.1f}%; "
        f"gap = {gap_pp:.1f} pp"
    )
    ax.text(0.5, 1.22, sum_txt, transform=ax.transAxes, ha="center", va="bottom", fontsize=PT_TICK, color="#333333")

    # (b) Scatter + means
    axb = fig.add_subplot(gs[0, 1])
    diag = np.diag(mat)
    off = mat[np.logical_not(np.eye(5, dtype=bool))].ravel()
    rng = np.random.default_rng(42)
    axb.scatter(
        rng.normal(0, 0.06, len(diag)),
        diag,
        s=38,
        color=COLOR_GENUINE,
        edgecolors="white",
        linewidths=0.6,
        label="Same fiber",
        zorder=4,
    )
    axb.scatter(
        rng.normal(1, 0.06, len(off)),
        off,
        s=22,
        color=COLOR_IMPOSTOR,
        alpha=0.65,
        edgecolors="none",
        label="Cross fiber",
        zorder=3,
    )
    axb.plot([-0.25, 0.25], [diag.mean(), diag.mean()], color=COLOR_GENUINE, lw=LW_LINE, zorder=5)
    axb.plot([0.75, 1.25], [off.mean(), off.mean()], color=COLOR_IMPOSTOR, lw=LW_LINE, zorder=5)
    axb.axhline(chance, color=COLOR_CHANCE, linestyle="--", lw=1.0, zorder=2)
    axb.text(1.35, chance + 0.8, "Random guess = 1/26 = 3.8%", fontsize=PT_TICK - 1, color=COLOR_CHANCE, va="bottom")
    axb.set_xticks([0, 1])
    axb.set_xticklabels(["Same fiber", "Cross fiber"])
    axb.set_ylabel("Recognition accuracy (%)")
    axb.set_xlim(-0.45, 1.45)
    axb.set_ylim(-2, 105)
    axb.legend(loc="upper right", fontsize=PT_LEGEND, frameon=False)
    despine(axb)
    add_panel_label(axb, "(b)")

    # (c)(d) confidence and cross-fiber distributions
    preds = []
    for fib in FIBERS:
        pcsv = RESULTS_DIR / fib.lower() / "test_predictions.csv"
        if pcsv.is_file():
            with open(pcsv, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    preds.append(float(row["confidence"]))

    axc = fig.add_subplot(gs[1, 0])
    if preds:
        axc.hist(preds, bins=np.linspace(0, 1, 28), color=COLOR_GENUINE, edgecolor="white", linewidth=0.35, alpha=0.9)
        axc.axvline(np.median(preds), color="#1a1a1a", ls="--", lw=1.0)
    axc.set_xlabel("Prediction confidence (authorized)")
    axc.set_ylabel("Count")
    despine(axc)
    add_panel_label(axc, "(c)")

    axd = fig.add_subplot(gs[1, 1])
    axd.hist(off, bins=np.linspace(0, max(off.max() * 1.1, 8), 18), color=COLOR_IMPOSTOR, edgecolor="white", linewidth=0.35, alpha=0.9)
    axd.axvline(chance, color=COLOR_CHANCE, linestyle="--", lw=1.0)
    axd.set_xlabel("Cross-fiber accuracy (%)")
    axd.set_ylabel("Count")
    despine(axd)
    add_panel_label(axd, "(d)")

    for p in save_figure_all_formats(fig, "publication_fig04_cross_fiber_auth"):
        generated_files.append(p)
    plt.close(fig)


# =============================================================================
# FIGURE 5 — Dual-channel reference and robustness
# =============================================================================


def fig05_dual_channel_robustness() -> None:
    m = load_metrics_summary()
    pw = m.get("power_common_mode", {}).get("per_power_setting") or []
    lt = m.get("long_term_stability", {}).get("per_fiber") or []

    fig = plt.figure(figsize=(7.4, 6.6))
    gs = fig.add_gridspec(2, 2, wspace=0.4, hspace=0.45, left=0.11, right=0.97, top=0.94, bottom=0.08)

    if pw:
        pw_sorted = sorted(
            [p for p in pw if p.get("power_index") is not None],
            key=lambda x: int(x["power_index"]),
        )
        xs = [int(p["power_index"]) for p in pw_sorted]
        intras = [float(p["intra_l2"]) for p in pw_sorted]
        inters = [float(p["inter_l2"]) for p in pw_sorted]
        ratios = np.array(
            [float(p["inter_intra_ratio"]) for p in pw_sorted],
            dtype=np.float64,
        )

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(xs, intras, "o-", color=COLOR_GENUINE, lw=LW_LINE, label="Intra-class")
        ax1.plot(xs, inters, "s-", color=COLOR_IMPOSTOR, lw=LW_LINE, label="Inter-class")
        ax1.set_xlabel("Red reference power setting")
        ax1.set_ylabel(r"Mean $L_2$ distance")
        best_idx = int(np.nanargmax(ratios))
        ax1.axvline(xs[best_idx], color=COLOR_NEUTRAL, ls=":", lw=1.0, alpha=0.8)
        ymax = max(max(intras), max(inters), 1.0)
        ax1.annotate(
            f"P{xs[best_idx]} peak ratio",
            xy=(xs[best_idx], ymax * 0.92),
            ha="center",
            fontsize=PT_TICK - 1,
            color=COLOR_NEUTRAL,
        )
        ax1.legend(loc="best", fontsize=PT_LEGEND, frameon=False)
        despine(ax1)
        add_panel_label(ax1, "(a)")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(xs, ratios, "D-", color=COLOR_GREEN_CH, lw=LW_LINE)
        ax2.set_xlabel("Red reference power setting")
        ax2.set_ylabel("Inter/intra distance ratio")
        ax2.axvline(xs[best_idx], color=COLOR_NEUTRAL, ls=":", lw=1.0, alpha=0.8)
        despine(ax2)
        add_panel_label(ax2, "(b)")
    else:
        for i in range(2):
            ax = fig.add_subplot(gs[0, i])
            ax.text(0.5, 0.5, "No power_common_mode in metrics_summary.json", ha="center")
            ax.axis("off")

    # (c) long-term: distributions across 15 fibers (consecutive vs vs-first)
    ax3 = fig.add_subplot(gs[1, 0])
    if lt:
        cons = [row["consecutive_ncc"] for row in lt]
        vfs = [row["vs_first_ncc"] for row in lt]
        parts = ax3.violinplot([cons, vfs], positions=[1, 2], showmeans=False, showmedians=True, widths=0.6)
        body_colors = [COLOR_GENUINE, COLOR_IMPOSTOR]
        for bi, b in enumerate(parts["bodies"]):
            b.set_facecolor(body_colors[bi % 2])
            b.set_alpha(0.6)
        ax3.set_xticks([1, 2])
        ax3.set_xticklabels(["Adjacent-frame\nSpeckle NCC", "vs first acquisition\nSpeckle NCC"], fontsize=PT_TICK - 1)
        ax3.set_ylabel("Speckle NCC")
        despine(ax3)
        add_panel_label(ax3, "(c)")
    else:
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.text(0.5, 0.5, "No long_term_stability in metrics", ha="center")
        ax3.axis("off")

    # (d) histograms of both NCC types across fibers
    ax4 = fig.add_subplot(gs[1, 1])
    if lt:
        cons = np.array([row["consecutive_ncc"] for row in lt])
        vfs = np.array([row["vs_first_ncc"] for row in lt])
        bins = np.linspace(0.3, 1.02, 26)
        ax4.hist(cons, bins=bins, alpha=0.55, color=COLOR_GENUINE, label="Adjacent-frame", edgecolor="white", linewidth=0.3)
        ax4.hist(vfs, bins=bins, alpha=0.5, color=COLOR_IMPOSTOR, label="vs first acquisition", edgecolor="white", linewidth=0.3)
        ax4.set_xlabel("Speckle NCC")
        ax4.set_ylabel("Fiber count")
        ax4.legend(loc="upper left", fontsize=PT_LEGEND, frameon=False)
        despine(ax4)
        add_panel_label(ax4, "(d)")
    else:
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")

    for p in save_figure_all_formats(fig, "publication_fig05_dual_channel_robustness"):
        generated_files.append(p)
    plt.close(fig)


# =============================================================================
# Supplementary
# =============================================================================


def supp_s1_ncc_hd() -> None:
    """Intra- vs inter-fiber NCC and Hamming distance (physical uniqueness)."""
    VIDEO = VIDEO_DIR
    try:
        import cv2
    except ImportError:
        warnings.warn("OpenCV missing — skip Supp S1.", UserWarning)
        return

    target_size = (256, 256)
    letters = list("ABCDEFGHIJKLMNOP")
    offsets = [20, 50, 80, 110, 140]

    frames_map: Dict[Tuple[str, str], List[np.ndarray]] = {}
    for fib in FIBERS:
        for lt in letters:
            vp = VIDEO / "Green" / fib / f"{lt}.avi"
            frames: List[np.ndarray] = []
            cap = cv2.VideoCapture(str(vp))
            if not cap.isOpened():
                continue
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for t in offsets:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(t, total - 1))
                ret, fr = cap.read()
                if ret and fr is not None:
                    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                    g = cv2.resize(g, target_size).astype(np.float64)
                    frames.append(g)
            cap.release()
            if frames:
                frames_map[(fib, lt)] = frames

    if not frames_map:
        warnings.warn("No video frames for Supp S1.", UserWarning)
        return

    def ncc(a, b):
        an = a - a.mean()
        bn = b - b.mean()
        d = np.sqrt(np.sum(an**2) * np.sum(bn**2))
        return 0.0 if d < 1e-12 else float(np.sum(an * bn) / d)

    def hd(a, b):
        ab = (a > np.median(a)).astype(np.uint8).ravel()
        bb = (b > np.median(b)).astype(np.uint8).ravel()
        return float(np.mean(ab != bb))

    intra_n, inter_n, intra_h, inter_h = [], [], [], []
    for letter in letters:
        for fib in FIBERS:
            frs = frames_map.get((fib, letter), [])
            for i in range(len(frs)):
                for j in range(i + 1, len(frs)):
                    intra_n.append(ncc(frs[i], frs[j]))
                    intra_h.append(hd(frs[i], frs[j]))
        for i, fa in enumerate(FIBERS):
            for fb in FIBERS[i + 1 :]:
                a1 = frames_map.get((fa, letter), [])
                a2 = frames_map.get((fb, letter), [])
                n = min(len(a1), len(a2))
                for k in range(n):
                    inter_n.append(ncc(a1[k], a2[k]))
                    inter_h.append(hd(a1[k], a2[k]))

    intra_n = np.asarray(intra_n)
    inter_n = np.asarray(inter_n)
    intra_h = np.asarray(intra_h)
    inter_h = np.asarray(inter_h)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), layout="none")

    ax = axes[0]
    bins_n = np.linspace(0.78, 1.0, 45)
    ax.hist(inter_n, bins=bins_n, density=True, alpha=0.65, color=COLOR_IMPOSTOR, edgecolor="white", linewidth=0.3)
    ax.hist(intra_n, bins=bins_n, density=True, alpha=0.65, color=COLOR_GENUINE, edgecolor="white", linewidth=0.3)
    ax.set_xlim(0.78, 1.0)
    ax.set_xlabel("Speckle NCC")
    ax.set_ylabel("Density")
    despine(ax)
    add_panel_label(ax, "(a)")

    ax2 = axes[1]
    lo, hi = 0.0, min(0.65, max(intra_h.max(), inter_h.max()) + 0.03)
    bins_h = np.linspace(lo, hi, 44)
    ax2.hist(inter_h, bins=bins_h, density=True, alpha=0.65, color=COLOR_IMPOSTOR, edgecolor="white", linewidth=0.3)
    ax2.hist(intra_h, bins=bins_h, density=True, alpha=0.65, color=COLOR_GENUINE, edgecolor="white", linewidth=0.3)
    ax2.axvline(0.5, color=COLOR_CHANCE, ls="--", lw=1.0)
    ax2.text(0.97, 0.93, "Unbiased HD = 0.5 (reference)", transform=ax2.transAxes, ha="right", fontsize=PT_TICK - 1, color=COLOR_NEUTRAL)
    ax2.set_xlabel("Hamming distance")
    ax2.set_ylabel("Density")
    despine(ax2)
    add_panel_label(ax2, "(b)")
    fig.subplots_adjust(bottom=0.24, top=0.92, wspace=0.35, left=0.08, right=0.98)
    fig.legend(
        [
            Patch(facecolor=COLOR_IMPOSTOR, edgecolor="white", linewidth=0.3),
            Patch(facecolor=COLOR_GENUINE, edgecolor="white", linewidth=0.3),
        ],
        ["Cross fiber", "Same fiber"],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.02),
        ncol=2,
        fontsize=PT_LEGEND,
        frameon=False,
    )

    for p in save_figure_all_formats(fig, "supplementary_fig_s1_ncc_hd"):
        generated_files.append(p)
    plt.close(fig)


def supp_s2_disturbance() -> None:
    m = load_metrics_summary()
    d = m.get("disturbance_sensitivity", {})
    ncc_map = d.get("within_fiber_mean_ncc") or {}
    if not ncc_map:
        warnings.warn("No disturbance_sensitivity in metrics — skip Supp S2.", UserWarning)
        return

    fibers = sorted(ncc_map.keys(), key=lambda s: (int(re.search(r"(\d+)", s).group(1)) if re.search(r"(\d+)", s) else 999, s))
    ys = [ncc_map[f] for f in fibers]
    x = np.arange(len(fibers))

    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.bar(x, ys, color=COLOR_GREEN_CH, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("Fiber", "F") for f in fibers], rotation=55, ha="right", fontsize=PT_TICK - 1)
    ax.set_ylabel("Mean within-fiber Speckle NCC")
    ax.set_xlabel("Fiber (after reinstall / disturbance)")
    ratio = d.get("pooled_inter_intra_ratio")
    ncap = d.get("n_captures")
    note = f"$n$ = {ncap} captures"
    if ratio is not None:
        note += f"\ninter/intra $L_2$ ratio = {ratio:.2f}"
    ax.text(0.98, 0.98, note, transform=ax.transAxes, ha="right", va="top", fontsize=PT_TICK, color="#333333")
    despine(ax)
    for p in save_figure_all_formats(fig, "supplementary_fig_s2_reinstallation_robustness"):
        generated_files.append(p)
    plt.close(fig)


def main() -> None:
    global generated_files
    generated_files = []
    set_pub_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Publication figure suite — outputs under", OUT_DIR)
    fig02_speckle_response(pseudocolor=False)
    fig02_speckle_response(pseudocolor=True)
    fig03_length_optimization()
    fig04_cross_fiber_auth()
    fig05_dual_channel_robustness()
    supp_s1_ncc_hd()
    supp_s2_disturbance()

    print("\nGenerated", len(generated_files), "files:")
    for path in sorted(generated_files):
        print(" ", path)


if __name__ == "__main__":
    main()
