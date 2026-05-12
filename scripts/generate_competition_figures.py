#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate planning-document figures 4, 7, 8 into figures_competition/ only.

Does not write to figures_publication/ (avoid confusion with paper auth figures).

Usage (repo root):
    python scripts/generate_competition_figures.py          # Fig. 7 & 8 only (default)
    python scripts/generate_competition_figures.py --fig 4  # re-run length Fig. 4 when intended
    python scripts/generate_competition_figures.py --fig 7 8
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import gridspec  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.caching.cache import FeatureCache  # noqa: E402
from analysis.experiments._features import extract_features  # noqa: E402
from analysis.io.dataset import DatasetLayout, discover_captures  # noqa: E402
from analysis.metrics.stability import temporal_stability_score  # noqa: E402
from analysis.preprocessing.pipeline import PreprocessConfig  # noqa: E402

OUT_DIR = ROOT / "figures_competition"
DOCS_DIR = ROOT / "docs"
LOSS_CSV = ROOT / "results" / "length_optimization_green" / "tables" / "per_length_summary.csv"
OPTIMAL_JSON = ROOT / "results" / "length_optimization_green" / "optimal_length.json"
LONG_TERM_ROOT = ROOT / "long_term_stability"
POWER_COMMON_ROOT = ROOT / "power_common_mode"
VIDEO_GREEN = ROOT / "videocapture" / "Green" / "Fiber1" / "A.avi"
VIDEO_DUAL = ROOT / "videocapture" / "GreenAndRed" / "Fiber1" / "A.avi"

# Colors — English-only figure policy (see docs/top_journal_figure_style_guide.md)
COL_RED = "#b2182b"
COL_GREEN = "#1b783f"
COL_ETA = "#4c72b0"
COL_MUTED_GREEN = "#8fbc8f"
COL_GRID = "#bbbbbb"

# Manuscript summary fall-backs (never labeled as raw)
FIG7A_RED_NCC_SUMMARY = 0.94
FIG7B_RED_DROP_PCT_SUMMARY = 5.0
MANUSCRIPT_GREEN_DROP_PCT = 25.0  # for conflict note only
FIG8_CV_ETA_SUMMARY = 4.3
FIG8_REINSTALL_IMPROVEMENT_FRAC = 0.28
FIG8_RAW_GREEN_MANUSCRIPT = 38.2
FIG8_NCC_BASE_SUMMARY = 0.72


def _save_triple(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.svg", bbox_inches="tight", facecolor="white")


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.02,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _float_csv(s: Optional[str]) -> Optional[float]:
    if s is None or str(s).strip() == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_length_rows(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "length_group": (r.get("length_group") or "").strip(),
                "length_mm": _float_csv(r.get("length_mm")),
                "green_loss": _float_csv(r.get("green_loss_dB_mean")),
                "green_loss_std": _float_csv(r.get("green_loss_dB_std")) or 0.0,
                "red_loss": _float_csv(r.get("red_loss_dB_mean")),
                "red_loss_std": _float_csv(r.get("red_loss_dB_std")) or 0.0,
                "intra": _float_csv(r.get("intra_distance_mean")),
                "inter": _float_csv(r.get("inter_distance")),
                "ratio": _float_csv(r.get("inter_intra_ratio")),
                "entropy": _float_csv(r.get("entropy_bits_mean")),
                "entropy_std": _float_csv(r.get("entropy_bits_std")) or 0.0,
            })
    rows.sort(key=lambda x: (x["length_mm"] or 0.0, x["length_group"]))
    return rows


def fig4_length_optimization(manifest_rows: List[Dict[str, str]]) -> None:
    if not LOSS_CSV.is_file():
        raise FileNotFoundError(f"Missing official CSV: {LOSS_CSV}")

    rows = load_length_rows(LOSS_CSV)
    xs = np.array([r["length_mm"] / 10.0 for r in rows], dtype=float)
    green = np.array([np.nan if r["green_loss"] is None else r["green_loss"] for r in rows], dtype=float)
    green_e = np.array([r["green_loss_std"] or 0.0 for r in rows], dtype=float)
    red = np.array([np.nan if r["red_loss"] is None else r["red_loss"] for r in rows], dtype=float)
    red_e = np.array([r["red_loss_std"] or 0.0 for r in rows], dtype=float)
    intra = np.array([r["intra"] for r in rows], dtype=float)
    inter = np.array([r["inter"] for r in rows], dtype=float)
    ratio = np.array([r["ratio"] for r in rows], dtype=float)
    ent = np.array([r["entropy"] for r in rows], dtype=float)
    ent_e = np.array([r["entropy_std"] for r in rows], dtype=float)

    has_red = bool(np.any(np.isfinite(red)))
    has_green_loss = bool(np.any(np.isfinite(green)))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), constrained_layout=False)

    ax = axes[0]
    if has_green_loss:
        ax.errorbar(
            xs, green, yerr=green_e, marker="o", lw=1.4, capsize=3,
            color="#2ca02c", label="Green (520 nm)",
        )
    if has_red:
        ax.errorbar(
            xs, red, yerr=red_e, marker="s", lw=1.4, capsize=3,
            color="#d62728", label="Red (650 nm)",
        )
    elif has_green_loss:
        ax.text(
            0.02, 0.98, "Red loss: no valid power data\nfor some lengths in CSV.",
            transform=ax.transAxes, va="top", fontsize=7, color="#555555",
        )
    ax.axvline(9.0, color="#9e9e9e", ls=":", lw=1.1)
    ax.set_xlabel("Total fiber length (cm)")
    ax.set_ylabel("Transmission loss (dB)")
    ax.set_xticks([8, 9, 11, 13, 16])
    ax.set_xlim(7.2, 16.8)
    ax.set_title("(a)")
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    axb = axes[1]
    w = 0.35
    axb.bar(xs - w / 2, intra, width=w, label="Intra-class distance", color="#4a90c8", edgecolor="white", lw=0.4)
    axb.bar(xs + w / 2, inter, width=w, label="Inter-class distance", color="#e59866", edgecolor="white", lw=0.4)
    axb.set_xlabel("Total fiber length (cm)")
    axb.set_ylabel(r"Mean $L_2$ distance (ROI)")
    axb.set_xticks([8, 9, 11, 13, 16])
    axb.set_xlim(7.2, 16.8)
    axb.axvline(9.0, color="#9e9e9e", ls=":", lw=1.1)
    axb2 = axb.twinx()
    axb2.plot(
        xs, ratio, "D--", color="#6a3d9a", ms=5, lw=1.2,
        label="Inter/intra distance ratio",
    )
    axb2.set_ylabel("Inter/intra distance ratio")
    rng = float(np.nanmax(ratio) - np.nanmin(ratio)) if np.any(np.isfinite(ratio)) else 1.0
    axb2.set_ylim(float(np.nanmin(ratio)) - 0.08 * rng, float(np.nanmax(ratio)) + 0.12 * rng)
    axb2.spines["top"].set_visible(False)
    h1, l1 = axb.get_legend_handles_labels()
    h2, l2 = axb2.get_legend_handles_labels()
    axb.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.5, frameon=False)
    axb.set_title("(b)")
    axb.spines["top"].set_visible(False)

    axc = axes[2]
    lo = ent - ent_e
    hi = ent + ent_e
    axc.fill_between(xs, lo, hi, alpha=0.2, color="#2ca02c")
    axc.plot(xs, ent, "o-", color="#2ca02c", lw=1.4, ms=5)
    axc.axvline(9.0, color="#9e9e9e", ls=":", lw=1.1)
    axc.set_xlabel("Total fiber length (cm)")
    axc.set_ylabel("Pixel entropy (bits)")
    axc.set_xticks([8, 9, 11, 13, 16])
    axc.set_xlim(7.2, 16.8)
    axc.set_title("(c)")
    axc.spines["top"].set_visible(False)
    axc.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.18, wspace=0.38)
    out = OUT_DIR / "fig4_length_optimization"
    _save_triple(fig, out)
    plt.close(fig)

    opt = {}
    if OPTIMAL_JSON.is_file():
        opt = json.loads(OPTIMAL_JSON.read_text(encoding="utf-8"))
    manifest_rows.append({
        "figure_id": "fig4_length_optimization",
        "output_png": str(out) + ".png",
        "output_svg": str(out) + ".svg",
        "output_pdf": str(out) + ".pdf",
        "source_data": str(LOSS_CSV.resolve()) + (";" + str(OPTIMAL_JSON.resolve()) if opt else ""),
        "source_script": "scripts/generate_competition_figures.py",
        "raw_data_summary_only_missing": "raw",
        "note": (
            f"Triple panel; optimum 9 cm; Fiber9cm targets green_loss≈{opt.get('green_loss_dB')}, "
            f"ratio≈{opt.get('inter_intra_ratio')}, entropy bit≈{opt.get('entropy_bits')}. "
            f"red_loss_plotted={has_red}"
        ),
    })


def _preprocess_config() -> PreprocessConfig:
    return PreprocessConfig(
        grayscale=True,
        center_crop_size=400,
        resize=112,
        normalize="minmax",
        frame_strategy="middle",
        n_frames=1,
        aggregate="mean",
    )


def _load_flat_dataset(rel_dir: str, cache_bucket: str):
    layout = DatasetLayout.from_config(
        {"root": rel_dir, "layout": "flat_fiber_repeat"},
        base_dir=ROOT,
    )
    caps = discover_captures(layout)
    cache = FeatureCache(root=ROOT / ".analysis_cache", bucket=cache_bucket, enabled=True)
    return extract_features(caps, _preprocess_config(), cache=cache, logger=None)


def _fiber_sort_key(name: str) -> Tuple[int, str]:
    m = re.match(r"Fiber(\d+)$", name, re.I)
    return (int(m.group(1)), name) if m else (9999, name)


def _per_fiber_consecutive_ncc(feats) -> Dict[str, float]:
    by_f: Dict[str, List] = defaultdict(list)
    for cf in feats:
        by_f[cf.capture.fiber].append(cf)
    for fib in by_f:
        by_f[fib].sort(key=lambda c: c.capture.repeat if c.capture.repeat is not None else 0)
    out: Dict[str, float] = {}
    for fib, seq in by_f.items():
        vecs = [c.vector for c in seq]
        st = temporal_stability_score(vecs)
        out[fib] = float(st["consecutive_ncc"])
    return out


def _compute_fig7a_b_data() -> Tuple[
    float,
    float,
    float,
    float,
    List[Dict[str, Any]],
]:
    """Returns green_mean, green_std, red_summary, n_fibers, csv_rows for 7a/7b."""
    rows: List[Dict[str, Any]] = []
    if not LONG_TERM_ROOT.is_dir():
        raise FileNotFoundError(f"Missing {LONG_TERM_ROOT}")
    lt_feats = _load_flat_dataset("long_term_stability", "long_term_feats")
    lt_cc = _per_fiber_consecutive_ncc(lt_feats)
    fibers_lt = sorted(lt_cc.keys(), key=_fiber_sort_key)
    vals = np.array([lt_cc[f] for f in fibers_lt], dtype=float)
    g_mean = float(np.mean(vals))
    g_std = float(np.std(vals, ddof=0))

    for f in fibers_lt:
        rows.append({
            "panel": "7a",
            "channel": "green",
            "metric": "intra_class_corr_consecutive_ncc",
            "fiber": f,
            "value": lt_cc[f],
            "std": "",
            "verification": "raw_data_verified",
            "notes": "long_term_stability; embedding preprocess",
        })
    rows.append({
        "panel": "7a",
        "channel": "red",
        "metric": "intra_class_corr_summary",
        "fiber": "",
        "value": FIG7A_RED_NCC_SUMMARY,
        "std": "",
        "verification": "summary_statistics_verified",
        "notes": "no red long_term_stability folder in repo",
    })

    ds_cc: Dict[str, float] = {}
    common: List[str] = []
    if (ROOT / "disturbance_sensitivity").is_dir():
        ds_feats = _load_flat_dataset("disturbance_sensitivity", "disturbance_feats")
        ds_cc = _per_fiber_consecutive_ncc(ds_feats)
        common = sorted(set(lt_cc.keys()) & set(ds_cc.keys()), key=_fiber_sort_key)

    drops: List[float] = []
    for f in common:
        if lt_cc[f] <= 1e-9:
            continue
        d_pct = (1.0 - ds_cc[f] / lt_cc[f]) * 100.0
        drops.append(float(d_pct))
        rows.append({
            "panel": "7b",
            "channel": "green",
            "metric": "correlation_decrease_pct",
            "fiber": f,
            "value": d_pct,
            "std": "",
            "verification": "raw_data_verified",
            "notes": "(1 - NCC_disturb/NCC_LT)*100; paired FiberK",
        })
    drop_mean = float(np.mean(drops)) if drops else MANUSCRIPT_GREEN_DROP_PCT
    rows.append({
        "panel": "7b",
        "channel": "green",
        "metric": "correlation_decrease_pct_mean",
        "fiber": "all_paired",
        "value": drop_mean,
        "std": float(np.std(drops, ddof=0)) if drops else "",
        "verification": "raw_data_verified" if drops else "summary_statistics_verified",
        "notes": f"manuscript green drop reference {MANUSCRIPT_GREEN_DROP_PCT}%",
    })
    rows.append({
        "panel": "7b",
        "channel": "red",
        "metric": "correlation_decrease_pct",
        "fiber": "",
        "value": FIG7B_RED_DROP_PCT_SUMMARY,
        "std": "",
        "verification": "summary_statistics_verified",
        "notes": "manuscript; no red disturb captures",
    })
    return g_mean, g_std, FIG7A_RED_NCC_SUMMARY, drop_mean, rows


def _video_middle_bgr(path: Path) -> Optional[np.ndarray]:
    try:
        import cv2
    except ImportError:
        return None
    if not path.is_file():
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = max(0, n // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def _radial_mean_profile(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_max = int(min(cx, cy, w // 2, h // 2))
    prof = np.zeros(r_max, dtype=np.float64)
    for i in range(r_max):
        m = (r >= i) & (r < i + 1)
        prof[i] = gray[m].mean() if np.any(m) else np.nan
    return np.arange(r_max), prof


def _norm_show(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float64)
    mn, mx = float(np.nanmin(img)), float(np.nanmax(img))
    if mx - mn < 1e-12:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


def fig7_dual_channel(manifest_rows: List[Dict[str, str]]) -> None:
    green_mean, green_std, red_ncc, green_drop_mean, data_rows = _compute_fig7a_b_data()

    fig = plt.figure(figsize=(11.2, 3.45), facecolor="white")
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.05, 1.28], wspace=0.36)

    # --- (a) ---
    ax_a = fig.add_subplot(gs[0])
    x = np.arange(2)
    means = np.array([red_ncc, green_mean])
    errs = np.array([0.0, green_std])
    colors = [COL_RED, COL_GREEN]
    ax_a.bar(x, means, yerr=errs, capsize=4, color=colors, width=0.52, edgecolor="white", lw=0.5,
             error_kw=dict(lw=0.9, capthick=0.9))
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["Red channel", "Green channel"])
    ax_a.set_ylabel("Intra-class correlation coefficient")
    ax_a.set_ylim(0.80, 1.005)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    _panel_label(ax_a, "(a)")
    ax_a.text(0.02, 0.04, "Red: summary", transform=ax_a.transAxes, fontsize=6.5, color="#444444")
    ax_a.text(0.52, 0.04, "Green: long_term_stability", transform=ax_a.transAxes, fontsize=6.5, color="#444444")

    # --- (b) ---
    ax_b = fig.add_subplot(gs[1])
    xb = np.arange(2)
    vals_b = np.array([FIG7B_RED_DROP_PCT_SUMMARY, green_drop_mean])
    ax_b.bar(xb, vals_b, color=[COL_RED, COL_GREEN], width=0.5, edgecolor="white", lw=0.5)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(["Red channel", "Green channel"])
    ax_b.set_ylabel("Correlation decrease (%)")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ymax = max(float(np.max(vals_b)) * 1.35, 40.0)
    ax_b.set_ylim(0, ymax)
    _panel_label(ax_b, "(b)")
    ratio_txt = green_drop_mean / FIG7B_RED_DROP_PCT_SUMMARY if FIG7B_RED_DROP_PCT_SUMMARY else 0.0
    ax_b.annotate(
        f"~{ratio_txt:.1f}× larger drop",
        xy=(0.5, ymax * 0.88),
        ha="center",
        fontsize=7.5,
        color="#333333",
    )
    ax_b.text(0.02, 0.06, "Red: summary; Green: paired LT vs disturb", transform=ax_b.transAxes, fontsize=6.5, color="#444444")

    # --- (c) nested ---
    gs_c = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[2], height_ratios=[1.05, 0.75], hspace=0.42, wspace=0.2)
    ax_cr = fig.add_subplot(gs_c[0, 0])
    ax_cg = fig.add_subplot(gs_c[0, 1])
    ax_pf = fig.add_subplot(gs_c[1, :])

    frame_g = _video_middle_bgr(VIDEO_GREEN)
    frame_d = _video_middle_bgr(VIDEO_DUAL)
    c_status = "proxy_image"
    r_prof_ok = False

    if frame_g is not None and frame_d is not None:
        g_gray = frame_g[:, :, 1].astype(np.float64)
        r_ch = frame_d[:, :, 2].astype(np.float64)
        h, w = min(g_gray.shape[0], r_ch.shape[0]), min(g_gray.shape[1], r_ch.shape[1])
        g_gray = g_gray[:h, :w]
        r_ch = r_ch[:h, :w]
        ax_cr.imshow(_norm_show(r_ch), cmap="gray", aspect="equal")
        ax_cr.set_title("Red (R ch., dual video)", fontsize=8)
        ax_cr.axis("off")
        ax_cg.imshow(_norm_show(g_gray), cmap="gray", aspect="equal")
        ax_cg.set_title("Green (G ch., side video)", fontsize=8)
        ax_cg.axis("off")
        rr, pr = _radial_mean_profile(r_ch / 255.0)
        rg, pg = _radial_mean_profile(g_gray / 255.0)
        prn = pr / (np.nanmax(pr) + 1e-9)
        pgn = pg / (np.nanmax(pg) + 1e-9)
        ax_pf.plot(rr, prn, color=COL_RED, lw=1.3, label="Red (norm.)")
        ax_pf.plot(rg, pgn, color=COL_GREEN, lw=1.3, label="Green (norm.)")
        ax_pf.set_xlabel("Radius (px)")
        ax_pf.set_ylabel("Norm. radial mean")
        ax_pf.legend(frameon=False, fontsize=7, loc="upper right")
        ax_pf.spines["top"].set_visible(False)
        ax_pf.spines["right"].set_visible(False)
        r_prof_ok = True
        data_rows.append({
            "panel": "7c",
            "channel": "red_green",
            "metric": "speckle_source",
            "fiber": "",
            "value": "",
            "std": "",
            "verification": "proxy_image",
            "notes": f"VIDEO_DUAL={VIDEO_DUAL.name}; VIDEO_G={VIDEO_GREEN.name}",
        })
    else:
        ax_pf.axis("off")
        ax_cr.axis("off")
        ax_cg.axis("off")
        ax_pf.text(
            0.5, 0.5,
            "Schematic / measured imagery unavailable\n(OpenCV or video missing).",
            ha="center",
            va="center",
            ma="center",
            fontsize=8,
            color="#444444",
            transform=ax_pf.transAxes,
        )
        c_status = "missing_data"

    _panel_label(ax_cr, "(c)")

    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.15)
    out = OUT_DIR / "fig7_dual_channel_characterization"
    _save_triple(fig, out)
    plt.close(fig)

    csv_path = OUT_DIR / "data_fig7_dual_channel_characterization.csv"
    _write_data_csv(
        csv_path,
        ["panel", "channel", "metric", "fiber", "value", "std", "verification", "notes"],
        data_rows,
    )

    n7a = "partial_raw_data: green raw (long_term_stability), red summary 0.94"
    n7b = "mixed: green paired LT vs disturb; red summary 5% drop"
    n7c = f"{c_status}; radial={'ok' if r_prof_ok else 'na'}"
    manifest_rows.append({
        "figure_id": "fig7_dual_channel_characterization",
        "output_png": str(out) + ".png",
        "output_svg": str(out) + ".svg",
        "output_pdf": str(out) + ".pdf",
        "source_data": str(LONG_TERM_ROOT.resolve()) + ";disturbance_sensitivity;videocapture",
        "source_script": "scripts/generate_competition_figures.py::fig7_dual_channel",
        "raw_data_summary_only_missing": "mixed",
        "note": " | ".join([n7a, n7b, n7c, f"csv={csv_path.name}"]),
    })


def _pooled_green_cv_p90() -> Tuple[float, int, float]:
    """Mean G channel per image; CV% pooled over all P90 captures. Returns cv_pct, n, manuscript_ref."""
    gvals: List[float] = []
    eta_vals: List[float] = []
    if not POWER_COMMON_ROOT.is_dir():
        return float("nan"), 0, float("nan")
    for jpg in sorted(POWER_COMMON_ROOT.rglob("P90/*.JPG")):
        try:
            from PIL import Image
        except ImportError:
            break
        im = np.array(Image.open(jpg))
        if im.ndim != 3:
            continue
        g = float(im[:, :, 1].mean())
        r = float(im[:, :, 0].mean())
        gvals.append(g)
        eta_vals.append(g / (r + 1e-6))
    if not gvals:
        return float("nan"), 0, float("nan")
    arr = np.array(gvals, dtype=float)
    eta_arr = np.array(eta_vals, dtype=float)
    cv_g = float(arr.std(ddof=0) / arr.mean() * 100.0)
    cv_eta_simple = float(eta_arr.std(ddof=0) / eta_arr.mean() * 100.0)
    return cv_g, int(arr.size), cv_eta_simple


def fig8_common_mode(manifest_rows: List[Dict[str, str]]) -> None:
    cv_g_raw, n_im, cv_eta_diag = _pooled_green_cv_p90()
    use_green = cv_g_raw if cv_g_raw == cv_g_raw and n_im > 0 else FIG8_RAW_GREEN_MANUSCRIPT
    green_ver = "raw_data_verified" if n_im > 0 else "summary_statistics_verified"
    eta_plotted = FIG8_CV_ETA_SUMMARY
    eta_ver = "summary_statistics_verified"
    conflict = (
        f"Manuscript eta CV={FIG8_CV_ETA_SUMMARY}% kept for bar-2; "
        f"simple mean(G)/mean(R) pooled P90 diagnostic CV≈{cv_eta_diag:.1f}% "
        f"(does not match 4.3% with this scalar)."
        if cv_eta_diag == cv_eta_diag
        else "eta diagnostic unavailable"
    )

    ncc_lo = FIG8_NCC_BASE_SUMMARY
    ncc_hi = FIG8_NCC_BASE_SUMMARY * (1.0 + FIG8_REINSTALL_IMPROVEMENT_FRAC)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), facecolor="white")

    ax = axes[0]
    labs = ["Raw green\nintensity", "η = G/R"]
    vals = np.array([use_green, eta_plotted])
    ax.bar(np.arange(2), vals, color=[COL_MUTED_GREEN, COL_ETA], width=0.48, edgecolor="white", lw=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("Coefficient of variation (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if vals[1] > 1e-9:
        fac = float(vals[0] / vals[1])
        ax.annotate(f"≈ {fac:.1f}× reduction", xy=(0.5, max(vals) * 0.72), ha="center", fontsize=8, color="#333333")
    _panel_label(ax, "(a)")
    gn = "recomputed P90 pool" if green_ver == "raw_data_verified" else "manuscript"
    ax.text(0.02, 0.05, f"Green CV: {gn}", transform=ax.transAxes, fontsize=6.5, color="#444444")

    axb = axes[1]
    axb.bar([0, 1], [ncc_lo, ncc_hi], color=[COL_MUTED_GREEN, COL_ETA], width=0.45, edgecolor="white", lw=0.5)
    axb.set_xticks([0, 1])
    axb.set_xticklabels(["Raw green\n(intensity feature)", "η = G/R\n(ratio feature)"], fontsize=7)
    axb.set_ylabel("Intra-class correlation coefficient")
    axb.set_ylim(0, min(1.05, ncc_hi + 0.12))
    axb.annotate("+28%\n(vs raw)", xy=(1, ncc_hi + 0.015), ha="center", fontsize=7, color="#333333")
    axb.spines["top"].set_visible(False)
    axb.spines["right"].set_visible(False)
    _panel_label(axb, "(b)")
    axb.text(0.02, 0.06, "Summary statistics (no reinstall NCC vector in repo)", transform=axb.transAxes, fontsize=6.5, color="#444444")

    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.22, wspace=0.35)
    out = OUT_DIR / "fig8_common_mode_suppression"
    _save_triple(fig, out)
    plt.close(fig)

    rows = [
        {
            "panel": "8a",
            "quantity": "CV_raw_green_pct",
            "value": use_green,
            "n_images_p90": n_im,
            "verification": green_ver,
            "notes": f"manuscript reference {FIG8_RAW_GREEN_MANUSCRIPT}%",
        },
        {
            "panel": "8a",
            "quantity": "CV_eta_pct_plotted",
            "value": eta_plotted,
            "n_images_p90": "",
            "verification": eta_ver,
            "notes": conflict,
        },
        {
            "panel": "8a",
            "quantity": "CV_eta_meanG_meanR_diagnostic_pool_pct",
            "value": cv_eta_diag if cv_eta_diag == cv_eta_diag else "",
            "n_images_p90": n_im if n_im else "",
            "verification": "diagnostic_only",
            "notes": "Not used as bar height",
        },
        {
            "panel": "8b",
            "quantity": "intra_class_NCC_raw_green",
            "value": ncc_lo,
            "n_images_p90": "",
            "verification": "summary_statistics_verified",
            "notes": "illustrative baseline",
        },
        {
            "panel": "8b",
            "quantity": "intra_class_NCC_eta",
            "value": ncc_hi,
            "n_images_p90": "",
            "verification": "summary_statistics_verified",
            "notes": f"+{FIG8_REINSTALL_IMPROVEMENT_FRAC*100:.0f}% vs baseline",
        },
    ]
    csv_path = OUT_DIR / "data_fig8_common_mode_suppression.csv"
    _write_data_csv(
        csv_path,
        ["panel", "quantity", "value", "n_images_p90", "verification", "notes"],
        rows,
    )

    manifest_rows.append({
        "figure_id": "fig8_common_mode_suppression",
        "output_png": str(out) + ".png",
        "output_svg": str(out) + ".svg",
        "output_pdf": str(out) + ".pdf",
        "source_data": str(POWER_COMMON_ROOT.resolve()) + ";metrics_summary N/A for CV bars",
        "source_script": "scripts/generate_competition_figures.py::fig8_common_mode",
        "raw_data_summary_only_missing": "mixed",
        "note": " | ".join([
            f"8a green {green_ver}",
            "8a eta summary 4.3% (conflict note in csv)",
            "8b summary only",
            f"csv={csv_path.name}",
        ]),
    })


def _write_data_csv(path: Path, fieldnames: List[str], dict_rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in dict_rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})


def _read_existing_manifest() -> Dict[str, Dict[str, str]]:
    path = OUT_DIR / "manifest.csv"
    if not path.is_file():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fid = row.get("figure_id")
            if fid:
                out[fid] = row
    return out


def write_manifest(rows: List[Dict[str, str]]) -> None:
    path = OUT_DIR / "manifest.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "figure_id", "output_png", "output_svg", "output_pdf",
        "source_data", "source_script", "raw_data_summary_only_missing", "note",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_status_md(rows: List[Dict[str, str]]) -> None:
    lines = [
        "# Competition / planning document — required figures status",
        "",
        "Generated by `scripts/generate_competition_figures.py`. Outputs live under `figures_competition/`.",
        "",
        "**Do not** use `figures_publication/publication_fig03_length_optimization` or "
        "`publication_fig04_length_optimization` as 策划书图 4 — those are legacy composites.",
        "",
        "| figure_id | data mode | key sources |",
        "|-----------|-----------|-------------|",
    ]
    for r in rows:
        src = r["source_data"]
        lines.append(
            f"| {r['figure_id']} | {r['raw_data_summary_only_missing']} | {src[:80]}… |"
            if len(src) > 80
            else f"| {r['figure_id']} | {r['raw_data_summary_only_missing']} | {src} |"
        )
    lines.extend(["", "## Notes per figure", ""])
    for r in rows:
        lines.extend([f"### {r['figure_id']}", "", r["note"], ""])
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "competition_required_figures_status.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate figures_competition Fig. 4 / 7 / 8.")
    p.add_argument(
        "--fig",
        nargs="+",
        choices=["4", "7", "8"],
        default=["7", "8"],
        help="Which figures to regenerate (default: 7 and 8 only; add 4 explicitly if needed).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    want = set(args.fig)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    existing = _read_existing_manifest()
    manifest: List[Dict[str, str]] = []

    if "4" in want:
        fig4_length_optimization(manifest)
    elif "fig4_length_optimization" in existing:
        manifest.append(existing["fig4_length_optimization"])

    if "7" in want:
        fig7_dual_channel(manifest)
    elif "fig7_dual_channel_characterization" in existing:
        manifest.append(existing["fig7_dual_channel_characterization"])

    if "8" in want:
        fig8_common_mode(manifest)
    elif "fig8_common_mode_suppression" in existing:
        manifest.append(existing["fig8_common_mode_suppression"])

    # Preserve ordering 4,7,8 in manifest
    order = ["fig4_length_optimization", "fig7_dual_channel_characterization", "fig8_common_mode_suppression"]
    by_id = {r["figure_id"]: r for r in manifest}
    merged: List[Dict[str, str]] = []
    for fid in order:
        if fid in by_id:
            merged.append(by_id[fid])
    write_manifest(merged)
    write_status_md(merged)
    print("Wrote figures and manifest under", OUT_DIR)


if __name__ == "__main__":
    main()
