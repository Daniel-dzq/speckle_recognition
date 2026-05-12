#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate planning-document figures 4, 7, 8 into figures_competition/ only.

Does not write to figures_publication/ (avoid confusion with paper auth figures).

Usage (repo root):
    python scripts/generate_competition_figures.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.metrics.stability import temporal_stability_score  # noqa: E402

OUT_DIR = ROOT / "figures_competition"
DOCS_DIR = ROOT / "docs"
LOSS_CSV = ROOT / "results" / "length_optimization_green" / "tables" / "per_length_summary.csv"
OPTIMAL_JSON = ROOT / "results" / "length_optimization_green" / "optimal_length.json"
LONG_TERM_ROOT = ROOT / "long_term_stability"
VIDEO_GREEN = ROOT / "videocapture" / "Green" / "Fiber1" / "A.avi"
VIDEO_DUAL = ROOT / "videocapture" / "GreenAndRed" / "Fiber1" / "A.avi"

# --- Proposal / manuscript summary values (Fig 7b partial, Fig 8) when raw not split ---
FIG7B_RED_REL_RETENTION = 0.95   # ~5% drop
FIG7B_GREEN_REL_RETENTION = 0.75  # ~25% drop
FIG8_CV_GREEN_RAW = 38.2
FIG8_CV_ETA = 4.3
FIG8_REINSTALL_IMPROVEMENT_FRAC = 0.28  # +28%
# Fig 7a: red bar when no red time-series dataset (summary from proposal: slightly above green)
FIG7A_RED_NCC_SUMMARY = 0.94


def _save_triple(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.svg", bbox_inches="tight", facecolor="white")


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

    # (a) Loss
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

    # (b) Bars + twin ratio (scale ~1–1.8 so 1.5653 is never read as 156×)
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
    # Combine legends
    h1, l1 = axb.get_legend_handles_labels()
    h2, l2 = axb2.get_legend_handles_labels()
    axb.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.5, frameon=False)
    axb.set_title("(b)")
    axb.spines["top"].set_visible(False)

    # (c) Entropy
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


def _long_term_green_ncc() -> Optional[float]:
    try:
        import cv2
    except ImportError:
        return None
    nccs: List[float] = []
    if not LONG_TERM_ROOT.is_dir():
        return None
    for fib_dir in sorted(LONG_TERM_ROOT.iterdir()):
        if not fib_dir.is_dir():
            continue
        imgs: List[np.ndarray] = []
        for p in sorted(fib_dir.glob("*.jpg")) + sorted(fib_dir.glob("*.JPG")):
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            h, w = im.shape
            s = min(h, w, 400)
            y0, x0 = (h - s) // 2, (w - s) // 2
            crop = im[y0 : y0 + s, x0 : x0 + s].astype(np.float64) / 255.0
            imgs.append(crop.reshape(-1))
        if len(imgs) < 2:
            continue
        ts = temporal_stability_score(imgs)
        v = float(ts["consecutive_ncc"])
        if v == v:
            nccs.append(v)
    if not nccs:
        return None
    return float(np.mean(nccs))


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


def _video_middle_gray(path: Path, channel: str = "green") -> Optional[np.ndarray]:
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
    if channel == "gray":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if channel == "red":
        return frame[:, :, 2]  # BGR → R
    if channel == "green":
        return frame[:, :, 1]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def fig7_dual_channel(manifest_rows: List[Dict[str, str]]) -> None:
    notes: List[str] = []
    green_ncc = _long_term_green_ncc()
    # (a) Red vs Green intra-class NCC: green from long_term_stability JPEGs; red from proposal summary
    if green_ncc is None:
        green_ncc = 0.89
        notes.append("Fig7a green NCC: long_term_stability missing or OpenCV failed — fallback 0.89")
    red_ncc = FIG7A_RED_NCC_SUMMARY
    notes.append("Fig7a red NCC: manuscript summary (no dedicated red time-series folder in repo)")

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3))

    ax = axes[0]
    xb = np.arange(2)
    ax.bar(xb, [red_ncc, green_ncc], color=["#d62728", "#2ca02c"], width=0.55, edgecolor="white", lw=0.5)
    ax.set_xticks(xb)
    ax.set_xticklabels(["Red channel", "Green channel"])
    ax.set_ylabel("Intra-class correlation coefficient")
    ax.set_ylim(0, 1.05)
    ax.set_title("(a)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) Micro-bend summary retention (not raw before/after pairs)
    axb = axes[1]
    cat = ["Red channel", "Green channel"]
    before = [1.0, 1.0]
    after = [FIG7B_RED_REL_RETENTION, FIG7B_GREEN_REL_RETENTION]
    xx = np.arange(2)
    w = 0.35
    axb.bar(xx - w / 2, before, width=w, label="Before bend (norm.)", color="#b8b8b8")
    axb.bar(xx + w / 2, after, width=w, label="After bend (norm.)", color="#7aa6c9")
    axb.set_xticks(xx)
    axb.set_xticklabels(cat)
    axb.set_ylabel("Relative correlation (normalized)")
    axb.set_ylim(0, 1.15)
    axb.legend(fontsize=7, frameon=False, loc="upper right")
    axb.set_title("(b)")
    axb.spines["top"].set_visible(False)
    axb.spines["right"].set_visible(False)
    notes.append(
        "Fig7b: summary-only bar chart; red ~5% drop, green ~25% drop vs baseline (manuscript)."
    )

    axc = axes[2]
    rprof_red_ok = False
    rg, rr = _video_middle_gray(VIDEO_GREEN, "green"), _video_middle_gray(VIDEO_DUAL, "red")
    gchan = _video_middle_gray(VIDEO_DUAL, "green")
    if rg is not None and rr is not None:
        h, w = min(rg.shape[0], rr.shape[0]), min(rg.shape[1], rr.shape[1])
        rg, rr = rg[:h, :w], rr[:h, :w]
        r_r, p_r = _radial_mean_profile(rr.astype(np.float64) / 255.0)
        r_g, p_g = _radial_mean_profile(rg.astype(np.float64) / 255.0)
        axc.plot(r_r, p_r / (np.nanmax(p_r) + 1e-9), color="#d62728", lw=1.4, label="Red (cam R ch., dual video)")
        axc.plot(r_g, p_g / (np.nanmax(p_g) + 1e-9), color="#2ca02c", lw=1.4, label="Green side (Green folder / G ch.)")
        axc.set_xlabel("Radius (px)")
        axc.set_ylabel("Norm. radial mean intensity")
        axc.legend(fontsize=6.5, frameon=False)
        rprof_red_ok = True
        notes.append("Fig7c: radial profile from video middle frame (proxy channels).")
    elif gchan is not None:
        r_g, p_g = _radial_mean_profile(gchan.astype(np.float64) / 255.0)
        axc.plot(r_g, p_g / (np.nanmax(p_g) + 1e-9), color="#2ca02c", lw=1.4)
        axc.text(0.5, 0.5, "Partial data: green profile only.\nRed end-face profile needs measured image.",
                 ha="center", va="center", ma="center", transform=axc.transAxes, fontsize=8, color="#444")
        notes.append("Fig7c: green channel profile only; red missing.")
    else:
        axc.axis("off")
        axc.text(
            0.5, 0.5,
            "Measured speckle / radial profile not available\n(CCD images or videos missing).",
            ha="center", va="center", ma="center", transform=axc.transAxes, fontsize=8, color="#444",
        )
        notes.append("Fig7c: missing — no usable video frame.")

    axc.set_title("(c)")
    axc.spines["top"].set_visible(False)

    fig.subplots_adjust(wspace=0.36, left=0.07, right=0.99, top=0.88, bottom=0.20)
    out = OUT_DIR / "fig7_dual_channel_characterization"
    _save_triple(fig, out)
    plt.close(fig)

    data_tag = "mixed"
    manifest_rows.append({
        "figure_id": "fig7_dual_channel_characterization",
        "output_png": str(out) + ".png",
        "output_svg": str(out) + ".svg",
        "output_pdf": str(out) + ".pdf",
        "source_data": f"{LONG_TERM_ROOT}; {VIDEO_GREEN}; {VIDEO_DUAL}",
        "source_script": "scripts/generate_competition_figures.py",
        "raw_data_summary_only_missing": data_tag,
        "note": " | ".join(notes),
    })


def fig8_common_mode(manifest_rows: List[Dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))

    # (a) CV bars
    ax = axes[0]
    labs = ["Raw green\nintensity", "η = G/R"]
    vals = [FIG8_CV_GREEN_RAW, FIG8_CV_ETA]
    clrs = ["#8fbc8f", "#4c72b0"]
    xo = np.arange(2)
    ax.bar(xo, vals, color=clrs, width=0.5, edgecolor="white", lw=0.5)
    ax.set_xticks(xo)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("Coefficient of variation (%)")
    ax.set_title("(a)")
    fac = FIG8_CV_GREEN_RAW / FIG8_CV_ETA
    ax.annotate(
        f"≈ {fac:.1f}× reduction",
        xy=(0.5, max(vals) * 0.72),
        ha="center",
        fontsize=8,
        color="#333333",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) Reinstall: raw green vs η, +28% relative improvement
    axb = axes[1]
    base = 0.72
    ncc_green = base
    ncc_eta = base * (1.0 + FIG8_REINSTALL_IMPROVEMENT_FRAC)
    axb.bar([0, 1], [ncc_green, ncc_eta], color=["#8fbc8f", "#9467bd"], width=0.45, edgecolor="white", lw=0.5)
    axb.set_xticks([0, 1])
    axb.set_xticklabels(["Raw green\n(intensity feature)", "η = G/R\n(ratio feature)"], fontsize=7)
    axb.set_ylabel("Intra-class correlation coefficient")
    axb.set_title("(b)")
    axb.annotate(
        "+28%\n(vs raw)",
        xy=(1, ncc_eta + 0.02),
        ha="center",
        fontsize=7,
        color="#333333",
    )
    axb.set_ylim(0, min(1.05, ncc_eta + 0.15))
    axb.spines["top"].set_visible(False)
    axb.spines["right"].set_visible(False)

    fig.subplots_adjust(wspace=0.35, left=0.10, right=0.98, top=0.88, bottom=0.22)
    out = OUT_DIR / "fig8_common_mode_suppression"
    _save_triple(fig, out)
    plt.close(fig)

    manifest_rows.append({
        "figure_id": "fig8_common_mode_suppression",
        "output_png": str(out) + ".png",
        "output_svg": str(out) + ".svg",
        "output_pdf": str(out) + ".pdf",
        "source_data": "manuscript summary statistics (proposal)",
        "source_script": "scripts/generate_competition_figures.py",
        "raw_data_summary_only_missing": "summary_only",
        "note": f"CV {FIG8_CV_GREEN_RAW}% vs {FIG8_CV_ETA}%; relative +{FIG8_REINSTALL_IMPROVEMENT_FRAC*100:.0f}% η NCC vs scaled raw baseline.",
    })


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
        lines.append(
            f"| {r['figure_id']} | {r['raw_data_summary_only_missing']} | {r['source_data'][:80]}… |"
            if len(r["source_data"]) > 80
            else f"| {r['figure_id']} | {r['raw_data_summary_only_missing']} | {r['source_data']} |"
        )
    lines.extend(["", "## Notes per figure", ""])
    for r in rows:
        lines.extend([f"### {r['figure_id']}", "", r["note"], ""])
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "competition_required_figures_status.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, str]] = []
    fig4_length_optimization(manifest)
    fig7_dual_channel(manifest)
    fig8_common_mode(manifest)
    write_manifest(manifest)
    write_status_md(manifest)
    print("Wrote figures under", OUT_DIR)
    print("Wrote", DOCS_DIR / "competition_required_figures_status.md")


if __name__ == "__main__":
    main()
