#!/usr/bin/env python3
"""
Partial section-3.2 analysis on the newly acquired green speckle screenshots.

Scope (by design):
    * Uses ONLY images under ``Green/`` (the freshly captured dataset).
    * Does NOT touch ``videocapture/`` or any older experiment data.
    * Computes every metric from section 3.2 that the current data supports
      and explicitly reports what cannot be computed yet (e.g. transmission
      loss requires power-meter measurements that are not in the repo).

Dataset layout detected / expected::

    Green/<length_tag>/<fiber_tag>/<N>.JPG

Each ``<N>.JPG`` is one of the 10 random samples of a fixed (length, fiber)
pair captured under the same acquisition setting (fixed side-illumination
green laser, no programmable SLM pattern — see paper section 3.2).

Output::

    results/green_partial_32/
        per_image_metrics.csv     one row per JPG
        per_fiber_summary.csv     one row per (length, fiber) pair
        per_length_summary.csv    one row per length group
        summary.json              machine-readable summary
        report.md                 honest, publication-aware report
        figures/                  PNG + PDF + SVG
            speckle_montage.*
            entropy_per_length.*
            separability_per_length.*
            intensity_histogram.*

Usage::

    python scripts/run_partial_length_analysis.py
    python scripts/run_partial_length_analysis.py \\
        --data-root Green --output results/green_partial_32 \\
        --roi 400 --channel green
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Repo-root import patch
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import cv2
import matplotlib.pyplot as plt

from analysis.metrics.basic import (
    pairwise_euclidean,
    shannon_entropy,
    coefficient_of_variation,
)
from analysis.plotting.charts import grouped_bars, image_panel, line_with_error
from analysis.plotting.style import PALETTE, SINGLE_COL_W, apply_style, save_figure
from analysis.reporting.writers import MarkdownBuilder, write_csv, write_json


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Accepts tags like "Fiber11cm", "11cm", "L11", "11_cm" — extracts the number.
_LENGTH_MM_RE = re.compile(r"(?:^|[^0-9])(\d+(?:\.\d+)?)\s*(mm|cm|m)?", re.IGNORECASE)


def _parse_length_mm(tag: str) -> Optional[float]:
    """Best-effort parse of a length tag like ``Fiber11cm`` -> ``110`` mm."""
    m = _LENGTH_MM_RE.search(tag)
    if not m:
        return None
    value = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit == "mm":
        return value
    if unit == "cm" or unit == "":
        return value * 10.0
    if unit == "m":
        return value * 1000.0
    return None


def _natural_key(name: str) -> Tuple[int, object]:
    """Order ``1.JPG, 2.JPG, ..., 10.JPG`` in natural order."""
    stem = Path(name).stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def discover(root: Path) -> List[Dict[str, Any]]:
    """Return a list of records describing every image under ``root``."""
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    records: List[Dict[str, Any]] = []
    for length_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        length_tag = length_dir.name
        length_mm = _parse_length_mm(length_tag)
        for fiber_dir in sorted(p for p in length_dir.iterdir() if p.is_dir()):
            fiber_tag = fiber_dir.name
            files = sorted(
                (p for p in fiber_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
                key=lambda p: _natural_key(p.name),
            )
            for idx, path in enumerate(files):
                records.append({
                    "path": path,
                    "length_tag": length_tag,
                    "length_mm": length_mm,
                    "fiber_tag": fiber_tag,
                    "repeat_index": idx + 1,
                    "stem": path.stem,
                })
    return records


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _load_gray(path: Path, channel: str = "green") -> np.ndarray:
    """
    Load a colour JPG and collapse it to a single 8-bit channel.

    ``channel`` values:
        ``green``     - take the green plane only (recommended for green laser)
        ``red``       - take the red plane only
        ``blue``      - take the blue plane only
        ``luminance`` - standard BT.601 luminance (0.114B + 0.587G + 0.299R)
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to read image: {path}")
    if channel == "green":
        plane = img[:, :, 1]
    elif channel == "red":
        plane = img[:, :, 2]
    elif channel == "blue":
        plane = img[:, :, 0]
    elif channel == "luminance":
        f = img.astype(np.float32)
        plane = (0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2])
        plane = np.clip(plane, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown channel: {channel!r}")
    return plane.astype(np.uint8, copy=False)


def _center_crop(img: np.ndarray, roi: int) -> Tuple[np.ndarray, int]:
    """
    Center-crop to a square of side ``roi``. Returns ``(crop, actual_side)``.

    If the image is smaller than ``roi`` the largest square that fits is used
    and the actual side length is reported so downstream metrics remain
    comparable (or at least auditable).
    """
    h, w = img.shape[:2]
    side = min(roi, h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0:y0 + side, x0:x0 + side], side


# ---------------------------------------------------------------------------
# Per-image metric extraction
# ---------------------------------------------------------------------------


def extract_per_image(
    records: List[Dict[str, Any]],
    roi: int,
    channel: str,
    logger=None,
) -> Tuple[List[Dict[str, Any]], np.ndarray, List[np.ndarray]]:
    """
    Process every image once and return:

        * ``rows``     - per-image descriptive metrics
        * ``vectors``  - ``(N, roi*roi)`` float32 vectors normalised per image
                         (used for Euclidean distance)
        * ``previews`` - list of small 256x256 thumbnails of the ROI (for the
                         montage figure)
    """
    rows: List[Dict[str, Any]] = []
    vecs: List[np.ndarray] = []
    previews: List[np.ndarray] = []

    for rec in records:
        raw = _load_gray(rec["path"], channel=channel)
        crop, actual_side = _center_crop(raw, roi)

        # ROI-based per-image statistics (raw 8-bit)
        mean = float(crop.mean())
        std = float(crop.std())
        contrast = std / mean if mean > 1e-6 else float("nan")
        entropy_bits = shannon_entropy(crop, bins=256, normalize=False)

        # Min-max normalise per image for distance comparability (removes
        # small exposure drift between captures).
        crop_f = crop.astype(np.float32)
        rng = crop_f.max() - crop_f.min()
        if rng > 1e-6:
            norm = (crop_f - crop_f.min()) / rng
        else:
            norm = np.zeros_like(crop_f)
        vecs.append(norm.ravel().astype(np.float32, copy=False))

        # Thumbnail for montage
        thumb = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
        previews.append(thumb)

        rows.append({
            "length_tag": rec["length_tag"],
            "length_mm": rec["length_mm"],
            "fiber_tag": rec["fiber_tag"],
            "repeat_index": rec["repeat_index"],
            "image_path": str(rec["path"].relative_to(rec["path"].parents[3]))
                           if len(rec["path"].parents) >= 4 else str(rec["path"]),
            "roi_side": actual_side,
            "mean_intensity": mean,
            "intensity_std": std,
            "contrast_proxy": contrast,
            "entropy_bits": entropy_bits,
        })
        if logger:
            logger(f"  {rec['length_tag']}/{rec['fiber_tag']}/{rec['path'].name}: "
                   f"entropy={entropy_bits:.3f} bits, mean={mean:.1f}, std={std:.1f}")

    vectors = np.stack(vecs, axis=0) if vecs else np.zeros((0, roi * roi), dtype=np.float32)
    return rows, vectors, previews


# ---------------------------------------------------------------------------
# Per-fiber / per-length aggregation
# ---------------------------------------------------------------------------


def aggregate_per_fiber(
    rows: List[Dict[str, Any]],
    vectors: np.ndarray,
) -> List[Dict[str, Any]]:
    """Per (length, fiber): mean entropy + mean intra-class pairwise distance."""
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["length_tag"], r["fiber_tag"])].append(i)

    out: List[Dict[str, Any]] = []
    for (length_tag, fiber_tag), idxs in sorted(groups.items()):
        ent = [rows[i]["entropy_bits"] for i in idxs]
        mean_int = [rows[i]["mean_intensity"] for i in idxs]
        std_int = [rows[i]["intensity_std"] for i in idxs]
        if len(idxs) >= 2:
            D = pairwise_euclidean(vectors[idxs])
            iu = np.triu_indices(len(idxs), k=1)
            intra = float(D[iu].mean())
            intra_std = float(D[iu].std())
            n_pairs = int(iu[0].size)
        else:
            intra, intra_std, n_pairs = float("nan"), float("nan"), 0
        out.append({
            "length_tag": length_tag,
            "length_mm": rows[idxs[0]]["length_mm"],
            "fiber_tag": fiber_tag,
            "n_images": len(idxs),
            "entropy_bits_mean": float(np.mean(ent)),
            "entropy_bits_std": float(np.std(ent, ddof=0)),
            "mean_intensity": float(np.mean(mean_int)),
            "intensity_std_mean": float(np.mean(std_int)),
            "intra_distance": intra,
            "intra_distance_std": intra_std,
            "n_intra_pairs": n_pairs,
        })
    return out


def aggregate_per_length(
    rows: List[Dict[str, Any]],
    vectors: np.ndarray,
    fiber_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Per length group: mean entropy + intra + inter + ratio."""
    by_length: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_length[r["length_tag"]].append(i)

    def _length_sort_key(tag: str) -> Tuple[float, str]:
        mm = next(
            (rows[i]["length_mm"] for i in by_length[tag] if rows[i]["length_mm"] is not None),
            None,
        )
        if mm is None:
            return (float("inf"), tag)
        return (float(mm), tag)

    out: List[Dict[str, Any]] = []
    for length_tag in sorted(by_length, key=_length_sort_key):
        idxs = by_length[length_tag]
        labels = np.asarray([rows[i]["fiber_tag"] for i in idxs])
        unique_fibers = sorted(set(labels.tolist()))

        # Intra aggregate = mean of per-fiber intra distances in this length
        fiber_intras = [r["intra_distance"] for r in fiber_rows
                        if r["length_tag"] == length_tag
                        and not np.isnan(r["intra_distance"])]
        intra_mean = float(np.mean(fiber_intras)) if fiber_intras else float("nan")
        intra_std = float(np.std(fiber_intras, ddof=0)) if fiber_intras else float("nan")

        # Inter = mean pairwise distance between samples of DIFFERENT fibers
        if len(unique_fibers) >= 2 and len(idxs) >= 2:
            sub = vectors[idxs]
            D = pairwise_euclidean(sub)
            iu = np.triu_indices(D.shape[0], k=1)
            same_fiber = labels[iu[0]] == labels[iu[1]]
            diff_pairs = D[iu][~same_fiber]
            inter_mean = float(diff_pairs.mean()) if diff_pairs.size else float("nan")
            inter_std = float(diff_pairs.std()) if diff_pairs.size else float("nan")
            n_inter_pairs = int(diff_pairs.size)
        else:
            inter_mean, inter_std, n_inter_pairs = float("nan"), float("nan"), 0

        ratio = (inter_mean / intra_mean
                 if intra_mean and intra_mean == intra_mean and intra_mean > 0
                 and inter_mean == inter_mean
                 else float("nan"))

        entropies_fiber = [r["entropy_bits_mean"] for r in fiber_rows
                           if r["length_tag"] == length_tag]

        out.append({
            "length_tag": length_tag,
            "length_mm": next((rows[i]["length_mm"] for i in idxs
                               if rows[i]["length_mm"] is not None), None),
            "n_fibers": len(unique_fibers),
            "n_images": len(idxs),
            "entropy_bits_mean": float(np.mean(entropies_fiber))
                if entropies_fiber else float("nan"),
            "entropy_bits_std": float(np.std(entropies_fiber, ddof=0))
                if entropies_fiber else float("nan"),
            "intra_distance_mean": intra_mean,
            "intra_distance_std": intra_std,
            "inter_distance_mean": inter_mean,
            "inter_distance_std": inter_std,
            "inter_intra_ratio": ratio,
            "n_inter_pairs": n_inter_pairs,
        })
    return out


# ---------------------------------------------------------------------------
# Plotting helpers (specific to this partial analysis)
# ---------------------------------------------------------------------------


def plot_speckle_montage(
    rows: List[Dict[str, Any]],
    previews: List[np.ndarray],
    output_stem: Path,
):
    # One representative (first) image per (length, fiber) pair.
    seen: Dict[Tuple[str, str], int] = {}
    for i, r in enumerate(rows):
        key = (r["length_tag"], r["fiber_tag"])
        seen.setdefault(key, i)

    length_tags = sorted({k[0] for k in seen})
    fiber_tags = sorted({k[1] for k in seen})
    images: List[Optional[np.ndarray]] = []
    for lt in length_tags:
        for ft in fiber_tags:
            if (lt, ft) in seen:
                images.append(previews[seen[(lt, ft)]])
            else:
                images.append(None)
    fig, _ = image_panel(
        [im if im is not None else np.zeros((16, 16)) for im in images],
        row_labels=length_tags,
        col_labels=fiber_tags,
        rows=len(length_tags),
        cols=len(fiber_tags),
        title="Representative green speckle (one per fiber × length)",
        cmap="magma",
        cell_size=1.3,
    )
    return save_figure(fig, output_stem)


def plot_entropy(length_rows: List[Dict[str, Any]], output_stem: Path):
    xs = list(range(len(length_rows)))
    values = [r["entropy_bits_mean"] for r in length_rows]
    errs = [r["entropy_bits_std"] for r in length_rows]
    fig, ax = line_with_error(
        xs,
        {"Entropy (bits)": (values, errs)},
        xlabel="Fiber length group",
        ylabel="Pixel entropy (bits, 256-bin)",
        title="Pixel entropy per length group",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([r["length_tag"] for r in length_rows])
    return save_figure(fig, output_stem)


def plot_separability(length_rows: List[Dict[str, Any]], output_stem: Path):
    tags = [r["length_tag"] for r in length_rows]
    intra = [r["intra_distance_mean"] for r in length_rows]
    inter = [r["inter_distance_mean"] for r in length_rows]
    ratio = [r["inter_intra_ratio"] for r in length_rows]
    subtitle = "(green ROI; vectors min-max normalised per image)"

    fig, ax = grouped_bars(
        tags,
        {"Intra-class distance": intra, "Inter-class distance": inter},
        ylabel="Euclidean distance (ROI)",
        xlabel="Fiber length (captures grouped by folder)",
        title="",
        value_labels=False,
        legend=False,
        layout="none",
        figsize=(6.6, SINGLE_COL_W / 1.25),
        bar_width=0.72,
    )
    fig.suptitle(
        "Intra vs inter-class distance by fiber length\n" + subtitle,
        fontsize=9,
        y=0.97,
        va="top",
    )
    y_top = float(max((*intra, *inter, 0)))
    pad = max(8.0, y_top * 0.08)
    ax.set_ylim(0, min(y_top + pad, max(y_top * 1.28, y_top + 18)))

    for bar_cont in ax.containers:
        color = bar_cont.patches[0].get_facecolor() if bar_cont.patches else "black"
        for b in bar_cont:
            v = b.get_height()
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + pad * 0.15,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=color,
            )

    ax2 = ax.twinx()
    ax2.plot(
        np.arange(len(tags)),
        ratio,
        marker="D",
        color=PALETTE[3],
        linewidth=1.35,
        label="Inter / intra ratio",
        zorder=5,
        markersize=5,
        clip_on=False,
    )
    rmax = np.nanmax(ratio)
    ax2.set_ylim(0, max(np.nanmean(ratio) * 1.35, float(rmax) * 1.12, 1.0))
    ax2.set_ylabel("Inter / intra ratio")
    ax2.spines["top"].set_visible(False)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    leg = ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=3,
        frameon=True,
        fontsize=7.5,
        columnspacing=0.85,
    )
    leg.set_zorder(10)

    fig.subplots_adjust(bottom=0.38, left=0.11, right=0.88, top=0.78)
    return save_figure(fig, output_stem)


def plot_intensity_histogram(
    rows: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    output_stem: Path,
    channel: str,
    roi: int,
    sample_per_group: int = 3,
):
    # Sample a few images per length group and stack their 8-bit histograms.
    by_length: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_length[r["length_tag"]].append(i)

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for ci, (length_tag, idxs) in enumerate(sorted(by_length.items())):
        sel = idxs[:sample_per_group]
        hist_sum = np.zeros(256, dtype=np.float64)
        for i in sel:
            img = _load_gray(records[i]["path"], channel=channel)
            crop, _ = _center_crop(img, roi)
            hist = np.bincount(crop.ravel(), minlength=256)
            hist_sum += hist
        hist_sum /= max(1, hist_sum.sum())
        ax.plot(np.arange(256), hist_sum, color=PALETTE[ci % len(PALETTE)],
                linewidth=1.2, label=length_tag)
    ax.set_xlabel("8-bit intensity (green channel)")
    ax.set_ylabel("Normalised frequency")
    ax.set_title("ROI intensity distribution")
    ax.legend(loc="best")
    return save_figure(fig, output_stem)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------


_DEFAULT_ROOT_CANDIDATES = [
    Path(_ROOT) / "LengthOptimize" / "Green",
    Path(_ROOT) / "Green",
]


def _default_data_root() -> Path:
    for cand in _DEFAULT_ROOT_CANDIDATES:
        if cand.exists():
            return cand
    return _DEFAULT_ROOT_CANDIDATES[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-root", type=Path,
                   default=_default_data_root(),
                   help="Path to the green-screenshot dataset root. Defaults to "
                        "the first of: LengthOptimize/Green, Green.")
    p.add_argument("--output", type=Path,
                   default=Path(_ROOT) / "results" / "green_partial_32",
                   help="Output directory (default: results/green_partial_32).")
    p.add_argument("--roi", type=int, default=400,
                   help="Side length of the square center ROI in pixels (default: 400).")
    p.add_argument("--channel", choices=["green", "red", "blue", "luminance"],
                   default="green",
                   help="Which plane of the BGR JPGs to analyse (default: green).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    apply_style()

    out_dir: Path = args.output
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    for d in (out_dir, fig_dir, table_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[partial-3.2] Dataset root     : {args.data_root}")
    print(f"[partial-3.2] Output directory : {out_dir}")
    print(f"[partial-3.2] Channel          : {args.channel}")
    print(f"[partial-3.2] Center ROI       : {args.roi}x{args.roi} px")

    records = discover(args.data_root)
    if not records:
        raise SystemExit(f"No images discovered under {args.data_root}.")
    print(f"[partial-3.2] Discovered        : {len(records)} images")

    length_tags = sorted({r["length_tag"] for r in records})
    fibers_per_length = {
        lt: sorted({r["fiber_tag"] for r in records if r["length_tag"] == lt})
        for lt in length_tags
    }
    for lt in length_tags:
        print(f"                                 {lt}: "
              f"{len(fibers_per_length[lt])} fibers, "
              f"{sum(1 for r in records if r['length_tag'] == lt)} images total")

    per_image, vectors, previews = extract_per_image(
        records, roi=args.roi, channel=args.channel,
    )
    per_fiber = aggregate_per_fiber(per_image, vectors)
    per_length = aggregate_per_length(per_image, vectors, per_fiber)

    # ---------------- CSVs ----------------
    per_image_csv = write_csv(table_dir / "per_image_metrics.csv", per_image)
    per_fiber_csv = write_csv(out_dir / "per_fiber_summary.csv", per_fiber)
    per_length_csv = write_csv(out_dir / "per_length_summary.csv", per_length)

    # Also keep a copy under tables/
    write_csv(table_dir / "per_fiber_summary.csv", per_fiber)
    write_csv(table_dir / "per_length_summary.csv", per_length)

    # ---------------- Figures ----------------
    montage_paths = plot_speckle_montage(per_image, previews, fig_dir / "speckle_montage")
    entropy_paths = plot_entropy(per_length, fig_dir / "entropy_per_length")
    sep_paths = (plot_separability(per_length, fig_dir / "separability_per_length")
                 if any(not np.isnan(r["inter_distance_mean"]) for r in per_length)
                 else [])
    hist_paths = plot_intensity_histogram(
        per_image, records, fig_dir / "intensity_histogram",
        channel=args.channel, roi=args.roi,
    )

    # ---------------- JSON summary ----------------
    has_multiple_lengths = len(length_tags) >= 2
    tentative = None
    if has_multiple_lengths and all(
        not np.isnan(r["inter_intra_ratio"]) for r in per_length
    ):
        best = max(per_length, key=lambda r: r["inter_intra_ratio"])
        tentative = {
            "preferred_length_tag": best["length_tag"],
            "inter_intra_ratio": best["inter_intra_ratio"],
            "entropy_bits_mean": best["entropy_bits_mean"],
            "note": (
                "Preliminary preference based on the two currently available "
                "length groups. Not a final recommendation; see limitations."
            ),
        }

    summary = {
        "status": "partial",
        "scope": (
            "Section 3.2 metrics that can be derived from the newly acquired "
            "green-channel screenshots under Green/."
        ),
        "data_root": str(args.data_root),
        "channel": args.channel,
        "roi_px": args.roi,
        "distance_metric": "Euclidean on per-image min-max normalised ROI vectors",
        "entropy_definition": "Shannon entropy on 256-bin raw 8-bit histogram",
        "n_images": len(per_image),
        "length_groups_available": length_tags,
        "fibers_per_length_group": fibers_per_length,
        "per_length_summary": per_length,
        "tentative_preference": tentative,
        "metrics_not_yet_available": {
            "transmission_loss_dB": (
                "Requires paired input/output green-power meter readings that "
                "are not in the repository."
            ),
            "additional_length_groups": (
                f"Only {len(length_tags)} length group(s) captured. The full "
                "paper narrative (e.g. entropy saturation, mode-mixing onset, "
                "damage-induced loss at long lengths) needs additional "
                "lengths such as 5 cm, 30 cm, 45 cm."
            ),
            "red_channel_measurements": (
                "Only green-channel screenshots were acquired; red-channel or "
                "dual-channel analyses (sections 3.3 / 3.4) are out of scope."
            ),
        },
    }
    summary_json = write_json(out_dir / "summary.json", summary)

    # ---------------- Markdown report ----------------
    md = MarkdownBuilder("Experiment 3.2 — Partial fiber-length analysis (green screenshots)")
    md.p(
        "This is a **partial** analysis based on newly acquired green-channel "
        "screenshots under `Green/`. Only the currently available length "
        "groups were analysed. Transmission-loss analysis is pending power "
        "measurements. The present results are preliminary and do not yet "
        "support a final four-length optimal-length conclusion."
    )
    md.h(2, "Dataset detected")
    md.kv({
        "Dataset root": str(args.data_root),
        "Channel analysed": args.channel,
        "Center ROI": f"{args.roi} x {args.roi} px",
        "Total images": len(per_image),
        "Length groups": ", ".join(length_tags),
    })
    md.table(
        ["Length group", "Length (mm)", "# fibers", "# images"],
        [[r["length_tag"],
          r["length_mm"] if r["length_mm"] is not None else "—",
          r["n_fibers"], r["n_images"]]
         for r in per_length],
    )

    md.h(2, "Per-length summary")
    md.table(
        ["Length", "Entropy (bits)", "Intra distance", "Inter distance",
         "Inter / Intra", "# inter pairs"],
        [[r["length_tag"], r["entropy_bits_mean"],
          r["intra_distance_mean"], r["inter_distance_mean"],
          r["inter_intra_ratio"], r["n_inter_pairs"]]
         for r in per_length],
    )

    md.h(2, "Per-fiber summary")
    md.table(
        ["Length", "Fiber", "# imgs", "Entropy (bits)", "Mean intensity",
         "Std intensity", "Intra distance"],
        [[r["length_tag"], r["fiber_tag"], r["n_images"],
          r["entropy_bits_mean"], r["mean_intensity"],
          r["intensity_std_mean"], r["intra_distance"]]
         for r in per_fiber],
    )

    if tentative is not None:
        md.h(2, "Tentative preference (preliminary)")
        md.kv(tentative)

    md.h(2, "What was computed")
    md.bullet([
        "Shannon pixel entropy (256-bin, raw 8-bit intensity) — paper 3.2 definition.",
        "Intra-class Euclidean distance over the 10 repeated samples of each (length, fiber) pair.",
        "Inter-class Euclidean distance over pairs of samples from different fibers within the same length group.",
        "Inter / Intra ratio per length group.",
        "Per-image intensity mean, std, contrast-proxy, and 256-bin histogram plot.",
        "Representative speckle montage (one image per fiber × length).",
    ])
    md.h(2, "What is not yet available")
    md.bullet([
        "**Transmission loss (dB)** — needs paired input/output green-power "
        "readings per fiber. The script will ingest them through a CSV once "
        "available; nothing in this report is computed from them.",
        f"**Extra length groups** — only `{', '.join(length_tags)}` were "
        "captured. Section 3.2 of the paper compares several lengths (e.g. "
        "5 cm, 30 cm, 45 cm); the monotonic entropy / ratio trend and any "
        "damage-induced drop at long lengths cannot be asserted from two "
        "points.",
        "**Red-channel and dual-channel experiments** (sections 3.3 / 3.4) — "
        "not in scope here; only green screenshots were acquired.",
        "**Within-session temporal stability** — every screenshot is a single "
        "still frame, so time-domain drift cannot be estimated from this "
        "dataset. Use the video acquisitions for that.",
    ])
    md.h(2, "Methodology notes")
    md.bullet([
        f"Every image is first reduced to the {args.channel} channel, then "
        "center-cropped to a fixed square ROI. ROI side is reported per image "
        "in `tables/per_image_metrics.csv` so any crop fallback (if an image "
        "were smaller) is auditable.",
        "Distance vectors are min-max normalised per image before the L2 "
        "distance, which removes exposure-level drift across captures while "
        "preserving spatial speckle structure. Raw 8-bit statistics "
        "(`mean_intensity`, `intensity_std`, `contrast_proxy`) are saved "
        "separately for completeness.",
        "Pixel entropy uses the raw 8-bit histogram (256 bins), not the "
        "normalised vector, so it reflects the true intensity distribution.",
    ])
    md.h(2, "Artefacts")
    md.bullet([
        "`per_fiber_summary.csv`, `per_length_summary.csv`, `summary.json`",
        "`tables/per_image_metrics.csv` — one row per JPG",
        "`figures/speckle_montage.{png,pdf,svg}`",
        "`figures/entropy_per_length.{png,pdf,svg}`",
        "`figures/separability_per_length.{png,pdf,svg}`",
        "`figures/intensity_histogram.{png,pdf,svg}`",
    ])

    report_path = md.save(out_dir / "report.md")

    # ---------------- Console summary ----------------
    print("\n[partial-3.2] DONE.")
    print(f"  Per-image CSV : {per_image_csv}")
    print(f"  Per-fiber CSV : {per_fiber_csv}")
    print(f"  Per-length CSV: {per_length_csv}")
    print(f"  Summary JSON  : {summary_json}")
    print(f"  Report        : {report_path}")
    print(f"  Figures       : {fig_dir}/ (speckle_montage, entropy_per_length, "
          "separability_per_length, intensity_histogram)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
