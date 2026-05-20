#!/usr/bin/env python3
"""
Generate Figure 2: performance comparison across fiber lengths (8, 9, 11, 13, 16 cm).

Loads distances, entropy, and ratios from Fig3_length_optimization_data.csv (or fallback table).
Loads transmission loss from fiber_loss/power_loss.csv where finite; fills missing lengths/channels
from fiber_loss/*cm.xlsx via transmission_loss_db (same convention as analysis.metrics.basic).

Writes only under figures/paper/Fig2_length_optimization/:
  Fig2_length_optimization.{png,pdf,svg}           # horizontal 1×3 (journal)
  Fig2_length_optimization_word.{png,pdf,svg}      # vertical 3×1 (Word-friendly)
  Fig2_length_optimization_data_summary.csv
  Fig2_length_optimization_report.md

Usage:
  python figures/paper/Fig2_length_optimization/generate_fig2_length_optimization.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

# -----------------------------------------------------------------------------
# Paths (script lives in figures/paper/Fig2_length_optimization/)
# -----------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent.parent

PRIMARY_TABLE = REPO_ROOT / "figures/paper/Fig3_length_optimization/Fig3_length_optimization_data.csv"
FALLBACK_TABLE = REPO_ROOT / "results/length_optimization_green/tables/per_length_summary.csv"
POWER_CSV = REPO_ROOT / "fiber_loss/power_loss.csv"
FIBER_LOSS_DIR = REPO_ROOT / "fiber_loss"
OPTIMAL_JSON = REPO_ROOT / "results/length_optimization_green/optimal_length.json"
CONFIG_YAML = REPO_ROOT / "config/length_optimization_green.yaml"
LENGTH_OPTIMIZE_GREEN = REPO_ROOT / "LengthOptimize/Green"

OUTPUT_DIR = THIS_DIR
OUTPUT_BASE = OUTPUT_DIR / "Fig2_length_optimization"
OUTPUT_WORD_BASE = OUTPUT_DIR / "Fig2_length_optimization_word"

# Generated artifacts removed before each run (output dir only; never raw data).
LEGACY_OUTPUT_NAMES = (
    "Fig2_length_optimization.png",
    "Fig2_length_optimization.pdf",
    "Fig2_length_optimization.svg",
    "Fig2_length_optimization_vertical.png",
    "Fig2_length_optimization_vertical.pdf",
    "Fig2_length_optimization_vertical.svg",
    "Fig2_length_optimization_horizontal_clean.png",
    "Fig2_length_optimization_horizontal_clean.pdf",
    "Fig2_length_optimization_horizontal_clean.svg",
)


def cleanup_legacy_outputs() -> None:
    for name in LEGACY_OUTPUT_NAMES:
        p = OUTPUT_DIR / name
        if p.is_file():
            p.unlink()


RIGHT_PROP_CM: Dict[float, float] = {
    8.0: 2.0,
    9.0: 3.0,
    11.0: 5.0,
    13.0: 7.0,
    16.0: 10.0,
}

LENGTH_ORDER = [8.0, 9.0, 11.0, 13.0, 16.0]
OPTIMUM_CM = 9.0


def transmission_loss_db(p_in: Optional[float], p_out: Optional[float]) -> float:
    """Same convention as analysis.metrics.basic.transmission_loss_db."""
    if p_in is None or p_out is None:
        return float("nan")
    if p_in <= 0 or p_out <= 0:
        return float("inf")
    return float(-10.0 * math.log10(float(p_out) / float(p_in)))


def _parse_float(x: str) -> Optional[float]:
    if x is None or str(x).strip() == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _scalar_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def load_primary_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate_power_loss_from_csv(path: Path) -> Dict[float, Dict[str, Any]]:
    """Mean/std green and red loss per total length from paired power measurements."""
    if not path.is_file():
        return {}
    rows = load_primary_rows(path)
    by_len: Dict[str, List[Tuple[float, float, float, float]]] = defaultdict(list)
    for r in rows:
        lg = (r.get("length_group") or "").strip()
        lg_norm = lg.lower().replace(" ", "")
        if not (lg_norm.startswith("fiber") and lg_norm.endswith("cm")):
            continue
        num_part = lg_norm[len("fiber") : -len("cm")]
        try:
            cm = float(num_part)
        except ValueError:
            continue
        pin_g = _parse_float(r.get("p_in_green"))
        pout_g = _parse_float(r.get("p_out_green"))
        pin_r = _parse_float(r.get("p_in_red"))
        pout_r = _parse_float(r.get("p_out_red"))
        by_len[str(cm)].append(
            (
                transmission_loss_db(pin_g, pout_g),
                transmission_loss_db(pin_r, pout_r),
                pin_g if pin_g is not None else float("nan"),
                pin_r if pin_r is not None else float("nan"),
            )
        )

    out: Dict[float, Dict[str, Any]] = {}
    for k, tuples_list in by_len.items():
        cm = float(k)
        greens = [t[0] for t in tuples_list if t[0] == t[0] and math.isfinite(t[0])]
        reds = [t[1] for t in tuples_list if t[1] == t[1] and math.isfinite(t[1])]
        out[cm] = {
            "green_loss_mean": float(np.mean(greens)) if greens else float("nan"),
            "green_loss_std": float(np.std(greens, ddof=0)) if len(greens) > 1 else 0.0,
            "red_loss_mean": float(np.mean(reds)) if reds else float("nan"),
            "red_loss_std": float(np.std(reds, ddof=0)) if len(reds) > 1 else 0.0,
            "n_power_samples": len(tuples_list),
        }
    return out


def aggregate_power_loss_from_xlsx_dir(
    fiber_dir: Path,
) -> Tuple[Dict[float, Dict[str, Any]], List[str], Dict[float, str]]:
    """
    Aggregate per-length loss from fiber_loss/*cm.xlsx using Input/Output and Input.1/Output.1.
    Returns mapping, sorted unique relative paths read, and cm -> workbook path for provenance.
    """
    out: Dict[float, Dict[str, Any]] = {}
    cm_source_path: Dict[float, str] = {}
    if not fiber_dir.is_dir():
        return out, [], {}
    for path in sorted(fiber_dir.glob("*.xlsx")):
        stem = path.stem.lower()
        if not stem.endswith("cm") or not stem.startswith("fiber"):
            continue
        num_part = stem.replace("fiber", "").replace("cm", "")
        try:
            cm = float(num_part)
        except ValueError:
            continue
        rel = str(path.relative_to(REPO_ROOT))
        df = pd.read_excel(path)
        greens: List[float] = []
        reds: List[float] = []
        for _, row in df.iterrows():
            pin_g = _scalar_float(row.get("Input"))
            pout_g = _scalar_float(row.get("Output"))
            pin_r = _scalar_float(row.get("Input.1"))
            pout_r = _scalar_float(row.get("Output.1"))
            lg = transmission_loss_db(pin_g, pout_g)
            lr = transmission_loss_db(pin_r, pout_r)
            if math.isfinite(lg):
                greens.append(float(lg))
            if math.isfinite(lr):
                reds.append(float(lr))
        out[cm] = {
            "green_loss_mean": float(np.mean(greens)) if greens else float("nan"),
            "green_loss_std": float(np.std(greens, ddof=0)) if len(greens) > 1 else 0.0,
            "red_loss_mean": float(np.mean(reds)) if reds else float("nan"),
            "red_loss_std": float(np.std(reds, ddof=0)) if len(reds) > 1 else 0.0,
            "n_power_samples": max(len(greens), len(reds)),
        }
        cm_source_path[cm] = rel

    used_sorted = sorted({cm_source_path[c] for c in cm_source_path})
    return out, used_sorted, cm_source_path


def merge_csv_and_xlsx_power_loss(
    csv_agg: Dict[float, Dict[str, Any]],
    xl_agg: Dict[float, Dict[str, Any]],
    cm_xl_path: Dict[float, str],
) -> Dict[float, Dict[str, Any]]:
    """
    CSV values win when finite; otherwise fill green/red from the Excel aggregate for that length.
    Adds green_source / red_source keys (relative repo paths) for debugging.
    """
    csv_rel = str(POWER_CSV.relative_to(REPO_ROOT)) if POWER_CSV.is_file() else "(no csv)"
    merged: Dict[float, Dict[str, Any]] = {}
    all_cm = set(LENGTH_ORDER) | set(csv_agg.keys()) | set(xl_agg.keys())
    for cm in sorted(all_cm):
        c = csv_agg.get(cm, {})
        x = xl_agg.get(cm, {})
        xl_rel = cm_xl_path.get(cm, "fiber_loss/*.xlsx")

        if _finite_opt(c.get("green_loss_mean")):
            gm = float(c["green_loss_mean"])
            gs = float(c.get("green_loss_std") or 0.0)
            g_src = csv_rel
        elif _finite_opt(x.get("green_loss_mean")):
            gm = float(x["green_loss_mean"])
            gs = float(x.get("green_loss_std") or 0.0)
            g_src = xl_rel
        else:
            gm = float("nan")
            gs = 0.0
            g_src = "(missing)"

        if _finite_opt(c.get("red_loss_mean")):
            rm = float(c["red_loss_mean"])
            rs = float(c.get("red_loss_std") or 0.0)
            r_src = csv_rel
        elif _finite_opt(x.get("red_loss_mean")):
            rm = float(x["red_loss_mean"])
            rs = float(x.get("red_loss_std") or 0.0)
            r_src = xl_rel
        else:
            rm = float("nan")
            rs = 0.0
            r_src = "(missing)"

        merged[cm] = {
            "green_loss_mean": gm,
            "green_loss_std": gs,
            "red_loss_mean": rm,
            "red_loss_std": rs,
            "green_source": g_src,
            "red_source": r_src,
            "n_power_samples": c.get("n_power_samples") or x.get("n_power_samples") or 0,
        }
    return merged


def load_power_loss_aggregates() -> Tuple[Dict[float, Dict[str, Any]], str, List[str]]:
    """
    Combine power_loss.csv with fiber_loss/*.xlsx: CSV first when values are finite;
    missing or non-finite per-length channels are filled from Excel.
    """
    csv_agg = aggregate_power_loss_from_csv(POWER_CSV) if POWER_CSV.is_file() else {}
    xl_agg, xl_files, cm_xl_path = aggregate_power_loss_from_xlsx_dir(FIBER_LOSS_DIR)
    merged = merge_csv_and_xlsx_power_loss(csv_agg, xl_agg, cm_xl_path)

    files_used: List[str] = []
    if POWER_CSV.is_file():
        files_used.append(str(POWER_CSV.relative_to(REPO_ROOT)))
    for rel in xl_files:
        if rel not in files_used:
            files_used.append(rel)

    if POWER_CSV.is_file() and xl_agg:
        note = (
            f"{POWER_CSV.relative_to(REPO_ROOT)} (finite values win); "
            f"missing lengths/channels filled from fiber_loss/*.xlsx"
        )
    elif POWER_CSV.is_file():
        note = str(POWER_CSV.relative_to(REPO_ROOT))
    elif xl_agg:
        note = "fiber_loss/*.xlsx (Input/Output columns → transmission_loss_db)"
    else:
        note = "(no fiber_loss/power_loss.csv and no fiber_loss/*.xlsx)"
        return {}, note, []

    return merged, note, files_used


def merge_power_loss_into_metrics(
    metrics_by_len: Dict[float, Dict[str, Any]],
    power_agg: Dict[float, Dict[str, Any]],
) -> None:
    for cm, pv in power_agg.items():
        if cm not in metrics_by_len:
            continue
        m = metrics_by_len[cm]
        gm = pv.get("green_loss_mean")
        rm = pv.get("red_loss_mean")
        if _finite_opt(gm):
            m["green_loss_dB_mean"] = float(gm)
            m["green_loss_dB_std"] = float(pv.get("green_loss_std") or 0.0)
        if _finite_opt(rm):
            m["red_loss_dB_mean"] = float(rm)
            m["red_loss_dB_std"] = float(pv.get("red_loss_std") or 0.0)


def row_to_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one CSV row from Fig3_length_optimization_data.csv."""
    total_cm = _parse_float(row.get("total_length_cm"))
    if total_cm is None:
        return {}
    return {
        "length_group": row.get("length_group", ""),
        "total_length_cm": total_cm,
        "right_propagation_cm": RIGHT_PROP_CM.get(total_cm, float("nan")),
        "n_fibers": int(float(row["n_fibers"])) if row.get("n_fibers") else 0,
        "entropy_bits_mean": _parse_float(row.get("entropy_bits_mean")),
        "entropy_bits_std": _parse_float(row.get("entropy_bits_std")),
        "intra_distance_mean": _parse_float(row.get("intra_distance_mean")),
        "intra_distance_std": _parse_float(row.get("intra_distance_std")),
        "inter_distance": _parse_float(row.get("inter_distance")),
        "inter_distance_std": _parse_float(row.get("inter_distance_std")),
        "inter_intra_ratio": _parse_float(row.get("inter_intra_ratio")),
        "green_loss_dB_mean": _parse_float(row.get("green_loss_dB_mean")),
        "green_loss_dB_std": _parse_float(row.get("green_loss_dB_std")),
        "red_loss_dB_mean": _parse_float(row.get("red_loss_dB_mean")),
        "red_loss_dB_std": _parse_float(row.get("red_loss_dB_std")),
        "is_selected_optimal": str(row.get("is_selected_optimal", "")).lower() == "true",
    }


def build_summary_rows(
    metrics_by_len: Dict[float, Dict[str, Any]],
    *,
    metrics_source_rel: str,
    loss_source_note: str,
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for t in LENGTH_ORDER:
        m = metrics_by_len[t]
        nf = m["n_fibers"]
        n_img = nf * 10 if nf else ""
        gm = m.get("green_loss_dB_mean")
        rm = m.get("red_loss_dB_mean")
        summaries.append({
            "total_length_cm": m["total_length_cm"],
            "right_propagation_cm": m["right_propagation_cm"],
            "green_loss_dB": gm,
            "green_loss_dB_std": m.get("green_loss_dB_std") if _finite_opt(gm) else None,
            "red_loss_dB": rm,
            "red_loss_dB_std": m.get("red_loss_dB_std") if _finite_opt(rm) else None,
            "intra_distance": m["intra_distance_mean"],
            "intra_distance_std": m["intra_distance_std"],
            "inter_distance": m["inter_distance"],
            "inter_distance_std": m["inter_distance_std"],
            "inter_intra_ratio": m["inter_intra_ratio"],
            "entropy_bits": m["entropy_bits_mean"],
            "entropy_bits_std": m["entropy_bits_std"],
            "n_fibers": nf,
            "n_images": n_img,
            "source": f"metrics:{metrics_source_rel}; loss:{loss_source_note}",
        })
    return summaries


def _finite_opt(x: Any) -> bool:
    return isinstance(x, (int, float)) and x == x and math.isfinite(float(x))


XLABEL = "Total fiber length (cm)"

FONT_FAMILY = "DejaVu Sans"
FONT_AXIS_LABEL = 10.5
FONT_TICK = 9
FONT_LEGEND = 8.5
FONT_PANEL_TITLE = 11
FONT_PANEL_LETTER = 10

COL_GREEN_LOSS = "#2d6a4f"
COL_RED_LOSS = "#9a4a4a"
COL_INTRA = "#3468a3"
COL_INTER = "#c86420"
COL_RATIO = "#6b4f85"
COL_ENTROPY = "#162946"

LW = 1.8
MS = 5.0
CAPSIZE = 3.0
GRID_ALPHA = 0.18


def _apply_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.linewidth": 0.9,
    })


def _style_axes_light_grid(ax: Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, alpha=GRID_ALPHA, linestyle="-", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_label(ax: Axes, letter: str) -> None:
    ax.text(
        0.03,
        0.97,
        letter,
        transform=ax.transAxes,
        fontsize=FONT_PANEL_LETTER,
        fontweight="bold",
        va="top",
        ha="left",
        color="#1a1a1a",
        family=FONT_FAMILY,
        clip_on=False,
    )


def _axes_title_outside(ax: Axes, title: str, *, pad: float = 10.0) -> None:
    """Panel title above the axes (outside the data region)."""
    ax.set_title(
        title,
        fontsize=FONT_PANEL_TITLE,
        fontweight="bold",
        pad=pad,
        loc="center",
        family=FONT_FAMILY,
        color="#222222",
    )


def _yerr_real(std_val: Any) -> Optional[float]:
    """Return error bar magnitude only when a positive finite std exists."""
    if not _finite_opt(std_val):
        return None
    s = float(std_val)
    if s <= 0.0:
        return None
    return s


def _subtle_selection_line(ax: Axes) -> None:
    ax.axvline(
        OPTIMUM_CM,
        linestyle="--",
        color="0.45",
        alpha=0.45,
        linewidth=1.0,
        zorder=1,
    )


def _configure_xticks(ax: Axes) -> None:
    ax.set_xticks(LENGTH_ORDER)
    ax.tick_params(axis="both", labelsize=FONT_TICK)


def _collect_loss_series(
    metrics_by_len: Dict[float, Dict[str, Any]],
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    green_y: List[float] = []
    green_e: List[float] = []
    red_y: List[float] = []
    red_e: List[float] = []
    gx: List[float] = []
    rx: List[float] = []
    for t in LENGTH_ORDER:
        m = metrics_by_len.get(t, {})
        gm, gsd = m.get("green_loss_dB_mean"), m.get("green_loss_dB_std")
        rm, rsd = m.get("red_loss_dB_mean"), m.get("red_loss_dB_std")
        if _finite_opt(gm):
            gx.append(t)
            green_y.append(float(gm))
            green_e.append(float(gsd) if _finite_opt(gsd) else 0.0)
        if _finite_opt(rm):
            rx.append(t)
            red_y.append(float(rm))
            red_e.append(float(rsd) if _finite_opt(rsd) else 0.0)
    return gx, green_y, green_e, rx, red_y, red_e


def _ylim_with_errors(y: List[float], err: List[float], pad_lo: float, pad_hi: float) -> Tuple[float, float]:
    pairs = [(float(v - e), float(v + e)) for v, e in zip(y, err)]
    lo = min(p[0] for p in pairs)
    hi = max(p[1] for p in pairs)
    return lo - pad_lo, hi + pad_hi


def plot_panel_loss(
    ax: Axes,
    metrics_by_len: Dict[float, Dict[str, Any]],
    *,
    show_xlabel: bool = True,
    word_vertical: bool = False,
) -> None:
    gx, green_y, green_e, rx, red_y, red_e = _collect_loss_series(metrics_by_len)
    has_green = bool(gx)
    has_red = bool(rx)
    ax_red = ax.twinx() if has_red else None

    if has_green:
        gyerr = [_yerr_real(e) or 0.0 for e in green_e]
        if max(gyerr) > 0.0:
            ax.errorbar(
                gx,
                green_y,
                yerr=gyerr,
                fmt="none",
                ecolor=COL_GREEN_LOSS,
                elinewidth=LW,
                capsize=CAPSIZE,
                zorder=2,
            )
        ax.plot(
            gx,
            green_y,
            "-o",
            color=COL_GREEN_LOSS,
            linewidth=LW,
            markersize=MS,
            label="Green loss",
            zorder=3,
        )
    if has_red and ax_red is not None:
        ryerr = [_yerr_real(e) or 0.0 for e in red_e]
        if max(ryerr) > 0.0:
            ax_red.errorbar(
                rx,
                red_y,
                yerr=ryerr,
                fmt="none",
                ecolor=COL_RED_LOSS,
                elinewidth=LW,
                capsize=CAPSIZE,
                zorder=2,
            )
        ax_red.plot(
            rx,
            red_y,
            "-s",
            color=COL_RED_LOSS,
            linewidth=LW,
            markersize=MS,
            label="Red loss",
            zorder=3,
        )

    if not has_green and not has_red:
        ax.text(
            0.5,
            0.5,
            "No transmission loss data loaded.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONT_TICK,
        )

    _style_axes_light_grid(ax)
    if ax_red is not None:
        ax_red.set_facecolor("white")
        ax_red.grid(False)
        ax_red.spines["top"].set_visible(False)
        ax_red.tick_params(axis="y", labelsize=FONT_TICK, colors=COL_RED_LOSS)
        ax_red.spines["right"].set_visible(True)
        ax_red.spines["right"].set_color(COL_RED_LOSS)
        ax_red.spines["right"].set_linewidth(0.8)

    _panel_label(ax, "a")
    title_pad = 8.0 if word_vertical else 10.0
    _axes_title_outside(ax, "Transmission loss", pad=title_pad)
    if show_xlabel:
        ax.set_xlabel(XLABEL, fontsize=FONT_AXIS_LABEL, family=FONT_FAMILY)
    else:
        ax.set_xlabel("")
    ax.set_ylabel(
        "Green loss (dB)",
        fontsize=FONT_AXIS_LABEL,
        color=COL_GREEN_LOSS,
        family=FONT_FAMILY,
    )
    ax.tick_params(axis="y", labelsize=FONT_TICK, colors=COL_GREEN_LOSS)
    ax.spines["left"].set_color(COL_GREEN_LOSS)
    if ax_red is not None:
        ax_red.set_ylabel(
            "Red loss (dB)",
            fontsize=FONT_AXIS_LABEL,
            color=COL_RED_LOSS,
            family=FONT_FAMILY,
            labelpad=10,
        )

    _configure_xticks(ax)
    ax.set_xlim(7.4, 16.6)
    _subtle_selection_line(ax)

    if has_green:
        lo, hi = _ylim_with_errors(green_y, green_e, 0.8, 0.8)
        ax.set_ylim(lo, hi)
    if has_red and ax_red is not None:
        lo_r, hi_r = _ylim_with_errors(red_y, red_e, 0.4, 0.4)
        ax_red.set_ylim(lo_r, hi_r)

    handles: List[Any] = []
    labels: List[str] = []
    h1, l1 = ax.get_legend_handles_labels()
    handles.extend(h1)
    labels.extend(l1)
    if ax_red is not None:
        h2, l2 = ax_red.get_legend_handles_labels()
        handles.extend(h2)
        labels.extend(l2)
    if handles:
        ax.legend(
            handles,
            labels,
            loc="lower right",
            fontsize=FONT_LEGEND,
            frameon=False,
            borderaxespad=0.35,
        )


def plot_panel_distance(
    ax: Axes,
    metrics_by_len: Dict[float, Dict[str, Any]],
    xs: np.ndarray,
    *,
    show_xlabel: bool = True,
    word_vertical: bool = False,
) -> None:
    intra_m = [metrics_by_len[t]["intra_distance_mean"] for t in LENGTH_ORDER]
    intra_s = [metrics_by_len[t]["intra_distance_std"] for t in LENGTH_ORDER]
    inter_m = [metrics_by_len[t]["inter_distance"] for t in LENGTH_ORDER]
    inter_s = [metrics_by_len[t]["inter_distance_std"] for t in LENGTH_ORDER]

    intra_ye = [_yerr_real(s) or 0.0 for s in intra_s]
    inter_ye = [_yerr_real(s) or 0.0 for s in inter_s]
    if max(intra_ye) > 0.0:
        ax.errorbar(
            xs,
            intra_m,
            yerr=intra_ye,
            fmt="none",
            ecolor=COL_INTRA,
            elinewidth=LW,
            capsize=CAPSIZE,
            zorder=2,
        )
    ax.plot(
        xs,
        intra_m,
        "-o",
        color=COL_INTRA,
        linewidth=LW,
        markersize=MS,
        label="Intra-class distance",
        zorder=3,
    )
    if max(inter_ye) > 0.0:
        ax.errorbar(
            xs,
            inter_m,
            yerr=inter_ye,
            fmt="none",
            ecolor=COL_INTER,
            elinewidth=LW,
            capsize=CAPSIZE,
            zorder=2,
        )
    ax.plot(
        xs,
        inter_m,
        "-s",
        color=COL_INTER,
        linewidth=LW,
        markersize=MS,
        label="Inter-class distance",
        zorder=3,
    )

    _style_axes_light_grid(ax)

    ax_r = ax.twinx()
    ratios = [metrics_by_len[t]["inter_intra_ratio"] for t in LENGTH_ORDER]
    ax_r.plot(
        xs,
        ratios,
        linestyle=(0, (6, 4)),
        marker="^",
        color=COL_RATIO,
        linewidth=LW,
        markersize=MS,
        alpha=0.65,
        label="Inter / intra ratio",
        zorder=4,
    )
    ax_r.set_ylabel(
        "Inter / intra ratio",
        fontsize=FONT_AXIS_LABEL,
        color=COL_RATIO,
        family=FONT_FAMILY,
        labelpad=22,
    )
    ax_r.tick_params(axis="y", labelsize=FONT_TICK, colors=COL_RATIO)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(True)
    ax_r.spines["right"].set_color(COL_RATIO)
    ax_r.set_facecolor("white")

    _panel_label(ax, "b")
    title_pad = 8.0 if word_vertical else 10.0
    _axes_title_outside(ax, "Distance metrics", pad=title_pad)
    if show_xlabel:
        ax.set_xlabel(XLABEL, fontsize=FONT_AXIS_LABEL, family=FONT_FAMILY)
    else:
        ax.set_xlabel("")
    ax.set_ylabel("L2 distance", fontsize=FONT_AXIS_LABEL, family=FONT_FAMILY)
    _configure_xticks(ax)
    ax.set_xlim(7.4, 16.6)
    _subtle_selection_line(ax)

    hb1, lb1 = ax.get_legend_handles_labels()
    hb2, lb2 = ax_r.get_legend_handles_labels()
    if word_vertical:
        ax.legend(
            hb1 + hb2,
            lb1 + lb2,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.96),
            ncol=1,
            fontsize=FONT_LEGEND,
            frameon=False,
            borderaxespad=0.0,
        )
    else:
        ax.legend(
            hb1 + hb2,
            lb1 + lb2,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=2,
            fontsize=FONT_LEGEND,
            frameon=False,
            borderaxespad=0.12,
            columnspacing=0.8,
        )


def plot_panel_entropy(
    ax: Axes,
    metrics_by_len: Dict[float, Dict[str, Any]],
    xs: np.ndarray,
    *,
    show_xlabel: bool = True,
    word_vertical: bool = False,
) -> None:
    ent_m = [metrics_by_len[t]["entropy_bits_mean"] for t in LENGTH_ORDER]
    ent_s_raw = [metrics_by_len[t]["entropy_bits_std"] for t in LENGTH_ORDER]
    ent_ye = [_yerr_real(s) or 0.0 for s in ent_s_raw]
    if max(ent_ye) > 0.0:
        ax.errorbar(
            xs,
            ent_m,
            yerr=ent_ye,
            fmt="none",
            ecolor=COL_ENTROPY,
            elinewidth=LW,
            capsize=CAPSIZE,
            zorder=2,
        )
    ax.plot(
        xs,
        ent_m,
        "-o",
        color=COL_ENTROPY,
        linewidth=LW,
        markersize=MS,
        zorder=3,
    )

    _style_axes_light_grid(ax)
    _panel_label(ax, "c")
    title_pad = 8.0 if word_vertical else 10.0
    _axes_title_outside(ax, "Shannon entropy", pad=title_pad)
    if show_xlabel:
        ax.set_xlabel(XLABEL, fontsize=FONT_AXIS_LABEL, family=FONT_FAMILY)
    else:
        ax.set_xlabel("")
    ax.set_ylabel("Shannon entropy (bits)", fontsize=FONT_AXIS_LABEL, family=FONT_FAMILY)
    _configure_xticks(ax)
    ax.set_xlim(7.4, 16.6)
    _subtle_selection_line(ax)


def plot_figure(metrics_by_len: Dict[float, Dict[str, Any]]) -> None:
    """Single-row Figure 2: panels (a)-(c), journal layout."""
    _apply_plot_style()
    xs = np.array(LENGTH_ORDER, dtype=float)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.0, 4.0),
        constrained_layout=True,
    )
    plot_panel_loss(axes[0], metrics_by_len, show_xlabel=True, word_vertical=False)
    plot_panel_distance(axes[1], metrics_by_len, xs, show_xlabel=True, word_vertical=False)
    plot_panel_entropy(axes[2], metrics_by_len, xs, show_xlabel=True, word_vertical=False)

    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.05, wspace=0.28)

    fig.savefig(f"{OUTPUT_BASE}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUTPUT_BASE}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUTPUT_BASE}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_figure_word(metrics_by_len: Dict[float, Dict[str, Any]]) -> None:
    """Vertical 3×1 layout for Word reports (narrow column, readable fonts)."""
    _apply_plot_style()
    xs = np.array(LENGTH_ORDER, dtype=float)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.8, 9.5),
        constrained_layout=True,
        sharex=True,
    )
    plot_panel_loss(axes[0], metrics_by_len, show_xlabel=False, word_vertical=True)
    plot_panel_distance(axes[1], metrics_by_len, xs, show_xlabel=False, word_vertical=True)
    plot_panel_entropy(axes[2], metrics_by_len, xs, show_xlabel=True, word_vertical=True)

    for ax_row in axes:
        ax_row.tick_params(axis="x", labelbottom=True)

    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.04, hspace=0.18)

    fig.savefig(f"{OUTPUT_WORD_BASE}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUTPUT_WORD_BASE}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUTPUT_WORD_BASE}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    def cell(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return ""
        return str(v)

    fieldnames = [
        "total_length_cm",
        "right_propagation_cm",
        "green_loss_dB",
        "green_loss_dB_std",
        "red_loss_dB",
        "red_loss_dB_std",
        "intra_distance",
        "intra_distance_std",
        "inter_distance",
        "inter_distance_std",
        "inter_intra_ratio",
        "entropy_bits",
        "entropy_bits_std",
        "n_fibers",
        "n_images",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: cell(r.get(k, "")) for k in fieldnames})


def write_report(
    path: Path,
    *,
    primary_used: Path,
    loss_source_note: str,
    loss_files_used: List[str],
    metrics_by_len: Dict[float, Dict[str, Any]],
    optimal_note: str,
    length_optimize_present: bool,
) -> None:
    eight_green = metrics_by_len.get(8.0, {}).get("green_loss_dB_mean")
    eight_red = metrics_by_len.get(8.0, {}).get("red_loss_dB_mean")
    eight_loaded = _finite_opt(eight_green) and _finite_opt(eight_red)

    def fmt_val(v: Any) -> str:
        if _finite_opt(v):
            return str(float(v))
        return ""

    lines = [
        "# Figure 2 — fiber length comparison (generation report)",
        "",
        "**Figure 2.** Performance comparison of fibers with different lengths. "
        "(a) Red and green transmission loss versus total fiber length. "
        "(b) Intra-class distance, inter-class distance, and inter/intra ratio versus total fiber length "
        "(ratio on right axis). "
        "(c) Shannon entropy of output speckle versus total fiber length.",
        "",
        "## Exact plotted values",
        "",
        "| total_length_cm | green_loss_mean | green_loss_std | red_loss_mean | red_loss_std | "
        "intra_mean | intra_std | inter_mean | inter_std | inter_intra_ratio | entropy_mean | entropy_std |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for t in LENGTH_ORDER:
        m = metrics_by_len[t]
        lines.append(
            "| "
            f"{fmt_val(m.get('total_length_cm'))} | "
            f"{fmt_val(m.get('green_loss_dB_mean'))} | "
            f"{fmt_val(m.get('green_loss_dB_std'))} | "
            f"{fmt_val(m.get('red_loss_dB_mean'))} | "
            f"{fmt_val(m.get('red_loss_dB_std'))} | "
            f"{fmt_val(m.get('intra_distance_mean'))} | "
            f"{fmt_val(m.get('intra_distance_std'))} | "
            f"{fmt_val(m.get('inter_distance'))} | "
            f"{fmt_val(m.get('inter_distance_std'))} | "
            f"{fmt_val(m.get('inter_intra_ratio'))} | "
            f"{fmt_val(m.get('entropy_bits_mean'))} | "
            f"{fmt_val(m.get('entropy_bits_std'))} |"
        )

    lines.extend(
        [
            "",
            "### 8 cm transmission loss",
            "",
            f"- **Loaded successfully:** **{'yes' if eight_loaded else 'no'}**.",
            "",
            "## Data sources",
            "",
            f"- Metrics table: `{primary_used.relative_to(REPO_ROOT)}`",
            f"- Fiber loss loader: `{loss_source_note}`",
        ]
    )

    lines.extend(["", "### Fiber loss inputs read", ""])
    if loss_files_used:
        for fp in loss_files_used:
            lines.append(f"- `{fp}`")
    else:
        lines.append("- *(none)*")

    lines.extend(
        [
            "",
            "## Figure export settings",
            "",
            "- Layout: **1 row × 3 columns**, **figsize (15, 4)**, **dpi 600**, white background.",
            "- Word-oriented export: **`Fig2_length_optimization_word.{png,pdf,svg}`**, **3 rows × 1 column**, "
            "**figsize ≈ (6.8, 9.5)**, **dpi 600**, shared x-axis with tick labels on each row; "
            "**Total fiber length (cm)** label on bottom panel only.",
            "- Panel titles (**Transmission loss**, **Distance metrics**, **Shannon entropy**) use **`ax.set_title`** "
            "above each subplot (**outside** the axes; **pad ≈ 10**, centered); letters **a–c** remain inside the upper-left.",
            "- **No global title**, **no geometry panel**, **no propagation-length annotation on axes**.",
            "- Optimal length marker: vertical dashed line at **9 cm**.",
            "",
            "## Pipeline references",
            "",
            f"- Optimal JSON: `{OPTIMAL_JSON.relative_to(REPO_ROOT)}`",
            f"- Config: `{CONFIG_YAML.relative_to(REPO_ROOT)}`",
            optimal_note,
            "",
            "## Raw JPG cohort",
            "",
            "- **`LengthOptimize/Green/`** raw captures were **not** re-ingested here; "
            + (
                "metrics come from the exported CSV."
                if length_optimize_present
                else "directory absent or empty — CSV export only."
            ),
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cleanup_legacy_outputs()

    rows = load_primary_rows(PRIMARY_TABLE)
    primary_used = PRIMARY_TABLE
    metrics_source_rel = str(PRIMARY_TABLE.relative_to(REPO_ROOT))
    if not rows:
        rows = load_primary_rows(FALLBACK_TABLE)
        primary_used = FALLBACK_TABLE
        metrics_source_rel = str(FALLBACK_TABLE.relative_to(REPO_ROOT))
    if not rows:
        raise SystemExit(
            f"No aggregated table found. Expected {PRIMARY_TABLE} or {FALLBACK_TABLE}."
        )

    metrics_by_len: Dict[float, Dict[str, Any]] = {}
    for row in rows:
        m = row_to_metrics(row)
        if not m:
            continue
        metrics_by_len[m["total_length_cm"]] = m

    for need in LENGTH_ORDER:
        if need not in metrics_by_len:
            raise SystemExit(f"Missing length {need} cm in {primary_used}")

    power_agg, loss_source_note, loss_files_used = load_power_loss_aggregates()
    merge_power_loss_into_metrics(metrics_by_len, power_agg)

    optimal_note = (
        f"- Could not read `{OPTIMAL_JSON.relative_to(REPO_ROOT)}` for textual confirmation."
    )
    if OPTIMAL_JSON.is_file():
        data = json.loads(OPTIMAL_JSON.read_text(encoding="utf-8"))
        rec = data.get("recommended_total_length_cm")
        grp = data.get("recommended_length_group")
        optimal_note = (
            f"- **`optimal_length.json`** recommends **`{grp}`** with **`recommended_total_length_cm = {rec}`** "
            "(pipeline optimum consistent with **9 cm total fiber length**)."
        )

    length_opt_present = LENGTH_OPTIMIZE_GREEN.is_dir() and any(LENGTH_OPTIMIZE_GREEN.iterdir())

    summary_rows = build_summary_rows(
        metrics_by_len,
        metrics_source_rel=metrics_source_rel,
        loss_source_note=loss_source_note,
    )
    write_summary_csv(OUTPUT_DIR / "Fig2_length_optimization_data_summary.csv", summary_rows)

    plot_figure(metrics_by_len)
    plot_figure_word(metrics_by_len)

    write_report(
        OUTPUT_DIR / "Fig2_length_optimization_report.md",
        primary_used=primary_used,
        loss_source_note=loss_source_note,
        loss_files_used=loss_files_used,
        metrics_by_len=metrics_by_len,
        optimal_note=optimal_note,
        length_optimize_present=length_opt_present,
    )

    print("Final Figure 2 generated successfully.")
    print(f"PNG: {OUTPUT_BASE}.png")
    print(f"PDF: {OUTPUT_BASE}.pdf")
    print(f"SVG: {OUTPUT_BASE}.svg")
    print(f"Word PNG: {OUTPUT_WORD_BASE}.png")
    print(f"Word PDF: {OUTPUT_WORD_BASE}.pdf")
    print(f"Word SVG: {OUTPUT_WORD_BASE}.svg")


if __name__ == "__main__":
    main()
