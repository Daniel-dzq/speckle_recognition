"""
Reusable chart primitives styled consistently with the paper figures.

Each function returns a ``(fig, ax)`` tuple (or a figure for multi-axis
panels) so callers can further customise before saving.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .style import PALETTE, SINGLE_COL_W


# ---------------------------------------------------------------------------
# Line + bar
# ---------------------------------------------------------------------------


def line_with_error(
    x: Sequence[float],
    series: Mapping[str, Tuple[Sequence[float], Optional[Sequence[float]]]],
    *,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    figsize: Tuple[float, float] = (SINGLE_COL_W, SINGLE_COL_W / 1.4),
    colors: Optional[Mapping[str, str]] = None,
    markers: Optional[Mapping[str, str]] = None,
    ax=None,
):
    """
    Plot one or more line series with optional shaded error bands.

    ``series`` maps a label to ``(mean_values, std_values_or_None)``.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    x_arr = np.asarray(list(x), dtype=np.float64)
    for i, (name, (mean, std)) in enumerate(series.items()):
        color = (colors or {}).get(name, PALETTE[i % len(PALETTE)])
        marker = (markers or {}).get(name, "o")
        mean_arr = np.asarray(list(mean), dtype=np.float64)
        ax.plot(x_arr, mean_arr, color=color, marker=marker, label=name, linewidth=1.3)
        if std is not None:
            std_arr = np.asarray(list(std), dtype=np.float64)
            ax.fill_between(x_arr, mean_arr - std_arr, mean_arr + std_arr,
                            color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if len(series) > 1:
        ax.legend(loc="best")
    if fig is None:
        fig = ax.figure
    return fig, ax


def grouped_bars(
    categories: Sequence[str],
    series: Mapping[str, Sequence[float]],
    errors: Optional[Mapping[str, Sequence[float]]] = None,
    *,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    colors: Optional[Mapping[str, str]] = None,
    figsize: Tuple[float, float] = (SINGLE_COL_W, SINGLE_COL_W / 1.4),
    bar_width: float = 0.8,
    value_labels: bool = False,
    ax=None,
):
    """Grouped bar plot with optional error bars and on-bar value labels."""
    n_series = max(1, len(series))
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(categories), dtype=np.float64)
    w = bar_width / n_series
    for i, (name, values) in enumerate(series.items()):
        offset = (i - (n_series - 1) / 2.0) * w
        color = (colors or {}).get(name, PALETTE[i % len(PALETTE)])
        err = None
        if errors and name in errors:
            err = list(errors[name])
        bars = ax.bar(
            x + offset, list(values), width=w * 0.95,
            color=color, label=name, edgecolor="white", linewidth=0.4,
            yerr=err, capsize=2,
        )
        if value_labels:
            for b, v in zip(bars, values):
                if v is None or np.isnan(v):
                    continue
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if n_series > 1:
        ax.legend(loc="best")
    if fig is None:
        fig = ax.figure
    return fig, ax


def dual_axis_line(
    x: Sequence[float],
    left_series: Mapping[str, Sequence[float]],
    right_series: Mapping[str, Sequence[float]],
    *,
    left_label: str = "",
    right_label: str = "",
    xlabel: str = "",
    title: str = "",
    left_colors: Optional[Mapping[str, str]] = None,
    right_colors: Optional[Mapping[str, str]] = None,
    figsize: Tuple[float, float] = (SINGLE_COL_W, SINGLE_COL_W / 1.4),
):
    fig, ax_l = plt.subplots(figsize=figsize)
    ax_r = ax_l.twinx()
    ax_r.spines["top"].set_visible(False)
    x_arr = np.asarray(list(x), dtype=np.float64)
    handles: List = []
    labels: List[str] = []
    for i, (name, vals) in enumerate(left_series.items()):
        color = (left_colors or {}).get(name, PALETTE[i % len(PALETTE)])
        h, = ax_l.plot(x_arr, list(vals), marker="o", color=color, linewidth=1.3)
        handles.append(h)
        labels.append(name)
    for i, (name, vals) in enumerate(right_series.items()):
        color = (right_colors or {}).get(name, PALETTE[(i + len(left_series)) % len(PALETTE)])
        h, = ax_r.plot(x_arr, list(vals), marker="s", color=color, linewidth=1.3, linestyle="--")
        handles.append(h)
        labels.append(name)
    ax_l.set_xlabel(xlabel)
    ax_l.set_ylabel(left_label)
    ax_r.set_ylabel(right_label)
    if title:
        ax_l.set_title(title)
    ax_l.legend(handles, labels, loc="best")
    return fig, (ax_l, ax_r)


# ---------------------------------------------------------------------------
# Heatmap / ROC / distribution
# ---------------------------------------------------------------------------


def heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    *,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    cmap: str = "YlOrRd",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fmt: str = ".2f",
    annotate: bool = True,
    cbar_label: str = "",
    figsize: Optional[Tuple[float, float]] = None,
):
    mat = np.asarray(matrix, dtype=np.float64)
    n_rows, n_cols = mat.shape
    if figsize is None:
        w = max(SINGLE_COL_W, 0.35 * n_cols + 1.2)
        h = max(SINGLE_COL_W / 1.4, 0.35 * n_rows + 1.2)
        figsize = (w, h)
    fig, ax = plt.subplots(figsize=figsize)
    vmin_ = float(np.nanmin(mat)) if vmin is None else vmin
    vmax_ = float(np.nanmax(mat)) if vmax is None else vmax
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin_, vmax=vmax_)
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if annotate:
        mid = (vmin_ + vmax_) / 2.0
        integer_fmt = fmt.endswith("d")
        for i in range(n_rows):
            for j in range(n_cols):
                val = mat[i, j]
                color = "white" if val >= mid else "black"
                formatted = format(int(val), fmt) if integer_fmt else format(val, fmt)
                ax.text(j, i, formatted, ha="center", va="center",
                        fontsize=6, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.5)
    return fig, ax


def roc_panel(
    curves: Mapping[str, Tuple[np.ndarray, np.ndarray, float]],
    *,
    title: str = "",
    figsize: Tuple[float, float] = (SINGLE_COL_W, SINGLE_COL_W),
    colors: Optional[Mapping[str, str]] = None,
):
    fig, ax = plt.subplots(figsize=figsize)
    for i, (name, (fpr, tpr, auc)) in enumerate(curves.items()):
        color = (colors or {}).get(name, PALETTE[i % len(PALETTE)])
        ax.plot(fpr, tpr, color=color, linewidth=1.3,
                label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#888", linestyle=":", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right")
    return fig, ax


def boxplot(
    data: Mapping[str, Sequence[float]],
    *,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    colors: Optional[Mapping[str, str]] = None,
    figsize: Tuple[float, float] = (SINGLE_COL_W, SINGLE_COL_W / 1.4),
):
    fig, ax = plt.subplots(figsize=figsize)
    labels = list(data.keys())
    values = [list(data[k]) for k in labels]
    bp = ax.boxplot(values, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black"))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor((colors or {}).get(labels[i], PALETTE[i % len(PALETTE)]))
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.6)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax


def violinplot(
    data: Mapping[str, Sequence[float]],
    *,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    colors: Optional[Mapping[str, str]] = None,
    figsize: Tuple[float, float] = (SINGLE_COL_W, SINGLE_COL_W / 1.4),
):
    fig, ax = plt.subplots(figsize=figsize)
    labels = list(data.keys())
    values = [list(data[k]) for k in labels]
    parts = ax.violinplot(values, showmeans=False, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor((colors or {}).get(labels[i], PALETTE[i % len(PALETTE)]))
        pc.set_edgecolor("black")
        pc.set_alpha(0.6)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax


def image_panel(
    images: Sequence[np.ndarray],
    *,
    row_labels: Optional[Sequence[str]] = None,
    col_labels: Optional[Sequence[str]] = None,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    title: str = "",
    cmap: str = "magma",
    cell_size: float = 1.1,
    value_range: Optional[Tuple[float, float]] = None,
    scale_consistent: bool = True,
):
    """Grid of images with scale-consistent colour range by default."""
    n = len(images)
    if rows is None and cols is None:
        cols = min(n, 4)
        rows = int(np.ceil(n / cols))
    elif rows is None:
        rows = int(np.ceil(n / cols))
    elif cols is None:
        cols = int(np.ceil(n / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell_size, rows * cell_size))
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[a] for a in axes])

    if scale_consistent and value_range is None and n > 0:
        finite = [im for im in images if im is not None and im.size]
        vmin = float(min(im.min() for im in finite)) if finite else 0.0
        vmax = float(max(im.max() for im in finite)) if finite else 1.0
    elif value_range is not None:
        vmin, vmax = value_range
    else:
        vmin = vmax = None

    for i in range(rows * cols):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        if i >= n or images[i] is None:
            ax.axis("off")
            continue
        ax.imshow(images[i], cmap=cmap, vmin=vmin, vmax=vmax)
        if row_labels and c == 0:
            label = row_labels[r] if r < len(row_labels) else ""
            ax.set_ylabel(label, fontsize=7)
        if col_labels and r == 0:
            label = col_labels[c] if c < len(col_labels) else ""
            ax.set_title(label, fontsize=7)
    if title:
        fig.suptitle(title)
    return fig, axes


__all__ = [
    "line_with_error",
    "grouped_bars",
    "dual_axis_line",
    "heatmap",
    "roc_panel",
    "boxplot",
    "violinplot",
    "image_panel",
]
