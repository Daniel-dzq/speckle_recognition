"""
Global matplotlib style + shared colour palettes.

Keeps the visual language consistent with the existing
``scripts/plot_style.py`` but extends it to the analysis framework and
provides a consistent save helper that writes PNG + PDF + SVG in one call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Palette ---------------------------------------------------------------
DEEP_BLUE = "#2C5F8A"
TEAL = "#2A9D8F"
MUTED_ORANGE = "#E07A3A"
MUTED_RED = "#C14953"
SOFT_PURPLE = "#7B6D9B"
SLATE_GRAY = "#6C757D"
GOLD = "#D4A843"

PALETTE: List[str] = [
    DEEP_BLUE, TEAL, MUTED_ORANGE, MUTED_RED, SOFT_PURPLE, SLATE_GRAY, GOLD,
]

FIBER_COLORS = {
    "Fiber1": DEEP_BLUE,
    "Fiber2": TEAL,
    "Fiber3": MUTED_ORANGE,
    "Fiber4": MUTED_RED,
    "Fiber5": SOFT_PURPLE,
}

CHANNEL_COLORS = {
    "green": TEAL,
    "red": MUTED_RED,
    "green_red_fixed": MUTED_ORANGE,
    "green_red_dynamic": MUTED_RED,
    "ratio": SOFT_PURPLE,
}

DOMAIN_COLORS = {
    "Green": TEAL,
    "GreenAndRed": MUTED_ORANGE,
    "RedChange": MUTED_RED,
    "green_only": TEAL,
    "red_green_fixed": MUTED_ORANGE,
    "red_green_dynamic": MUTED_RED,
}

# --- Figure sizing ---------------------------------------------------------
SINGLE_COL_W = 3.5
DOUBLE_COL_W = 7.0
GOLDEN_RATIO = 1.618

FONT_SIZE_SMALL = 7
FONT_SIZE_NORMAL = 8
FONT_SIZE_LARGE = 9
FONT_SIZE_TITLE = 10


def apply_style() -> None:
    """Globally apply the journal-quality matplotlib style."""
    try:
        from cycler import cycler  # type: ignore
    except ImportError:  # pragma: no cover
        cycler = None

    plt.rcdefaults()
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT_SIZE_NORMAL,

        "axes.linewidth": 0.6,
        "axes.labelsize": FONT_SIZE_NORMAL,
        "axes.titlesize": FONT_SIZE_LARGE,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelpad": 4,
        "axes.titlepad": 6,
        "axes.grid": False,

        "xtick.labelsize": FONT_SIZE_SMALL,
        "ytick.labelsize": FONT_SIZE_SMALL,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "legend.fontsize": FONT_SIZE_SMALL,
        "legend.frameon": False,
        "legend.handlelength": 1.2,

        "lines.linewidth": 1.2,
        "lines.markersize": 4,

        "figure.dpi": 150,
        "figure.facecolor": "white",
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.transparent": False,
        "figure.constrained_layout.use": True,
    }
    if cycler is not None:
        rc["axes.prop_cycle"] = cycler("color", PALETTE)
    mpl.rcParams.update(rc)


def add_panel_label(ax, label: str, x: float = -0.12, y: float = 1.08, **kwargs) -> None:
    defaults = dict(
        fontsize=FONT_SIZE_TITLE,
        fontweight="bold",
        transform=ax.transAxes,
        va="top",
        ha="right",
    )
    defaults.update(kwargs)
    ax.text(x, y, label, **defaults)


def save_figure(
    fig,
    path_stem: str | Path,
    formats: Sequence[str] = ("png", "pdf", "svg"),
) -> List[str]:
    """Save a figure in several formats and return the saved paths."""
    saved: List[str] = []
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = f"{stem}.{fmt}"
        fig.savefig(out, format=fmt)
        saved.append(out)
    return saved


__all__ = [
    "apply_style",
    "add_panel_label",
    "save_figure",
    "PALETTE",
    "FIBER_COLORS",
    "CHANNEL_COLORS",
    "DOMAIN_COLORS",
    "SINGLE_COL_W",
    "DOUBLE_COL_W",
    "GOLDEN_RATIO",
]
