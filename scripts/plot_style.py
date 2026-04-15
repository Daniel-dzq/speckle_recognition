"""
Shared publication-quality plotting style for all paper figures.

Usage:
    from plot_style import apply_style, PALETTE, FIBER_COLORS, DOMAIN_COLORS
    apply_style()
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# ── Colour palette (colorblind-friendly, muted, journal-grade) ──────────
DEEP_BLUE   = "#2C5F8A"
TEAL        = "#2A9D8F"
MUTED_ORANGE = "#E07A3A"
MUTED_RED   = "#C14953"
SOFT_PURPLE = "#7B6D9B"
SLATE_GRAY  = "#6C757D"
GOLD        = "#D4A843"

PALETTE = [DEEP_BLUE, TEAL, MUTED_ORANGE, MUTED_RED, SOFT_PURPLE, SLATE_GRAY, GOLD]

FIBER_COLORS = {
    "Fiber1": DEEP_BLUE,
    "Fiber2": TEAL,
    "Fiber3": MUTED_ORANGE,
    "Fiber4": MUTED_RED,
    "Fiber5": SOFT_PURPLE,
}

DOMAIN_COLORS = {
    "green_only":         TEAL,
    "red_green_fixed":    MUTED_ORANGE,
    "red_green_dynamic":  MUTED_RED,
}

DOMAIN_LABELS = {
    "green_only":         "Green only",
    "red_green_fixed":    "Green + Red (fixed)",
    "red_green_dynamic":  "Green + Red (dynamic)",
}

# ── Heatmap colour maps ─────────────────────────────────────────────────
HEATMAP_CMAP = "YlOrRd"

# ── Dimensions (inches) for two-column journal layout ───────────────────
SINGLE_COL_W = 3.5    # single-column figure width
DOUBLE_COL_W = 7.0    # two-column figure width
GOLDEN_RATIO = 1.618

# ── Font sizes ──────────────────────────────────────────────────────────
FONT_SIZE_SMALL  = 7
FONT_SIZE_NORMAL = 8
FONT_SIZE_LARGE  = 9
FONT_SIZE_TITLE  = 10


def apply_style():
    """Apply a clean, journal-quality matplotlib style globally."""
    plt.rcdefaults()

    mpl.rcParams.update({
        # Font
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":         FONT_SIZE_NORMAL,

        # Axes
        "axes.linewidth":    0.6,
        "axes.labelsize":    FONT_SIZE_NORMAL,
        "axes.titlesize":    FONT_SIZE_LARGE,
        "axes.titleweight":  "bold",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.prop_cycle":   cycler("color", PALETTE),
        "axes.labelpad":     4,
        "axes.titlepad":     6,

        # Ticks
        "xtick.labelsize":   FONT_SIZE_SMALL,
        "ytick.labelsize":   FONT_SIZE_SMALL,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size":  3,
        "ytick.major.size":  3,
        "xtick.direction":   "out",
        "ytick.direction":   "out",

        # Legend
        "legend.fontsize":   FONT_SIZE_SMALL,
        "legend.frameon":    False,
        "legend.handlelength": 1.2,

        # Lines
        "lines.linewidth":   1.2,
        "lines.markersize":  4,

        # Figure
        "figure.dpi":        150,
        "figure.facecolor":  "white",
        "savefig.dpi":       600,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.05,
        "savefig.transparent": False,

        # Grid (off by default, enable explicitly)
        "axes.grid":         False,

        # Layout
        "figure.constrained_layout.use": True,
    })


def add_panel_label(ax, label, x=-0.12, y=1.08, **kwargs):
    """Add a panel label like (a), (b), etc. to an axes."""
    defaults = dict(
        fontsize=FONT_SIZE_TITLE,
        fontweight="bold",
        transform=ax.transAxes,
        va="top",
        ha="right",
    )
    defaults.update(kwargs)
    ax.text(x, y, label, **defaults)


def save_figure(fig, path_stem, formats=("png", "pdf", "svg")):
    """Save a figure in multiple formats. path_stem has no extension."""
    saved = []
    for fmt in formats:
        out = f"{path_stem}.{fmt}"
        fig.savefig(out, format=fmt)
        saved.append(out)
    return saved
