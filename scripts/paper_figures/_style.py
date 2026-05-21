"""Journal-style matplotlib settings for paper figures."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.plotting.style import (  # noqa: E402
    DEEP_BLUE,
    MUTED_ORANGE,
    MUTED_RED,
    SLATE_GRAY,
    SOFT_PURPLE,
    TEAL,
    apply_style as _apply_analysis_style,
)

DEEP_GREEN = TEAL
MUTED_RED_CHANNEL = MUTED_RED
INTRA_COLOR = DEEP_BLUE
INTER_COLOR = MUTED_ORANGE
RATIO_COLOR = SOFT_PURPLE
DIAGONAL_COLOR = DEEP_BLUE
OFF_DIAG_COLOR = SLATE_GRAY
RANDOM_BASELINE_COLOR = SLATE_GRAY

FONT_PANEL = 13
FONT_AXIS = 10
FONT_TICK = 8
FONT_METRIC = 20
FONT_METRIC_TITLE = 9


def apply_paper_style() -> None:
    _apply_analysis_style()


def panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.08,
        1.05,
        letter,
        transform=ax.transAxes,
        fontsize=FONT_PANEL,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


__all__ = [
    "apply_paper_style",
    "panel_label",
    "FONT_PANEL",
    "FONT_AXIS",
    "FONT_TICK",
    "FONT_METRIC",
    "FONT_METRIC_TITLE",
    "DEEP_GREEN",
    "MUTED_RED_CHANNEL",
    "INTRA_COLOR",
    "INTER_COLOR",
    "RATIO_COLOR",
    "DIAGONAL_COLOR",
    "OFF_DIAG_COLOR",
    "RANDOM_BASELINE_COLOR",
]
