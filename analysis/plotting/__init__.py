"""Publication-grade plotting helpers."""

from .style import (
    CHANNEL_COLORS,
    DOMAIN_COLORS,
    FIBER_COLORS,
    PALETTE,
    add_panel_label,
    apply_style,
    save_figure,
)
from .charts import (
    boxplot,
    dual_axis_line,
    grouped_bars,
    heatmap,
    image_panel,
    line_with_error,
    roc_panel,
    violinplot,
)

__all__ = [
    "apply_style",
    "add_panel_label",
    "save_figure",
    "CHANNEL_COLORS",
    "DOMAIN_COLORS",
    "FIBER_COLORS",
    "PALETTE",
    "line_with_error",
    "grouped_bars",
    "dual_axis_line",
    "heatmap",
    "roc_panel",
    "boxplot",
    "violinplot",
    "image_panel",
]
