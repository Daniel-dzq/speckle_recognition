"""
Unified style for journal paper figures (English labels, colorblind-safe, high-res export).

All figures must call `apply_style()` before plotting.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

# --- Colorblind-oriented palette (Paul Tol–inspired, muted) ---
COL_RED = "#CC6677"
COL_GREEN = "#117733"
COL_BLUE = "#332288"
COL_ORANGE = "#DDCC77"
COL_CYAN = "#88CCEE"
COL_MAROON = "#AA4499"
COL_GRAY = "#888888"

PALETTE = [COL_BLUE, COL_ORANGE, COL_CYAN, COL_RED, COL_MAROON, COL_GREEN, COL_GRAY]

# Dimensions (inches)
SINGLE_COL_W = 3.46  # ~88 mm
DOUBLE_COL_W = 7.08  # ~180 mm
GOLDEN_RATIO = 1.618

FONT_SIZE_TICK = 7
FONT_SIZE_LABEL = 8
FONT_SIZE_PANEL = 9
FONT_SIZE_LEGEND = 7

DPI_PNG = 600


def apply_style() -> None:
    plt.rcdefaults()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": FONT_SIZE_LABEL,
        "axes.linewidth": 0.6,
        "axes.labelsize": FONT_SIZE_LABEL,
        "axes.titlesize": FONT_SIZE_LABEL,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": cycler("color", PALETTE),
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def panel_label(ax: plt.Axes, letter: str, x: float = -0.12, y: float = 1.02) -> None:
    ax.text(
        x, y, f"({letter})",
        transform=ax.transAxes,
        fontsize=FONT_SIZE_PANEL,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def figure_root(repo_root: Path) -> Path:
    return repo_root / "figures" / "paper"


def save_figure_bundle(
    fig: plt.Figure,
    output_dir: Path,
    base_name: str,
    script_name: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Save PNG (600 dpi), PDF, SVG plus metadata JSON. CSV must be saved separately by caller.
    Returns dict of written paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    for ext, kwargs in (
        ("png", {"dpi": DPI_PNG}),
        ("pdf", {}),
        ("svg", {}),
    ):
        p = output_dir / f"{base_name}.{ext}"
        fig.savefig(p, **kwargs)
        paths[ext] = str(p.resolve())
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": script_name,
        "output_dir": str(output_dir.resolve()),
        "base_name": base_name,
        "dpi_png": DPI_PNG,
    }
    if extra_meta:
        meta.update(extra_meta)
    meta_path = output_dir / f"{base_name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    paths["meta"] = str(meta_path.resolve())
    return paths


def write_table_csv(df, path: Path) -> None:
    """Write pandas DataFrame to CSV (requires pandas)."""
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be pandas.DataFrame")
    df.to_csv(path, index=False)
