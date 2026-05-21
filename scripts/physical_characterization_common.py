"""Shared paths and summary sync for root-level physical-characterization wrappers."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]

EXP_LENGTH = ROOT / "experiments" / "length_optimization"
EXP_LOSS = ROOT / "experiments" / "fiber_loss"
EXP_LONG_TERM = ROOT / "experiments" / "long_term_stability"
EXP_DISTURBANCE = ROOT / "experiments" / "disturbance_sensitivity"

PC_ROOT = ROOT / "outputs" / "physical_characterization"
PC_LENGTH = PC_ROOT / "length_optimization"
PC_LOSS = PC_ROOT / "fiber_loss"
PC_LONG_TERM = PC_ROOT / "long_term_stability"
PC_DISTURBANCE = PC_ROOT / "disturbance_sensitivity"

FIGURES_DIR = ROOT / "figures"
FIG2_PAPER = FIGURES_DIR / "paper" / "Fig2_length_optimization"
LEN_OUT = EXP_LENGTH / "outputs" / "length_optimization_green"


def sync_length_summaries() -> List[Path]:
    """Copy JSON/CSV/MD summaries only (no figures) into physical_characterization."""
    PC_LENGTH.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    pairs = [
        (LEN_OUT / "summary.json", PC_LENGTH / "summary.json"),
        (LEN_OUT / "optimal_length.json", PC_LENGTH / "optimal_length.json"),
        (FIG2_PAPER / "Fig2_length_optimization_data_summary.csv", PC_LENGTH / "Fig2_length_optimization_data_summary.csv"),
        (FIG2_PAPER / "Fig2_length_optimization_report.md", PC_LENGTH / "Fig2_length_optimization_report.md"),
    ]
    for src, dst in pairs:
        if src.is_file():
            shutil.copy2(src, dst)
            written.append(dst)
    return written


def sync_fiber_loss_summary_csv() -> Path:
    """Build fiber_loss_summary.csv from Fig2 data summary (loss columns only)."""
    src = FIG2_PAPER / "Fig2_length_optimization_data_summary.csv"
    if not src.is_file():
        src = PC_LENGTH / "Fig2_length_optimization_data_summary.csv"
    if not src.is_file():
        raise FileNotFoundError(
            "Missing Fig2_length_optimization_data_summary.csv. "
            "Run scripts/run_length_optimization.py or scripts/run_fiber_loss_analysis.py after Fig2 exists."
        )
    PC_LOSS.mkdir(parents=True, exist_ok=True)
    cols = [
        "total_length_cm",
        "green_loss_dB",
        "green_loss_dB_std",
        "red_loss_dB",
        "red_loss_dB_std",
    ]
    rows_out = []
    with src.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows_out.append({c: row.get(c, "") for c in cols})
    dst = PC_LOSS / "fiber_loss_summary.csv"
    with dst.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_out)
    return dst


def sync_long_term_summary() -> Path:
    src = EXP_LONG_TERM / "outputs" / "metrics_summary.json"
    if not src.is_file():
        raise FileNotFoundError(f"Missing {src}. Run scripts/run_long_term_stability.py first.")
    PC_LONG_TERM.mkdir(parents=True, exist_ok=True)
    dst = PC_LONG_TERM / "metrics_summary.json"
    shutil.copy2(src, dst)
    return dst


def sync_disturbance_summary() -> Path:
    src = EXP_DISTURBANCE / "outputs" / "metrics_summary.json"
    if not src.is_file():
        raise FileNotFoundError(f"Missing {src}. Run scripts/run_disturbance_sensitivity.py first.")
    PC_DISTURBANCE.mkdir(parents=True, exist_ok=True)
    dst = PC_DISTURBANCE / "metrics_summary.json"
    shutil.copy2(src, dst)
    return dst


def promote_fig2_regen_to_standard() -> List[Path]:
    """Fig2 writes directly to figures/paper/Fig2_length_optimization/ (no regen step)."""
    promoted: List[Path] = []
    for ext in (".png", ".pdf", ".svg"):
        p = FIG2_PAPER / f"Fig2_length_optimization{ext}"
        if p.is_file():
            promoted.append(p)
    return promoted


def load_fig2_module():
    import importlib.util

    path = ROOT / "scripts" / "paper_figures" / "_fig2_length_optimization_impl.py"
    spec = importlib.util.spec_from_file_location("generate_fig2_length_optimization", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_length_pipeline() -> None:
    import sys

    script = EXP_LENGTH / "scripts" / "run_length_optimization.py"
    argv = [
        "length_optimization",
        "--config",
        str(EXP_LENGTH / "scripts" / "length_optimization_green.yaml"),
    ]
    script_dir = str(EXP_LENGTH / "scripts")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from run_experiment import main as run_exp_main  # noqa: WPS433

    raise SystemExit(run_exp_main(argv))
