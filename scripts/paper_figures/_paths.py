"""Project-root-relative paths for paper figure generation."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FIGURES_PAPER = ROOT / "figures" / "paper"
FIGURE_GEN_OUT = ROOT / "outputs" / "figure_generation"

TRAINING_OUT = ROOT / "outputs" / "final_15fiber_training"
SUMMARY_CSV = TRAINING_OUT / "summary_15fibers.csv"
AUTH_MATRIX_CSV = TRAINING_OUT / "auth_matrix_15x15.csv"
AUTH_REPORT_MD = TRAINING_OUT / "auth_matrix_report.md"

CHALLENGE_DIR = ROOT / "challenge_inputs"
CHALLENGE_MANIFEST = CHALLENGE_DIR / "manifest.json"
SPECKLE_VIDEO_DIR = ROOT / "data" / "recognition_dataset" / "GreenAndRed" / "Fiber1"

FIG2_OUT_DIR = FIGURES_PAPER / "Fig2_length_optimization"
FIG2_SUMMARY_CSV = FIG2_OUT_DIR / "Fig2_length_optimization_data_summary.csv"
FIG2_REPORT_MD = FIG2_OUT_DIR / "Fig2_length_optimization_report.md"

PHYS_CHAR = ROOT / "outputs" / "physical_characterization"
LT_METRICS = PHYS_CHAR / "long_term_stability" / "metrics_summary.json"
LT_METRICS_EXP = ROOT / "experiments" / "long_term_stability" / "outputs" / "metrics_summary.json"
DS_METRICS = PHYS_CHAR / "disturbance_sensitivity" / "metrics_summary.json"
DS_METRICS_EXP = ROOT / "experiments" / "disturbance_sensitivity" / "outputs" / "metrics_summary.json"
