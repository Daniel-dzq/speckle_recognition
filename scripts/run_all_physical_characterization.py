#!/usr/bin/env python3
"""Run all physical-characterization analyses in recommended order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _run(label: str, script: str, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(SCRIPTS / script)] + (extra_args or [])
    print(f"\n=== {label} ===")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run fiber loss, length optimization, long-term stability, and disturbance analyses."
    )
    parser.add_argument(
        "--run-length-pipeline",
        action="store_true",
        help="Pass --run-pipeline to run_length_optimization.py (slow).",
    )
    parser.add_argument(
        "--overwrite-fig2",
        action="store_true",
        help="Pass --overwrite to run_length_optimization.py.",
    )
    parser.add_argument(
        "--skip-stability",
        action="store_true",
        help="Skip long-term and disturbance analyses.",
    )
    args = parser.parse_args()

    len_args: list[str] = []
    if args.run_length_pipeline:
        len_args.append("--run-pipeline")
    if args.overwrite_fig2:
        len_args.append("--overwrite")

    try:
        _run("1. Fiber loss summary", "run_fiber_loss_analysis.py")
        _run("2. Length optimization + Fig2", "run_length_optimization.py", len_args or None)
        _run("1b. Fiber loss (refresh from Fig2 CSV)", "run_fiber_loss_analysis.py", ["--from-fig2-csv"])

        if not args.skip_stability:
            _run("3. Long-term stability", "run_long_term_stability.py")
            _run("4. Disturbance sensitivity", "run_disturbance_sensitivity.py")
        else:
            print("\nSkipped long-term and disturbance (--skip-stability).")

    except subprocess.CalledProcessError as exc:
        print(f"\nFatal error (exit {exc.returncode}).", file=sys.stderr)
        return exc.returncode or 1

    print("\n=== All requested steps completed ===")
    print("Figures: figures/Fig2_length_optimization_regen.* (and experiments/*/outputs/figures/)")
    print("Summaries: outputs/physical_characterization/ (CSV/JSON/MD only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
