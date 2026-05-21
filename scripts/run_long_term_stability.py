#!/usr/bin/env python3
"""Run long-term stability analysis (experiment script + summary sync)."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = _SCRIPT_DIR.parent
for _p in (str(ROOT), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from physical_characterization_common import (  # noqa: E402
    EXP_LONG_TERM,
    ROOT,
    sync_long_term_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze long_term_stability JPEG data and sync metrics summary."
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Only sync existing experiments/.../metrics_summary.json to outputs/physical_characterization/.",
    )
    args = parser.parse_args()

    script = EXP_LONG_TERM / "scripts" / "analyze_long_term_stability.py"
    if not args.skip_analysis:
        print(f"Running {script.relative_to(ROOT)}...")
        runpy.run_path(str(script), run_name="__main__")

    dst = sync_long_term_summary()
    print(f"Synced {dst.relative_to(ROOT)}")
    print(f"Plots: {EXP_LONG_TERM / 'outputs/figures/'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
