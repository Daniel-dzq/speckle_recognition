#!/usr/bin/env python3
"""
Figure 2 — Fiber length optimization (real experiment data).

Writes to figures/paper/Fig2_length_optimization/
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
IMPL = SCRIPT_DIR / "_fig2_length_optimization_impl.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 2 (length optimization).")
    parser.add_argument(
        "--include-word",
        action="store_true",
        help="Also write Fig2_length_optimization_word.* (optional Word layout).",
    )
    parser.add_argument("--archive-old", action="store_true", help="Archive existing outputs before overwrite.")
    args = parser.parse_args()

    if not IMPL.is_file():
        raise FileNotFoundError(IMPL)

    import os
    if args.include_word:
        os.environ["FIG2_INCLUDE_WORD"] = "1"
    else:
        os.environ.pop("FIG2_INCLUDE_WORD", None)

    print(f"Running {IMPL.relative_to(ROOT)} ...")
    runpy.run_path(str(IMPL), run_name="__main__")


if __name__ == "__main__":
    main()
