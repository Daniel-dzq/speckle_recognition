#!/usr/bin/env python3
"""Convenience wrapper for the length-optimisation experiment."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_RELEASE_ROOT = _SCRIPT_DIR.parents[2]
for _p in (_RELEASE_ROOT, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_experiment import main  # noqa: E402


if __name__ == "__main__":
    argv = ["length_optimization"] + sys.argv[1:]
    raise SystemExit(main(argv))
