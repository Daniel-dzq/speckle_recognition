#!/usr/bin/env python3
"""Convenience wrapper for the common-mode suppression experiment."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_experiment import main  # noqa: E402


if __name__ == "__main__":
    argv = ["common_mode"] + sys.argv[1:]
    raise SystemExit(main(argv))
