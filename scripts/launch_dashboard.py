#!/usr/bin/env python3
"""
Launch the Speckle-PUF experiment dashboard (PySide6).

Usage:
    python scripts/launch_dashboard.py [results_dir]

If ``results_dir`` is omitted, ``<repo>/results`` is used. The dashboard
discovers every experiment run folder that contains a ``manifest.json`` /
``summary.json`` / ``report.md`` and displays it in a browsable dark-themed
interface with figure gallery, table viewer, and report reader tabs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gui.experiment_dashboard import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
