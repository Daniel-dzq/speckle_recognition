#!/usr/bin/env python3
"""
Regenerate all paper figures that have verified inputs available.

Does not push; does not modify Word documents.

Usage (repo root):
    python3 scripts/generate_all_paper_figures.py
    python3 scripts/paper_figures/sanity.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

SUBSCRIPTS = [
    "scripts/paper_figures/plot_fig3_length_optimization.py",
    "scripts/paper_figures/plot_fig5_dual_channel.py",
    "scripts/paper_figures/plot_fig6_common_mode.py",
    "scripts/paper_figures/plot_fig7_authentication.py",
]


def main() -> int:
    for rel in SUBSCRIPTS:
        script = ROOT / rel
        print("==>", rel)
        subprocess.check_call([PY, str(script)], cwd=str(ROOT))
    print("Running sanity...")
    subprocess.check_call([PY, str(ROOT / "scripts/paper_figures/sanity.py")], cwd=str(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
