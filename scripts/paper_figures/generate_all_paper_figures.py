#!/usr/bin/env python3
"""Run all paper figure generators in recommended order."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent

ORDER = [
    ("Figure 2 length optimization", "generate_fig2_length_optimization.py", []),
    ("Figure 3 authentication", "generate_fig3_auth_performance.py", []),
    ("Figure 4 challenge speckle", "generate_fig4_challenge_speckle_examples.py", []),
    ("Figure 5 stability", "generate_fig5_stability.py", []),
    ("Figure 6 disturbance", "generate_fig6_disturbance.py", []),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all available paper figures.")
    parser.add_argument("--skip-fig2", action="store_true", help="Skip Fig2 (requires pandas).")
    parser.add_argument("--full-physical", action="store_true", help="Pass --full-analysis to Fig5/Fig6.")
    args = parser.parse_args()

    extra = ["--full-analysis"] if args.full_physical else []
    generated = []
    for title, script, flags in ORDER:
        if args.skip_fig2 and script.startswith("generate_fig2"):
            print(f"SKIP: {title}")
            continue
        path = SCRIPTS / script
        cmd = [sys.executable, str(path)] + flags + (extra if script in ("generate_fig5_stability.py", "generate_fig6_disturbance.py") and args.full_physical else [])
        print(f"\n=== {title} ===")
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        generated.append(title)

    print("\n=== Summary ===")
    for g in generated:
        print(f"  OK: {g}")
    print(f"Outputs under: {ROOT / 'figures' / 'paper'}")


if __name__ == "__main__":
    main()
