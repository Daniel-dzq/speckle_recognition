#!/usr/bin/env python3
"""
Run the 3-domain ablation for Fiber2-5 sequentially and print results.

Usage (run from project root):
    python scripts/run_all_fiber_ablations.py
    python scripts/run_all_fiber_ablations.py --fibers Fiber3 Fiber5
"""
import subprocess, sys, os, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "diagnose_domains.py")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fibers", nargs="+", default=["Fiber2", "Fiber3", "Fiber4", "Fiber5"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--img_size", type=int, default=112)
    args = p.parse_args()

    for fiber in args.fibers:
        print(f"\n{'#'*70}")
        print(f"#  Running: {fiber}")
        print(f"{'#'*70}\n", flush=True)

        cmd = [
            sys.executable, "-u", SCRIPT,
            "--fiber", fiber,
            "--epochs", str(args.epochs),
            "--img_size", str(args.img_size),
        ]
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"\n[ERROR] {fiber} exited with code {result.returncode}")

    print("\n\nAll fiber ablations complete.")

if __name__ == "__main__":
    main()
