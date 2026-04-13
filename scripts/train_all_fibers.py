#!/usr/bin/env python3
"""
Train speckle recognition models for ALL fibers sequentially.

Usage:
    python scripts/train_all_fibers.py
    python scripts/train_all_fibers.py --epochs 50 --lr 1e-4
    python scripts/train_all_fibers.py --skip Fiber3 Fiber4
    python scripts/train_all_fibers.py --only Fiber1 Fiber2

Output:
    checkpoints/<fiber_key>_best.pth  for each fiber
    results/<fiber_key>/              for each fiber
    results/training_summary.json     overall summary
"""

import os
import sys
import json
import argparse
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from train_fiber import discover_fibers, train_single_fiber, VIDEO_CAPTURE_DIR, RESULTS_DIR


class _Args:
    """Simple namespace for passing args to train_single_fiber."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train speckle models for all fibers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model_type", type=str,   default="cnn_pool", choices=["cnn_pool", "r3d"])
    p.add_argument("--clip_len",   type=int,   default=16)
    p.add_argument("--stride",     type=int,   default=8)
    p.add_argument("--img_size",   type=int,   default=224)
    p.add_argument("--epochs",     type=int,   default=40)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--patience",   type=int,   default=10)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--skip",       nargs="*",  default=[],
                   help="Fibers to skip, e.g. --skip Fiber3 Fiber4")
    p.add_argument("--only",       nargs="*",  default=[],
                   help="Only train these fibers, e.g. --only Fiber1 Fiber2")
    return p.parse_args()


def main():
    args = parse_args()

    all_fibers = discover_fibers(VIDEO_CAPTURE_DIR)
    if not all_fibers:
        sys.exit(f"[ERROR] No fiber directories found in {VIDEO_CAPTURE_DIR}")

    if args.only:
        fibers_to_train = [f for f in all_fibers if f in args.only]
        missing = [f for f in args.only if f not in all_fibers]
        if missing:
            print(f"[WARNING] These fibers not found: {missing}")
    else:
        fibers_to_train = [f for f in all_fibers if f not in args.skip]

    if not fibers_to_train:
        sys.exit("[ERROR] No fibers to train after applying filters.")

    print("=" * 80)
    print("  Speckle-PUF  |  Training ALL fibers")
    print("=" * 80)
    print(f"  All available fibers : {all_fibers}")
    print(f"  Fibers to train      : {fibers_to_train}")
    print(f"  Model type           : {args.model_type}")
    print(f"  Clip len / stride    : {args.clip_len} / {args.stride}")
    print(f"  Epochs / LR / Batch  : {args.epochs} / {args.lr} / {args.batch_size}")
    print("=" * 80)

    summary      = {}
    total_start  = time.time()

    for i, fiber in enumerate(fibers_to_train, 1):
        print(f"\n\n{'#' * 80}")
        print(f"  [{i}/{len(fibers_to_train)}] Training: {fiber}")
        print(f"{'#' * 80}")
        t0 = time.time()

        try:
            fiber_args = _Args(
                fiber      = fiber,
                model_type = args.model_type,
                clip_len   = args.clip_len,
                stride     = args.stride,
                img_size   = args.img_size,
                epochs     = args.epochs,
                lr         = args.lr,
                batch_size = args.batch_size,
                patience   = args.patience,
                seed       = args.seed,
            )
            best_val_acc, test_acc = train_single_fiber(fiber_args)
            elapsed = time.time() - t0
            summary[fiber] = {
                "status":       "ok",
                "best_val_acc": round(best_val_acc, 4),
                "test_acc":     round(test_acc, 4),
                "elapsed_sec":  round(elapsed, 1),
            }
            print(f"\n  [{fiber}] Done in {elapsed:.0f}s  |  val={best_val_acc:.1f}%  test={test_acc:.1f}%")

        except Exception as e:
            elapsed = time.time() - t0
            summary[fiber] = {
                "status":      "error",
                "error":       str(e),
                "elapsed_sec": round(elapsed, 1),
            }
            print(f"\n  [{fiber}] FAILED after {elapsed:.0f}s: {e}")
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - total_start

    # Save summary JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print(f"\n\n{'=' * 80}")
    print(f"  Training Summary  (total: {total_elapsed:.0f}s)")
    print(f"{'=' * 80}")
    print(f"  {'Fiber':<20}  {'Status':<8}  {'Val Acc':>8}  {'Test Acc':>9}  {'Time':>8}")
    print(f"  {'-' * 60}")
    for fiber, info in summary.items():
        status = info["status"]
        val    = f"{info['best_val_acc']:.2f}%" if status == "ok" else "N/A"
        test   = f"{info['test_acc']:.2f}%"     if status == "ok" else "N/A"
        t      = f"{info['elapsed_sec']:.0f}s"
        print(f"  {fiber:<20}  {status:<8}  {val:>8}  {test:>9}  {t:>8}")

    print(f"\n  Summary saved: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
