#!/usr/bin/env python3
"""
Train a 26-class speckle recognition model for a single fiber.

Usage:
    python scripts/train_fiber.py --fiber Fiber1
    python scripts/train_fiber.py --fiber "test fiber"
    python scripts/train_fiber.py --fiber Fiber1 --epochs 50 --lr 1e-4
    python scripts/train_fiber.py --fiber Fiber1 --model_type r3d

Output:
    checkpoints/<fiber_key>_best.pth
    results/<fiber_key>/best_model.pth
    results/<fiber_key>/confusion_matrix.png
    results/<fiber_key>/per_class_metrics.csv
    results/<fiber_key>/test_predictions.csv
    results/<fiber_key>/training_log.csv
    results/<fiber_key>/metrics.json
    results/<fiber_key>/classification_report.txt
    results/<fiber_key>/loss_acc_curve.png
"""

import os
import sys
import json
import shutil
import argparse
import random
import glob

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Add project root to path so we can import dataset/models/train_eval
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import prepare_data, SpeckleClipDataset
from models import get_model
from train_eval import train_model, test_model, evaluate


# ============================================================================
#  Constants & defaults
# ============================================================================

VIDEO_CAPTURE_DIR = os.path.join(ROOT, "video_capture")
CHECKPOINTS_DIR   = os.path.join(ROOT, "checkpoints")
RESULTS_DIR       = os.path.join(ROOT, "results")


def fiber_key(fiber_name: str) -> str:
    """Normalize fiber name to a filesystem-safe key, e.g. 'test fiber' -> 'test_fiber'."""
    return fiber_name.lower().replace(" ", "_")


def discover_fibers(video_dir: str):
    """Scan video_capture/ and return all subdirs that contain .avi files."""
    fibers = []
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"video_capture directory not found: {video_dir}")
    for d in sorted(os.listdir(video_dir)):
        dpath = os.path.join(video_dir, d)
        if not os.path.isdir(dpath):
            continue
        avis = glob.glob(os.path.join(dpath, "*.avi")) + glob.glob(os.path.join(dpath, "*.AVI"))
        if avis:
            fibers.append(d)
    return fibers


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def select_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device="cuda")
            _ = t + t
            return torch.device("cuda")
        except RuntimeError as e:
            print(f"  [WARNING] GPU detected but CUDA kernel failed ({e}), falling back to next option")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            t = torch.zeros(1, device="mps")
            _ = t + t
            return torch.device("mps")
        except Exception as e:
            print(f"  [WARNING] MPS detected but failed ({e}), falling back to CPU")
    return torch.device("cpu")


# ============================================================================
#  Training curves
# ============================================================================

def save_loss_acc_curves(history: list, output_dir: str):
    """Save training loss/accuracy curves to PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs    = [h["epoch"] for h in history]
        tr_loss   = [h["train_loss"] for h in history]
        val_loss  = [h["val_loss"] for h in history]
        tr_acc    = [h["train_acc"] for h in history]
        val_acc   = [h["val_acc"] for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(epochs, tr_loss,  label="train loss",  linewidth=1.5)
        ax1.plot(epochs, val_loss, label="val loss",    linewidth=1.5, linestyle="--")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curves")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, tr_acc,  label="train acc",  linewidth=1.5)
        ax2.plot(epochs, val_acc, label="val acc",    linewidth=1.5, linestyle="--")
        ax2.axhline(y=100.0 / 26.0, color="red", linestyle=":", linewidth=1, label=f"chance ({100/26:.1f}%)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Accuracy Curves")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        save_path = os.path.join(output_dir, "loss_acc_curve.png")
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Training curves saved: {save_path}")
    except Exception as e:
        print(f"  [WARNING] Could not save training curves: {e}")


# ============================================================================
#  Leakage / sanity checks
# ============================================================================

def sanity_check(history: list, best_val_acc: float, test_acc: float) -> list:
    """Return a list of warning strings based on training dynamics."""
    warnings = []
    chance = 100.0 / 26.0

    if len(history) < 3:
        return warnings

    early_val = history[min(2, len(history) - 1)]["val_acc"]
    last_train = history[-1]["train_acc"]
    last_val   = history[-1]["val_acc"]

    if best_val_acc > 90 and len(history) <= 5:
        warnings.append(
            f"val_acc reached {best_val_acc:.1f}% in only {len(history)} epochs — check for data leakage."
        )
    if last_train > 90 and last_val < 30:
        warnings.append(
            f"train_acc={last_train:.1f}% but val_acc={last_val:.1f}% — possible overfitting."
        )
    if best_val_acc < chance * 1.5:
        warnings.append(
            f"best_val_acc={best_val_acc:.1f}% is near chance ({chance:.1f}%) — "
            "speckle patterns may not be discriminative enough with current settings."
        )
    if test_acc > best_val_acc + 15:
        warnings.append(
            f"test_acc={test_acc:.1f}% significantly exceeds val_acc={best_val_acc:.1f}% — unexpected."
        )
    return warnings


# ============================================================================
#  Metrics JSON
# ============================================================================

def save_metrics_json(
    fiber_name: str,
    best_val_acc: float,
    test_acc: float,
    history: list,
    class_names: list,
    output_dir: str,
    warnings: list,
):
    chance = 100.0 / 26.0
    best_epoch = max(history, key=lambda h: h["val_acc"])["epoch"] if history else 0
    metrics = {
        "fiber":          fiber_name,
        "num_classes":    len(class_names),
        "class_names":    class_names,
        "best_val_acc":   round(best_val_acc, 4),
        "test_acc":       round(test_acc, 4),
        "best_epoch":     best_epoch,
        "total_epochs":   len(history),
        "chance_level":   round(chance, 4),
        "above_chance_x": round(test_acc / chance, 2) if chance > 0 else None,
        "warnings":       warnings,
    }
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics JSON saved: {path}")


# ============================================================================
#  Classification report text file
# ============================================================================

def save_classification_report(
    labels: list, preds: list, class_names: list, output_dir: str
):
    from sklearn.metrics import classification_report
    report = classification_report(
        labels, preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    path = os.path.join(output_dir, "classification_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Classification report saved: {path}")
    return report


# ============================================================================
#  Main training function
# ============================================================================

def train_single_fiber(args):
    set_seed(args.seed)
    device = select_device()

    fkey       = fiber_key(args.fiber)
    fiber_dir  = os.path.join(VIDEO_CAPTURE_DIR, args.fiber)
    output_dir = os.path.join(RESULTS_DIR, fkey)
    ckpt_dir   = CHECKPOINTS_DIR

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    if not os.path.isdir(fiber_dir):
        sys.exit(f"[ERROR] Fiber directory not found: {fiber_dir}")

    print("=" * 80)
    print(f"  Speckle-PUF  |  Training fiber: {args.fiber}")
    print("=" * 80)
    print(f"  Device       : {device}")
    if device.type == "cuda":
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    elif device.type == "mps":
        print(f"  GPU          : Apple Silicon GPU (MPS)")
    print(f"  Data dir     : {fiber_dir}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Checkpoint   : {ckpt_dir}/{fkey}_best.pth")
    print(f"  Model type   : {args.model_type}")
    print(f"  Clip len     : {args.clip_len}")
    print(f"  Stride       : {args.stride}")
    print(f"  Image size   : {args.img_size}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  LR           : {args.lr}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Patience     : {args.patience}")
    print()

    # ── 1. Load video data ──────────────────────────────────────────────
    print("[1/5] Loading videos and building clips ...")
    all_frames, train_clips, val_clips, test_clips, class_names = prepare_data(
        fiber_dir, args.clip_len, args.stride, args.img_size
    )
    num_classes = len(class_names)

    if len(train_clips) == 0:
        sys.exit("[ERROR] No training clips generated. Check video files.")

    # ── 2. Build DataLoaders ─────────────────────────────────────────────
    print("\n[2/5] Building DataLoaders ...")
    pin = device.type == "cuda"  # pin_memory only works with CUDA

    train_ds = SpeckleClipDataset(train_clips, all_frames, args.clip_len, augment=True)
    val_ds   = SpeckleClipDataset(val_clips,   all_frames, args.clip_len, augment=False)
    test_ds  = SpeckleClipDataset(test_clips,  all_frames, args.clip_len, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=pin)

    print(f"  train clips={len(train_ds)}, val clips={len(val_ds)}, test clips={len(test_ds)}")

    # ── 3. Build model ───────────────────────────────────────────────────
    print("\n[3/5] Building model ...")
    model = get_model(args.model_type, num_classes, pretrained=True).to(device)
    total_p    = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_p:,}  |  Trainable: {trainable:,}")

    # Attach extra info to args for train_eval.py compatibility
    args.num_classes = num_classes
    args.class_names = class_names

    # ── 4. Train ─────────────────────────────────────────────────────────
    print("\n[4/5] Training ...")
    model, best_val_acc, history = train_model(
        model, train_loader, val_loader, args, device, output_dir
    )

    # ── 5. Test ──────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating on test set ...")
    if len(test_clips) > 0:
        test_acc = test_model(model, test_loader, test_clips, class_names, device, output_dir)
    else:
        test_acc = 0.0
        print("  [WARNING] Test set is empty, skipping test evaluation.")

    # ── Save training curves ─────────────────────────────────────────────
    save_loss_acc_curves(history, output_dir)

    # ── Save classification report ───────────────────────────────────────
    if len(test_clips) > 0:
        import torch.nn as nn
        from train_eval import evaluate as eval_fn
        _, _, all_preds, all_labels, _ = eval_fn(
            model, test_loader, nn.CrossEntropyLoss(), device, "test-report"
        )
        save_classification_report(all_labels, all_preds, class_names, output_dir)

    # ── Sanity checks & metrics JSON ─────────────────────────────────────
    warns = sanity_check(history, best_val_acc, test_acc)
    save_metrics_json(args.fiber, best_val_acc, test_acc, history, class_names, output_dir, warns)

    # ── Copy best checkpoint to checkpoints/ dir ─────────────────────────
    src_ckpt  = os.path.join(output_dir, "best_model.pth")
    dest_ckpt = os.path.join(ckpt_dir, f"{fkey}_best.pth")
    if os.path.exists(src_ckpt):
        # Re-save with fiber name embedded for cross-fiber eval
        ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)
        ckpt["fiber_name"] = args.fiber
        ckpt["fiber_key"]  = fkey
        torch.save(ckpt, dest_ckpt)
        print(f"\n  Checkpoint saved: {dest_ckpt}")

    # ── Summary ──────────────────────────────────────────────────────────
    chance = 100.0 / 26.0
    print(f"\n{'=' * 80}")
    print(f"  Results for fiber: {args.fiber}")
    print(f"{'=' * 80}")
    print(f"  Best val accuracy : {best_val_acc:.2f}%")
    print(f"  Test accuracy     : {test_acc:.2f}%")
    print(f"  Chance level (1/26): {chance:.2f}%")
    print(f"  Above-chance ratio : {test_acc / chance:.1f}x")
    if warns:
        print(f"\n  Warnings:")
        for w in warns:
            print(f"    - {w}")
    print(f"\n  Results saved to: {output_dir}")
    print("=" * 80)

    return best_val_acc, test_acc


# ============================================================================
#  CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train speckle recognition model for a single fiber",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--fiber",      type=str, required=True,
                   help='Fiber folder name, e.g. Fiber1 or "test fiber"')
    p.add_argument("--model_type", type=str, default="cnn_pool", choices=["cnn_pool", "r3d"],
                   help="Model architecture (default: cnn_pool)")
    p.add_argument("--clip_len",   type=int, default=16,
                   help="Frames per clip (default: 16)")
    p.add_argument("--stride",     type=int, default=8,
                   help="Clip stride (default: 8)")
    p.add_argument("--img_size",   type=int, default=224,
                   help="Input image size (default: 224)")
    p.add_argument("--epochs",     type=int, default=40,
                   help="Max training epochs (default: 40)")
    p.add_argument("--lr",         type=float, default=1e-4,
                   help="Initial learning rate (default: 1e-4)")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size (default: 8)")
    p.add_argument("--patience",   type=int, default=10,
                   help="Early stopping patience (default: 10)")
    p.add_argument("--seed",       type=int, default=42,
                   help="Random seed (default: 42)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Validate fiber exists
    available = discover_fibers(VIDEO_CAPTURE_DIR)
    if args.fiber not in available:
        print(f"[ERROR] Fiber '{args.fiber}' not found.")
        print(f"  Available fibers: {available}")
        sys.exit(1)

    train_single_fiber(args)
