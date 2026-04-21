#!/usr/bin/env python3
"""
Speckle-PUF Video Recognition — Main Entry Point
=================================================

Background:
  This is a speckle recognition experiment based on SLM + multimode/plastic fiber.
  Input:  one speckle video per letter (A-Z)
  Goal:   verify that deep learning can reliably distinguish the speckle patterns
          produced by different SLM letter encodings after fiber transmission.

Usage examples:
  # Default (CNN + Temporal Pooling):
  python main.py --data_dir video_capture --epochs 30 --batch_size 8

  # Use 3D CNN (R3D-18):
  python main.py --data_dir video_capture --model_type r3d --epochs 30

  # Fully customized:
  python main.py --data_dir "C:\\Users\\daizi\\Desktop\\recognition\\video_capture" \\
                 --model_type cnn_pool --clip_len 16 --stride 8 \\
                 --img_size 224 --epochs 30 --batch_size 8 --lr 1e-4
"""

import os

# Avoid duplicate OpenMP library error on Windows with PyTorch + NumPy
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if os.name == "nt":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys
import io

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import prepare_data, SpeckleClipDataset
from models import get_model
from train_eval import train_model, test_model


def parse_args():
    p = argparse.ArgumentParser(
        description="Speckle-PUF Video Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    p.add_argument("--data_dir", type=str, default="video_capture",
                    help="Directory containing videos A.avi ~ Z.avi (default: video_capture)")
    p.add_argument("--output_dir", type=str, default="output",
                    help="Output directory for results (default: output)")

    # Clip parameters
    p.add_argument("--clip_len", type=int, default=16,
                    help="Number of frames per clip (default: 16)")
    p.add_argument("--stride", type=int, default=8,
                    help="Stride between adjacent clips (default: 8)")
    p.add_argument("--img_size", type=int, default=224,
                    help="Input image size (default: 224)")

    # Model
    p.add_argument("--model_type", type=str, default="cnn_pool",
                    choices=["cnn_pool", "r3d"],
                    help="Model type: cnn_pool (ResNet18 + temporal pooling) or r3d (3D CNN) (default: cnn_pool)")

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=30,
                    help="Max training epochs (default: 30)")
    p.add_argument("--batch_size", type=int, default=8,
                    help="Batch size (default: 8)")
    p.add_argument("--lr", type=float, default=1e-4,
                    help="Initial learning rate (default: 1e-4)")
    p.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience (default: 10)")
    p.add_argument("--num_workers", type=int, default=0,
                    help="DataLoader worker processes; 0 recommended on Windows (default: 0)")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def _select_device() -> torch.device:
    """Select compute device; falls back to CPU if GPU is incompatible."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        t = torch.zeros(1, device="cuda")
        _ = t + t
        return torch.device("cuda")
    except RuntimeError:
        print("  [Warning] GPU detected but CUDA kernel is incompatible with this PyTorch build. Falling back to CPU.")
        print("  [Hint] For GPU support, upgrade PyTorch: pip install --pre torch torchvision "
              "--index-url https://download.pytorch.org/whl/nightly/cu128")
        return torch.device("cpu")


def main():
    args = parse_args()
    set_seed(args.seed)

    device = _select_device()

    print(f"\n{'=' * 80}")
    print(f"  Speckle-PUF Video Recognition")
    print(f"{'=' * 80}")
    print(f"  Device       : {device}")
    if device.type == "cuda":
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print(f"  Data dir     : {os.path.abspath(args.data_dir)}")
    print(f"  Output dir   : {os.path.abspath(args.output_dir)}")
    print(f"  Model        : {args.model_type}")
    print(f"  Clip length  : {args.clip_len} frames")
    print(f"  Clip stride  : {args.stride} frames")
    print(f"  Image size   : {args.img_size}x{args.img_size}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max epochs   : {args.epochs}")
    print(f"  Early stop   : {args.patience} epochs")

    os.makedirs(args.output_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════
    #  1. Load videos and generate clip metadata (temporal split)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  Loading videos and generating clips ...")
    print(f"{'=' * 80}")

    all_frames, train_clips, val_clips, test_clips, class_names = prepare_data(
        args.data_dir, args.clip_len, args.stride, args.img_size
    )

    args.num_classes = len(class_names)
    args.class_names = class_names

    if len(train_clips) == 0:
        sys.exit("Error: no training clips generated. Check video files and parameters.")
    if len(val_clips) == 0:
        print("Warning: validation set is empty! Too few frames; training results may be unreliable.")
    if len(test_clips) == 0:
        print("Warning: test set is empty! Too few frames; test evaluation unavailable.")

    # ══════════════════════════════════════════════════════════════════════
    #  2. Build Dataset / DataLoader
    # ══════════════════════════════════════════════════════════════════════
    train_ds = SpeckleClipDataset(train_clips, all_frames, args.clip_len, augment=True)
    val_ds = SpeckleClipDataset(val_clips, all_frames, args.clip_len, augment=False)
    test_ds = SpeckleClipDataset(test_clips, all_frames, args.clip_len, augment=False)

    pin = device.type == "cuda"
    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=pin)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    print(f"  Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # ══════════════════════════════════════════════════════════════════════
    #  3. Create model
    # ══════════════════════════════════════════════════════════════════════
    model = get_model(args.model_type, args.num_classes, pretrained=True).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: total {total_params:,}  trainable {trainable_params:,}")

    # ══════════════════════════════════════════════════════════════════════
    #  4. Train
    # ══════════════════════════════════════════════════════════════════════
    model, best_val_acc, history = train_model(
        model, train_loader, val_loader, args, device, args.output_dir
    )

    # ══════════════════════════════════════════════════════════════════════
    #  5. Test
    # ══════════════════════════════════════════════════════════════════════
    if len(test_clips) > 0:
        test_acc = test_model(
            model, test_loader, test_clips, class_names, device, args.output_dir
        )
    else:
        test_acc = 0.0
        print("\n  Skipping test evaluation (test set is empty).")

    # ══════════════════════════════════════════════════════════════════════
    #  6. Summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print(f"  Experiment Summary")
    print(f"{'=' * 80}")
    print(f"  Best val accuracy  : {best_val_acc:.2f}%")
    print(f"  Test accuracy      : {test_acc:.2f}%")
    print(f"  Best model         : {os.path.join(os.path.abspath(args.output_dir), 'best_model.pth')}")
    print(f"  Confusion matrix   : {os.path.join(os.path.abspath(args.output_dir), 'confusion_matrix.png')}")
    print(f"  Training log       : {os.path.join(os.path.abspath(args.output_dir), 'training_log.csv')}")
    print(f"  Per-class metrics  : {os.path.join(os.path.abspath(args.output_dir), 'per_class_metrics.csv')}")
    print(f"  Test predictions   : {os.path.join(os.path.abspath(args.output_dir), 'test_predictions.csv')}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
