#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a unified 26-letter speckle recognizer across all domains and fibers.

Checkpoint is saved to:  checkpoints/unified_best.pth

Usage
-----
    # Cross-fiber split (Fiber1-3 train, Fiber4 val, Fiber5 test):
    python train_unified.py --data_root videocapture --split_mode cross_fiber

    # Within-fiber split (all fibers in every split, no leakage):
    python train_unified.py --data_root videocapture --split_mode within_fiber

    # Random clip sampling + cap dynamic videos:
    python train_unified.py --data_root videocapture --split_mode within_fiber \
        --clip_sampling random --max_clips_per_video 30

    # Custom fiber assignment (cross_fiber only):
    python train_unified.py --data_root videocapture --split_mode cross_fiber \
        --train_fibers Fiber1 Fiber2 Fiber3 Fiber4 \
        --val_fibers Fiber5 --test_fibers Fiber5
"""

import os
import sys
import io
import json
import csv
import copy
import argparse
import random
from collections import defaultdict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if os.name == "nt":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
from torch.utils.data import DataLoader

from unified_dataset import (
    discover_videos, assign_splits, prepare_unified_data, build_manifest,
    UnifiedSpeckleDataset, CLASS_NAMES, NUM_CLASSES, CHECKPOINT_PATH,
    DEFAULT_CACHE_DIR,
    verify_no_leakage, build_split_summary, save_split_summary,
    print_split_table, compute_group_accuracy,
    build_accuracy_table, print_accuracy_table, save_accuracy_table,
    DEFAULT_TRAIN_FIBERS, DEFAULT_VAL_FIBERS, DEFAULT_TEST_FIBERS,
)
from models import get_model
from train_eval import train_one_epoch, evaluate, _save_confusion_matrix


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Train unified 26-letter speckle recognizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_root", type=str, default="videocapture",
                   help="Root of the videocapture directory (default: videocapture)")
    p.add_argument("--output_dir", type=str, default=os.path.join("results", "unified"),
                   help="Output directory for logs/metrics (default: results/unified)")

    # Split
    p.add_argument("--split_mode", type=str, default="cross_fiber",
                   choices=["cross_fiber", "within_fiber", "deploy"],
                   help="cross_fiber = split by fiber; "
                        "within_fiber = rotation split by video; "
                        "deploy = temporal split within every video (all data)")
    p.add_argument("--train_fibers", nargs="+", default=DEFAULT_TRAIN_FIBERS,
                   help="(cross_fiber only) fibers for training")
    p.add_argument("--val_fibers",   nargs="+", default=DEFAULT_VAL_FIBERS)
    p.add_argument("--test_fibers",  nargs="+", default=DEFAULT_TEST_FIBERS)

    # Input
    p.add_argument("--input_mode", type=str, default="rgb", choices=["rgb", "gray"])
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--stride",   type=int, default=8)
    p.add_argument("--img_size", type=int, default=224)

    # Clip sampling
    p.add_argument("--clip_sampling", type=str, default="uniform",
                   choices=["uniform", "random"],
                   help="Clip sampling strategy for training "
                        "(val/test always use uniform)")
    p.add_argument("--max_clips_per_video", type=int, default=0,
                   help="Cap clips per video; 0 = unlimited (default: 0)")

    # Caching & parallelism
    p.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR,
                   help="Directory for manifest + frame caches "
                        f"(default: {DEFAULT_CACHE_DIR})")
    p.add_argument("--no_cache", action="store_true",
                   help="Disable all caching (decode from scratch every time)")
    p.add_argument("--index_workers", type=int, default=8,
                   help="Threads for parallel video indexing (default: 8)")
    p.add_argument("--load_workers", type=int, default=4,
                   help="Threads for parallel frame loading (default: 4)")

    # Model
    p.add_argument("--model_type", type=str, default="cnn_pool",
                   choices=["cnn_pool", "r3d"])

    # Training
    p.add_argument("--epochs",     type=int,   default=40)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--patience",   type=int,   default=12)
    p.add_argument("--num_workers", type=int,  default=0)
    p.add_argument("--seed",       type=int,   default=42)

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

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
            t = torch.zeros(1, device="cuda"); _ = t + t
            return torch.device("cuda")
        except RuntimeError:
            pass
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            t = torch.zeros(1, device="mps"); _ = t + t
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_unified_model(model, train_loader, val_loader, args, device, ckpt_path):
    """AdamW + CosineAnnealing + EarlyStopping.  Saves checkpoint to *ckpt_path*."""
    import torch.nn as nn
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    best_val_acc = 0.0
    best_state = None
    patience_cnt = 0
    history = []

    print(f"\n{'='*80}")
    print(f"  Training: model={args.model_type}  epochs={args.epochs}  "
          f"lr={args.lr}  device={device}")
    print(f"{'='*80}\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs,
        )
        val_loss, val_acc, _, _, _ = evaluate(
            model, val_loader, criterion, device, "val",
        )

        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        marker = "*" if val_acc > best_val_acc else " "
        print(f" {marker} Epoch {epoch:03d}/{args.epochs} | "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:6.2f}% | "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:6.2f}% | lr={lr:.2e}")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 4),
            "lr": lr,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_cnt = 0

            torch.save({
                "model_state_dict": best_state,
                "model_type": args.model_type,
                "num_classes": NUM_CLASSES,
                "class_names": CLASS_NAMES,
                "clip_len": args.clip_len,
                "img_size": args.img_size,
                "input_mode": args.input_mode,
                "best_val_acc": best_val_acc,
                "epoch": epoch,
                "unified_model": True,
                "fiber_name": "unified",
                "split_mode": args.split_mode,
                "train_fibers": args.train_fibers,
                "val_fibers": args.val_fibers,
                "test_fibers": args.test_fibers,
                "seed": args.seed,
            }, ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\n  Early stopping after {args.patience} epochs without improvement")
                break

    print(f"\n{'='*80}")
    print(f"  Training complete.  Best val accuracy: {best_val_acc:.2f}%")
    print(f"{'='*80}\n")

    model.load_state_dict(best_state)
    return model, best_val_acc, history


# ═══════════════════════════════════════════════════════════════════════════
#  Comprehensive evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_unified(model, test_loader, test_clips, device, output_dir, class_names):
    """Run inference and save per-domain / per-fiber / per-letter results."""
    import torch.nn as nn
    from sklearn.metrics import classification_report, precision_recall_fscore_support

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, all_preds, all_labels, all_probs = evaluate(
        model, test_loader, criterion, device, "test",
    )

    print(f"\n{'='*80}")
    print(f"  Test accuracy (overall): {test_acc:.2f}%")
    print(f"{'='*80}\n")

    # ── Per-letter ──────────────────────────────────────────────────────
    letter_acc = compute_group_accuracy(test_clips, all_preds, all_labels, "label_name")
    print("  Per-letter accuracy:")
    for lt, info in letter_acc.items():
        print(f"    {lt}: {info['accuracy']:6.2f}%  ({info['correct']}/{info['total']})")

    # ── Per-domain ──────────────────────────────────────────────────────
    domain_acc = compute_group_accuracy(test_clips, all_preds, all_labels, "domain")
    print("\n  Per-domain accuracy:")
    for dom, info in domain_acc.items():
        print(f"    {dom:25s}: {info['accuracy']:6.2f}%  ({info['correct']}/{info['total']})")

    # ── Per-fiber ───────────────────────────────────────────────────────
    fiber_acc = compute_group_accuracy(test_clips, all_preds, all_labels, "fiber")
    print("\n  Per-fiber accuracy:")
    for fib, info in fiber_acc.items():
        print(f"    {fib:15s}: {info['accuracy']:6.2f}%  ({info['correct']}/{info['total']})")

    # ── Domain × Fiber table ────────────────────────────────────────────
    table, domains, fibers = build_accuracy_table(test_clips, all_preds, all_labels)
    print("\n  Domain x Fiber accuracy (%):")
    print_accuracy_table(table, domains, fibers)
    save_accuracy_table(table, domains, fibers, output_dir)

    # ── Classification report ───────────────────────────────────────────
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4, zero_division=0,
    )
    rpt_path = os.path.join(output_dir, "classification_report.txt")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write(report)

    # ── Per-class CSV ───────────────────────────────────────────────────
    prec, rec, f1, sup = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(class_names))), zero_division=0,
    )
    met_path = os.path.join(output_dir, "per_class_metrics.csv")
    with open(met_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for i, name in enumerate(class_names):
            w.writerow([name, f"{prec[i]:.4f}", f"{rec[i]:.4f}", f"{f1[i]:.4f}", int(sup[i])])

    # ── Predictions CSV ─────────────────────────────────────────────────
    pred_path = os.path.join(output_dir, "test_predictions.csv")
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true_label", "pred_label", "confidence",
                     "domain", "fiber", "video_id", "start_frame", "end_frame"])
        for i, clip in enumerate(test_clips):
            true = class_names[clip["label"]]
            pred = class_names[all_preds[i]]
            conf = float(all_probs[i][all_preds[i]])
            w.writerow([true, pred, f"{conf:.4f}", clip["domain"], clip["fiber"],
                        clip["video_id"], clip["start_frame"], clip["end_frame"]])

    # ── Confusion matrices ──────────────────────────────────────────────
    _save_confusion_matrix(all_labels, all_preds, class_names, output_dir)
    for dom in sorted(set(c["domain"] for c in test_clips)):
        idxs = [i for i, c in enumerate(test_clips) if c["domain"] == dom]
        if not idxs:
            continue
        dom_dir = os.path.join(output_dir, f"domain_{dom}")
        os.makedirs(dom_dir, exist_ok=True)
        _save_confusion_matrix(
            [all_labels[i] for i in idxs], [all_preds[i] for i in idxs],
            class_names, dom_dir,
        )

    # ── Summary JSON ────────────────────────────────────────────────────
    eval_summary = {
        "test_loss": round(test_loss, 6),
        "test_accuracy": round(test_acc, 4),
        "per_letter": {k: round(v["accuracy"], 4) for k, v in letter_acc.items()},
        "per_domain": {k: round(v["accuracy"], 4) for k, v in domain_acc.items()},
        "per_fiber":  {k: round(v["accuracy"], 4) for k, v in fiber_acc.items()},
    }
    eval_path = os.path.join(output_dir, "eval_summary.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved: {rpt_path}")
    print(f"  Saved: {met_path}")
    print(f"  Saved: {pred_path}")
    print(f"  Saved: {eval_path}")

    return eval_summary


# ═══════════════════════════════════════════════════════════════════════════
#  Training curves
# ═══════════════════════════════════════════════════════════════════════════

def save_curves(history: list, output_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [h["epoch"] for h in history]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ax1.plot(epochs, [h["train_loss"] for h in history], label="train", lw=1.5)
        ax1.plot(epochs, [h["val_loss"]   for h in history], label="val", lw=1.5, ls="--")
        ax1.set(xlabel="Epoch", ylabel="Loss", title="Loss Curves")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, [h["train_acc"] for h in history], label="train", lw=1.5)
        ax2.plot(epochs, [h["val_acc"]   for h in history], label="val", lw=1.5, ls="--")
        chance = 100.0 / NUM_CLASSES
        ax2.axhline(y=chance, color="red", ls=":", lw=1, label=f"chance ({chance:.1f}%)")
        ax2.set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy Curves")
        ax2.legend(); ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        path = os.path.join(output_dir, "loss_acc_curve.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Curves saved: {path}")
    except Exception as e:
        print(f"  [WARNING] Could not save curves: {e}")


def save_training_log(history: list, output_dir: str):
    log_path = os.path.join(output_dir, "training_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
        w.writeheader()
        w.writerows(history)
    print(f"  Training log: {log_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    device = select_device()

    output_dir = args.output_dir
    ckpt_dir = os.path.join(ROOT, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "unified_best.pth")

    # ── Banner ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Unified Speckle Recognizer -- Training")
    print(f"{'='*80}")
    print(f"  Device          : {device}")
    if device.type == "cuda":
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
    print(f"  Data root       : {os.path.abspath(args.data_root)}")
    print(f"  Output dir      : {os.path.abspath(output_dir)}")
    print(f"  Checkpoint      : {ckpt_path}")
    print(f"  Split mode      : {args.split_mode}")
    if args.split_mode == "cross_fiber":
        print(f"  Train fibers    : {args.train_fibers}")
        print(f"  Val fibers      : {args.val_fibers}")
        print(f"  Test fibers     : {args.test_fibers}")
    print(f"  Model           : {args.model_type}")
    print(f"  Input mode      : {args.input_mode}")
    print(f"  Clip/stride     : {args.clip_len}/{args.stride}")
    cache_dir = None if args.no_cache else args.cache_dir
    print(f"  Clip sampling   : {args.clip_sampling}")
    if args.max_clips_per_video > 0:
        print(f"  Max clips/video : {args.max_clips_per_video}")
    print(f"  Frame cache     : {cache_dir or 'DISABLED'}")
    print(f"  Index workers   : {args.index_workers}")
    print(f"  Load workers    : {args.load_workers}")
    print(f"  Image size      : {args.img_size}")
    print()

    # ── 1. Discover & index ─────────────────────────────────────────────
    print("[1/6] Discovering and indexing videos ...")
    videos = discover_videos(args.data_root)
    if not videos:
        sys.exit(f"[ERROR] No videos found under {args.data_root}")
    print(f"  Found {len(videos)} video files")

    build_manifest(videos, cache_dir=cache_dir, index_workers=args.index_workers)

    videos = assign_splits(
        videos,
        split_mode=args.split_mode,
        train_fibers=args.train_fibers,
        val_fibers=args.val_fibers,
        test_fibers=args.test_fibers,
        seed=args.seed,
    )

    # ── 2. Generate clips & load frames ─────────────────────────────────
    print("\n[2/6] Generating clips and loading frames ...")
    all_frames, train_clips, val_clips, test_clips = prepare_unified_data(
        videos, args.clip_len, args.stride, args.img_size, args.input_mode,
        clip_sampling=args.clip_sampling,
        max_clips_per_video=args.max_clips_per_video,
        cache_dir=cache_dir,
        load_workers=args.load_workers,
    )

    print_split_table(videos, train_clips, val_clips, test_clips)

    # Leakage check + split metadata
    print()
    leakage = verify_no_leakage(train_clips, val_clips, test_clips)
    summary = build_split_summary(
        videos, train_clips, val_clips, test_clips,
        split_mode=args.split_mode, leakage_result=leakage,
    )
    save_split_summary(summary, output_dir)

    if not train_clips:
        sys.exit("[ERROR] No training clips generated.")

    # ── 3. Datasets & loaders ───────────────────────────────────────────
    print("\n[3/6] Building datasets ...")
    train_ds = UnifiedSpeckleDataset(
        train_clips, all_frames, args.clip_len, args.input_mode, augment=True)
    val_ds = UnifiedSpeckleDataset(
        val_clips, all_frames, args.clip_len, args.input_mode, augment=False)
    test_ds = UnifiedSpeckleDataset(
        test_clips, all_frames, args.clip_len, args.input_mode, augment=False)

    pin = device.type == "cuda"
    persist = args.num_workers > 0
    kw = dict(num_workers=args.num_workers, pin_memory=pin, persistent_workers=persist)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, **kw)

    print(f"  train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,} clips")

    # ── 4. Model ────────────────────────────────────────────────────────
    print("\n[4/6] Building model ...")
    model = get_model(args.model_type, NUM_CLASSES, pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,} total, {n_train:,} trainable")

    args.num_classes = NUM_CLASSES
    args.class_names = CLASS_NAMES

    # ── 5. Train ────────────────────────────────────────────────────────
    print("\n[5/6] Training ...")
    model, best_val_acc, history = train_unified_model(
        model, train_loader, val_loader, args, device, ckpt_path,
    )
    save_training_log(history, output_dir)
    save_curves(history, output_dir)

    # ── 6. Evaluate on test set ─────────────────────────────────────────
    print("\n[6/6] Evaluating on test set ...")
    test_acc = 0.0
    if test_clips:
        eval_result = evaluate_unified(
            model, test_loader, test_clips, device, output_dir, CLASS_NAMES,
        )
        test_acc = eval_result["test_accuracy"]
    else:
        print("  [WARNING] Test set is empty -- skipping evaluation")

    # ── Final summary ───────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Training complete")
    print(f"{'='*80}")
    print(f"  Split mode        : {args.split_mode}")
    print(f"  Best val accuracy : {best_val_acc:.2f}%")
    print(f"  Test accuracy     : {test_acc:.2f}%")
    print(f"  Chance level      : {100.0/NUM_CLASSES:.2f}%")
    print(f"  Checkpoint        : {ckpt_path}")
    print(f"  Results           : {os.path.abspath(output_dir)}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
