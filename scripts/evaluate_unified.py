#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained unified speckle model.

Loads a checkpoint, rebuilds the same split, and produces detailed metrics
including a domain x fiber accuracy table.

Usage
-----
    # Default (reads split_mode from checkpoint):
    python evaluate_unified.py

    # Explicit checkpoint:
    python evaluate_unified.py --checkpoint checkpoints/unified_best.pth

    # Override split mode / fibers for ad-hoc evaluation:
    python evaluate_unified.py --split_mode within_fiber --split val

    # Override test fibers (cross_fiber only):
    python evaluate_unified.py --split_mode cross_fiber \
        --test_fibers Fiber3 Fiber4 Fiber5
"""

import os
import sys
import io
import json
import csv
import argparse
from collections import defaultdict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if os.name == "nt":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, precision_recall_fscore_support

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
from train_eval import evaluate, _save_confusion_matrix


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a unified speckle recognition model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH,
                   help=f"Path to checkpoint (default: {CHECKPOINT_PATH})")
    p.add_argument("--data_root", type=str, default="videocapture")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (default: results/unified_eval)")
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])

    p.add_argument("--split_mode", type=str, default=None,
                   choices=["cross_fiber", "within_fiber", "deploy"],
                   help="Override split mode (default: read from checkpoint)")
    p.add_argument("--train_fibers", nargs="+", default=None)
    p.add_argument("--val_fibers",   nargs="+", default=None)
    p.add_argument("--test_fibers",  nargs="+", default=None)

    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--index_workers", type=int, default=8)
    p.add_argument("--load_workers", type=int, default=4)
    return p.parse_args()


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


def main():
    args = parse_args()
    device = select_device()

    if not os.path.isfile(args.checkpoint):
        sys.exit(f"[ERROR] Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    model_type  = ckpt.get("model_type",  "cnn_pool")
    num_classes = ckpt.get("num_classes",  NUM_CLASSES)
    class_names = ckpt.get("class_names",  CLASS_NAMES)
    clip_len    = ckpt.get("clip_len",     16)
    img_size    = ckpt.get("img_size",     224)
    input_mode  = ckpt.get("input_mode",   "gray")
    seed        = ckpt.get("seed",         42)

    split_mode   = args.split_mode   or ckpt.get("split_mode",   "cross_fiber")
    train_fibers = args.train_fibers or ckpt.get("train_fibers", DEFAULT_TRAIN_FIBERS)
    val_fibers   = args.val_fibers   or ckpt.get("val_fibers",   DEFAULT_VAL_FIBERS)
    test_fibers  = args.test_fibers  or ckpt.get("test_fibers",  DEFAULT_TEST_FIBERS)

    output_dir = args.output_dir or os.path.join("results", "unified_eval")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"  Unified Model Evaluation")
    print(f"{'='*80}")
    print(f"  Checkpoint    : {args.checkpoint}")
    print(f"  Model         : {model_type}")
    print(f"  Input mode    : {input_mode}")
    print(f"  Clip length   : {clip_len}")
    print(f"  Eval split    : {args.split}")
    print(f"  Split mode    : {split_mode}")
    if split_mode == "cross_fiber":
        print(f"  Train fibers  : {train_fibers}")
        print(f"  Val fibers    : {val_fibers}")
        print(f"  Test fibers   : {test_fibers}")
    print(f"  Device        : {device}")
    print()

    # ── Load model ──────────────────────────────────────────────────────
    model = get_model(model_type, num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # ── Discover and prepare data ───────────────────────────────────────
    cache_dir = None if args.no_cache else args.cache_dir
    print("\n  Discovering and indexing videos ...")
    videos = discover_videos(args.data_root)
    if not videos:
        sys.exit(f"[ERROR] No videos found under {args.data_root}")

    build_manifest(videos, cache_dir=cache_dir, index_workers=args.index_workers)

    videos = assign_splits(
        videos, split_mode=split_mode,
        train_fibers=train_fibers, val_fibers=val_fibers, test_fibers=test_fibers,
        seed=seed,
    )

    print(f"  Loading frames (input_mode={input_mode}) ...")
    all_frames, train_clips, val_clips, test_clips = prepare_unified_data(
        videos, clip_len, stride=clip_len, img_size=img_size, input_mode=input_mode,
        cache_dir=cache_dir, load_workers=args.load_workers,
    )

    print_split_table(videos, train_clips, val_clips, test_clips)
    print()
    leakage = verify_no_leakage(train_clips, val_clips, test_clips)

    split_info = build_split_summary(
        videos, train_clips, val_clips, test_clips,
        split_mode=split_mode, leakage_result=leakage,
    )
    save_split_summary(split_info, output_dir)

    eval_clips = test_clips if args.split == "test" else val_clips
    if not eval_clips:
        sys.exit(f"[ERROR] {args.split} split has 0 clips")

    eval_ds = UnifiedSpeckleDataset(
        eval_clips, all_frames, clip_len, input_mode=input_mode, augment=False,
    )
    persist = args.num_workers > 0
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=persist,
    )

    # ── Evaluate ────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    loss, acc, all_preds, all_labels, all_probs = evaluate(
        model, eval_loader, criterion, device, args.split,
    )

    print(f"\n{'='*80}")
    print(f"  {args.split.upper()} -- Overall accuracy: {acc:.2f}%  (loss: {loss:.4f})")
    print(f"{'='*80}\n")

    # ── Per-letter ──────────────────────────────────────────────────────
    letter_acc = compute_group_accuracy(eval_clips, all_preds, all_labels, "label_name")
    print("  Per-letter accuracy:")
    for lt, info in letter_acc.items():
        bar = "#" * int(info["accuracy"] / 5)
        print(f"    {lt}: {info['accuracy']:6.2f}% |{bar}")

    # ── Per-domain ──────────────────────────────────────────────────────
    domain_acc = compute_group_accuracy(eval_clips, all_preds, all_labels, "domain")
    print("\n  Per-domain accuracy:")
    for dom, info in domain_acc.items():
        print(f"    {dom:25s}: {info['accuracy']:6.2f}%  ({info['correct']}/{info['total']})")

    # ── Per-fiber ───────────────────────────────────────────────────────
    fiber_acc = compute_group_accuracy(eval_clips, all_preds, all_labels, "fiber")
    print("\n  Per-fiber accuracy:")
    for fib, info in fiber_acc.items():
        print(f"    {fib:15s}: {info['accuracy']:6.2f}%  ({info['correct']}/{info['total']})")

    # ── Domain × Fiber table ────────────────────────────────────────────
    table, domains, fibers = build_accuracy_table(eval_clips, all_preds, all_labels)
    print("\n  Domain x Fiber accuracy (%):")
    print_accuracy_table(table, domains, fibers)
    save_accuracy_table(table, domains, fibers, output_dir)

    # ── Sklearn report ──────────────────────────────────────────────────
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
    pred_path = os.path.join(output_dir, f"{args.split}_predictions.csv")
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true_label", "pred_label", "confidence",
                     "domain", "fiber", "video_id", "start_frame", "end_frame"])
        for i, clip in enumerate(eval_clips):
            true = class_names[clip["label"]]
            pred = class_names[all_preds[i]]
            conf = float(all_probs[i][all_preds[i]])
            w.writerow([true, pred, f"{conf:.4f}", clip["domain"], clip["fiber"],
                        clip["video_id"], clip["start_frame"], clip["end_frame"]])

    # ── Confusion matrices ──────────────────────────────────────────────
    _save_confusion_matrix(all_labels, all_preds, class_names, output_dir)
    for dom in sorted(set(c["domain"] for c in eval_clips)):
        idxs = [i for i, c in enumerate(eval_clips) if c["domain"] == dom]
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
        "checkpoint": args.checkpoint,
        "split": args.split,
        "split_mode": split_mode,
        "overall_loss": round(loss, 6),
        "overall_accuracy": round(acc, 4),
        "per_letter": {k: round(v["accuracy"], 4) for k, v in letter_acc.items()},
        "per_domain": {k: round(v["accuracy"], 4) for k, v in domain_acc.items()},
        "per_fiber":  {k: round(v["accuracy"], 4) for k, v in fiber_acc.items()},
        "leakage_check": leakage["status"],
    }
    eval_path = os.path.join(output_dir, "eval_summary.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved: {rpt_path}")
    print(f"  Saved: {met_path}")
    print(f"  Saved: {pred_path}")
    print(f"  Saved: {eval_path}")

    print(f"\n{'='*80}")
    print(f"  Evaluation complete.  Overall {args.split} accuracy: {acc:.2f}%")
    print(f"  Results: {os.path.abspath(output_dir)}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
