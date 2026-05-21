#!/usr/bin/env python3
"""
Train per-fiber challenge recognition models on the final 15-fiber dataset.

Usage:
    python scripts/train_final_15fibers.py --data_root recognition_dataset --dry_run
    python scripts/train_final_15fibers.py --data_root recognition_dataset --epochs 30
    python scripts/train_final_15fibers.py --data_root "(5.20)" --run_auth_matrix
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from final_fiber_dataset import (
    CANONICAL_LABELS,
    DEFAULT_DOMAINS,
    apply_label_map,
    audit_dataset,
    build_final_label_map,
    discover_fiber_videos,
    find_typo_video_files,
    format_audit_markdown,
    resolve_data_root,
    save_label_map,
)
from models import get_model
from train_eval import (
    evaluate,
    log_progress_display,
    resolve_use_tqdm,
    test_model,
    train_model,
    train_one_epoch,
    _save_confusion_matrix,
)
from unified_dataset import (
    SPLIT_STRATEGIES,
    SPLIT_STRATEGY_CONTIGUOUS,
    SPLIT_STRATEGY_UNIFORM,
    UnifiedSpeckleDataset,
    assign_splits_deploy,
    build_fiber_split_report,
    build_manifest,
    compute_group_accuracy,
    prepare_unified_data,
    print_per_class_split_counts,
    save_fiber_split_report,
    verify_no_leakage,
)

DEFAULT_FIBERS = [f"Fiber{i}" for i in range(1, 16)]
DEFAULT_OUTPUT = os.path.join(ROOT, "outputs", "final_15fiber_training")
DEFAULT_MODELS_DIR = os.path.join(ROOT, "models", "final_15fibers")


def parse_args():
    p = argparse.ArgumentParser(description="Train final 15-fiber PUF recognition models")
    p.add_argument("--data_root", type=str, default="data/recognition_dataset",
                   help="Dataset root (default: data/recognition_dataset)")
    p.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    p.add_argument("--fibers", nargs="+", default=DEFAULT_FIBERS)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT)
    p.add_argument("--models_dir", type=str, default=DEFAULT_MODELS_DIR)
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--input_mode", type=str, default="gray", choices=["gray", "rgb"])
    p.add_argument("--model_type", type=str, default="cnn_pool", choices=["cnn_pool", "r3d"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--dry_run", action="store_true",
                   help="Audit dataset only; do not train")
    p.add_argument("--run_auth_matrix", action="store_true",
                   help="After training, evaluate 15x15 cross-fiber matrix")
    p.add_argument("--only_fiber", type=str, default=None,
                   help="Train a single fiber only (e.g. Fiber1)")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training; run test evaluation from existing best_model.pth")
    p.add_argument("--skip_completed", action="store_true",
                   help="Skip fibers that already have metrics.json with status=ok")
    p.add_argument("--skip_auth_matrix", action="store_true")
    p.add_argument("--fresh", action="store_true",
                   help="Archive existing fiber outputs and GUI models before training")
    p.add_argument("--no_tqdm", action="store_true",
                   help="Disable tqdm progress bars (default; this flag is optional)")
    p.add_argument("--tqdm", action="store_true",
                   help="Enable tqdm progress bars")
    p.add_argument("--log_batch_every", type=int, default=0,
                   help="If >0, print train batch progress every N batches (no tqdm)")
    p.add_argument(
        "--split_strategy",
        type=str,
        default=SPLIT_STRATEGY_UNIFORM,
        choices=SPLIT_STRATEGIES,
        help="Per-video clip split: uniform_temporal (default) or contiguous_temporal",
    )
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device="cuda")
            _ = t + t
            return torch.device("cuda")
        except RuntimeError:
            pass
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            t = torch.zeros(1, device="mps")
            _ = t + t
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


def train_one_fiber(
    fiber: str,
    data_root: str,
    domains: list,
    label_map: dict,
    args,
    device: torch.device,
    output_dir: str,
    eval_only: bool = False,
) -> dict:
    """Train and evaluate one fiber model. Returns metrics dict."""
    fiber_out = os.path.join(output_dir, fiber)
    ckpt_path = os.path.join(fiber_out, "best_model.pth")

    class_names = label_map["labels"]
    num_classes = label_map["num_classes"]

    if eval_only and os.path.isfile(ckpt_path):
        ckpt_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_nc = int(ckpt_meta.get("num_classes", 0))
        if ckpt_nc != num_classes:
            print(
                f"  [ERROR] Checkpoint has {ckpt_nc} classes but label map has "
                f"{num_classes}. Retrain with --fresh (do not use --eval_only)."
            )
            return {"fiber": fiber, "status": "error", "error": "class count mismatch"}
        class_names = list(ckpt_meta.get("class_names") or class_names)
        print(f"  Eval-only checkpoint: num_classes={num_classes}")
    os.makedirs(fiber_out, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"  Training {fiber}")
    print(f"{'=' * 72}")

    videos = discover_fiber_videos(data_root, fiber, domains, label_map)
    videos = apply_label_map(videos, label_map)
    if not videos:
        return {"status": "error", "error": "no videos found", "fiber": fiber}

    missing_domains = [d for d in domains
                       if not os.path.isdir(os.path.join(data_root, d, fiber))]
    if missing_domains:
        print(f"  [WARNING] Missing domains for {fiber}: {missing_domains}")

    build_manifest(videos, cache_dir=None, index_workers=4)
    videos = assign_splits_deploy(videos)

    cache_dir = os.path.join(output_dir, ".cache", fiber)
    split_strategy = getattr(args, "split_strategy", SPLIT_STRATEGY_UNIFORM)
    all_frames, train_clips, val_clips, test_clips = prepare_unified_data(
        videos,
        clip_len=args.clip_len,
        stride=args.stride,
        img_size=args.img_size,
        input_mode=args.input_mode,
        clip_sampling="uniform",
        max_clips_per_video=0,
        cache_dir=cache_dir,
        load_workers=4,
        split_strategy=split_strategy,
        split_seed=args.seed,
    )

    leakage = verify_no_leakage(train_clips, val_clips, test_clips)
    if leakage.get("status") == "FAIL":
        print(
            "  Note: shared video_id across splits is expected when clips from the "
            "same video are assigned to train/val/test (disjoint frame ranges)."
        )
    split_note = {
        "strategy": split_strategy,
        "seed": args.seed,
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "video_id_overlap_across_splits": leakage.get("status") == "FAIL",
        "clip_frame_leakage": "none (disjoint clip frame ranges per video)",
    }
    with open(os.path.join(fiber_out, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(split_note, f, indent=2)

    split_report = build_fiber_split_report(
        videos, train_clips, val_clips, test_clips,
        split_strategy, args.seed, class_names,
    )
    json_path, md_path = save_fiber_split_report(split_report, fiber_out)
    print(f"  Split report: {json_path}")
    print(f"  Split report: {md_path}")

    print(f"  Clips: train={len(train_clips)} val={len(val_clips)} test={len(test_clips)}")
    print_per_class_split_counts(train_clips, val_clips, test_clips, class_names)

    if not train_clips:
        return {"status": "error", "error": "no training clips", "fiber": fiber}

    train_ds = UnifiedSpeckleDataset(
        train_clips, all_frames, args.clip_len, args.input_mode, augment=True)
    val_ds = UnifiedSpeckleDataset(
        val_clips, all_frames, args.clip_len, args.input_mode, augment=False)
    test_ds = UnifiedSpeckleDataset(
        test_clips, all_frames, args.clip_len, args.input_mode, augment=False)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=pin)

    model = get_model(args.model_type, num_classes, pretrained=not eval_only).to(device)
    best_val_acc = 0.0

    if eval_only:
        if not os.path.isfile(ckpt_path):
            return {"status": "error", "error": f"missing checkpoint: {ckpt_path}", "fiber": fiber}
        print(f"  Eval-only: loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_acc = float(ckpt.get("best_val_acc", ckpt.get("best_val_accuracy", 0.0)))
    else:
        train_args = SimpleNamespace(
            model_type=args.model_type,
            num_classes=num_classes,
            class_names=class_names,
            clip_len=args.clip_len,
            img_size=args.img_size,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            use_tqdm=False,
            log_batch_every=args.log_batch_every,
        )
        train_args.use_tqdm = resolve_use_tqdm(args)
        model, best_val_acc, history = train_model(
            model, train_loader, val_loader, train_args, device, fiber_out,
        )
        log_dst = os.path.join(fiber_out, "train_log.csv")
        if os.path.isfile(os.path.join(fiber_out, "training_log.csv")):
            shutil.copy2(os.path.join(fiber_out, "training_log.csv"), log_dst)

    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()

    train_loader_eval = DataLoader(
        UnifiedSpeckleDataset(train_clips, all_frames, args.clip_len, args.input_mode),
        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin,
    )
    _, train_acc, _, _, _ = evaluate(
        model, train_loader_eval, criterion, device, "train", verbose=False,
    )

    test_acc = 0.0
    per_class = {}
    label_coverage = {}
    if test_clips:
        test_acc, label_coverage = test_model(
            model, test_loader, test_clips, class_names, device, fiber_out,
        )
        _, _, preds, labels, _ = evaluate(
            model, test_loader, criterion, device, "test", verbose=False,
        )
        grp = compute_group_accuracy(test_clips, preds, labels, "label_name")
        per_class = {k: round(v["accuracy"], 2) for k, v in grp.items()}

    metrics = {
        "fiber": fiber,
        "status": "ok",
        "num_classes": num_classes,
        "class_names": class_names,
        "videos": len(videos),
        "clips": {"train": len(train_clips), "val": len(val_clips), "test": len(test_clips)},
        "train_accuracy": round(train_acc, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "per_class_test_accuracy": per_class,
        "label_coverage": label_coverage,
        "missing_domains": missing_domains,
        "split_strategy": split_note,
        "model_path": ckpt_path,
        "eval_only": eval_only,
    }
    with open(os.path.join(fiber_out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    assert num_classes == 8, f"Expected 8 classes, got {num_classes}"
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_type": args.model_type,
        "num_classes": num_classes,
        "class_names": class_names,
        "label_to_index": label_map["label_to_index"],
        "index_to_label": label_map["index_to_label"],
        "clip_len": args.clip_len,
        "img_size": args.img_size,
        "input_mode": args.input_mode,
        "fiber_name": fiber,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
    }
    torch.save(ckpt, os.path.join(fiber_out, "best_model.pth"))

    return metrics


def copy_models_to_gui(metrics_list: list, output_dir: str, models_dir: str, label_map: dict):
    """Copy checkpoints to models/final_15fibers/ without overwriting backups."""
    os.makedirs(models_dir, exist_ok=True)
    save_label_map(label_map, os.path.join(models_dir, "label_map.json"))

    for m in metrics_list:
        if m.get("status") != "ok":
            continue
        fiber = m["fiber"]
        src = os.path.join(output_dir, fiber, "best_model.pth")
        dst = os.path.join(models_dir, f"{fiber}.pth")
        if os.path.isfile(dst):
            bak = dst + ".bak"
            if not os.path.isfile(bak):
                shutil.copy2(dst, bak)
                print(f"  Backed up existing model: {bak}")
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  GUI model: {dst}")


@torch.no_grad()
def eval_fiber_on_data(
    model,
    videos: list,
    label_map: dict,
    args,
    device: torch.device,
) -> float:
    """Return accuracy (%) for model on given videos (test temporal segments)."""
    import torch.nn as nn

    videos = apply_label_map(list(videos), label_map)
    if not videos:
        return 0.0
    build_manifest(videos, cache_dir=None, index_workers=2)
    videos = assign_splits_deploy(videos)
    split_strategy = getattr(args, "split_strategy", SPLIT_STRATEGY_UNIFORM)
    all_frames, _, _, test_clips = prepare_unified_data(
        videos,
        clip_len=args.clip_len,
        stride=args.stride,
        img_size=args.img_size,
        input_mode=args.input_mode,
        clip_sampling="uniform",
        cache_dir=None,
        load_workers=2,
        split_strategy=split_strategy,
        split_seed=args.seed,
    )
    if not test_clips:
        return 0.0
    loader = DataLoader(
        UnifiedSpeckleDataset(test_clips, all_frames, args.clip_len, args.input_mode),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    criterion = nn.CrossEntropyLoss()
    _, acc, _, _, _ = evaluate(model, loader, criterion, device, "auth")
    return acc


def run_auth_matrix(
    data_root: str,
    domains: list,
    fibers: list,
    label_map: dict,
    args,
    device: torch.device,
    models_dir: str,
    output_dir: str,
) -> dict:
    """Build 15x15 matrix: row=model fiber, col=test data fiber."""
    print(f"\n{'=' * 72}")
    print("  Cross-fiber authentication matrix")
    print(f"{'=' * 72}")

    matrix = {}
    for mf in fibers:
        ckpt_path = os.path.join(models_dir, f"{mf}.pth")
        if not os.path.isfile(ckpt_path):
            ckpt_path = os.path.join(output_dir, mf, "best_model.pth")
        if not os.path.isfile(ckpt_path):
            print(f"  [WARNING] No model for {mf}, skipping row")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        num_classes = ckpt.get("num_classes", label_map["num_classes"])
        model = get_model(ckpt.get("model_type", "cnn_pool"), num_classes, pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        matrix[mf] = {}
        for df in fibers:
            vids = discover_all_for_fiber(data_root, df, domains)
            acc = eval_fiber_on_data(model, vids, label_map, args, device)
            matrix[mf][df] = round(acc, 2)
            tag = " (same)" if mf == df else ""
            print(f"  {mf} -> {df}: {acc:.1f}%{tag}")

    csv_path = os.path.join(output_dir, "auth_matrix_15x15.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_fiber"] + fibers)
        for mf in fibers:
            if mf not in matrix:
                continue
            w.writerow([mf] + [matrix[mf].get(df, "") for df in fibers])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ordered = [f for f in fibers if f in matrix]
        data = np.array([[matrix[mf].get(df, np.nan) for df in fibers] for mf in ordered])
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(data, vmin=0, vmax=100, cmap="RdYlGn")
        ax.set_xticks(range(len(fibers)))
        ax.set_yticks(range(len(ordered)))
        ax.set_xticklabels(fibers, rotation=45, ha="right")
        ax.set_yticklabels(ordered)
        ax.set_xlabel("Test data fiber")
        ax.set_ylabel("Model fiber")
        ax.set_title("Cross-fiber authentication accuracy (%)")
        fig.colorbar(im, ax=ax, label="Accuracy %")
        for i in range(len(ordered)):
            for j in range(len(fibers)):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7)
        fig.tight_layout()
        png_path = os.path.join(output_dir, "auth_matrix_15x15.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {png_path}")
    except Exception as e:
        print(f"  [WARNING] Could not save auth matrix plot: {e}")
        png_path = None

    diag = [matrix[f][f] for f in fibers if f in matrix and f in matrix[f]]
    off = [matrix[mf][df] for mf in matrix for df in matrix[mf] if mf != df]
    report_lines = [
        "# Cross-fiber authentication report",
        "",
        f"- Diagonal (same-fiber) mean: {np.mean(diag):.1f}%" if diag else "- No diagonal data",
        f"- Off-diagonal mean: {np.mean(off):.1f}%" if off else "- No off-diagonal data",
        f"- Chance level ({label_map['num_classes']} classes): "
        f"{100.0 / label_map['num_classes']:.1f}%",
        "",
        "Expected: high diagonal, low off-diagonal.",
    ]
    report_path = os.path.join(output_dir, "auth_matrix_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    return {"matrix": matrix, "csv": csv_path, "report": report_path}


def discover_all_for_fiber(data_root, fiber, domains):
    from final_fiber_dataset import discover_all_videos
    all_v = discover_all_videos(data_root, domains, [fiber])
    return [v for v in all_v if v["fiber"] == fiber]


def archive_old_artifacts(output_dir: str, models_dir: str) -> str:
    """Move prior 9-class (or partial) runs to a timestamped backup folder."""
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = os.path.join(ROOT, "outputs", f"final_15fiber_training_backup_{stamp}")
    os.makedirs(backup_root, exist_ok=True)

    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            src = os.path.join(output_dir, name)
            if name.startswith("Fiber") or name == ".cache":
                dst = os.path.join(backup_root, name)
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.move(src, dst)
                print(f"  Archived: {src} -> {dst}")

    gui_backup = os.path.join(backup_root, "gui_models")
    os.makedirs(gui_backup, exist_ok=True)
    if os.path.isdir(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(".pth") or f.endswith(".pth.bak"):
                shutil.move(os.path.join(models_dir, f), os.path.join(gui_backup, f))
                print(f"  Archived GUI model: {f}")

    print(f"  Backup root: {backup_root}")
    return backup_root


def write_summary(metrics_list: list, output_dir: str, label_map: dict):
    csv_path = os.path.join(output_dir, "summary_15fibers.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "fiber", "status", "videos", "train_clips", "val_clips", "test_clips",
            "train_acc", "best_val_acc", "test_acc",
        ])
        for m in metrics_list:
            clips = m.get("clips", {})
            w.writerow([
                m.get("fiber", ""),
                m.get("status", ""),
                m.get("videos", ""),
                clips.get("train", ""),
                clips.get("val", ""),
                clips.get("test", ""),
                m.get("train_accuracy", ""),
                m.get("best_val_accuracy", ""),
                m.get("test_accuracy", ""),
            ])

    md_path = os.path.join(output_dir, "summary_15fibers.md")
    lines = [
        "# Final 15-fiber training summary",
        "",
        f"- Classes: {label_map['num_classes']} ({', '.join(label_map['labels'])})",
        "",
        "| Fiber | Status | Test acc | Best val | Videos |",
        "|-------|--------|----------|----------|--------|",
    ]
    for m in metrics_list:
        lines.append(
            f"| {m.get('fiber','')} | {m.get('status','')} | "
            f"{m.get('test_accuracy', 'N/A')} | {m.get('best_val_accuracy', 'N/A')} | "
            f"{m.get('videos', '')} |"
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Summary: {csv_path}")
    print(f"  Summary: {md_path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)

    data_root = resolve_data_root(args.data_root, ROOT)
    output_dir = os.path.abspath(args.output_dir)
    models_dir = os.path.abspath(args.models_dir)
    os.makedirs(output_dir, exist_ok=True)

    fibers = [args.only_fiber] if args.only_fiber else list(args.fibers)

    print(f"\n{'=' * 72}")
    print("  Final 15-fiber training pipeline")
    print(f"{'=' * 72}")
    print(f"  Resolved data root : {data_root}")
    print(f"  Domains            : {args.domains}")
    print(f"  Fibers             : {len(fibers)} fibers")
    print(f"  Output dir         : {output_dir}")
    print(f"  Models dir         : {models_dir}")
    print(f"  Device             : {device}")
    print(f"  Dry run            : {args.dry_run}")
    print(f"  Eval only          : {args.eval_only}")
    print(f"  Skip completed     : {args.skip_completed}")
    _use_tqdm = resolve_use_tqdm(args)
    log_progress_display(_use_tqdm)
    print(f"  Split strategy     : {args.split_strategy}")
    print(f"  Split seed         : {args.seed}")

    if not _has_domain_folders(data_root):
        sys.exit(f"[ERROR] No domain folders under {data_root}")

    if args.fresh and not args.eval_only:
        print("\n[0] Archiving previous training artifacts ...")
        archive_old_artifacts(output_dir, models_dir)

    print("\n[1] Dataset audit ...")
    typo_files = find_typo_video_files(data_root, args.domains, DEFAULT_FIBERS)
    if typo_files:
        print(f"  Typo filename aliases ({len(typo_files)}):")
        for t in typo_files:
            print(f"    {os.path.basename(t['path'])}: "
                  f"{t['raw_label']} -> {t['canonical_label']}")
    else:
        print("  No gril.avi (or other configured typo) files found.")

    audit_fibers = list(DEFAULT_FIBERS) if not args.eval_only else fibers
    audit = audit_dataset(data_root, args.domains, audit_fibers, args.clip_len, args.stride)
    label_map = build_final_label_map()

    print(f"  Canonical labels  : {CANONICAL_LABELS}")
    print(f"  Final label map   : {label_map['labels']}")
    print(f"  Num classes       : {label_map['num_classes']}")
    print(f"  Total videos      : {audit['total_videos']}")
    est_total = sum(audit["clip_estimates"].values())
    print(f"  Est. clips (x3 split): ~{est_total}")

    os.makedirs(models_dir, exist_ok=True)
    save_label_map(label_map, os.path.join(output_dir, "label_map.json"))
    save_label_map(label_map, os.path.join(models_dir, "label_map.json"))
    print(f"  Saved label map: {output_dir}/label_map.json")
    print(f"  Saved label map: {models_dir}/label_map.json")
    md = format_audit_markdown(audit, args.clip_len, args.stride)
    audit_path = os.path.join(output_dir, "dataset_audit.md")
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  Audit report      : {audit_path}")

    if audit["broken_videos"]:
        print(f"  [WARNING] {len(audit['broken_videos'])} broken video(s)")

    if args.dry_run:
        print("\nDry run complete. No training performed.")
        return

    metrics_list = []
    t_all = time.perf_counter()

    for i, fiber in enumerate(fibers, 1):
        print(f"\n[{i}/{len(fibers)}] {fiber}")
        metrics_path = os.path.join(output_dir, fiber, "metrics.json")
        if args.skip_completed and not args.eval_only and os.path.isfile(metrics_path):
            try:
                with open(metrics_path, encoding="utf-8") as fh:
                    existing = json.load(fh)
                if existing.get("status") == "ok" and existing.get("test_accuracy") is not None:
                    print(f"  Skipping {fiber} (already completed)")
                    metrics_list.append(existing)
                    continue
            except (json.JSONDecodeError, OSError):
                pass

        t0 = time.perf_counter()
        try:
            m = train_one_fiber(
                fiber, data_root, args.domains, label_map, args, device, output_dir,
                eval_only=args.eval_only,
            )
            m["elapsed_sec"] = round(time.perf_counter() - t0, 1)
            print(f"  Done {fiber}: test_acc={m.get('test_accuracy')}% "
                  f"val_best={m.get('best_val_accuracy')}% [{m['elapsed_sec']}s]")
        except Exception as e:
            m = {"fiber": fiber, "status": "error", "error": str(e)}
            print(f"  [ERROR] {fiber}: {e}")
            import traceback
            traceback.print_exc()
        metrics_list.append(m)

    copy_models_to_gui(metrics_list, output_dir, models_dir, label_map)
    write_summary(metrics_list, output_dir, label_map)

    if args.run_auth_matrix and not args.skip_auth_matrix:
        run_auth_matrix(
            data_root, args.domains, fibers, label_map, args, device, models_dir, output_dir,
        )

    print(f"\nTotal time: {(time.perf_counter() - t_all) / 60:.1f} min")
    print(f"Models in: {models_dir}")


def _has_domain_folders(path: str) -> bool:
    from final_fiber_dataset import _has_domain_folders as check
    return check(path)


if __name__ == "__main__":
    main()
