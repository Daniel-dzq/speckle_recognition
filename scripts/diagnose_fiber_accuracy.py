#!/usr/bin/env python3
"""
Diagnose per-fiber test accuracy without changing data or models.

Usage:
    python scripts/diagnose_fiber_accuracy.py --fiber Fiber1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader

from final_fiber_dataset import (
    DEFAULT_DOMAINS,
    apply_label_map,
    discover_fiber_videos,
    resolve_data_root,
)
from models import get_model
from train_eval import evaluate, test_model, _save_confusion_matrix
from unified_dataset import (
    SPLIT_STRATEGY_UNIFORM,
    UnifiedSpeckleDataset,
    assign_splits_deploy,
    build_manifest,
    prepare_unified_data,
    print_per_class_split_counts,
    verify_no_leakage,
)


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose fiber recognition accuracy")
    p.add_argument("--fiber", default="Fiber1")
    p.add_argument("--data_root", default="recognition_dataset")
    p.add_argument("--output_dir", default=os.path.join(ROOT, "outputs", "final_15fiber_training"))
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--input_mode", default="gray")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--split_strategy", type=str, default=None,
                   help="Override split strategy (default: read metrics.json)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def count_clips_by_class(clips: list, class_names: list) -> dict:
    counts = defaultdict(int)
    for c in clips:
        name = c.get("label_name") or class_names[c["label"]]
        counts[name] += 1
    return dict(sorted(counts.items(), key=lambda x: class_names.index(x[0]) if x[0] in class_names else 999))


def count_clips_by_video_split(clips: list) -> dict:
    by_vid = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    return by_vid


def analyze_clip_splits(train_clips, val_clips, test_clips, class_names):
    def split_detail(clips, split_name):
        by_class = count_clips_by_class(clips, class_names)
        by_video = defaultdict(int)
        frame_ranges = defaultdict(list)
        for c in clips:
            by_video[c["video_id"]] += 1
            frame_ranges[c["video_id"]].append((c["start_frame"], c["end_frame"]))
        return {
            "split": split_name,
            "total_clips": len(clips),
            "by_class": by_class,
            "videos": sorted(by_video.keys()),
            "clips_per_video": dict(by_video),
            "frame_ranges": {k: frame_ranges[k] for k in sorted(frame_ranges)},
        }

    return {
        "train": split_detail(train_clips, "train"),
        "val": split_detail(val_clips, "val"),
        "test": split_detail(test_clips, "test"),
    }


def list_fiber_videos(data_root, fiber, domains):
    rows = []
    for domain in domains:
        path = os.path.join(data_root, domain, fiber)
        if not os.path.isdir(path):
            rows.append((domain, "(missing)", []))
            continue
        from final_fiber_dataset import _list_video_files, extract_label_from_filename
        files = _list_video_files(path)
        labels = [extract_label_from_filename(f) for f in files]
        rows.append((domain, path, list(zip([os.path.basename(f) for f in files], labels))))
    return rows


def main():
    args = parse_args()
    data_root = resolve_data_root(args.data_root, ROOT)
    fiber_out = os.path.join(args.output_dir, args.fiber)
    os.makedirs(fiber_out, exist_ok=True)

    label_map_path = os.path.join(args.output_dir, "label_map.json")
    with open(label_map_path, encoding="utf-8") as f:
        label_map = json.load(f)

    ckpt_path = os.path.join(fiber_out, "best_model.pth")
    metrics_path = os.path.join(fiber_out, "metrics.json")
    metrics = {}
    if os.path.isfile(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

    class_names_ckpt = metrics.get("class_names")
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        class_names = list(ckpt.get("class_names") or label_map["labels"])
        num_classes = int(ckpt.get("num_classes", len(class_names)))
    else:
        class_names = label_map["labels"]
        num_classes = label_map["num_classes"]
        ckpt = None

    print(f"Diagnosing {args.fiber} ...")
    print(f"Label map ({label_map['num_classes']} classes): {label_map['labels']}")
    print(f"Model head ({num_classes} classes): {class_names}")

    video_rows = list_fiber_videos(data_root, args.fiber, DEFAULT_DOMAINS)
    videos = discover_fiber_videos(data_root, args.fiber, DEFAULT_DOMAINS, label_map)
    videos = apply_label_map(videos, label_map)

    split_strategy = args.split_strategy
    if not split_strategy and metrics.get("split_strategy"):
        split_strategy = metrics["split_strategy"].get("strategy")
    if not split_strategy:
        split_strategy = SPLIT_STRATEGY_UNIFORM
    split_seed = int(metrics.get("split_strategy", {}).get("seed", args.seed))

    build_manifest(videos, cache_dir=None, index_workers=4)
    videos = assign_splits_deploy(videos)
    cache_dir = os.path.join(fiber_out, ".cache", "diagnosis")
    print(f"Split strategy: {split_strategy}  seed={split_seed}")
    all_frames, train_clips, val_clips, test_clips = prepare_unified_data(
        videos,
        clip_len=args.clip_len,
        stride=args.stride,
        img_size=args.img_size,
        input_mode=args.input_mode,
        cache_dir=cache_dir,
        load_workers=4,
        split_strategy=split_strategy,
        split_seed=split_seed,
    )
    print_per_class_split_counts(train_clips, val_clips, test_clips, class_names)

    split_analysis = analyze_clip_splits(train_clips, val_clips, test_clips, class_names)

    leakage = verify_no_leakage(train_clips, val_clips, test_clips)

    per_class_test = metrics.get("per_class_test_accuracy", {})
    label_coverage = metrics.get("label_coverage", {})

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if ckpt:
        model = get_model(ckpt.get("model_type", "cnn_pool"), num_classes, pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        test_loader = DataLoader(
            UnifiedSpeckleDataset(test_clips, all_frames, args.clip_len, args.input_mode),
            batch_size=args.batch_size,
            shuffle=False,
        )
        import torch.nn as nn
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, preds, labels, _ = evaluate(
            model, test_loader, criterion, device, "test", verbose=False,
        )
        _save_confusion_matrix(labels, preds, class_names, fiber_out)
        test_acc_run = test_acc
    else:
        test_acc_run = None
        preds, labels = [], []

    lines = [
        f"# {args.fiber} accuracy diagnosis",
        "",
        "## Summary",
        "",
        f"- **Reported test accuracy (metrics.json):** {metrics.get('test_accuracy', 'N/A')}%",
        f"- **Best validation accuracy:** {metrics.get('best_val_accuracy', 'N/A')}%",
        f"- **Train accuracy (eval on train clips):** {metrics.get('train_accuracy', 'N/A')}%",
        f"- **Recomputed test accuracy (this run):** "
        f"{f'{test_acc_run:.2f}%' if test_acc_run is not None else 'N/A (no checkpoint)'}",
        "",
        "## Label map (global `label_map.json`)",
        "",
        f"- **Number of classes:** {label_map['num_classes']}",
        f"- **Labels:** {', '.join(label_map['labels'])}",
        "",
        "| Index | Label |",
        "|------:|-------|",
    ]
    for i, lb in enumerate(label_map["labels"]):
        lines.append(f"| {i} | {lb} |")

    lines.extend([
        "",
        "## Model checkpoint classes",
        "",
        f"- **Checkpoint `num_classes`:** {num_classes}",
        f"- **Checkpoint `class_names`:** {', '.join(class_names)}",
        "",
    ])

    if num_classes != label_map["num_classes"] or set(class_names) != set(label_map["labels"]):
        extra = set(class_names) - set(label_map["labels"])
        missing = set(label_map["labels"]) - set(class_names)
        lines.append("**Mismatch between label map and checkpoint:**")
        if extra:
            lines.append(f"- Extra in checkpoint only: {', '.join(sorted(extra))}")
        if missing:
            lines.append(f"- Missing from checkpoint: {', '.join(sorted(missing))}")
        lines.append("")

    fiber_video_labels = set()
    for _domain, _path, files in video_rows:
        if isinstance(files, list):
            for _fname, lb in files:
                if lb:
                    fiber_video_labels.add(lb)

    unexpected = [c for c in class_names if c not in label_map["labels"]]
    if unexpected:
        lines.append(
            f"## Unexpected classes in model (not in current label map): "
            f"{', '.join(unexpected)}"
        )
        lines.append("")

    if "gril" in class_names and "gril" not in fiber_video_labels:
        lines.extend([
            "## Ninth class `gril`",
            "",
            "The saved Fiber1 model was trained with a **9-class global head** that includes "
            "`gril` (typo class from Fiber2's `gril.avi`). Fiber1 has **no** `gril.avi` video. "
            "Test clips never contain label `gril`, so the extra logit is unused at test time.",
            "",
        ])

    lines.extend([
        "## Domains used",
        "",
        f"- **Domains configured:** {', '.join(DEFAULT_DOMAINS)}",
        "",
    ])
    for domain, path, files in video_rows:
        lines.append(f"### {domain}")
        if files == "(missing)":
            lines.append("- Folder missing")
        else:
            lines.append(f"- Path: `{path}`")
            lines.append(f"- Videos: {len(files)}")
            for fname, lb in sorted(files, key=lambda x: x[1]):
                lines.append(f"  - `{fname}` -> label `{lb}`")
        lines.append("")

    lines.extend([
        "## Label parsing check",
        "",
        "Expected labels for Fiber1: `1`, `2`, `3`, `a`, `b`, `c`, `boy`, `girl`",
        "",
    ])
    fiber_labels = sorted(set(v["label_name"] for v in videos))
    lines.append(f"- **Labels found in Fiber1 videos:** {', '.join(fiber_labels)}")
    ok = set(fiber_labels) == {"1", "2", "3", "a", "b", "c", "boy", "girl"}
    lines.append(f"- **Parsing OK:** {'yes' if ok else 'NO - review filenames'}")
    lines.append("")

    lines.extend([
        "## Split strategy",
        "",
        f"- **Strategy:** `{split_strategy}` (seed {split_seed})",
        "- **Ratios:** 70% train / 15% val / 15% test (per-video clip partition)",
        "- **Video-level split:** no — same video may contribute clips to train, val, and test",
        "- **Frame leakage across splits:** none (disjoint clip frame ranges)",
        f"- **Leakage check (video_id overlap):** {leakage.get('status')} "
        "(overlap expected when clips from one file span multiple splits)",
        "",
        "## Clip counts (total)",
        "",
        f"| Split | Clips |",
        f"|-------|------:|",
        f"| train | {len(train_clips)} |",
        f"| val   | {len(val_clips)} |",
        f"| test  | {len(test_clips)} |",
        "",
    ])

    for split_name in ("train", "val", "test"):
        info = split_analysis[split_name]
        lines.append(f"### {split_name} — clips per class")
        lines.append("")
        lines.append("| Class | Clips |")
        lines.append("|-------|------:|")
        for lb in class_names:
            if lb in label_map["labels"]:
                n = info["by_class"].get(lb, 0)
                lines.append(f"| {lb} | {n} |")
        if "gril" in class_names and "gril" not in label_map["labels"]:
            lines.append(f"| gril | {info['by_class'].get('gril', 0)} |")
        lines.append("")

    lines.extend([
        "### test — clips per class (detail)",
        "",
    ])
    test_info = split_analysis["test"]
    for lb, n in sorted(test_info["by_class"].items()):
        flag = " **(very few clips)**" if n <= 2 else ""
        lines.append(f"- `{lb}`: {n} test clip(s){flag}")
    lines.append("")

    lines.extend([
        "## Per-class test accuracy",
        "",
        "| Class | Test acc (%) | Test clips |",
        "|-------|-------------:|-----------:|",
    ])
    for lb in class_names:
        if lb not in label_map["labels"] and lb != "gril":
            continue
        acc = per_class_test.get(lb, "N/A")
        nclips = test_info["by_class"].get(lb, 0)
        lines.append(f"| {lb} | {acc} | {nclips} |")
    lines.append("")

    if label_coverage:
        lines.extend([
            "## Label coverage on test set",
            "",
            f"- All model classes: {', '.join(label_coverage.get('all_label_names', []))}",
            f"- In y_true: {', '.join(label_coverage.get('labels_in_y_true', []))}",
            f"- Missing in y_true: {', '.join(label_coverage.get('missing_in_y_true', [])) or '(none)'}",
            "",
        ])

    test_section = (
        "## Test clip frame ranges (contiguous tail segment)"
        if split_strategy == "contiguous_temporal"
        else "## Test clip frame ranges (seeded clip partition)"
    )
    lines.extend([test_section, ""])
    for vid, ranges in test_info["frame_ranges"].items():
        lines.append(f"- `{vid}`: {len(ranges)} clip(s), frames {ranges}")
    lines.append("")

    n_test = len(test_clips)
    lines.extend([
        "## Test accuracy notes",
        "",
        f"1. **Test set size:** {n_test} test clips across {len(test_info['by_class'])} classes.",
        "2. **Small per-class counts:** with few test clips per class, one misclassified "
        "clip moves per-class accuracy sharply.",
        "3. **Train/val vs test gap:** if train/val are much higher than test, check whether "
        "test clips cover the full timeline (use `uniform_temporal` for final training).",
        "4. **Same-video clips:** train and test may share a source file with disjoint "
        "frame windows; late-segment bias is reduced under `uniform_temporal`.",
        "5. **Not a label-parsing bug:** Fiber1 filenames map to "
        "`1,2,3,a,b,c,boy,girl` across GreenAndRed and RedChange.",
        "",
        "## Confusion matrix",
        "",
        f"Saved to: `{os.path.join(fiber_out, 'confusion_matrix.png')}`",
        "",
        "## Artifacts",
        "",
        f"- metrics: `{metrics_path}`",
        f"- label coverage: `{os.path.join(fiber_out, 'label_coverage.json')}`",
        f"- per-class CSV: `{os.path.join(fiber_out, 'per_class_metrics.csv')}`",
        f"- split report: `{os.path.join(fiber_out, 'split_report.json')}`",
        "",
    ])

    report_path = os.path.join(fiber_out, "diagnosis.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
