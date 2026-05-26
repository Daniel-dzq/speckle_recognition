#!/usr/bin/env python3
"""
Offline evaluation of trained fiber models on GUI-captured (or any) video files.

Use this to separate domain shift from live GUI pipeline issues:
  - Low accuracy here → optics / dataset mismatch.
  - High accuracy here but poor live GUI → timing, smoothing, or buffer bug.

Example:
  python scripts/evaluate_gui_captured_clips.py \\
    --model models/final_15fibers/Fiber1.pth \\
    --video_dir path/to/gui_recorded_avis \\
    --output_dir outputs/gui_diagnostics/offline_eval_Fiber1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from final_fiber_dataset import build_final_label_map, extract_label_from_filename, normalize_label
from models import get_model
from train_eval import evaluate
from unified_dataset import UnifiedSpeckleDataset, load_video_frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate model on captured videos (offline)")
    p.add_argument("--model", type=str, required=True, help="Path to FiberN.pth")
    p.add_argument("--video_dir", type=str, required=True, help="Folder with .avi/.mp4 files")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--clip_len", type=int, default=None)
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--input_mode", type=str, default=None, choices=["gray", "rgb"])
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4)
    return p.parse_args()


def discover_videos(video_dir: str) -> list[dict]:
    paths: list[str] = []
    for ext in ("*.avi", "*.AVI", "*.mp4", "*.MP4", "*.mov", "*.mkv"):
        paths.extend(glob.glob(os.path.join(video_dir, ext)))
    paths = sorted(set(paths))
    label_map = build_final_label_map()
    label_to_idx = {normalize_label(l): i for i, l in enumerate(label_map["labels"])}
    videos = []
    for path in paths:
        stem = extract_label_from_filename(os.path.basename(path))
        if not stem:
            continue
        key = normalize_label(stem)
        if key not in label_to_idx:
            print(f"[skip] unknown label for {path}")
            continue
        videos.append({
            "path": path,
            "label": label_to_idx[key],
            "label_name": key,
            "video_name": os.path.basename(path),
        })
    return videos


def main() -> int:
    args = parse_args()
    if not os.path.isfile(args.model):
        print(f"Model not found: {args.model}")
        return 1
    if not os.path.isdir(args.video_dir):
        print(f"video_dir not found: {args.video_dir}")
        return 1

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    clip_len = args.clip_len or ckpt.get("clip_len", 16)
    img_size = args.img_size or ckpt.get("img_size", 224)
    input_mode = args.input_mode or ckpt.get("input_mode", "gray")
    class_names = ckpt.get("class_names") or []
    if not class_names and ckpt.get("index_to_label"):
        idx_map = ckpt["index_to_label"]
        class_names = [idx_map[str(i)] for i in range(len(idx_map))]
    num_classes = len(class_names)

    model = get_model(ckpt.get("model_type", "cnn_pool"), num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    videos = discover_videos(args.video_dir)
    if not videos:
        print(f"No labeled videos in {args.video_dir}")
        return 1

    clips: list[dict] = []
    all_frames: dict[str, object] = {}
    for v in videos:
        frames = load_video_frames(v["path"], img_size, mode=input_mode)
        all_frames[v["video_name"]] = frames
        n = frames.shape[0]
        for start in range(0, max(1, n - clip_len + 1), args.stride):
            clips.append({
                "video_name": v["video_name"],
                "label": v["label"],
                "start_frame": start,
                "end_frame": start + clip_len,
            })

    dataset = UnifiedSpeckleDataset(clips, all_frames, clip_len, input_mode, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    loss, acc, preds, labels, _ = evaluate(
        model, loader, criterion, device, "gui_captured", verbose=True,
    )

    out_dir = args.output_dir or os.path.join(
        ROOT, "outputs", "gui_diagnostics",
        f"offline_eval_{os.path.splitext(os.path.basename(args.model))[0]}",
    )
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        "model": args.model,
        "video_dir": args.video_dir,
        "n_videos": len(videos),
        "n_clips": len(clips),
        "clip_len": clip_len,
        "img_size": img_size,
        "input_mode": input_mode,
        "accuracy_percent": acc,
        "loss": loss,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nOffline GUI-capture eval: accuracy={acc:.2f}% clips={len(clips)}")
    print(f"Summary: {out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
