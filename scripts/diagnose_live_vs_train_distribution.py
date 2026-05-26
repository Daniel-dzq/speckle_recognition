#!/usr/bin/env python3
"""
Compare training-video vs live-GUI frame distributions (same preprocessing path).

Usage:
  python scripts/diagnose_live_vs_train_distribution.py \\
    --fiber Fiber1 --label a \\
    --data_root data/recognition_dataset \\
    --live_dir outputs/gui_diagnostics/session_YYYYMMDD_HHMMSS/raw_frames

If --live_dir is omitted, uses the newest session_* under outputs/gui_diagnostics/.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from final_fiber_dataset import build_final_label_map, discover_fiber_videos, resolve_data_root
from gui.gui_diagnostics import (
    DIAG_ROOT,
    frame_stats_uint8,
    preprocess_frame_like_inference,
    tensor_stats,
)
from unified_dataset import load_video_frames

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train vs live GUI distribution diagnostic")
    p.add_argument("--fiber", type=str, default="Fiber1")
    p.add_argument("--label", type=str, default="a")
    p.add_argument("--domain", type=str, default="GreenAndRed")
    p.add_argument("--data_root", type=str, default="data/recognition_dataset")
    p.add_argument("--live_dir", type=str, default=None,
                   help="Folder with raw_*.png from GUI diagnostics session")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--input_mode", type=str, default="gray", choices=["gray", "rgb"])
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--max_train_frames", type=int, default=64)
    p.add_argument("--max_live_frames", type=int, default=32)
    p.add_argument("--output_dir", type=str, default=DIAG_ROOT)
    return p.parse_args()


def newest_live_dir() -> Optional[str]:
    sessions = sorted(glob.glob(os.path.join(DIAG_ROOT, "session_*")))
    for path in reversed(sessions):
        raw = os.path.join(path, "raw_frames")
        if os.path.isdir(raw) and glob.glob(os.path.join(raw, "*.png")):
            return raw
    return None


def load_live_frames(live_dir: str, limit: int) -> List[np.ndarray]:
    paths = sorted(glob.glob(os.path.join(live_dir, "*.png")))
    paths += sorted(glob.glob(os.path.join(live_dir, "*.jpg")))
    frames = []
    for path in paths[:limit]:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is not None:
            frames.append(im)
    return frames


def load_train_frames(
    data_root: str,
    domain: str,
    fiber: str,
    label: str,
    img_size: int,
    input_mode: str,
    limit: int,
) -> Tuple[List[np.ndarray], str]:
    root = resolve_data_root(data_root, ROOT)
    label_map = build_final_label_map()
    videos = discover_fiber_videos(root, fiber, [domain], label_map=label_map)
    target = None
    label_l = label.lower()
    for v in videos:
        if v.get("label_name", "").lower() == label_l:
            target = v
            break
    if target is None:
        raise FileNotFoundError(
            f"No training video for {fiber} label={label} under {root}/{domain}"
        )
    stack = load_video_frames(target["path"], img_size, mode=input_mode)
    n = min(limit, stack.shape[0])
    if input_mode == "gray":
        frames = [stack[i] for i in range(n)]
    else:
        frames = [cv2.cvtColor(stack[i], cv2.COLOR_RGB2BGR) for i in range(n)]
    return frames, target["path"]


def central_roi_mean(frame: np.ndarray) -> float:
    if frame.ndim == 3:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        g = frame
    h, w = g.shape[:2]
    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4
    return float(np.mean(g[y0:y1, x0:x1]))


def frame_to_frame_std(frames: List[np.ndarray]) -> float:
    if len(frames) < 2:
        return float("nan")
    diffs = []
    for i in range(1, len(frames)):
        a = frames[i - 1]
        b = frames[i]
        if a.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        if b.ndim == 3:
            b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        diffs.append(float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))))
    return float(np.mean(diffs))


def pairwise_ncc(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if b.ndim == 3:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    af -= af.mean()
    bf -= bf.mean()
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(af, bf) / denom)


def compare_ssim(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if not HAS_SSIM:
        return None
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if b.ndim == 3:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    size = 224
    a = cv2.resize(a, (size, size))
    b = cv2.resize(b, (size, size))
    return float(ssim_fn(a, b, data_range=255))


def build_histogram(frame: np.ndarray, bins: int = 64) -> np.ndarray:
    if frame.ndim == 3:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        g = frame
    hist, _ = np.histogram(g.ravel(), bins=bins, range=(0, 256))
    hist = hist.astype(np.float64)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def write_preprocessing_audit(path: str) -> None:
    text = """# Training vs GUI preprocessing audit (code review)

## Training pipeline (`unified_dataset.py` + `scripts/train_final_15fibers.py`)

| Item | Value |
|------|--------|
| input_mode | CLI `--input_mode` (default **gray**) |
| clip_len | default **16** |
| img_size | default **224** |
| Resize | `cv2.resize(..., INTER_AREA)` on each frame |
| Gray path | BGR frame → resize → `cvtColor(BGR2GRAY)` → uint8 (N,H,W) |
| RGB path | BGR frame → resize → `cvtColor(BGR2RGB)` → uint8 (N,H,W,3) |
| Scale | `/255.0` float32 |
| Channels to model | Gray: stack single channel to **3 identical** (T,3,H,W) |
| Normalization | ImageNet mean/std per channel |
| Tensor layout | **(T, C, H, W)** batch → **(B, T, C, H, W)** |
| Center crop | **None** (full frame resized to square) |
| Augmentation | Train only (flip, brightness; RGB color jitter) |

## GUI live pipeline (`gui/inference_worker.py`)

| Item | Value |
|------|--------|
| input_mode | From checkpoint `input_mode` (default gray) |
| clip_len | From checkpoint `clip_len` |
| img_size | From checkpoint `img_size` |
| Resize | `cv2.resize(..., INTER_AREA)` — **matches training** |
| Gray path | BGR → resize → BGR2GRAY → `/255` → stack 3ch — **matches training** |
| Normalization | Same ImageNet mean/std — **matches training** |
| Tensor layout | Stack T frames (3,H,W) → **(1, T, 3, H, W)** — **matches training** |

## Known differences (not preprocessing bugs)

1. **Temporal sampling**: Training uses clips from recorded videos; GUI uses a **live sliding buffer** with `infer_every` (default 4) between inferences.
2. **Vote smoothing**: GUI majority-votes recent predictions; offline test does not.
3. **Optical domain**: Live setup (exposure, alignment, lasers, fiber coupling) may differ from `data/recognition_dataset` capture conditions.
4. **Camera path**: MindVision SDK may deliver mono or RGB; training videos were captured under dataset conditions.

## Preprocessing mismatch?

Per-code review: **core per-frame preprocessing matches** when `input_mode`, `img_size`, and `clip_len` match the checkpoint.

Poor live accuracy with good offline video accuracy strongly suggests **domain shift** or **temporal/smoothing differences**, not a missing `/255` or wrong channel order.

"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    live_dir = args.live_dir or newest_live_dir()
    train_frames, train_path = load_train_frames(
        args.data_root, args.domain, args.fiber, args.label,
        args.img_size, args.input_mode, args.max_train_frames,
    )
    live_frames: List[np.ndarray] = []
    if live_dir and os.path.isdir(live_dir):
        live_frames = load_live_frames(live_dir, args.max_live_frames)

    train_proc = [
        preprocess_frame_like_inference(f, img_size=args.img_size, input_mode=args.input_mode)
        for f in train_frames
    ]
    live_proc = [
        preprocess_frame_like_inference(f, img_size=args.img_size, input_mode=args.input_mode)
        for f in live_frames
    ]

    train_raw_stats = frame_stats_uint8(train_frames[0]) if train_frames else {}
    live_raw_stats = frame_stats_uint8(live_frames[0]) if live_frames else {}
    train_tensor_stats = tensor_stats(np.stack(train_proc[: args.clip_len], axis=0)) if train_proc else {}
    live_tensor_stats = tensor_stats(np.stack(live_proc[: args.clip_len], axis=0)) if live_proc else {}

    rows: List[Dict[str, Any]] = []

    def add_row(metric: str, train_val: Any, live_val: Any) -> None:
        rows.append({
            "metric": metric,
            "train": train_val,
            "live_gui": live_val,
        })

    for key in ("mean", "std", "min", "max", "p50"):
        add_row(f"raw_{key}", train_raw_stats.get(key), live_raw_stats.get(key))
    for key in ("mean", "std", "min", "max"):
        add_row(f"tensor_{key}", train_tensor_stats.get(key), live_tensor_stats.get(key))

    add_row("central_roi_mean", central_roi_mean(train_frames[0]) if train_frames else None,
            central_roi_mean(live_frames[0]) if live_frames else None)
    add_row("frame_to_frame_mean_abs_diff",
            frame_to_frame_std(train_frames), frame_to_frame_std(live_frames))

    hist_train = build_histogram(train_frames[0]) if train_frames else None
    hist_live = build_histogram(live_frames[0]) if live_frames else None
    if hist_train is not None and hist_live is not None:
        add_row("histogram_l1", float(np.sum(np.abs(hist_train - hist_live))), None)

    if train_frames and live_frames:
        add_row("ncc_first_frame", 1.0, pairwise_ncc(train_frames[0], live_frames[0]))
        ssim_v = compare_ssim(train_frames[0], live_frames[0])
        if ssim_v is not None:
            add_row("ssim_first_frame", ssim_v, ssim_v)

    csv_path = os.path.join(args.output_dir, "distribution_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "train", "live_gui"])
        writer.writeheader()
        writer.writerows(rows)

    audit_path = os.path.join(args.output_dir, "preprocessing_audit.md")
    write_preprocessing_audit(audit_path)

    conclusion = "inconclusive — more live captures needed"
    if not live_frames:
        conclusion = "inconclusive — no live frames found; run GUI with SPECKLE_GUI_DIAGNOSTICS=1"
    elif train_raw_stats and live_raw_stats:
        mean_ratio = live_raw_stats.get("mean", 0) / max(train_raw_stats.get("mean", 1), 1)
        std_ratio = live_raw_stats.get("std", 1) / max(train_raw_stats.get("std", 1), 1)
        if abs(mean_ratio - 1.0) > 0.35 or abs(std_ratio - 1.0) > 0.35:
            conclusion = "evidence suggests domain shift (intensity distribution differs)"
        elif train_tensor_stats and live_tensor_stats:
            if abs(train_tensor_stats.get("mean", 0) - live_tensor_stats.get("mean", 0)) > 0.5:
                conclusion = "evidence suggests preprocessing or intensity mismatch"
            else:
                conclusion = (
                    "preprocessing likely matched; poor live accuracy may be domain shift "
                    "or temporal/smoothing differences"
                )

    report_path = os.path.join(args.output_dir, "distribution_report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("# Train vs live GUI distribution report\n\n")
        fh.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
        fh.write(f"- Training video: `{train_path}`\n")
        fh.write(f"- Live frames dir: `{live_dir or 'none'}`\n")
        fh.write(f"- Fiber / label: {args.fiber} / {args.label}\n")
        fh.write(f"- input_mode={args.input_mode} img_size={args.img_size} clip_len={args.clip_len}\n\n")
        fh.write("## Summary metrics\n\n")
        fh.write(f"See `{csv_path}` and `{audit_path}`.\n\n")
        fh.write(f"## Conclusion\n\n**{conclusion}**\n\n")
        fh.write("## Likely physical causes to check\n\n")
        fh.write(
            "- Fiber reinsertion / clamp stress\n"
            "- Red laser power mismatch\n"
            "- Green laser power mismatch\n"
            "- SLM position or angle shift\n"
            "- Side-polish illumination spot shift\n"
            "- CCD exposure / gain / gamma mismatch\n"
            "- Camera focus or output-end position shift\n"
            "- Ambient light leakage\n"
            "- Dual-channel ratio mismatch (GreenAndRed vs RedChange domain)\n\n"
        )
        fh.write("## Offline evaluation on GUI-captured videos\n\n")
        fh.write(
            "Record or export AVI files from the live setup, then run:\n\n"
            "```bash\n"
            "python scripts/evaluate_gui_captured_clips.py \\\n"
            "  --model models/final_15fibers/Fiber1.pth \\\n"
            "  --video_dir path/to/gui_captured_avis\n"
            "```\n\n"
            "- If offline accuracy on GUI captures is **low** → domain shift / optics.\n"
            "- If offline accuracy is **high** but live GUI is **low** → GUI pipeline timing/smoothing.\n"
        )

    _save_comparison_figure(train_frames, live_frames, args.output_dir)

    print(f"Wrote {report_path}")
    print(f"Conclusion: {conclusion}")
    return 0


def _save_comparison_figure(
    train_frames: List[np.ndarray],
    live_frames: List[np.ndarray],
    output_dir: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping example_train_vs_live.png")
        return
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, frames, title in zip(
        axes[0],
        [train_frames, live_frames],
        ["Train (first frame)", "Live GUI (first frame)"],
    ):
        if frames:
            im = frames[0]
            if im.ndim == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    for ax, frames, title in zip(
        axes[1],
        [train_frames, live_frames],
        ["Train hist", "Live hist"],
    ):
        if frames:
            g = frames[0]
            if g.ndim == 3:
                g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
            ax.hist(g.ravel(), bins=64, range=(0, 255), alpha=0.85)
        ax.set_title(title)
    fig.tight_layout()
    out = os.path.join(output_dir, "example_train_vs_live.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
