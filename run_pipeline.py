#!/usr/bin/env python3
"""
Speckle-PUF End-to-End Pipeline
=================================
Starting from 26 letter videos in video_capture/, this script completes:
  1. Frame extraction  -> screenshots/<letter>/frame_*.png
  2. Dataset split     -> 70% train / 15% val / 15% test (temporal, no leakage)
  3. CNN training
  4. Test-set evaluation with full classification report + confusion matrix

Usage:
  python run_pipeline.py                    # all steps
  python run_pipeline.py --skip-extract     # skip extraction (frames already exist)
  python run_pipeline.py --frames 300       # extract 300 frames per video (default 300)
  python run_pipeline.py --epochs 80        # training epochs
"""

import os
import sys
import glob
import time
import argparse
import subprocess
from pathlib import Path

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR  = os.path.join(BASE_DIR, "video_capture")
SHOTS_DIR  = os.path.join(BASE_DIR, "screenshots")
MODEL_PATH = os.path.join(BASE_DIR, "model_pipeline.pth")

# ============================================================================
#  STEP 0 — locate ffmpeg
# ============================================================================

def get_ffmpeg():
    try:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"[ffmpeg] Using imageio-ffmpeg: {path}")
        return path
    except Exception:
        pass
    result = subprocess.run(
        ["ffmpeg", "-version"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if result.returncode == 0:
        print("[ffmpeg] Using system ffmpeg")
        return "ffmpeg"
    sys.exit("Error: ffmpeg not found. Install imageio-ffmpeg:  pip install imageio-ffmpeg")


# ============================================================================
#  STEP 1 — video frame extraction
# ============================================================================

def get_video_duration(video_path: str, ffmpeg: str) -> float | None:
    cmd = [ffmpeg, "-i", video_path]
    r = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
                       text=True, encoding="utf-8", errors="ignore")
    for line in r.stderr.splitlines():
        if "Duration" in line:
            ts = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = ts.split(":")
            return float(h) * 3600 + float(m) * 60 + float(s)
    return None


def extract_video(video_path: str, out_dir: str, ffmpeg: str, n_frames: int) -> int:
    """Uniformly extract n_frames grayscale PNG frames from a single video."""
    os.makedirs(out_dir, exist_ok=True)

    existing = glob.glob(os.path.join(out_dir, "*.png"))
    if len(existing) >= n_frames:
        print(f"  skip  ({len(existing)} frames already exist)")
        return len(existing)

    duration = get_video_duration(video_path, ffmpeg)
    if not duration or duration <= 0:
        print(f"  Warning: could not determine video duration, skipping: {video_path}")
        return 0

    fps_target = max(n_frames / duration, 0.1)
    out_pat    = os.path.join(out_dir, "frame_%05d.png")

    cmd = [
        ffmpeg, "-i", video_path,
        "-vf", f"fps={fps_target:.6f},scale=128:128,format=gray",
        "-vsync", "vfr",
        "-frames:v", str(n_frames + 5),
        "-y", out_pat,
    ]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        print(f"  ffmpeg error: {r.stderr[-400:]}")
        return 0

    saved = len(glob.glob(os.path.join(out_dir, "*.png")))
    return saved


def step_extract(n_frames: int, ffmpeg: str):
    print("\n" + "=" * 70)
    print("  STEP 1 — Video Frame Extraction")
    print("=" * 70)

    videos = sorted(
        glob.glob(os.path.join(VIDEO_DIR, "*.avi")) +
        glob.glob(os.path.join(VIDEO_DIR, "*.AVI")) +
        glob.glob(os.path.join(VIDEO_DIR, "*.mp4")) +
        glob.glob(os.path.join(VIDEO_DIR, "*.MP4"))
    )
    if not videos:
        sys.exit(f"Error: no video files found in {VIDEO_DIR}")

    print(f"  Video dir    : {VIDEO_DIR}")
    print(f"  Output dir   : {SHOTS_DIR}")
    print(f"  Video count  : {len(videos)}")
    print(f"  Frames/video : {n_frames}")
    print()

    total = 0
    for vp in videos:
        name    = os.path.splitext(os.path.basename(vp))[0]
        out_dir = os.path.join(SHOTS_DIR, name)
        print(f"  Processing {name} ...", end=" ", flush=True)
        t0    = time.time()
        saved = extract_video(vp, out_dir, ffmpeg, n_frames)
        dt    = time.time() - t0
        print(f"OK  {saved} frames  ({dt:.1f}s)")
        total += saved

    print(f"\n  Frame extraction complete: {total} frames total")


# ============================================================================
#  STEP 2/3 — data split + training + evaluation (reuses train_model.py logic)
# ============================================================================

def step_train(args_ns):
    """Call train_model.py's functions directly to avoid code duplication."""
    sys.path.insert(0, BASE_DIR)
    import importlib
    tm = importlib.import_module("train_model")

    import torch

    class TrainArgs:
        data       = SHOTS_DIR
        output     = MODEL_PATH
        task       = "decode"
        epochs     = args_ns.epochs
        lr         = args_ns.lr
        batch      = args_ns.batch
        image_size = args_ns.image_size
        patience   = args_ns.patience
        seed       = 42

    train_args = TrainArgs()
    tm.set_seed(train_args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("  STEP 2 — Dataset Split  (70% train / 15% val / 15% test)")
    print("=" * 70)
    train_s, val_s, test_s, class_names = tm.collect_samples(SHOTS_DIR, "decode")

    if not train_s:
        sys.exit("Error: no training samples found. Run frame extraction first.")

    print("\n" + "=" * 70)
    print("  STEP 3 — CNN Training & Test-set Evaluation")
    print("=" * 70)
    print(f"  Device       : {device}")
    if torch.cuda.is_available():
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print(f"  Classes      : {len(class_names)}")
    print(f"  Image size   : {train_args.image_size}x{train_args.image_size}")
    print(f"  Epochs       : {train_args.epochs}")
    print(f"  LR           : {train_args.lr}")
    print(f"  Batch size   : {train_args.batch}")
    print()

    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)

    train_loader, val_loader, test_loader = tm.make_loaders(
        train_s, val_s, test_s,
        train_args.image_size, train_args.batch,
    )
    history = tm.train(
        train_args, train_loader, val_loader, test_loader, class_names, device
    )

    print("\n" + "=" * 70)
    print("  Pipeline complete")
    print("=" * 70)
    print(f"  Best val accuracy  : {history['best_val_acc']:.2f}%")
    print(f"  Test accuracy      : {history['test_acc']:.2f}%")
    print(f"  Model saved to     : {MODEL_PATH}")
    cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    if os.path.exists(cm_path):
        print(f"  Confusion matrix   : {cm_path}")
    print()


# ============================================================================
#  Argument parsing & entry point
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Speckle-PUF end-to-end pipeline: frame extraction -> training -> evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--skip-extract", action="store_true",
                   help="Skip frame extraction (use existing screenshots/ folder)")
    p.add_argument("--frames",      type=int,   default=300,
                   help="Frames to extract per video (default 300)")
    p.add_argument("--epochs",      type=int,   default=60,
                   help="Training epochs (default 60)")
    p.add_argument("--lr",          type=float, default=1e-3,
                   help="Initial learning rate (default 0.001)")
    p.add_argument("--batch",       type=int,   default=32,
                   help="Batch size (default 32)")
    p.add_argument("--image-size",  type=int,   default=128,
                   dest="image_size",
                   help="Image size (default 128, must be a multiple of 16)")
    p.add_argument("--patience",    type=int,   default=12,
                   help="Early stopping patience (default 12)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  Speckle-PUF End-to-End Recognition Pipeline")
    print("=" * 70)
    print(f"  Video dir    : {VIDEO_DIR}")
    print(f"  Frames dir   : {SHOTS_DIR}")
    print(f"  Model output : {MODEL_PATH}")

    if not args.skip_extract:
        ffmpeg = get_ffmpeg()
        step_extract(args.frames, ffmpeg)
    else:
        print("\n[Skipping extraction] Using existing frames in screenshots/")

    step_train(args)


if __name__ == "__main__":
    main()
