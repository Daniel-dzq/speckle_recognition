#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speckle Video Clip Dataset
==========================

Why split train/val/test in time instead of randomly?
-----------------------------------------------------

Each letter class has only one source video. Adjacent frames are strongly
temporally correlated (pixel differences are small). Random clip splits put
neighboring frames from the same span in both train and test; the model can
get inflated accuracy by "memorizing" pixels — **data leakage**.

For trustworthy evaluation we use a strict **temporal split**:
    First 70% of the timeline  →  train
    Middle 15%                 →  val
    Last 15%                   →  test

The three sets do not overlap in time, minimizing leakage.

Limitations of the current setup
--------------------------------
  1. One video per class: the model may overfit to that capture (light drift,
     camera noise, etc.); generalization needs more independent videos.
  2. Adjacent clips in the same split can still overlap when stride < clip_len,
     but cross-split leakage is removed by temporal splitting.
  3. Prefer multiple independent videos per class and split at video level.
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def load_video_frames(video_path: str, img_size: int) -> List[np.ndarray]:
    """Load all video frames, convert to grayscale and resize. Returns list of uint8 numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
        frames.append(gray)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Video has 0 frames: {video_path}")

    return frames


def prepare_data(
    data_dir: str,
    clip_len: int = 16,
    stride: int = 8,
    img_size: int = 224,
) -> Tuple[Dict[str, np.ndarray], List[dict], List[dict], List[dict], List[str]]:
    """
    Load all videos, apply temporal train/val/test split, and generate clip metadata.

    Returns
    -------
    all_frames : dict  {video_filename: np.ndarray of shape (N, H, W), uint8}
    train_clips, val_clips, test_clips : list of clip info dicts
        Each dict contains: label, label_name, video_name, start_frame, end_frame
    class_names : sorted list of class name strings
    """
    video_files = []
    for ext in ["*.avi", "*.mp4", "*.mkv", "*.mov", "*.AVI", "*.MP4"]:
        video_files.extend(glob.glob(os.path.join(data_dir, ext)))
    video_files = sorted(set(os.path.normpath(f) for f in video_files))

    if not video_files:
        raise FileNotFoundError(f"No video files found in {data_dir}")

    class_names = sorted(set(
        os.path.splitext(os.path.basename(f))[0].upper() for f in video_files
    ))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    print(f"\nFound {len(video_files)} videos, {len(class_names)} classes: {class_names}")
    print(f"Clip length: {clip_len}, stride: {stride}, image size: {img_size}x{img_size}")
    print(f"Split ratio: train {TRAIN_RATIO*100:.0f}% / val {VAL_RATIO*100:.0f}% "
          f"/ test {(1 - TRAIN_RATIO - VAL_RATIO)*100:.0f}%\n")

    all_frames: Dict[str, np.ndarray] = {}
    train_clips: List[dict] = []
    val_clips: List[dict] = []
    test_clips: List[dict] = []

    header = (f"{'Class':<8} {'Frames':>8} {'Train':>8} {'Val':>8} "
              f"{'Test':>8} {'TrainC':>8} {'ValC':>8} {'TestC':>8}")
    print(header)
    print("-" * len(header))

    for vpath in video_files:
        vname = os.path.basename(vpath)
        label_name = os.path.splitext(vname)[0].upper()
        label_idx = class_to_idx[label_name]

        frames = load_video_frames(vpath, img_size)
        n_frames = len(frames)
        all_frames[vname] = np.stack(frames)  # (N, H, W), uint8

        train_end = int(n_frames * TRAIN_RATIO)
        val_end = int(n_frames * (TRAIN_RATIO + VAL_RATIO))

        splits_def = [
            ("train", 0, train_end, train_clips),
            ("val", train_end, val_end, val_clips),
            ("test", val_end, n_frames, test_clips),
        ]

        counts = {}
        for split_name, s_start, s_end, clip_list in splits_def:
            n_clips = 0
            for s in range(s_start, s_end - clip_len + 1, stride):
                clip_list.append({
                    "label": label_idx,
                    "label_name": label_name,
                    "video_name": vname,
                    "start_frame": s,
                    "end_frame": s + clip_len,
                })
                n_clips += 1
            counts[split_name] = n_clips

        frame_counts = (train_end, val_end - train_end, n_frames - val_end)
        print(f"{label_name:<8} {n_frames:>8} {frame_counts[0]:>8} {frame_counts[1]:>8} "
              f"{frame_counts[2]:>8} {counts['train']:>8} {counts['val']:>8} {counts['test']:>8}")

    print("-" * 80)
    print(f"{'Total':<8} {'':>8} {'':>8} {'':>8} {'':>8} "
          f"{len(train_clips):>8} {len(val_clips):>8} {len(test_clips):>8}\n")

    if len(train_clips) == 0:
        raise RuntimeError("No training clips generated. Check that video length >= clip_len")

    return all_frames, train_clips, val_clips, test_clips, class_names


class SpeckleClipDataset(Dataset):
    """
    Speckle video clip dataset.

    Each sample is a contiguous clip of clip_len frames. Grayscale frames are
    stacked to 3 channels for ImageNet pretrained weights and normalized with
    ImageNet mean and standard deviation.

    Why can similar-looking speckles still be classified?
      Speckle looks like random grains; speckles from different letters are
      nearly indistinguishable visually. However:
      - SLM encodings per letter change phase/amplitude of the incident field
      - After multi-mode / plastic fiber propagation, differences map to subtle
        statistics in speckle intensity (spatial frequency, grain correlation, etc.)
      - CNNs can extract subtle patterns from high-dimensional pixels that
        humans cannot see, enabling stable classification
      - Analogy: fingerprint minutiae are hard for humans but easy for algorithms
    """

    def __init__(
        self,
        clips: List[dict],
        all_frames: Dict[str, np.ndarray],
        clip_len: int,
        augment: bool = False,
    ):
        self.clips = clips
        self.all_frames = all_frames
        self.clip_len = clip_len
        self.augment = augment

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        info = self.clips[idx]
        video_frames = self.all_frames[info["video_name"]]

        start = info["start_frame"]
        end = info["end_frame"]
        frames = video_frames[start:end].copy()  # (T, H, W), uint8

        if len(frames) < self.clip_len:
            pad = np.repeat(frames[-1:], self.clip_len - len(frames), axis=0)
            frames = np.concatenate([frames, pad], axis=0)

        frames = frames.astype(np.float32) / 255.0  # (T, H, W)

        if self.augment:
            if np.random.random() > 0.5:
                frames = frames[:, :, ::-1].copy()
            if np.random.random() > 0.5:
                frames = frames[:, ::-1, :].copy()
            brightness = 1.0 + np.random.uniform(-0.1, 0.1)
            frames = np.clip(frames * brightness, 0.0, 1.0)

        # Grayscale -> 3 channels (T, 3, H, W), compatible with ImageNet pretrained models
        frames_3ch = np.stack([frames, frames, frames], axis=1)

        for c in range(3):
            frames_3ch[:, c] = (frames_3ch[:, c] - self.mean[c]) / self.std[c]

        return torch.from_numpy(frames_3ch), info["label"]
