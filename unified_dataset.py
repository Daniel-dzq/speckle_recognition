#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Multi-Domain Speckle Dataset
=====================================

Loads speckle videos from all illumination domains and fibers into a single
dataset for training a unified 26-letter recognizer.

Dataset layout on disk (``videocapture/`` root)::

    Green/          -> domain "green_only"
    GreenAndRed/    -> domain "red_green_fixed"
    RedChange/      -> domain "red_green_dynamic"

Each domain folder has Fiber1/ … Fiber5/ subfolders, each containing A.avi–Z.avi.

Two split modes
---------------
``cross_fiber``
    Entire fibers go to one split.  Default: Fiber1-3 train, Fiber4 val,
    Fiber5 test.  Good for measuring cross-fiber generalisation.

``within_fiber``
    All five fibers appear in every split.  Videos are assigned by a
    deterministic per-letter rotation so no video (session) is shared across
    splits.  Good for practical deployment training on all available data.
"""

import os
import re
import glob
import json
import csv
import time
import hashlib
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional

# ────────────────────────────────────────────────────────────────────────────
#  Constants
# ────────────────────────────────────────────────────────────────────────────

DOMAIN_FOLDERS = OrderedDict([
    ("Green",       "green_only"),
    ("GreenAndRed", "red_green_fixed"),
    ("RedChange",   "red_green_dynamic"),
])

DOMAIN_FOLDER_REVERSE = {v: k for k, v in DOMAIN_FOLDERS.items()}

LETTERS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
CLASS_NAMES = LETTERS
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

DEFAULT_TRAIN_FIBERS = ["Fiber1", "Fiber2", "Fiber3"]
DEFAULT_VAL_FIBERS   = ["Fiber4"]
DEFAULT_TEST_FIBERS  = ["Fiber5"]

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "checkpoints", "unified_best.pth"
)

DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".cache", "unified"
)


# ────────────────────────────────────────────────────────────────────────────
#  Caching infrastructure
# ────────────────────────────────────────────────────────────────────────────

def _fast_frame_count(video_path: str) -> int:
    """Count frames via container header — no pixel decoding."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        n = 0
        while cap.read()[0]:
            n += 1
    cap.release()
    return n


def _file_stat(path: str) -> Tuple[int, float]:
    """Return (file_size, mtime) or (0, 0) on error."""
    try:
        s = os.stat(path)
        return s.st_size, s.st_mtime
    except OSError:
        return 0, 0.0


def _index_one_video(v: dict) -> dict:
    """Worker callable: count frames + collect file stats."""
    n = _fast_frame_count(v["path"])
    sz, mt = _file_stat(v["path"])
    return {"video_id": v["video_id"], "n_frames": n, "file_size": sz, "file_mtime": mt}


def build_manifest(
    videos: List[dict],
    cache_dir: Optional[str] = None,
    index_workers: int = 4,
) -> List[dict]:
    """
    Fast-index all videos (parallel frame counting) with optional JSON cache.

    Populates ``n_frames``, ``file_size``, ``file_mtime`` on each video dict.
    Returns the same list, mutated in place.
    """
    t0 = time.perf_counter()

    # ── Try loading cached manifest ────────────────────────────────────
    manifest_path = os.path.join(cache_dir, "manifest.json") if cache_dir else None
    cache_hit = False

    if manifest_path and os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cmap = {e["video_id"]: e for e in cached.get("videos", [])}

            if len(cmap) == len(videos) and all(v["video_id"] in cmap for v in videos):
                all_fresh = True
                for v in videos:
                    ce = cmap[v["video_id"]]
                    sz, mt = _file_stat(v["path"])
                    if sz != ce.get("file_size") or abs(mt - ce.get("file_mtime", 0)) > 0.5:
                        all_fresh = False
                        break
                if all_fresh:
                    for v in videos:
                        ce = cmap[v["video_id"]]
                        v["n_frames"]  = ce["n_frames"]
                        v["file_size"] = ce["file_size"]
                        v["file_mtime"] = ce["file_mtime"]
                    cache_hit = True
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    elapsed = time.perf_counter() - t0
    if cache_hit:
        print(f"  Manifest cache HIT  ({manifest_path})  [{elapsed:.2f}s]")
        return videos

    # ── Cache miss: parallel indexing ──────────────────────────────────
    print(f"  Manifest cache MISS — indexing {len(videos)} videos "
          f"({index_workers} workers) ...")
    t1 = time.perf_counter()

    workers = max(1, index_workers)
    if workers == 1:
        results = [_index_one_video(v) for v in videos]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_index_one_video, v): v["video_id"] for v in videos}
            rmap = {}
            for fut in as_completed(futures):
                r = fut.result()
                rmap[r["video_id"]] = r
            results = [rmap[v["video_id"]] for v in videos]

    for v, r in zip(videos, results):
        v["n_frames"]  = r["n_frames"]
        v["file_size"] = r["file_size"]
        v["file_mtime"] = r["file_mtime"]

    # ── Persist manifest ──────────────────────────────────────────────
    if manifest_path:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        obj = {"videos": [
            {"video_id": v["video_id"], "n_frames": v["n_frames"],
             "file_size": v["file_size"], "file_mtime": v["file_mtime"]}
            for v in videos
        ]}
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(obj, f)

    elapsed = time.perf_counter() - t1
    print(f"  Indexed {len(videos)} videos [{elapsed:.2f}s]")
    if manifest_path:
        print(f"  Manifest saved: {manifest_path}")

    return videos


# ── Per-video .npy frame cache ─────────────────────────────────────────────

def _npy_cache_path(video_id: str, img_size: int, input_mode: str, cache_dir: str) -> str:
    key = f"{video_id}|{img_size}|{input_mode}"
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, "frames", f"{h}.npy")


def _load_or_decode_video(
    v: dict, img_size: int, input_mode: str, cache_dir: Optional[str],
) -> Optional[np.ndarray]:
    """Load decoded frames from .npy cache; on miss, decode from video and cache."""
    npy_path = _npy_cache_path(v["video_id"], img_size, input_mode, cache_dir) if cache_dir else None

    # Check cache freshness (npy must be newer than source video)
    if npy_path and os.path.isfile(npy_path):
        try:
            npy_mt = os.path.getmtime(npy_path)
            vid_mt = v.get("file_mtime") or os.path.getmtime(v["path"])
            if npy_mt >= vid_mt:
                return np.load(npy_path, mmap_mode=None)
        except (OSError, ValueError):
            pass

    # Decode
    frames = load_video_frames(v["path"], img_size, mode=input_mode)

    # Save to cache
    if npy_path:
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, frames)

    return frames


def _load_video_worker(args: tuple) -> Tuple[str, Optional[np.ndarray]]:
    """Worker for parallel frame loading. Returns (video_id, frames_or_None)."""
    v, img_size, input_mode, cache_dir = args
    try:
        frames = _load_or_decode_video(v, img_size, input_mode, cache_dir)
        return v["video_id"], frames
    except Exception as e:
        print(f"  [WARNING] {e}")
        return v["video_id"], None


# ────────────────────────────────────────────────────────────────────────────
#  Video discovery
# ────────────────────────────────────────────────────────────────────────────

def _extract_letter(filename: str) -> Optional[str]:
    """Extract letter from filenames like ``A.avi`` or ``A(1).avi``."""
    base = os.path.splitext(os.path.basename(filename))[0]
    base = re.sub(r"\(\d+\)$", "", base).strip().upper()
    if len(base) == 1 and base in CLASS_NAMES:
        return base
    return None


def discover_videos(data_root: str) -> List[dict]:
    """
    Scan ``data_root`` for all domain/fiber/letter videos.

    Returns a list of dicts, each with keys:
        path, letter, label, domain, domain_folder, fiber, video_id, filename
    """
    videos: List[dict] = []
    for domain_folder, domain_name in DOMAIN_FOLDERS.items():
        domain_path = os.path.join(data_root, domain_folder)
        if not os.path.isdir(domain_path):
            print(f"  [WARNING] Domain folder not found, skipping: {domain_path}")
            continue

        for fiber_name in sorted(os.listdir(domain_path)):
            fiber_path = os.path.join(domain_path, fiber_name)
            if not os.path.isdir(fiber_path):
                continue

            vfiles = sorted(set(
                os.path.normpath(f)
                for ext in ("*.avi", "*.AVI", "*.mp4", "*.MP4", "*.mkv", "*.mov")
                for f in glob.glob(os.path.join(fiber_path, ext))
            ))

            for vpath in vfiles:
                letter = _extract_letter(vpath)
                if letter is None:
                    print(f"  [WARNING] Cannot extract letter from: {vpath}")
                    continue
                video_id = f"{domain_name}/{fiber_name}/{os.path.basename(vpath)}"
                videos.append({
                    "path": vpath,
                    "letter": letter,
                    "label": CLASS_TO_IDX[letter],
                    "domain": domain_name,
                    "domain_folder": domain_folder,
                    "fiber": fiber_name,
                    "video_id": video_id,
                    "filename": os.path.basename(vpath),
                })

    return videos


# ────────────────────────────────────────────────────────────────────────────
#  Split assignment
# ────────────────────────────────────────────────────────────────────────────

def _all_fibers(videos: List[dict]) -> List[str]:
    return sorted(set(v["fiber"] for v in videos))


def assign_splits_cross_fiber(
    videos: List[dict],
    train_fibers: List[str] = None,
    val_fibers: List[str] = None,
    test_fibers: List[str] = None,
) -> List[dict]:
    """Assign videos to splits by fiber membership (whole fibers per split)."""
    train_set = set(train_fibers or DEFAULT_TRAIN_FIBERS)
    val_set   = set(val_fibers   or DEFAULT_VAL_FIBERS)
    test_set  = set(test_fibers  or DEFAULT_TEST_FIBERS)

    for v in videos:
        if v["fiber"] in train_set:
            v["split"] = "train"
        elif v["fiber"] in val_set:
            v["split"] = "val"
        elif v["fiber"] in test_set:
            v["split"] = "test"
        else:
            v["split"] = "train"
            print(f"  [WARNING] Fiber '{v['fiber']}' not in any split list -> train")
    return videos


def assign_splits_within_fiber(
    videos: List[dict],
    seed: int = 42,
) -> List[dict]:
    """
    Assign videos to splits so all fibers appear in every split.

    Uses a deterministic per-letter rotation: for each letter, a different
    set of fibers is assigned to train / val / test.  Over 26 letters each
    fiber contributes roughly equally to every split.

    Guarantees: entire videos stay in one split (no frame leakage).
    """
    fibers = _all_fibers(videos)
    n_fibers = len(fibers)
    fiber_idx = {f: i for i, f in enumerate(fibers)}

    n_train = max(1, int(n_fibers * 0.6))  # 3 of 5
    n_val   = max(1, (n_fibers - n_train) // 2)  # 1 of 5
    # rest goes to test

    by_letter = defaultdict(list)
    for v in videos:
        by_letter[v["letter"]].append(v)

    for letter in sorted(by_letter):
        letter_idx = CLASS_TO_IDX[letter]
        shift = letter_idx % n_fibers
        rotated = fibers[shift:] + fibers[:shift]

        train_set = set(rotated[:n_train])
        val_set   = set(rotated[n_train:n_train + n_val])
        # test_set = everything else

        for v in by_letter[letter]:
            if v["fiber"] in train_set:
                v["split"] = "train"
            elif v["fiber"] in val_set:
                v["split"] = "val"
            else:
                v["split"] = "test"

    return videos


DEPLOY_TRAIN_RATIO = 0.70
DEPLOY_VAL_RATIO   = 0.15


def assign_splits_deploy(videos: List[dict]) -> List[dict]:
    """
    Mark every video for temporal splitting (all fibers, all domains).

    The actual temporal split happens inside ``prepare_unified_data``
    when ``split_mode="deploy"`` — each video's frames are divided
    70 / 15 / 15 by timeline position so clips from the same video
    never cross splits.
    """
    for v in videos:
        v["split"] = "all"
    return videos


def assign_splits(
    videos: List[dict],
    split_mode: str = "cross_fiber",
    train_fibers: List[str] = None,
    val_fibers: List[str] = None,
    test_fibers: List[str] = None,
    seed: int = 42,
) -> List[dict]:
    """Dispatch to the appropriate split strategy."""
    if split_mode == "cross_fiber":
        return assign_splits_cross_fiber(videos, train_fibers, val_fibers, test_fibers)
    elif split_mode == "within_fiber":
        return assign_splits_within_fiber(videos, seed=seed)
    elif split_mode == "deploy":
        return assign_splits_deploy(videos)
    else:
        raise ValueError(f"Unknown split_mode: {split_mode!r}  "
                         f"(choose 'cross_fiber', 'within_fiber', or 'deploy')")


# ────────────────────────────────────────────────────────────────────────────
#  Video loading
# ────────────────────────────────────────────────────────────────────────────

def load_video_frames(video_path: str, img_size: int, mode: str = "rgb") -> np.ndarray:
    """
    Load all frames from a video file.

    Parameters
    ----------
    mode : ``"rgb"`` -> ``(N, H, W, 3)``  |  ``"gray"`` -> ``(N, H, W)``
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
        if mode == "gray":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Video has 0 frames: {video_path}")
    return np.stack(frames)


# ────────────────────────────────────────────────────────────────────────────
#  Clip generation
# ────────────────────────────────────────────────────────────────────────────

def _clips_uniform(n_frames, clip_len, stride, video_meta) -> List[dict]:
    """Fixed-stride sliding window."""
    clips = []
    for s in range(0, n_frames - clip_len + 1, stride):
        clips.append({**video_meta, "start_frame": s, "end_frame": s + clip_len})
    return clips


def _clips_random(n_frames, clip_len, n_clips, video_meta, seed_str) -> List[dict]:
    """Sample n_clips random start positions (deterministic per video)."""
    max_start = n_frames - clip_len
    if max_start < 0:
        return []
    h = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**31)
    rng = np.random.RandomState(h)
    starts = sorted(rng.randint(0, max_start + 1, size=n_clips))
    clips = []
    for s in starts:
        clips.append({**video_meta, "start_frame": int(s), "end_frame": int(s) + clip_len})
    return clips


def _generate_clips_for_video(
    v: dict, clip_len: int, stride: int,
    clip_sampling: str, max_clips_per_video: int,
) -> List[dict]:
    """Generate clip metadata for a single video using its ``n_frames``.

    When ``v["split"] == "all"`` (deploy mode), clips are generated for
    train/val/test by temporal position within the video (70/15/15).
    Returns clips with the correct ``"split"`` key set per clip.
    """
    n_frames = v.get("n_frames", 0)
    if n_frames <= 0:
        return []

    meta = {
        "label": v["label"], "label_name": v["letter"],
        "video_name": v["video_id"], "video_id": v["video_id"],
        "domain": v["domain"], "fiber": v["fiber"], "filename": v["filename"],
    }

    if v.get("split") == "all":
        # Deploy mode: temporal split within this video
        t_end = int(n_frames * DEPLOY_TRAIN_RATIO)
        v_end = int(n_frames * (DEPLOY_TRAIN_RATIO + DEPLOY_VAL_RATIO))
        clips = []
        for split_tag, s_start, s_end in [
            ("train", 0, t_end), ("val", t_end, v_end), ("test", v_end, n_frames),
        ]:
            is_train = split_tag == "train"
            use_random = is_train and clip_sampling == "random"
            if use_random:
                n_uniform = max(1, (s_end - s_start - clip_len) // stride + 1)
                n_want = min(n_uniform, max_clips_per_video) if max_clips_per_video > 0 else n_uniform
                seg_clips = _clips_random(s_end - s_start, clip_len, n_want, meta, v["video_id"] + split_tag)
                for c in seg_clips:
                    c["start_frame"] += s_start
                    c["end_frame"]   += s_start
            else:
                seg_clips = []
                for s in range(s_start, s_end - clip_len + 1, stride):
                    seg_clips.append({**meta, "start_frame": s, "end_frame": s + clip_len})
            if not seg_clips and (s_end - s_start) > 0:
                seg_clips = [{**meta, "start_frame": s_start, "end_frame": s_end}]
            if max_clips_per_video > 0 and len(seg_clips) > max_clips_per_video:
                step = len(seg_clips) / max_clips_per_video
                seg_clips = [seg_clips[int(j * step)] for j in range(max_clips_per_video)]
            for c in seg_clips:
                c["_split"] = split_tag
            clips.extend(seg_clips)
        return clips

    # Non-deploy: whole video belongs to one split
    is_train = v.get("split") == "train"
    use_random = is_train and clip_sampling == "random"

    if use_random:
        n_uniform = max(1, (n_frames - clip_len) // stride + 1)
        n_want = min(n_uniform, max_clips_per_video) if max_clips_per_video > 0 else n_uniform
        clips = _clips_random(n_frames, clip_len, n_want, meta, v["video_id"])
    else:
        clips = _clips_uniform(n_frames, clip_len, stride, meta)

    if not clips and n_frames > 0:
        clips = [{**meta, "start_frame": 0, "end_frame": n_frames}]

    if max_clips_per_video > 0 and len(clips) > max_clips_per_video:
        step = len(clips) / max_clips_per_video
        clips = [clips[int(j * step)] for j in range(max_clips_per_video)]

    return clips


def prepare_unified_data(
    videos: List[dict],
    clip_len: int = 16,
    stride: int = 8,
    img_size: int = 224,
    input_mode: str = "rgb",
    clip_sampling: str = "uniform",
    max_clips_per_video: int = 0,
    cache_dir: Optional[str] = None,
    load_workers: int = 0,
) -> Tuple[Dict[str, np.ndarray], List[dict], List[dict], List[dict]]:
    """
    Load all videos into memory and generate clip metadata per split.

    Parameters
    ----------
    clip_sampling : ``"uniform"`` | ``"random"``
        ``uniform`` — fixed stride (deterministic, used for eval).
        ``random``  — random start positions (train split only).
    max_clips_per_video : int
        If >0, cap clips per video.
    cache_dir : str or None
        If set, decoded frames are cached as ``.npy`` files under this
        directory.  Second run loads from cache (~20 s vs ~6 min).
    load_workers : int
        Number of threads for parallel frame loading/decoding.
        0 = sequential (single thread).

    Returns
    -------
    all_frames, train_clips, val_clips, test_clips
    """
    # ── Phase 1: generate clip indices (fast, uses n_frames from manifest) ─
    t0 = time.perf_counter()
    split_clips: Dict[str, list] = {"train": [], "val": [], "test": []}
    valid_videos = [v for v in videos if v.get("n_frames", 0) > 0]

    for v in valid_videos:
        clips = _generate_clips_for_video(v, clip_len, stride, clip_sampling, max_clips_per_video)
        if v.get("split") == "all":
            for c in clips:
                split_clips[c.pop("_split")].append(c)
        else:
            split_clips[v["split"]].extend(clips)

    n_clips = sum(len(c) for c in split_clips.values())
    clip_time = time.perf_counter() - t0
    print(f"  Generated {n_clips:,} clips [{clip_time:.2f}s]")

    # ── Phase 2: load / decode frames (parallel, .npy cached) ──────────
    t1 = time.perf_counter()
    all_frames: Dict[str, np.ndarray] = {}
    skipped = 0
    n_total = len(valid_videos)
    n_cache_hits = 0

    if load_workers >= 2:
        args_list = [(v, img_size, input_mode, cache_dir) for v in valid_videos]
        with ThreadPoolExecutor(max_workers=load_workers) as pool:
            futures = [pool.submit(_load_video_worker, a) for a in args_list]
            for i, fut in enumerate(as_completed(futures)):
                vid_id, frames = fut.result()
                if frames is None:
                    skipped += 1
                else:
                    all_frames[vid_id] = frames
                done = i + 1
                if done % 50 == 0 or done == n_total:
                    print(f"    Loaded {done}/{n_total} videos ...")
    else:
        for i, v in enumerate(valid_videos):
            if (i + 1) % 50 == 0 or i == 0 or i == n_total - 1:
                print(f"    Loading [{i+1}/{n_total}] {v['video_id']}")
            try:
                if cache_dir:
                    frames = _load_or_decode_video(v, img_size, input_mode, cache_dir)
                else:
                    frames = load_video_frames(v["path"], img_size, mode=input_mode)
            except RuntimeError as e:
                print(f"  [WARNING] {e}")
                skipped += 1
                continue
            all_frames[v["video_id"]] = frames

    load_time = time.perf_counter() - t1
    if cache_dir:
        n_cache_hits = sum(
            1 for v in valid_videos
            if os.path.isfile(_npy_cache_path(v["video_id"], img_size, input_mode, cache_dir))
        )
        print(f"  Frames loaded [{load_time:.1f}s]  "
              f"(cache hits: {n_cache_hits}/{n_total})")
    else:
        print(f"  Frames loaded [{load_time:.1f}s]  (no cache)")

    if skipped:
        print(f"  [WARNING] Skipped {skipped} unreadable video(s)")

    # Remove clips whose video failed to load
    loaded_ids = set(all_frames.keys())
    for split in split_clips:
        split_clips[split] = [c for c in split_clips[split] if c["video_id"] in loaded_ids]

    return (
        all_frames,
        split_clips["train"],
        split_clips["val"],
        split_clips["test"],
    )


# ────────────────────────────────────────────────────────────────────────────
#  PyTorch Dataset
# ────────────────────────────────────────────────────────────────────────────

class UnifiedSpeckleDataset(Dataset):
    """
    Multi-domain speckle clip dataset.

    Supports both ``rgb`` and ``gray`` input modes.  Output shape is always
    ``(T, 3, H, W)`` float32 with ImageNet normalisation.
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        clips: List[dict],
        all_frames: Dict[str, np.ndarray],
        clip_len: int,
        input_mode: str = "rgb",
        augment: bool = False,
    ):
        self.clips = clips
        self.all_frames = all_frames
        self.clip_len = clip_len
        self.input_mode = input_mode
        self.augment = augment

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        info = self.clips[idx]
        video = self.all_frames[info["video_name"]]
        start, end = info["start_frame"], info["end_frame"]
        frames = video[start:end].copy()

        if len(frames) < self.clip_len:
            pad = np.repeat(frames[-1:], self.clip_len - len(frames), axis=0)
            frames = np.concatenate([frames, pad], axis=0)

        frames = frames.astype(np.float32) / 255.0

        if self.input_mode == "gray":
            if self.augment:
                frames = self._augment_gray(frames)
            frames_3ch = np.stack([frames, frames, frames], axis=1)
        else:
            if self.augment:
                frames = self._augment_rgb(frames)
            frames_3ch = frames.transpose(0, 3, 1, 2)

        for c in range(3):
            frames_3ch[:, c] = (frames_3ch[:, c] - self.MEAN[c]) / self.STD[c]

        return torch.from_numpy(frames_3ch), info["label"]

    @staticmethod
    def _augment_gray(frames: np.ndarray) -> np.ndarray:
        if np.random.random() > 0.5:
            frames = frames[:, :, ::-1].copy()
        if np.random.random() > 0.5:
            frames = frames[:, ::-1, :].copy()
        b = 1.0 + np.random.uniform(-0.1, 0.1)
        return np.clip(frames * b, 0.0, 1.0)

    @staticmethod
    def _augment_rgb(frames: np.ndarray) -> np.ndarray:
        if np.random.random() > 0.5:
            frames = frames[:, :, ::-1, :].copy()
        if np.random.random() > 0.5:
            frames = frames[:, ::-1, :, :].copy()
        b = 1.0 + np.random.uniform(-0.1, 0.1)
        frames = np.clip(frames * b, 0.0, 1.0)
        if np.random.random() > 0.5:
            jitter = np.random.uniform(0.9, 1.1, size=(1, 1, 1, 3)).astype(np.float32)
            frames = np.clip(frames * jitter, 0.0, 1.0)
        return frames


# ────────────────────────────────────────────────────────────────────────────
#  Leakage verification
# ────────────────────────────────────────────────────────────────────────────

def verify_no_leakage(
    train_clips: List[dict],
    val_clips: List[dict],
    test_clips: List[dict],
) -> dict:
    """
    Verify no video_id appears in more than one split.

    Returns a dict with detailed results suitable for JSON export.
    """
    train_vids = sorted(set(c["video_id"] for c in train_clips))
    val_vids   = sorted(set(c["video_id"] for c in val_clips))
    test_vids  = sorted(set(c["video_id"] for c in test_clips))

    train_set, val_set, test_set = set(train_vids), set(val_vids), set(test_vids)
    tv = sorted(train_set & val_set)
    tt = sorted(train_set & test_set)
    vt = sorted(val_set & test_set)

    ok = not (tv or tt or vt)
    for tag, overlap in [("train∩val", tv), ("train∩test", tt), ("val∩test", vt)]:
        if overlap:
            print(f"  [LEAKAGE] {tag}: {overlap}")

    status = "PASS" if ok else "FAIL"
    print(f"  Leakage check: {status}")

    return {
        "status": status,
        "train_video_count": len(train_vids),
        "val_video_count": len(val_vids),
        "test_video_count": len(test_vids),
        "train_videos": train_vids,
        "val_videos": val_vids,
        "test_videos": test_vids,
        "overlap_train_val": tv,
        "overlap_train_test": tt,
        "overlap_val_test": vt,
    }


# ────────────────────────────────────────────────────────────────────────────
#  Split summary
# ────────────────────────────────────────────────────────────────────────────

def build_split_summary(
    videos: List[dict],
    train_clips: List[dict],
    val_clips: List[dict],
    test_clips: List[dict],
    split_mode: str = "cross_fiber",
    leakage_result: dict = None,
) -> dict:
    """Build a JSON-serialisable summary of the dataset split."""
    summary: dict = {
        "split_mode": split_mode,
        "total_videos": len(videos),
        "total_clips": {
            "train": len(train_clips), "val": len(val_clips), "test": len(test_clips),
        },
    }

    for split_name, clips in [("train", train_clips), ("val", val_clips), ("test", test_clips)]:
        by_domain = defaultdict(int)
        by_fiber  = defaultdict(int)
        by_letter = defaultdict(int)
        vid_ids: set = set()
        for c in clips:
            by_domain[c["domain"]] += 1
            by_fiber[c["fiber"]] += 1
            by_letter[c["label_name"]] += 1
            vid_ids.add(c["video_id"])
        summary[f"{split_name}_by_domain"] = dict(sorted(by_domain.items()))
        summary[f"{split_name}_by_fiber"]  = dict(sorted(by_fiber.items()))
        summary[f"{split_name}_by_letter"] = dict(sorted(by_letter.items()))
        summary[f"{split_name}_video_count"] = len(vid_ids)

    # Per-video assignment table
    vid_table = []
    for v in videos:
        vid_table.append({
            "video_id": v["video_id"],
            "domain": v["domain"],
            "fiber": v["fiber"],
            "letter": v["letter"],
            "n_frames": v.get("n_frames", -1),
            "split": v.get("split", "unknown"),
        })
    summary["video_assignments"] = vid_table

    if leakage_result is not None:
        summary["leakage_check"] = leakage_result

    return summary


def save_split_summary(summary: dict, output_dir: str) -> str:
    path = os.path.join(output_dir, "split_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Split summary saved: {path}")
    return path


# ────────────────────────────────────────────────────────────────────────────
#  Per-group accuracy helpers
# ────────────────────────────────────────────────────────────────────────────

def compute_group_accuracy(
    clips: List[dict], preds: list, labels: list, group_key: str,
) -> Dict[str, dict]:
    """Compute accuracy grouped by *group_key* (``"domain"`` / ``"fiber"`` / ``"label_name"``)."""
    groups: Dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for i, clip in enumerate(clips):
        g = clip[group_key]
        groups[g]["total"] += 1
        if preds[i] == labels[i]:
            groups[g]["correct"] += 1
    for g in groups:
        t = groups[g]["total"]
        groups[g]["accuracy"] = 100.0 * groups[g]["correct"] / t if t else 0.0
    return dict(sorted(groups.items()))


# ────────────────────────────────────────────────────────────────────────────
#  Domain × Fiber accuracy table
# ────────────────────────────────────────────────────────────────────────────

def build_accuracy_table(
    clips: List[dict], preds: list, labels: list,
) -> Tuple[dict, List[str], List[str]]:
    """
    Build a domain × fiber accuracy table.

    Returns
    -------
    table : ``{domain: {fiber: accuracy_float_or_None}}``
    domains : sorted list of domains present
    fibers  : sorted list of fibers present
    """
    cells: Dict[Tuple[str,str], dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for i, c in enumerate(clips):
        key = (c["domain"], c["fiber"])
        cells[key]["total"] += 1
        if preds[i] == labels[i]:
            cells[key]["correct"] += 1

    domains = sorted(set(c["domain"] for c in clips))
    fibers  = sorted(set(c["fiber"]  for c in clips))

    table: Dict[str, Dict[str, Optional[float]]] = {}
    for d in domains:
        table[d] = {}
        for f in fibers:
            info = cells.get((d, f))
            if info and info["total"] > 0:
                table[d][f] = round(100.0 * info["correct"] / info["total"], 2)
            else:
                table[d][f] = None

    return table, domains, fibers


def print_accuracy_table(table: dict, domains: list, fibers: list):
    """Pretty-print the domain × fiber accuracy table with averages."""
    col_w = max(12, max((len(f) for f in fibers), default=8) + 2)
    dom_w = max(24, max((len(d) for d in domains), default=8) + 2)

    header = f"  {'Domain':<{dom_w}}" + "".join(f"{f:>{col_w}}" for f in fibers) + f"{'Avg':>{col_w}}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    overall_sum = 0.0
    overall_cnt = 0

    for d in domains:
        row = f"  {d:<{dom_w}}"
        vals = []
        for f in fibers:
            v = table[d][f]
            if v is not None:
                row += f"{v:>{col_w}.2f}"
                vals.append(v)
            else:
                row += f"{'--':>{col_w}}"
        avg = sum(vals) / len(vals) if vals else 0
        row += f"{avg:>{col_w}.2f}"
        print(row)
        overall_sum += sum(vals)
        overall_cnt += len(vals)

    # Fiber averages row
    row = f"  {'Avg':<{dom_w}}"
    for f in fibers:
        col_vals = [table[d][f] for d in domains if table[d][f] is not None]
        if col_vals:
            row += f"{sum(col_vals)/len(col_vals):>{col_w}.2f}"
        else:
            row += f"{'--':>{col_w}}"
    overall = overall_sum / overall_cnt if overall_cnt else 0
    row += f"{overall:>{col_w}.2f}"
    print(row)


def save_accuracy_table(
    table: dict, domains: list, fibers: list, output_dir: str,
) -> Tuple[str, str]:
    """Save the domain × fiber table as CSV and JSON. Returns (csv_path, json_path)."""

    # ── CSV ──
    csv_path = os.path.join(output_dir, "accuracy_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["domain"] + fibers + ["average"])
        for d in domains:
            vals = [table[d][fi] for fi in fibers]
            nums = [v for v in vals if v is not None]
            avg = round(sum(nums) / len(nums), 2) if nums else ""
            w.writerow([d] + [v if v is not None else "" for v in vals] + [avg])
        # Fiber avg row
        avgs = []
        for fi in fibers:
            col = [table[d][fi] for d in domains if table[d][fi] is not None]
            avgs.append(round(sum(col) / len(col), 2) if col else "")
        all_nums = [table[d][fi] for d in domains for fi in fibers if table[d][fi] is not None]
        overall = round(sum(all_nums) / len(all_nums), 2) if all_nums else ""
        w.writerow(["average"] + avgs + [overall])
    print(f"  Accuracy table CSV: {csv_path}")

    # ── JSON ──
    json_path = os.path.join(output_dir, "accuracy_table.json")
    domain_avgs = {}
    for d in domains:
        nums = [table[d][fi] for fi in fibers if table[d][fi] is not None]
        domain_avgs[d] = round(sum(nums) / len(nums), 2) if nums else None
    fiber_avgs = {}
    for fi in fibers:
        col = [table[d][fi] for d in domains if table[d][fi] is not None]
        fiber_avgs[fi] = round(sum(col) / len(col), 2) if col else None
    all_nums = [table[d][fi] for d in domains for fi in fibers if table[d][fi] is not None]
    overall = round(sum(all_nums) / len(all_nums), 2) if all_nums else None

    obj = {
        "cells": table,
        "domain_averages": domain_avgs,
        "fiber_averages": fiber_avgs,
        "overall_average": overall,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  Accuracy table JSON: {json_path}")

    return csv_path, json_path


# ────────────────────────────────────────────────────────────────────────────
#  Misc display helpers
# ────────────────────────────────────────────────────────────────────────────

def print_split_table(
    videos: List[dict],
    train_clips: List[dict],
    val_clips: List[dict],
    test_clips: List[dict],
):
    """Print a concise overview of the video / clip split."""
    counts = defaultdict(lambda: defaultdict(int))
    for v in videos:
        counts[v.get("split", "?")][v["domain"]] += 1

    domains = list(DOMAIN_FOLDERS.values())
    header = f"  {'Split':<8}" + "".join(f"{d:>22}" for d in domains) + f"{'Total':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for split in ("train", "val", "test"):
        row = f"  {split:<8}"
        total = 0
        for d in domains:
            n = counts[split][d]
            total += n
            row += f"{n:>22}"
        row += f"{total:>10}"
        print(row)

    print(f"\n  Clips: train={len(train_clips):,}  val={len(val_clips):,}  test={len(test_clips):,}")
