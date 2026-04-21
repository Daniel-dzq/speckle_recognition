"""
Thin wrappers around OpenCV for lazy video frame I/O.

All functions operate on a single file and never hold more than a handful of
frames in memory at once. Indices are validated and de-duplicated so the
caller can request e.g. ``[0, n-1, n//2]`` without thinking about order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Union

import numpy as np

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def _import_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "opencv-python is required for video/image I/O. Install with `pip install opencv-python`."
        ) from exc


def video_frame_count(path: Union[str, Path]) -> int:
    """Return ``cv2.CAP_PROP_FRAME_COUNT`` for videos, ``1`` for images."""
    cv2 = _import_cv2()
    p = Path(path)
    if _is_image(p):
        return 1
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {p}")
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def iter_video_frames(
    path: Union[str, Path],
    grayscale: bool = False,
    max_frames: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """Stream every frame from a video lazily."""
    cv2 = _import_cv2()
    p = Path(path)
    if _is_image(p):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to read image: {p}")
        yield img
        return

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {p}")
    count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if grayscale and frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield frame
            count += 1
            if max_frames is not None and count >= max_frames:
                break
    finally:
        cap.release()


def read_frames(
    path: Union[str, Path],
    indices: Sequence[int],
    grayscale: bool = False,
) -> List[np.ndarray]:
    """
    Read a specific list of frame indices.

    Uses a single pass over the video, seeking only when the stream falls
    behind the next target index, which keeps performance close to linear.
    """
    if not indices:
        return []

    cv2 = _import_cv2()
    p = Path(path)
    if _is_image(p):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to read image: {p}")
        return [img.copy() for _ in indices]

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {p}")

    try:
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if n_total > 0:
            clamped = [max(0, min(i, n_total - 1)) for i in indices]
        else:
            clamped = list(indices)
        order = sorted(range(len(clamped)), key=lambda k: clamped[k])
        sorted_idx = [clamped[k] for k in order]

        out: List[Optional[np.ndarray]] = [None] * len(clamped)
        pos = -1
        for target, original_pos in zip(sorted_idx, order):
            if target != pos + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                pos = target - 1
            frame = None
            while pos < target:
                ret, frame = cap.read()
                if not ret:
                    frame = None
                    break
                pos += 1
            if frame is None:
                # Fall back to a seek read attempt.
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise IOError(f"Could not read frame {target} from {p}")
                pos = target
            if grayscale and frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out[original_pos] = frame.copy()
        return [f for f in out if f is not None]
    finally:
        cap.release()


def read_frame_indices(
    n_total: int,
    n_samples: int,
    strategy: str = "middle",
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """
    Pick frame indices for a sampling strategy.

    Strategies:
        ``middle``   - return ``[n_total // 2]`` (n_samples is ignored).
        ``uniform``  - evenly spaced indices from 0..n_total-1.
        ``random``   - ``n_samples`` random indices (seeded via ``rng``).
        ``all``      - return every frame up to a safety cap of ``n_samples``.
    """
    if n_total <= 0:
        return []
    n = max(1, int(n_samples))

    if strategy == "middle":
        return [n_total // 2]
    if strategy == "uniform":
        if n == 1:
            return [n_total // 2]
        return list(np.linspace(0, n_total - 1, n, dtype=int))
    if strategy == "random":
        generator = rng if rng is not None else np.random.default_rng(0)
        return sorted(int(x) for x in generator.choice(n_total, size=min(n, n_total), replace=False))
    if strategy == "all":
        return list(range(min(n_total, n)))

    raise ValueError(f"Unknown frame sampling strategy: {strategy!r}")


def read_representative_frame(
    path: Union[str, Path],
    grayscale: bool = False,
    strategy: str = "middle",
) -> np.ndarray:
    """Read a single representative frame for preview / template use."""
    p = Path(path)
    if _is_image(p):
        return read_frames(p, [0], grayscale=grayscale)[0]
    n = video_frame_count(p)
    idx = read_frame_indices(n, 1, strategy=strategy)
    frames = read_frames(p, idx, grayscale=grayscale)
    if not frames:
        raise IOError(f"No frame could be read from {p}")
    return frames[0]


__all__ = [
    "VIDEO_EXTS",
    "IMAGE_EXTS",
    "video_frame_count",
    "iter_video_frames",
    "read_frames",
    "read_frame_indices",
    "read_representative_frame",
]
