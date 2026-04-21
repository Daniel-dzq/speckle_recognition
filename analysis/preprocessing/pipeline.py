"""
Configurable, deterministic frame preprocessing.

Each step in the pipeline is a pure function on a 2-D or 3-D NumPy array.
:class:`PreprocessConfig` groups all knobs so the experiments only ever see a
single, typed, serialisable object.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[-1] == 3:
        # BGR -> luminance weights (OpenCV convention)
        f = frame.astype(np.float32)
        return (0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]).astype(frame.dtype)
    if frame.ndim == 3 and frame.shape[-1] == 4:
        return to_grayscale(frame[..., :3])
    raise ValueError(f"Cannot convert array of shape {frame.shape} to grayscale")


def center_crop(frame: np.ndarray, size: Optional[int]) -> np.ndarray:
    if size is None:
        return frame
    h, w = frame.shape[:2]
    side = int(size)
    if side <= 0 or side >= min(h, w):
        # Crop to the smallest side if the requested size is larger.
        side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame[y0:y0 + side, x0:x0 + side, ...]


def crop_roi(frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    """Crop ``(x, y, w, h)``. Values outside the frame are clamped."""
    if roi is None:
        return frame
    x0, y0, w, h = [int(v) for v in roi]
    H, W = frame.shape[:2]
    x0 = max(0, min(x0, W))
    y0 = max(0, min(y0, H))
    x1 = max(x0, min(x0 + w, W))
    y1 = max(y0, min(y0 + h, H))
    return frame[y0:y1, x0:x1, ...]


def resize_if_needed(frame: np.ndarray, size: Optional[int]) -> np.ndarray:
    if size is None:
        return frame
    h, w = frame.shape[:2]
    if h == size and w == size:
        return frame
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("opencv-python required for resize") from exc
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def normalize_intensity(frame: np.ndarray, mode: str = "minmax") -> np.ndarray:
    f = frame.astype(np.float32)
    if mode == "none" or mode is None:
        return f
    if mode == "minmax":
        mn = float(f.min())
        mx = float(f.max())
        rng = mx - mn
        if rng < 1e-8:
            return np.zeros_like(f)
        return (f - mn) / rng
    if mode == "zscore":
        mean = float(f.mean())
        std = float(f.std())
        if std < 1e-8:
            return np.zeros_like(f)
        return (f - mean) / std
    if mode == "unit":
        return f / 255.0
    raise ValueError(f"Unknown normalisation mode: {mode!r}")


def subtract_background(frame: np.ndarray, bg: Optional[np.ndarray]) -> np.ndarray:
    if bg is None:
        return frame
    out = frame.astype(np.float32) - bg.astype(np.float32)
    return np.clip(out, 0, None)


# ---------------------------------------------------------------------------
# Config + pipeline
# ---------------------------------------------------------------------------


@dataclass
class PreprocessConfig:
    """Typed configuration for :class:`Pipeline`."""

    grayscale: bool = True
    center_crop_size: Optional[int] = 400
    roi: Optional[Tuple[int, int, int, int]] = None
    resize: Optional[int] = 112
    normalize: str = "minmax"
    frame_strategy: str = "middle"
    n_frames: int = 1
    aggregate: str = "mean"          # mean | first | none
    subtract_background: bool = False
    background_path: Optional[str] = None

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> "PreprocessConfig":
        data = dict(cfg)
        roi = data.get("roi")
        if roi is not None:
            roi = tuple(int(x) for x in roi)
            if len(roi) != 4:
                raise ValueError("roi must be (x, y, w, h)")
        return cls(
            grayscale=bool(data.get("grayscale", True)),
            center_crop_size=data.get("center_crop_size"),
            roi=roi,
            resize=data.get("resize"),
            normalize=str(data.get("normalize", "minmax")),
            frame_strategy=str(data.get("frame_strategy", "middle")),
            n_frames=int(data.get("n_frames", 1)),
            aggregate=str(data.get("aggregate", "mean")),
            subtract_background=bool(data.get("subtract_background", False)),
            background_path=data.get("background_path"),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("roi") is not None:
            d["roi"] = list(d["roi"])
        return d


class Pipeline:
    """
    Apply the preprocessing steps to a single frame or a batch.

    The pipeline reads nothing, owns no global state, and is safe to share
    across threads / processes for parallel preprocessing.
    """

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self._background: Optional[np.ndarray] = None

    # ----- single-frame operations ----------------------------------------
    def apply_frame(self, frame: np.ndarray) -> np.ndarray:
        cfg = self.config
        f = frame
        if cfg.grayscale:
            f = to_grayscale(f)
        if cfg.subtract_background:
            f = subtract_background(f, self._background)
        if cfg.roi is not None:
            f = crop_roi(f, cfg.roi)
        if cfg.center_crop_size is not None:
            f = center_crop(f, cfg.center_crop_size)
        if cfg.resize is not None:
            f = resize_if_needed(f, cfg.resize)
        f = normalize_intensity(f, cfg.normalize)
        return f.astype(np.float32, copy=False)

    # ----- batch aggregation ---------------------------------------------
    def aggregate_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        if not frames:
            raise ValueError("Cannot aggregate an empty frame list")
        mode = self.config.aggregate
        if mode == "first":
            return frames[0]
        if mode == "mean":
            stack = np.stack(frames, axis=0)
            return stack.mean(axis=0)
        if mode == "none":
            return np.stack(frames, axis=0)
        raise ValueError(f"Unknown aggregation mode: {mode!r}")


__all__ = [
    "PreprocessConfig",
    "Pipeline",
    "to_grayscale",
    "center_crop",
    "crop_roi",
    "resize_if_needed",
    "normalize_intensity",
    "subtract_background",
]
