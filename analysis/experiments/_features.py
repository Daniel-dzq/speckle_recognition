"""
Shared capture -> feature extraction pipeline.

Every experiment that needs per-capture feature vectors/images pumps captures
through :func:`extract_features`. The function:

* reads the right number of frames via the lazy video helpers
* pipes them through :class:`Pipeline`
* aggregates per-capture per the preprocessing config
* caches the result on disk (keyed by path + pipeline hash)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..caching.cache import FeatureCache
from ..io.video import read_frame_indices, read_frames, video_frame_count
from ..preprocessing.pipeline import Pipeline, PreprocessConfig
from ..utils.types import Capture


@dataclass
class CaptureFeature:
    """Per-capture feature bundle produced by the preprocessing pipeline."""

    capture: Capture
    image: np.ndarray        # 2-D aggregated image (after preprocessing)
    vector: np.ndarray       # flattened float32 view of ``image``
    n_frames: int
    frame_indices: List[int]

    @property
    def key(self) -> Tuple[str, str, str, Optional[str], Optional[str], Optional[int]]:
        c = self.capture
        return (c.fiber, c.channel, c.challenge, c.condition, c.session, c.repeat)


def _pp_cache_key(capture: Capture, pp_cfg: PreprocessConfig, extra: Dict) -> str:
    d = pp_cfg.to_dict()
    d.update(extra)
    payload = json.dumps({
        "path": str(capture.path),
        "pp": d,
    }, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def extract_features(
    captures: Iterable[Capture],
    pp_cfg: PreprocessConfig,
    *,
    cache: Optional[FeatureCache] = None,
    frame_strategy: Optional[str] = None,
    n_frames: Optional[int] = None,
    average: Optional[bool] = None,
    logger: Optional[logging.Logger] = None,
) -> List[CaptureFeature]:
    """Batch feature extraction."""
    pipeline = Pipeline(pp_cfg)
    out: List[CaptureFeature] = []
    strat = frame_strategy or pp_cfg.frame_strategy
    n = max(1, int(n_frames if n_frames is not None else pp_cfg.n_frames))
    agg_override = pp_cfg.aggregate
    if average is True:
        agg_override = "mean"
    elif average is False:
        agg_override = "first"

    for cap in captures:
        extra = {"strategy": strat, "n_frames": n, "aggregate": agg_override}
        key = _pp_cache_key(cap, pp_cfg, extra)

        if cache is not None:
            bundle = cache.get(key, source=cap.path)
            if bundle is not None and "image" in bundle:
                image = bundle["image"]
                indices = bundle.get("indices", np.array([], dtype=np.int64)).tolist()
                out.append(
                    CaptureFeature(
                        capture=cap,
                        image=image,
                        vector=image.ravel().astype(np.float32, copy=False),
                        n_frames=image.shape[0] if image.ndim == 3 else 1,
                        frame_indices=[int(i) for i in indices],
                    )
                )
                continue

        try:
            n_total = video_frame_count(cap.path)
        except Exception as exc:
            if logger:
                logger.warning("Skipping %s (failed to read frame count: %s)", cap.path, exc)
            continue
        if n_total <= 0:
            if logger:
                logger.warning("Skipping %s (no frames)", cap.path)
            continue

        indices = read_frame_indices(n_total, n, strategy=strat)
        try:
            frames = read_frames(cap.path, indices, grayscale=pp_cfg.grayscale)
        except Exception as exc:
            if logger:
                logger.warning("Skipping %s (read failed: %s)", cap.path, exc)
            continue
        if not frames:
            continue

        processed = [pipeline.apply_frame(f) for f in frames]
        # honour per-call aggregation override
        saved_aggregate = pipeline.config.aggregate
        pipeline.config.aggregate = agg_override
        try:
            image = pipeline.aggregate_frames(processed)
        finally:
            pipeline.config.aggregate = saved_aggregate
        vector = image.ravel().astype(np.float32, copy=False)

        if cache is not None:
            cache.put(
                key,
                payload={"image": image, "indices": np.asarray(indices, dtype=np.int64)},
                source=cap.path,
                meta={"fiber": cap.fiber, "channel": cap.channel, "challenge": cap.challenge},
            )

        out.append(
            CaptureFeature(
                capture=cap,
                image=image,
                vector=vector,
                n_frames=len(indices),
                frame_indices=list(indices),
            )
        )

    return out


__all__ = ["CaptureFeature", "extract_features"]
