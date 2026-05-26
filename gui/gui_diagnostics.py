"""
Optional live-GUI diagnostics (enable with SPECKLE_GUI_DIAGNOSTICS=1).

Writes under outputs/gui_diagnostics/session_<timestamp>/:
  - raw frames, preprocess montage, tensor stats, prediction_log.csv
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAG_ROOT = os.path.join(ROOT, "outputs", "gui_diagnostics")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def diagnostics_enabled() -> bool:
    v = os.environ.get("SPECKLE_GUI_DIAGNOSTICS", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def preprocess_frame_like_inference(
    frame: np.ndarray,
    *,
    img_size: int,
    input_mode: str,
) -> np.ndarray:
    """Match gui/inference_worker._preprocess_frame (BGR uint8 in)."""
    resized = cv2.resize(
        frame, (img_size, img_size), interpolation=cv2.INTER_AREA
    )
    if input_mode == "rgb":
        if len(resized.shape) == 2:
            rgb = np.stack([resized, resized, resized], axis=-1)
        else:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw = rgb.astype(np.float32) / 255.0
        chw = chw.transpose(2, 0, 1)
    else:
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        gray_f = gray.astype(np.float32) / 255.0
        chw = np.stack([gray_f, gray_f, gray_f], axis=0)
    for c in range(3):
        chw[c] = (chw[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    return chw


def tensor_stats(arr: np.ndarray) -> Dict[str, Any]:
    flat = arr.astype(np.float64).ravel()
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "p01": float(np.percentile(flat, 1)),
        "p50": float(np.percentile(flat, 50)),
        "p99": float(np.percentile(flat, 99)),
    }


def frame_stats_uint8(frame: np.ndarray) -> Dict[str, Any]:
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    flat = gray.astype(np.float64).ravel()
    return {
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "p01": float(np.percentile(flat, 1)),
        "p50": float(np.percentile(flat, 50)),
        "p99": float(np.percentile(flat, 99)),
    }


class GuiDiagnosticsSession:
    """Singleton-style session for one GUI run."""

    _instance: Optional["GuiDiagnosticsSession"] = None

    def __init__(self) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(DIAG_ROOT, f"session_{ts}")
        os.makedirs(self.session_dir, exist_ok=True)
        self.raw_dir = os.path.join(self.session_dir, "raw_frames")
        os.makedirs(self.raw_dir, exist_ok=True)
        self.prediction_csv = os.path.join(self.session_dir, "prediction_log.csv")
        self._pred_header_written = False
        self._raw_count = 0
        self._first_raw_logged = False
        self.model_meta: Dict[str, Any] = {}
        self.auth_challenge = ""
        print(f"[GUI diagnostics] session_dir={self.session_dir}", flush=True)

    @classmethod
    def get(cls) -> Optional["GuiDiagnosticsSession"]:
        if not diagnostics_enabled():
            return None
        if cls._instance is None:
            cls._instance = GuiDiagnosticsSession()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def set_model_meta(
        self,
        *,
        checkpoint_path: str,
        fiber_name: str,
        clip_len: int,
        img_size: int,
        input_mode: str,
        class_names: List[str],
    ) -> None:
        self.model_meta = {
            "checkpoint_path": checkpoint_path,
            "fiber_name": fiber_name,
            "clip_len": clip_len,
            "img_size": img_size,
            "input_mode": input_mode,
            "class_names": class_names,
        }
        path = os.path.join(self.session_dir, "model_meta.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.model_meta, fh, indent=2)

    def set_auth_challenge(self, label: str) -> None:
        self.auth_challenge = (label or "").strip()

    def log_raw_frame(self, frame: np.ndarray, *, tag: str = "live") -> None:
        if self._raw_count >= 32:
            return
        self._raw_count += 1
        name = f"raw_{tag}_{self._raw_count:03d}.png"
        path = os.path.join(self.raw_dir, name)
        if frame.ndim == 2:
            cv2.imwrite(path, frame)
        else:
            cv2.imwrite(path, frame)
        if not self._first_raw_logged:
            stats = frame_stats_uint8(frame)
            stats["path"] = path
            stats_path = os.path.join(self.session_dir, "first_raw_frame_stats.json")
            with open(stats_path, "w", encoding="utf-8") as fh:
                json.dump(stats, fh, indent=2)
            print(
                f"[GUI diagnostics] first raw frame shape={stats['shape']} "
                f"mean={stats['mean']:.2f}",
                flush=True,
            )
            self._first_raw_logged = True

    def log_inference_clip(
        self,
        *,
        raw_frames: List[np.ndarray],
        processed_frames: List[np.ndarray],
        tensor: np.ndarray,
        prediction: Dict[str, Any],
        infer_every: int,
        buffer_len: int,
    ) -> None:
        clip_len = len(processed_frames)
        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "auth_challenge": self.auth_challenge,
            "model": self.model_meta,
            "clip_len": clip_len,
            "infer_every": infer_every,
            "buffer_len": buffer_len,
            "tensor": tensor_stats(tensor),
            "processed_per_frame": [tensor_stats(f) for f in processed_frames],
            "raw_per_frame": [frame_stats_uint8(f) for f in raw_frames],
        }
        out = os.path.join(self.session_dir, "preprocess_summary.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        self._save_montage(raw_frames, "raw_clip_montage.png", uint8=True)
        proc_vis = self._processed_to_vis(processed_frames)
        self._save_montage(proc_vis, "preprocessed_clip_montage.png", uint8=True)

        self._append_prediction(prediction)

    def _processed_to_vis(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        out = []
        for chw in frames:
            denorm = chw.copy().astype(np.float32)
            for c in range(3):
                denorm[c] = denorm[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
            denorm = np.clip(denorm, 0.0, 1.0)
            gray = denorm[0]
            vis = (gray * 255.0).astype(np.uint8)
            out.append(vis)
        return out

    def _save_montage(
        self,
        frames: List[np.ndarray],
        filename: str,
        *,
        uint8: bool,
    ) -> None:
        if not frames:
            return
        n = len(frames)
        cols = 4
        rows = int(np.ceil(n / cols))
        cell = 112
        canvas = np.zeros((rows * cell, cols * cell), dtype=np.uint8)
        for i, fr in enumerate(frames[: rows * cols]):
            r, c = divmod(i, cols)
            if fr.ndim == 3:
                g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if uint8 else fr[..., 0]
            else:
                g = fr
            thumb = cv2.resize(g, (cell, cell), interpolation=cv2.INTER_AREA)
            if not uint8:
                thumb = np.clip(thumb, 0, 255).astype(np.uint8)
            canvas[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = thumb
        cv2.imwrite(os.path.join(self.session_dir, filename), canvas)

    def _append_prediction(self, prediction: Dict[str, Any]) -> None:
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "auth_challenge": self.auth_challenge,
            "top1": prediction.get("top1", ""),
            "confidence": prediction.get("confidence", ""),
            "smoothed": prediction.get("smoothed", ""),
            "frame_count": prediction.get("frame_count", ""),
        }
        write_header = not self._pred_header_written
        with open(self.prediction_csv, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._pred_header_written = True
            writer.writerow(row)
