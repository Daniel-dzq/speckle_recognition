"""
Inference worker thread.

Receives frames from the camera worker, accumulates them into a sliding
window, and runs the loaded speckle recognition model periodically.

Features:
  - Sliding window frame buffer (size = clip_len)
  - Configurable inference interval (every N new frames)
  - Majority-vote smoothing over recent predictions
  - EMA confidence smoothing
  - Top-k class output
"""

import os
import sys
import collections
import numpy as np
import time

import cv2
import torch
import torch.nn.functional as F
from PySide6.QtCore import QThread, Signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import get_model


class InferenceWorker(QThread):
    """
    Background inference thread.

    Signals:
        prediction_ready(dict):  Emitted after each inference run.
            dict keys:
                'top1'       : str   - top-1 predicted class name
                'confidence' : float - top-1 confidence (0-1)
                'topk'       : list  - [(class_name, prob), ...] for top-k
                'smoothed'   : str   - majority-vote smoothed prediction
                'frame_count': int
        model_loaded(str):  Emitted when a model is successfully loaded
        error(str):         Emitted on error
    """

    prediction_ready = Signal(dict)
    model_loaded     = Signal(str)
    error            = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._model       = None
        self._class_names = []
        self._clip_len    = 16
        self._img_size    = 224
        self._device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._frame_buffer  = collections.deque(maxlen=self._clip_len)
        self._vote_buffer   = collections.deque(maxlen=10)   # last N predictions for voting
        self._new_frame_cnt = 0
        self._infer_every   = 4    # run inference every N new frames
        self._top_k         = 5

        self._running = False

        # ImageNet normalization stats (matches dataset.py)
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # ── Model loading ──────────────────────────────────────────────────

    def load_model(self, checkpoint_path: str) -> bool:
        """Load model from checkpoint. Thread-safe (call from main thread before start)."""
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            model_type  = ckpt.get("model_type",  "cnn_pool")
            num_classes = ckpt.get("num_classes",  26)
            class_names = ckpt.get("class_names",  [chr(65 + i) for i in range(26)])
            clip_len    = ckpt.get("clip_len",      16)
            img_size    = ckpt.get("img_size",      224)
            fiber_name  = ckpt.get("fiber_name",    "unknown")

            model = get_model(model_type, num_classes, pretrained=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            model.to(self._device)

            self._model       = model
            self._class_names = class_names
            self._clip_len    = clip_len
            self._img_size    = img_size
            self._frame_buffer = collections.deque(maxlen=clip_len)
            self._vote_buffer.clear()

            msg = (f"Model loaded: {fiber_name}  |  {model_type}  |  "
                   f"classes={num_classes}  clip_len={clip_len}  "
                   f"device={self._device}")
            self.model_loaded.emit(msg)
            return True

        except Exception as e:
            self.error.emit(f"Model load failed: {e}")
            return False

    # ── Configuration ──────────────────────────────────────────────────

    def set_infer_every(self, n: int):
        self._infer_every = max(1, n)

    def set_top_k(self, k: int):
        self._top_k = max(1, k)

    def set_vote_window(self, n: int):
        maxlen = max(1, n)
        old = list(self._vote_buffer)
        self._vote_buffer = collections.deque(old[-maxlen:], maxlen=maxlen)

    # ── Frame ingestion ────────────────────────────────────────────────

    def push_frame(self, frame: np.ndarray):
        """
        Called from the main thread whenever a new camera frame arrives.
        Preprocesses and buffers the frame; triggers inference periodically.
        """
        if self._model is None:
            return

        processed = self._preprocess_frame(frame)
        self._frame_buffer.append(processed)
        self._new_frame_cnt += 1

        if (self._new_frame_cnt >= self._infer_every and
                len(self._frame_buffer) >= self._clip_len):
            self._new_frame_cnt = 0
            self._run_inference()

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """BGR uint8 -> normalized float32 (3, H, W) matching dataset.py."""
        # Resize
        resized = cv2.resize(frame, (self._img_size, self._img_size),
                             interpolation=cv2.INTER_AREA)
        # Convert to gray then replicate to 3ch (matches SpeckleClipDataset)
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        gray_f = gray.astype(np.float32) / 255.0  # (H, W)

        # 3 channels
        frame_3ch = np.stack([gray_f, gray_f, gray_f], axis=0)  # (3, H, W)
        for c in range(3):
            frame_3ch[c] = (frame_3ch[c] - self._mean[c]) / self._std[c]
        return frame_3ch

    # ── Inference ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_inference(self):
        frames = list(self._frame_buffer)[-self._clip_len:]
        if len(frames) < self._clip_len:
            return

        clip = np.stack(frames, axis=0)                  # (T, 3, H, W)
        tensor = torch.from_numpy(clip).unsqueeze(0)     # (1, T, 3, H, W)
        tensor = tensor.to(self._device, non_blocking=True)

        try:
            logits = self._model(tensor)                  # (1, num_classes)
            probs  = F.softmax(logits, dim=1)[0]          # (num_classes,)

            top_probs, top_idxs = torch.topk(probs, min(self._top_k, len(self._class_names)))
            top1_name = self._class_names[top_idxs[0].item()]
            top1_conf = top_probs[0].item()

            topk = [
                (self._class_names[idx.item()], prob.item())
                for idx, prob in zip(top_idxs, top_probs)
            ]

            # Voting smoothing
            self._vote_buffer.append(top1_name)
            counts  = collections.Counter(self._vote_buffer)
            smoothed = counts.most_common(1)[0][0]

            self.prediction_ready.emit({
                "top1":        top1_name,
                "confidence":  top1_conf,
                "topk":        topk,
                "smoothed":    smoothed,
                "frame_count": len(self._frame_buffer),
            })

        except Exception as e:
            self.error.emit(f"Inference error: {e}")

    # ── Thread run (not actually used as a thread - worker is driven by signals) ──

    def run(self):
        """Not used directly - inference is triggered via push_frame() from main thread."""
        self._running = True
        while self._running:
            time.sleep(0.1)

    def stop(self):
        self._running = False
        self.wait(2000)
