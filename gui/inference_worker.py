"""
Inference worker thread.

Receives frames from the camera worker via a bounded queue, accumulates them
into a sliding window, and runs the loaded speckle recognition model on this
thread—never blocking the Qt GUI thread.

Features:
  - Sliding window frame buffer (size = clip_len)
  - Configurable inference interval (every N new frames)
  - Majority-vote smoothing over recent predictions
  - EMA confidence smoothing
  - Top-k class output
"""

import os
import sys
import json
import queue
import collections
import threading
import numpy as np
import time

import cv2
import torch
import torch.nn.functional as F
from PySide6.QtCore import QThread, Signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import get_model

try:
    from gui.gui_diagnostics import GuiDiagnosticsSession, diagnostics_enabled
except ImportError:
    GuiDiagnosticsSession = None  # type: ignore

    def diagnostics_enabled() -> bool:
        return False


def _demo_trace(msg: str) -> None:
    if os.environ.get("SPECKLE_DEMO_TRACE", "").strip().lower() not in (
        "1", "true", "yes", "on",
    ):
        return
    t = threading.current_thread()
    ts = time.strftime("%H:%M:%S")
    print(
        f"[SPECKLE_DEMO_TRACE {ts} thread={t.name!r} ident={threading.get_ident()}] "
        f"{msg}",
        flush=True,
    )


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
    model_loaded = Signal(str)
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._model = None
        self._class_names = []
        self._clip_len = 16
        self._img_size = 224
        self._input_mode = "gray"    # "gray" (legacy) or "rgb" (unified model)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._frame_buffer = collections.deque(maxlen=self._clip_len)
        self._raw_frame_buffer = collections.deque(maxlen=self._clip_len)
        self._vote_buffer = collections.deque(maxlen=10)   # last N predictions for voting
        self._new_frame_cnt = 0
        self._infer_every = 4    # run inference every N new frames
        self._top_k = 5
        self._checkpoint_path = ""
        self._fiber_name = ""
        self._logged_first_inference = False
        self._diag_clip_saved = False
        self._warmup_skip_remaining = 0

        self._buf_lock = threading.Lock()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._quit = threading.Event()

        # ImageNet normalization stats (matches dataset.py)
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # ── Model loading ──────────────────────────────────────────────────

    @staticmethod
    def _load_label_map_from_dir(model_dir: str) -> list:
        path = os.path.join(model_dir, "label_map.json")
        if not os.path.isfile(path):
            return []
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("labels"):
                return list(data["labels"])
            idx_map = data.get("index_to_label", {})
            return [idx_map[str(i)] for i in range(len(idx_map))]
        except (json.JSONDecodeError, OSError, KeyError):
            return []

    def load_model(self, checkpoint_path: str) -> bool:
        """Load model from checkpoint. Thread-safe (call from main thread before start)."""
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            model_type = ckpt.get("model_type", "cnn_pool")
            num_classes = ckpt.get("num_classes")
            class_names = ckpt.get("class_names")
            if not class_names and ckpt.get("index_to_label"):
                idx_map = ckpt["index_to_label"]
                class_names = [idx_map[str(i)] for i in range(len(idx_map))]
            if not class_names:
                class_names = self._load_label_map_from_dir(
                    os.path.dirname(checkpoint_path)
                )
            if num_classes is None:
                num_classes = len(class_names) if class_names else 26
            if not class_names:
                num_classes = ckpt.get("num_classes", 26)
                class_names = [chr(65 + i) for i in range(num_classes)]
            clip_len = ckpt.get("clip_len", 16)
            img_size = ckpt.get("img_size", 224)
            fiber_name = ckpt.get("fiber_name", "unknown")

            model = get_model(model_type, num_classes, pretrained=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            model.to(self._device)

            input_mode = ckpt.get("input_mode", "gray")

            with self._buf_lock:
                self._model = model
                self._class_names = class_names
                self._clip_len = clip_len
                self._img_size = img_size
                self._input_mode = input_mode
                self._frame_buffer = collections.deque(maxlen=clip_len)
                self._raw_frame_buffer = collections.deque(maxlen=clip_len)
                self._vote_buffer.clear()
                self._new_frame_cnt = 0
                self._checkpoint_path = checkpoint_path
                self._fiber_name = fiber_name
                self._logged_first_inference = False
                self._diag_clip_saved = False

            if diagnostics_enabled() and GuiDiagnosticsSession is not None:
                sess = GuiDiagnosticsSession.get()
                if sess is not None:
                    sess.set_model_meta(
                        checkpoint_path=checkpoint_path,
                        fiber_name=fiber_name,
                        clip_len=clip_len,
                        img_size=img_size,
                        input_mode=input_mode,
                        class_names=list(class_names),
                    )

            msg = (
                f"Model loaded: {fiber_name}  |  {model_type}  |  "
                f"classes={num_classes}  clip_len={clip_len}  "
                f"input={self._input_mode}  device={self._device}"
            )
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
        with self._buf_lock:
            old = list(self._vote_buffer)
            self._vote_buffer = collections.deque(old[-maxlen:], maxlen=maxlen)

    def reset_inference_state(self) -> None:
        """Clear buffers and pending frames after recognition is stopped."""
        with self._buf_lock:
            self._frame_buffer.clear()
            self._raw_frame_buffer.clear()
            self._vote_buffer.clear()
            self._new_frame_cnt = 0
            self._warmup_skip_remaining = 0
        while True:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

    def set_warmup_skip(self, n: int) -> None:
        """Drop the next N frames before filling the clip buffer (unstable speckle)."""
        with self._buf_lock:
            self._warmup_skip_remaining = max(0, int(n))

    # ── Frame ingestion (GUI thread — non-blocking) ─────────────────────

    def push_frame(self, frame: np.ndarray):
        """Enqueue a frame copy for processing on the inference thread."""
        if self._model is None:
            return
        if not self.isRunning():
            _demo_trace("push_frame skipped: inference thread not running")
            return

        _demo_trace(f"push_frame enqueue shape={frame.shape}")
        try:
            self._frame_queue.put_nowait(np.copy(frame))
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(np.copy(frame))
            except queue.Full:
                pass

    def run(self):
        """Drain the queue and run inference here—not on the GUI thread."""
        while not self._quit.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if frame is None:
                break
            try:
                self._process_frame(frame)
            except Exception as exc:
                self.error.emit(f"Inference worker loop error: {exc}")

    def _process_frame(self, frame: np.ndarray):
        with self._buf_lock:
            if self._model is None:
                return

            if self._warmup_skip_remaining > 0:
                self._warmup_skip_remaining -= 1
                return

            processed = self._preprocess_frame(frame)
            self._frame_buffer.append(processed)
            self._raw_frame_buffer.append(np.copy(frame))
            self._new_frame_cnt += 1

            if (
                self._new_frame_cnt >= self._infer_every
                and len(self._frame_buffer) >= self._clip_len
            ):
                self._new_frame_cnt = 0
                t0 = time.perf_counter()
                _demo_trace("inference start")
                self._run_inference()
                _demo_trace(f"inference end dt_ms={(time.perf_counter()-t0)*1000:.1f}")

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """BGR uint8 -> normalized float32 (3, H, W) matching dataset.py."""
        resized = cv2.resize(
            frame, (self._img_size, self._img_size), interpolation=cv2.INTER_AREA
        )

        if self._input_mode == "rgb":
            if len(resized.shape) == 2:
                rgb = np.stack([resized, resized, resized], axis=-1)
            else:
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frame_3ch = rgb.astype(np.float32) / 255.0          # (H, W, 3)
            frame_3ch = frame_3ch.transpose(2, 0, 1)            # (3, H, W)
        else:
            if len(resized.shape) == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized
            gray_f = gray.astype(np.float32) / 255.0
            frame_3ch = np.stack([gray_f, gray_f, gray_f], axis=0)  # (3, H, W)

        for c in range(3):
            frame_3ch[c] = (frame_3ch[c] - self._mean[c]) / self._std[c]
        return frame_3ch

    # ── Inference ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_inference(self):
        """Call with self._buf_lock held."""
        frames = list(self._frame_buffer)[-self._clip_len:]
        if len(frames) < self._clip_len:
            return

        clip = np.stack(frames, axis=0)                  # (T, 3, H, W)
        tensor = torch.from_numpy(clip).unsqueeze(0)     # (1, T, 3, H, W)
        tensor = tensor.to(self._device, non_blocking=True)

        try:
            logits = self._model(tensor)                  # (1, num_classes)
            probs = F.softmax(logits, dim=1)[0]          # (num_classes,)

            top_probs, top_idxs = torch.topk(probs, min(self._top_k, len(self._class_names)))
            top1_name = self._class_names[top_idxs[0].item()]
            top1_conf = top_probs[0].item()

            topk = [
                (self._class_names[idx.item()], prob.item())
                for idx, prob in zip(top_idxs, top_probs)
            ]

            # Voting smoothing
            self._vote_buffer.append(top1_name)
            counts = collections.Counter(self._vote_buffer)
            smoothed = counts.most_common(1)[0][0]

            result = {
                "top1": top1_name,
                "confidence": top1_conf,
                "topk": topk,
                "smoothed": smoothed,
                "frame_count": len(self._frame_buffer),
            }
            self.prediction_ready.emit(result)

            if diagnostics_enabled() and GuiDiagnosticsSession is not None:
                sess = GuiDiagnosticsSession.get()
                if sess is not None:
                    sess._append_prediction(result)

            if not self._logged_first_inference:
                self._log_first_inference_tensor(tensor)

            if (
                diagnostics_enabled()
                and GuiDiagnosticsSession is not None
                and not self._diag_clip_saved
            ):
                sess = GuiDiagnosticsSession.get()
                if sess is not None:
                    raw_list = list(self._raw_frame_buffer)[-self._clip_len :]
                    sess.log_inference_clip(
                        raw_frames=raw_list,
                        processed_frames=frames,
                        tensor=tensor.detach().cpu().numpy(),
                        prediction=result,
                        infer_every=self._infer_every,
                        buffer_len=len(self._frame_buffer),
                    )
                    self._diag_clip_saved = True

        except Exception as e:
            self.error.emit(f"Inference error: {e}")

    def _log_first_inference_tensor(self, tensor: torch.Tensor) -> None:
        """Log model input statistics once per model load (not every frame)."""
        self._logged_first_inference = True
        arr = tensor.detach().cpu().numpy()
        flat = arr.astype(np.float64).ravel()
        challenge = ""
        if diagnostics_enabled() and GuiDiagnosticsSession is not None:
            sess = GuiDiagnosticsSession.get()
            if sess is not None:
                challenge = sess.auth_challenge or ""
        raw_stats = ""
        if self._raw_frame_buffer:
            raw = self._raw_frame_buffer[-1]
            rflat = raw.astype(np.float64).ravel()
            raw_stats = (
                f" | raw_shape={tuple(raw.shape)} raw_dtype={raw.dtype} "
                f"raw_min={float(np.min(rflat)):.1f} raw_max={float(np.max(rflat)):.1f} "
                f"raw_mean={float(np.mean(rflat)):.1f}"
            )
        msg = (
            f"[Inference] first batch: shape={tuple(arr.shape)} dtype={arr.dtype} "
            f"min={float(np.min(flat)):.4f} max={float(np.max(flat)):.4f} "
            f"mean={float(np.mean(flat)):.4f} std={float(np.std(flat)):.4f}{raw_stats} | "
            f"buffer_len={len(self._frame_buffer)} clip_len={self._clip_len} "
            f"input_mode={self._input_mode} resize={self._img_size} "
            f"model={self._fiber_name} path={self._checkpoint_path} "
            f"challenge={challenge!r} infer_every={self._infer_every} device={self._device}"
        )
        print(msg, flush=True)

    def stop(self):
        """Request shutdown and wait for the thread to exit."""
        self._quit.set()
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(None)
            except queue.Full:
                pass
        self.wait(3000)
