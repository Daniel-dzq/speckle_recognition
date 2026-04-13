"""
Camera / video capture worker thread.

Supports:
  - Live camera (OpenCV device index) with runtime parameter control
  - Local video file (for debugging without hardware)

Runtime-adjustable camera parameters (call while running):
  set_exposure(), set_gain(), set_brightness(), set_contrast(),
  set_saturation(), set_white_balance(), set_auto_exposure(),
  set_auto_wb(), set_flip(), set_gamma()
"""

import time
import threading
import numpy as np
import cv2
from PySide6.QtCore import QThread, Signal


class CameraWorker(QThread):
    """
    Background thread for frame capture.

    Signals:
        frame_ready(np.ndarray):   BGR uint8 frame (H, W, 3)
        error(str):                capture error message
        fps_updated(float):        current FPS, reported every 2 s
        props_read(dict):          camera property snapshot after open
    """

    frame_ready = Signal(object)
    error       = Signal(str)
    fps_updated = Signal(float)
    props_read  = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source     = 0
        self._is_file    = False
        self._running    = False
        self._target_fps = 30.0
        self._loop_file  = True

        self._cap_width  = None
        self._cap_height = None

        # ── pending property changes (applied inside capture loop) ──
        self._prop_lock    = threading.Lock()
        self._pending_props: dict = {}   # cv2.CAP_PROP_* -> value

        # ── post-processing flags (applied in Python, no OpenCV prop) ──
        self._flip_h = False
        self._flip_v = False

    # ───────────────────────────── configuration ──────────────────────

    def set_camera(self, device_index: int = 0,
                   width: int = None, height: int = None):
        self._source     = device_index
        self._is_file    = False
        self._cap_width  = width
        self._cap_height = height

    def set_video_file(self, path: str, loop: bool = True):
        self._source     = path
        self._is_file    = True
        self._loop_file  = loop

    def set_target_fps(self, fps: float):
        self._target_fps = max(1.0, fps)

    # ───────────────────────── runtime controls ────────────────────────

    def _queue(self, prop_id: int, value: float):
        """Queue an OpenCV property change to be applied in the capture thread."""
        with self._prop_lock:
            self._pending_props[prop_id] = value

    def set_auto_exposure(self, enable: bool):
        """
        0.25 = manual, 0.75 = auto  (OpenCV convention for V4L2/AVFoundation).
        On Windows MSMF the values differ; we try both conventions.
        """
        val = 0.75 if enable else 0.25
        self._queue(cv2.CAP_PROP_AUTO_EXPOSURE, val)

    def set_exposure(self, value: float):
        """
        Exposure value. Range depends on backend:
          V4L2/AVFoundation: typically -13 … 0 (log2 seconds)
          MindVision via DirectShow: absolute µs value
        """
        self._queue(cv2.CAP_PROP_EXPOSURE, value)

    def set_gain(self, value: float):
        self._queue(cv2.CAP_PROP_GAIN, value)

    def set_brightness(self, value: float):
        self._queue(cv2.CAP_PROP_BRIGHTNESS, value)

    def set_contrast(self, value: float):
        self._queue(cv2.CAP_PROP_CONTRAST, value)

    def set_saturation(self, value: float):
        self._queue(cv2.CAP_PROP_SATURATION, value)

    def set_gamma(self, value: float):
        self._queue(cv2.CAP_PROP_GAMMA, value)

    def set_sharpness(self, value: float):
        self._queue(cv2.CAP_PROP_SHARPNESS, value)

    def set_auto_wb(self, enable: bool):
        self._queue(cv2.CAP_PROP_AUTO_WB, 1.0 if enable else 0.0)

    def set_white_balance_temp(self, value: float):
        """Color temperature in K, typically 2800–6500."""
        self._queue(cv2.CAP_PROP_WB_TEMPERATURE, value)

    def set_resolution(self, width: int, height: int):
        self._queue(cv2.CAP_PROP_FRAME_WIDTH,  float(width))
        self._queue(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

    def set_fps_prop(self, fps: float):
        self._queue(cv2.CAP_PROP_FPS, fps)

    def set_flip(self, horizontal: bool, vertical: bool):
        self._flip_h = horizontal
        self._flip_v = vertical

    # ───────────────────────────── control ────────────────────────────

    def stop(self):
        self._running = False
        self.wait(3000)

    # ───────────────────────────── main loop ──────────────────────────

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(self._source)

        if not cap.isOpened():
            kind = "video file" if self._is_file else "camera device"
            self.error.emit(f"Cannot open {kind}: {self._source}")
            return

        # Apply initial resolution
        if not self._is_file and self._cap_width and self._cap_height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._cap_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cap_height)

        # Read back actual properties and notify UI
        self._emit_props(cap)

        interval       = 1.0 / self._target_fps
        frame_count    = 0
        t_fps_start    = time.time()
        fps_report_sec = 2.0

        while self._running:
            t_start = time.time()

            # Apply any pending property changes
            with self._prop_lock:
                pending = dict(self._pending_props)
                self._pending_props.clear()
            for prop_id, val in pending.items():
                cap.set(prop_id, val)
            if pending:
                self._emit_props(cap)

            ret, frame = cap.read()
            if not ret:
                if self._is_file and self._loop_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                elif self._is_file:
                    self.error.emit("End of video file reached.")
                    break
                else:
                    self.error.emit("Camera read failed.")
                    break

            # Post-processing
            if self._flip_h and self._flip_v:
                frame = cv2.flip(frame, -1)
            elif self._flip_h:
                frame = cv2.flip(frame, 1)
            elif self._flip_v:
                frame = cv2.flip(frame, 0)

            self.frame_ready.emit(frame.copy())
            frame_count += 1

            now = time.time()
            if now - t_fps_start >= fps_report_sec:
                self.fps_updated.emit(frame_count / (now - t_fps_start))
                frame_count = 0
                t_fps_start = now

            elapsed = time.time() - t_start
            sleep = interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

        cap.release()
        self._running = False

    # ───────────────────────── helpers ────────────────────────────────

    def _emit_props(self, cap: cv2.VideoCapture):
        props = {
            "width":        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height":       cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps":          cap.get(cv2.CAP_PROP_FPS),
            "exposure":     cap.get(cv2.CAP_PROP_EXPOSURE),
            "auto_exposure":cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            "gain":         cap.get(cv2.CAP_PROP_GAIN),
            "brightness":   cap.get(cv2.CAP_PROP_BRIGHTNESS),
            "contrast":     cap.get(cv2.CAP_PROP_CONTRAST),
            "saturation":   cap.get(cv2.CAP_PROP_SATURATION),
            "gamma":        cap.get(cv2.CAP_PROP_GAMMA),
            "sharpness":    cap.get(cv2.CAP_PROP_SHARPNESS),
            "auto_wb":      cap.get(cv2.CAP_PROP_AUTO_WB),
            "wb_temp":      cap.get(cv2.CAP_PROP_WB_TEMPERATURE),
            "backend":      cap.getBackendName(),
        }
        self.props_read.emit(props)
