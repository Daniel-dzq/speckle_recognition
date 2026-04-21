"""
MindVision CCD camera worker thread.

Drop-in replacement for CameraWorker when using HT-UBS300C (or any MindVision
USB camera) via the vendor SDK (libmvsdk.dylib) instead of OpenCV.

Emits the same signals as CameraWorker:
    frame_ready(np.ndarray)   BGR uint8 frame (H, W, 3) or grayscale (H, W)
    error(str)
    fps_updated(float)
    props_read(dict)

Runtime controls (can be called while running):
    set_exposure(us)          exposure in microseconds
    set_gain(value)           analog gain integer value
    set_auto_exposure(enable) enable/disable auto-exposure
    set_flip(h, v)            mirror frame in software
"""

import sys
import time
import threading

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from gui import mvsdk


class MvCameraWorker(QThread):
    """Background thread that streams frames from a MindVision CCD camera."""

    frame_ready = Signal(object)   # np.ndarray
    error       = Signal(str)
    fps_updated = Signal(float)
    props_read  = Signal(dict)

    def __init__(self, dev_info: mvsdk.tSdkCameraDevInfo, parent=None):
        super().__init__(parent)
        self._dev_info     = dev_info
        self._running      = False
        self._target_fps   = 30.0

        # ── runtime-adjustable settings (set from GUI thread) ──
        self._lock         = threading.Lock()
        self._pending: dict = {}   # key → value

        # ── post-processing (no SDK round-trip needed) ──
        self._flip_h = False
        self._flip_v = False

    # ──────────────────────────── runtime controls ────────────────────────────

    def set_target_fps(self, fps: float) -> None:
        self._target_fps = max(1.0, fps)

    def set_exposure(self, us: float) -> None:
        """Exposure time in microseconds."""
        with self._lock:
            self._pending["exposure_us"] = float(us)

    def set_gain(self, value: float) -> None:
        with self._lock:
            self._pending["gain"] = int(value)

    def set_auto_exposure(self, enable: bool) -> None:
        with self._lock:
            self._pending["auto_exposure"] = bool(enable)

    def set_flip(self, horizontal: bool, vertical: bool) -> None:
        self._flip_h = horizontal
        self._flip_v = vertical

    # ── stubs to satisfy the same interface as CameraWorker ──────────────────

    def set_brightness(self, value: float) -> None:  pass
    def set_contrast(self, value: float) -> None:    pass
    def set_saturation(self, value: float) -> None:  pass
    def set_gamma(self, value: float) -> None:       pass
    def set_sharpness(self, value: float) -> None:   pass
    def set_auto_wb(self, enable: bool) -> None:     pass
    def set_white_balance_temp(self, v: float) -> None: pass
    def set_resolution(self, w: int, h: int) -> None: pass
    def set_fps_prop(self, fps: float) -> None:      pass

    # ──────────────────────────────── control ─────────────────────────────────

    def stop(self) -> None:
        self._running = False
        self.wait(3000)

    # ──────────────────────────────── main loop ───────────────────────────────

    def run(self) -> None:  # noqa: C901
        self._running = True

        # ── open camera ──
        try:
            mvsdk.sdk_init()
            handle = mvsdk.camera_init(self._dev_info)
        except Exception as exc:
            self.error.emit(f"MindVision: cannot open camera – {exc}")
            return

        cap     = mvsdk.get_capability(handle)
        is_mono = bool(cap.sIspCapacity.bMonoSensor)
        fmt     = mvsdk.CAMERA_MEDIA_TYPE_MONO8 if is_mono else mvsdk.CAMERA_MEDIA_TYPE_RGB8
        mvsdk.set_isp_out_format(handle, fmt)

        max_w   = cap.sResolutionRange.iWidthMax  or 2592
        max_h   = cap.sResolutionRange.iHeightMax or 1944
        buf_sz  = max_w * max_h * (1 if is_mono else 3)

        try:
            frame_buf = mvsdk.align_malloc(buf_sz, 16)
        except MemoryError as exc:
            mvsdk.camera_uninit(handle)
            self.error.emit(f"MindVision: memory alloc failed – {exc}")
            return

        # start continuous acquisition, manual exposure by default
        mvsdk.set_ae_state(handle, False)
        mvsdk.set_exposure_time(handle, 10_000.0)   # 10 ms default
        mvsdk.set_trigger_mode(handle, 0)            # continuous
        mvsdk.camera_play(handle)

        # report initial properties
        self._emit_props(handle, cap, is_mono)

        interval       = 1.0 / self._target_fps
        frame_count    = 0
        t_fps_start    = time.time()
        fps_report_sec = 2.0

        while self._running:
            t_start = time.time()

            # ── apply pending control changes ──
            with self._lock:
                pending = dict(self._pending)
                self._pending.clear()

            for key, val in pending.items():
                if key == "exposure_us":
                    mvsdk.set_exposure_time(handle, val)
                elif key == "gain":
                    mvsdk.set_analog_gain(handle, val)
                elif key == "auto_exposure":
                    mvsdk.set_ae_state(handle, val)

            # ── grab frame ──
            head, raw_ptr = mvsdk.get_image_buffer(handle, timeout_ms=200)
            if raw_ptr is None:
                continue   # timeout, keep looping

            ok = mvsdk.image_process(handle, raw_ptr, frame_buf, head)
            mvsdk.release_image_buffer(handle, raw_ptr)

            if not ok:
                continue

            try:
                frame = mvsdk.frame_to_numpy(frame_buf, head)
            except Exception:
                continue

            # ── software flip ──
            if self._flip_h and self._flip_v:
                frame = cv2.flip(frame, -1)
            elif self._flip_h:
                frame = cv2.flip(frame, 1)
            elif self._flip_v:
                frame = cv2.flip(frame, 0)

            self.frame_ready.emit(frame)
            frame_count += 1

            now = time.time()
            if now - t_fps_start >= fps_report_sec:
                self.fps_updated.emit(frame_count / (now - t_fps_start))
                frame_count  = 0
                t_fps_start  = now

            elapsed = time.time() - t_start
            wait    = interval - elapsed
            if wait > 0:
                time.sleep(wait)

        # ── cleanup ──
        mvsdk.camera_stop(handle)
        mvsdk.camera_uninit(handle)
        mvsdk.align_free(frame_buf)
        self._running = False

    # ──────────────────────────────── helpers ─────────────────────────────────

    def _emit_props(
        self,
        handle: int,
        cap: mvsdk.tSdkCameraCapbility,
        is_mono: bool,
    ) -> None:
        exp_us = mvsdk.get_exposure_time(handle)
        gain   = mvsdk.get_analog_gain(handle)
        props  = {
            "width":         float(cap.sResolutionRange.iWidthMax),
            "height":        float(cap.sResolutionRange.iHeightMax),
            "fps":           self._target_fps,
            "exposure":      exp_us,        # µs
            "auto_exposure": 0.0,
            "gain":          float(gain),
            "brightness":    0.0,
            "contrast":      0.0,
            "saturation":    0.0,
            "gamma":         0.0,
            "sharpness":     0.0,
            "auto_wb":       0.0,
            "wb_temp":       0.0,
            "backend":       "MindVision-SDK",
            "is_mono":       is_mono,
        }
        self.props_read.emit(props)
