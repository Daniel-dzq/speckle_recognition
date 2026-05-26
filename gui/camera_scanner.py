"""
Unified camera discovery for the live demo (OpenCV indices + MindVision SDK).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

import cv2

LogFn = Optional[Callable[[str], None]]


@dataclass
class CameraDeviceEntry:
    """Structured camera source for the demo selector (stored in QComboBox data)."""

    backend: str
    label: str
    opencv_index: Optional[int] = None
    mv_device: Any = None
    width: int = 0
    height: int = 0


def _log(log_fn: LogFn, msg: str) -> None:
    if log_fn is not None:
        log_fn(msg)


def probe_opencv_index(
    index: int,
    *,
    log_fn: LogFn = None,
    device_name: str = "",
) -> Optional[CameraDeviceEntry]:
    """Try to open one OpenCV camera index on the main thread."""
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tag = f", {device_name}" if device_name else ""
        _log(
            log_fn,
            f"OpenCV camera found: index {index}, resolution {w}x{h}{tag}",
        )
        return CameraDeviceEntry(
            backend="opencv",
            label=f"Camera {index} — OpenCV ({w}x{h})",
            opencv_index=index,
            width=w,
            height=h,
        )
    finally:
        cap.release()


def scan_opencv_cameras(
    indices: Sequence[int],
    *,
    log_fn: LogFn = None,
    device_names: Optional[Sequence[str]] = None,
) -> List[CameraDeviceEntry]:
    """Probe OpenCV camera indices (must run on the GUI main thread on macOS)."""
    found: List[CameraDeviceEntry] = []
    names = list(device_names or [])
    for idx in indices:
        name = names[idx] if idx < len(names) else ""
        entry = probe_opencv_index(idx, log_fn=log_fn, device_name=name)
        if entry is not None:
            found.append(entry)
    return found


def scan_mindvision_cameras(*, log_fn: LogFn = None) -> List[CameraDeviceEntry]:
    """Enumerate MindVision SDK devices if the SDK is available."""
    try:
        from gui import mvsdk
    except ImportError as exc:
        _log(log_fn, f"MindVision SDK import failed: {exc}")
        return []

    try:
        mvsdk.sdk_init()
        devices = mvsdk.enumerate_devices()
    except FileNotFoundError as exc:
        _log(log_fn, f"MindVision SDK not found: {exc}")
        return []
    except Exception as exc:
        _log(log_fn, f"MindVision enumeration failed: {exc}")
        return []

    found: List[CameraDeviceEntry] = []
    for dev in devices:
        name = (dev.friendly_name or dev.product_name or "MindVision camera").strip()
        sn = (dev.sn or "").strip()
        sn_tag = f", SN {sn}" if sn else ""
        _log(log_fn, f"MindVision device found: {name}{sn_tag}")
        label = f"MindVision {name} — SDK"
        if "HT-UBS300C" in name.upper() or "HT-UBS300C" in (dev.product_name or "").upper():
            label = "MindVision HT-UBS300C — SDK"
        found.append(
            CameraDeviceEntry(
                backend="mindvision",
                label=label,
                mv_device=dev,
            )
        )
    return found


def scan_all_cameras(
    *,
    opencv_indices: Sequence[int] = range(6),
    log_fn: LogFn = None,
    device_names: Optional[Sequence[str]] = None,
    include_mindvision: bool = True,
) -> List[CameraDeviceEntry]:
    """Scan OpenCV and MindVision backends."""
    _log(log_fn, "Scanning cameras...")
    opencv = scan_opencv_cameras(
        opencv_indices, log_fn=log_fn, device_names=device_names
    )
    mindvision = scan_mindvision_cameras(log_fn=log_fn) if include_mindvision else []
    all_entries = opencv + mindvision
    count = len(all_entries)
    if count:
        _log(log_fn, f"Camera scan complete: {count} device(s) found.")
    else:
        _log(
            log_fn,
            "No cameras found. Close vendor camera software and try again.",
        )
    return all_entries
