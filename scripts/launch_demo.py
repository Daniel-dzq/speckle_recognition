#!/usr/bin/env python3
"""
Launch the Speckle-PUF live demo GUI.

Usage:
    python scripts/launch_demo.py

Supports:
    - macOS  (Apple Silicon M1-M4) : MindVision CCD via libmvsdk.dylib
    - Windows (64-bit)              : MindVision CCD via MVCAMSDK_X64.dll
    - Any platform                  : Webcam / video file via OpenCV

Requirements:
    pip install PySide6 opencv-python torch torchvision
"""

import os
import sys
import platform

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if os.name == "nt":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ── NOTE (macOS): Do NOT preset OPENCV_AVFOUNDATION_SKIP_AUTH here.
# The first VideoCapture call on the main thread (_start_camera / _scan_cameras)
# triggers the system camera-permission dialog.  Only after permission is
# confirmed do we set SKIP_AUTH=1 so the worker thread avoids re-requesting it.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

GUI_DIR = os.path.join(ROOT, "gui")

# ─────────────────────────────────────────────────────────────────────────────
# Windows: inject DLL search paths BEFORE importing anything that loads ctypes.
# Python 3.8+ no longer searches PATH for DLLs by default; os.add_dll_directory
# is the correct way to add extra search locations.
# ─────────────────────────────────────────────────────────────────────────────
if sys.platform == "win32":
    _dll_dirs = [
        os.path.join(GUI_DIR, "win_sdk"),                  # bundled DLLs (preferred)
        GUI_DIR,                                            # legacy: DLL placed next to mvsdk.py
        r"C:\Program Files\MindVision\SDK",
        r"C:\Program Files (x86)\MindVision\SDK",
        r"C:\MindVision\SDK",
    ]
    for _d in _dll_dirs:
        if os.path.isdir(_d):
            try:
                os.add_dll_directory(_d)                   # Python 3.8+
            except AttributeError:
                os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")


# ─────────────────────────────────────────────────────────────────────────────
# Dependency checks
# ─────────────────────────────────────────────────────────────────────────────
try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore    import Qt
except ImportError:
    print("[ERROR] PySide6 is not installed.")
    print("  Install with:  pip install PySide6")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV is not installed.")
    print("  Install with:  pip install opencv-python")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# MindVision SDK availability check (non-fatal: GUI still opens without it)
# ─────────────────────────────────────────────────────────────────────────────
def _check_mvsdk() -> str:
    """Return a one-line status string for the MindVision SDK."""
    try:
        import gui.mvsdk as mvsdk
        mvsdk.get_lib()          # loads the shared library
        return "OK"
    except FileNotFoundError:
        return "NOT FOUND"
    except OSError as e:
        return f"LOAD ERROR: {e}"
    except Exception as e:
        return f"ERROR: {e}"


def _print_startup_banner():
    line = "=" * 64
    print(line)
    print("  Speckle-PUF Demo  |  starting up")
    print(line)
    print(f"  Python     : {sys.version.split()[0]}")
    print(f"  Platform   : {platform.platform()}")
    print(f"  OpenCV     : {cv2.__version__}")

    sdk_status = _check_mvsdk()
    sdk_icon   = "[OK]" if sdk_status == "OK" else "[--]"
    print(f"  MindVision : {sdk_icon} {sdk_status}")

    if sdk_status != "OK":
        if sys.platform == "darwin":
            print()
            print("  To enable HT-UBS300C on macOS:")
            print(f"    1. Copy libmvsdk.dylib  →  {GUI_DIR}/")
            print( "    2. codesign --force --sign - gui/libmvsdk.dylib")
        elif sys.platform == "win32":
            print()
            print("  To enable HT-UBS300C on Windows (MindVision SDK path):")
            win_sdk = os.path.join(GUI_DIR, "win_sdk")
            print(f"    DLLs are bundled in: {win_sdk}")
            print( "    If DLLs are missing, re-extract them from the project archive.")
            print()
            print("  USB driver (needed once per machine):")
            drivers = os.path.join(win_sdk, "drivers")
            print(f"    Device Manager → right-click camera → Update Driver")
            print(f"    → Browse to: {drivers}")

    print(line)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
from gui.main_window import MainWindow


def main():
    _print_startup_banner()

    app = QApplication(sys.argv)
    app.setApplicationName("Speckle-PUF Demo")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
