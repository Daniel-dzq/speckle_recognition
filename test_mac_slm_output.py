#!/usr/bin/env python3
"""
Minimal PySide6 probe: borderless black window + white "SLM TEST" on selected display.

- Lists all screens to stdout.
- Default screen index: 1 if available, else 0.
- macOS: does NOT use showFullScreen() — uses setGeometry(screen.geometry()).
- Cycles subtitle A -> B -> C every second.

Run:
  python test_mac_slm_output.py
  python test_mac_slm_output.py --screen 1
"""

from __future__ import annotations

import argparse
import platform
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QLabel, QWidget


def log_screens(app: QGuiApplication) -> list:
    screens = app.screens()
    primary = app.primaryScreen()
    print(f"Platform: {platform.system()}")
    print(f"Screen count: {len(screens)}")
    for idx, s in enumerate(screens):
        g = s.geometry()
        name = s.name() or f"Screen {idx}"
        dpr = s.devicePixelRatio()
        is_pri = s == primary
        print(
            f"  [{idx}] name={name!r} geometry=({g.x()},{g.y()},{g.width()},{g.height()}) "
            f"dpr={dpr} primary={is_pri}"
        )
    if len(screens) <= 1:
        print(
            "WARNING: Only one screen reported. "
            "If SLM is a second monitor, check Displays (extend) and replug HDMI."
        )
    return screens


class ProbeWindow(QWidget):
    def __init__(self, screen, letter: str = "A"):
        super().__init__(None)
        self._screen = screen
        self.setWindowTitle("SLM TEST")
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setStyleSheet("background-color: #000000;")
        self._label = QLabel(f"SLM TEST\n{letter}", self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(
            "color: #ffffff; font-size: 120px; font-weight: bold; font-family: Arial, Helvetica;"
        )

    def resizeEvent(self, event):
        self._label.setGeometry(self.rect())
        super().resizeEvent(event)

    def show_on_screen(self) -> None:
        self.winId()
        QGuiApplication.processEvents()
        handle = self.windowHandle()
        if handle is not None and self._screen is not None:
            handle.setScreen(self._screen)
            QGuiApplication.processEvents()
        g = self._screen.geometry()
        print(
            f"Placing window: geometry x={g.x()} y={g.y()} w={g.width()} h={g.height()} "
            f"(showFullScreen={'skipped on Darwin' if platform.system() == 'Darwin' else 'N/A'})"
        )
        self.showNormal()
        self.setGeometry(g)
        self.move(g.topLeft())
        self.show()
        self.raise_()
        self.activateWindow()
        self.repaint()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--screen",
        type=int,
        default=None,
        help="Screen index (default: 1 if exists else 0)",
    )
    args = ap.parse_args()

    app = QApplication(sys.argv)
    screens = log_screens(app)
    if not screens:
        print("No screens — abort.")
        return 1

    idx = args.screen
    if idx is None:
        idx = 1 if len(screens) > 1 else 0
    idx = max(0, min(idx, len(screens) - 1))
    print(f"Using screen index: {idx}")

    probe = ProbeWindow(screens[idx], "A")
    probe.show_on_screen()

    QTimer.singleShot(1000, lambda: (probe._label.setText("SLM TEST\nB"), print("[+1s] SLM TEST / B")))
    QTimer.singleShot(2000, lambda: (probe._label.setText("SLM TEST\nC"), print("[+2s] SLM TEST / C")))
    print("Window stays open — close it or press Ctrl+C when done.")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
