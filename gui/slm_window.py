"""
SLM output window.

Improved version:
  - More reliable screen placement on multi-monitor Windows setups
  - Better fullscreen / windowed transitions
  - Adaptive letter sizing based on actual display area
  - Image mode rescales cleanly after move/resize/fullscreen changes
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PySide6.QtCore import Qt, Signal, QRect
from PySide6.QtGui import (
    QFont, QKeyEvent, QMouseEvent, QColor, QPalette,
    QPixmap, QScreen, QGuiApplication,
)


class SLMWindow(QWidget):
    """Dedicated output window for the SLM (Spatial Light Modulator)."""

    letter_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._current_letter = "A"
        self._base_font_size = 400
        self._bg_color = "#000000"
        self._fg_color = "#FFFFFF"
        self._is_fullscreen = False
        self._pixmap_source = None
        self._margin_ratio = 0.08

        self._setup_ui()
        self._apply_colors()
        self.set_letter("A")

    def _setup_ui(self):
        self.setWindowTitle("SLM Output")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setMinimumSize(400, 400)
        self.resize(900, 900)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._label = QLabel("A", self)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._label.setScaledContents(False)
        self._label.setWordWrap(False)

        layout.addWidget(self._label)
        self.setLayout(layout)

    def _apply_colors(self):
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(self._bg_color))
        pal.setColor(QPalette.WindowText, QColor(self._fg_color))
        self.setPalette(pal)
        self.setAutoFillBackground(True)
        self._label.setStyleSheet(
            f"color: {self._fg_color}; background-color: {self._bg_color};"
        )

    def _ensure_native_window(self):
        if self.windowHandle() is None:
            self.winId()
            QGuiApplication.processEvents()

    def _target_geometry(self, screen: QScreen, fullscreen: bool) -> QRect:
        if screen is None:
            return self.geometry()
        return screen.geometry() if fullscreen else screen.availableGeometry()

    def _fit_letter_font(self):
        if self._pixmap_source is not None:
            return

        text = self._current_letter if self._current_letter.strip() else " "
        rect = self.rect()
        short_side = max(1, min(rect.width(), rect.height()))
        auto_size = int(short_side * (1.0 - self._margin_ratio * 2) * 0.78)
        font_px = max(24, min(self._base_font_size, auto_size))

        font = QFont("Arial", 10, QFont.Bold)
        font.setPixelSize(font_px)
        self._label.setFont(font)
        self._label.setText(text)

    def _update_pixmap_display(self):
        if self._pixmap_source is None:
            return
        target = self._label.size()
        if target.width() <= 0 or target.height() <= 0:
            return
        scaled = self._pixmap_source.scaled(
            target,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._label.setPixmap(scaled)

    def _refresh_content(self):
        if self._pixmap_source is None:
            self._label.setPixmap(QPixmap())
            self._fit_letter_font()
        else:
            self._label.setText("")
            self._update_pixmap_display()

    def set_letter(self, letter: str):
        letter = letter.strip().upper() if letter.strip() else " "
        self._current_letter = letter
        self._pixmap_source = None
        self._refresh_content()
        self.letter_changed.emit(letter)

    def set_font_size(self, size: int):
        self._base_font_size = max(10, size)
        if self._pixmap_source is None:
            self._fit_letter_font()

    def set_colors(self, bg: str = "#000000", fg: str = "#FFFFFF"):
        self._bg_color = bg
        self._fg_color = fg
        self._apply_colors()

    def load_image(self, path: str) -> bool:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return False
        self._pixmap_source = pixmap
        self._current_letter = ""
        self._refresh_content()
        return True

    def show_on_screen(self, screen: QScreen, fullscreen: bool = True):
        self._ensure_native_window()

        if self.isVisible() and self._is_fullscreen and not fullscreen:
            self.showNormal()
            QGuiApplication.processEvents()

        handle = self.windowHandle()
        if handle is not None and screen is not None:
            handle.setScreen(screen)
            QGuiApplication.processEvents()

        geom = self._target_geometry(screen, fullscreen)
        self.setGeometry(geom)
        self.move(geom.topLeft())

        if not self.isVisible():
            self.show()
            QGuiApplication.processEvents()

        if fullscreen:
            self.showFullScreen()
            self._is_fullscreen = True
        else:
            self.showNormal()
            self.setGeometry(geom)
            self._is_fullscreen = False

        self._refresh_content()
        self.raise_()
        self.activateWindow()

    def toggle_fullscreen(self):
        handle = self.windowHandle()
        screen = handle.screen() if handle is not None else QGuiApplication.primaryScreen()
        self.show_on_screen(screen, fullscreen=not self._is_fullscreen)

    def current_letter(self) -> str:
        return self._current_letter

    def resizeEvent(self, event):
        self._refresh_content()
        super().resizeEvent(event)

    def showEvent(self, event):
        self._refresh_content()
        super().showEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key in (Qt.Key_F11, Qt.Key_F):
            self.toggle_fullscreen()
        elif key == Qt.Key_Escape:
            if self._is_fullscreen:
                self.toggle_fullscreen()
        elif Qt.Key_A <= key <= Qt.Key_Z:
            self.set_letter(chr(key))
        else:
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        self.toggle_fullscreen()
        super().mouseDoubleClickEvent(event)
