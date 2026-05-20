"""
SLM output window.

- macOS: avoids Qt showFullScreen(); borderless window with screen.geometry().
- Content is drawn with QPainter (not QLabel+pixmap) for reliable output on
  external / edid-only panels where QLabel sometimes paints nothing.
"""

from __future__ import annotations

import os
import platform

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy, QHBoxLayout
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import (
    QFont,
    QKeyEvent,
    QMouseEvent,
    QColor,
    QPalette,
    QPixmap,
    QScreen,
    QGuiApplication,
    QImage,
    QPainter,
    QPaintEvent,
)

_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_GUI_DIR)


def _is_darwin() -> bool:
    return platform.system() == "Darwin"


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _letter_image_search_roots() -> list[str]:
    roots: list[str] = []
    env = os.environ.get("SPECKLE_LETTER_IMAGES_DIR", "").strip()
    if env:
        roots.append(os.path.abspath(os.path.expanduser(env)))
    roots.append(os.path.join(_ROOT_DIR, "letter_images"))
    seen: set[str] = set()
    out: list[str] = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _find_letter_png(letter: str) -> str | None:
    if not (letter.isalpha() and len(letter) == 1):
        return None
    for base in _letter_image_search_roots():
        for name in (f"{letter}.png", f"{letter}.PNG"):
            path = os.path.join(base, name)
            if os.path.isfile(path):
                return path
    return None


def _find_challenge_png(label: str) -> str | None:
    """Look up a PNG for an arbitrary challenge label (letter, digit, avatar name, etc.)."""
    key = (label or "").strip()
    if not key:
        return None
    candidates = [
        key,
        key.lower(),
        key.upper(),
        key.replace(" ", "_"),
        key.replace(" ", "-"),
    ]
    if len(key) == 1 and key.isalpha():
        found = _find_letter_png(key.upper())
        if found:
            return found
    seen: set[str] = set()
    for base in _letter_image_search_roots():
        for stem in candidates:
            if stem in seen:
                continue
            seen.add(stem)
            for ext in (".png", ".PNG", ".jpg", ".jpeg", ".bmp"):
                path = os.path.join(base, stem + ext)
                if os.path.isfile(path):
                    return path
    return None


class SLMViewport(QWidget):
    """
    Paints SLM content. QLabel+scaled QPixmap is unreliable on some macOS
    external displays; QPainter.drawPixmap is used instead.
    """

    def __init__(self, slm: "SLMWindow"):
        super().__init__(slm)
        self._slm = slm
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: ARG002
        slm = self._slm
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.fillRect(self.rect(), QColor(slm._bg_color))

        if slm._diagnostic_active:
            p.setPen(QColor(slm._fg_color))
            short = max(1, min(self.width(), self.height()))
            f_title = QFont()
            f_title.setFamilies(["Helvetica Neue", "Arial", "PingFang SC"])
            f_title.setPixelSize(max(48, short // 12))
            f_title.setWeight(QFont.Weight.Black)
            f_letter = QFont(f_title)
            f_letter.setPixelSize(max(160, short // 2))
            r = self.rect()
            p.setFont(f_title)
            p.drawText(
                r.adjusted(8, int(short * 0.06), -8, -int(short * 0.45)),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                "SLM TEST",
            )
            p.setFont(f_letter)
            p.drawText(
                r,
                Qt.AlignmentFlag.AlignCenter,
                slm._diag_letter,
            )
            p.end()
            return

        pm = slm._pixmap_source
        if pm is not None and not pm.isNull():
            im = pm.toImage()
            if im.hasAlphaChannel() or im.format() == QImage.Format.Format_ARGB32:
                im = im.convertToFormat(QImage.Format.Format_RGB888)
            pm_work = QPixmap.fromImage(im)
            mode = (
                Qt.AspectRatioMode.IgnoreAspectRatio
                if slm._stretch_to_fill
                else Qt.AspectRatioMode.KeepAspectRatio
            )
            scaled = pm_work.scaled(
                self.size(),
                mode,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            p.drawPixmap(max(0, x), max(0, y), scaled)
            p.end()
            return

        text = slm._current_letter if slm._current_letter.strip() else " "
        short_side = max(1, min(self.width(), self.height()))
        auto_size = int(short_side * (1.0 - slm._margin_ratio * 2) * 0.78)
        font_px = max(24, min(slm._base_font_size, auto_size))
        font = QFont()
        font.setFamilies(["Helvetica Neue", "Arial", "PingFang SC", "Segoe UI"])
        font.setPixelSize(font_px)
        font.setWeight(QFont.Weight.Black)
        p.setFont(font)
        p.setPen(QColor(slm._fg_color))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)
        p.end()


class SLMWindow(QWidget):
    """Dedicated output window for the SLM (Spatial Light Modulator)."""

    letter_changed = Signal(str)
    diagnostic_log = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._current_letter = "A"
        self._base_font_size = 400
        self._bg_color = "#000000"
        self._fg_color = "#FFFFFF"
        self._is_fullscreen = False
        self._pixmap_source: QPixmap | None = None
        self._margin_ratio = 0.08
        self._stretch_to_fill = True
        self._last_png_path: str | None = None
        self._last_png_diagnostic: str | None = None
        self._diagnostic_active = False
        self._diag_letter = "A"

        self._setup_ui()
        if _is_darwin():
            # Helps some external panels / secondary-GPU paths flush a real framebuffer.
            self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self._apply_colors()
        self.set_letter("A")

    def _slm_log(self, msg: str) -> None:
        print(msg, flush=True)
        self.diagnostic_log.emit(msg)

    def _setup_ui(self):
        self.setWindowTitle("SLM Output")
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setMinimumSize(400, 400)
        self.resize(900, 900)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._viewport = SLMViewport(self)
        outer.addWidget(self._viewport, stretch=1)

        self._color_bar = QWidget(self)
        self._color_bar.setFixedHeight(40)
        hb = QHBoxLayout(self._color_bar)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(0)
        for hex_c in ("#cc0000", "#00aa00", "#2244ff", "#ffffff"):
            seg = QLabel(self._color_bar)
            seg.setStyleSheet(f"background-color: {hex_c};")
            hb.addWidget(seg, stretch=1)
        self._color_bar.hide()
        outer.addWidget(self._color_bar, stretch=0)
        self.setLayout(outer)

    def _apply_colors(self):
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(self._bg_color))
        pal.setColor(QPalette.ColorRole.WindowText, QColor(self._fg_color))
        self.setPalette(pal)
        self.setAutoFillBackground(True)

    def _ensure_native_window(self):
        if self.windowHandle() is None:
            self.winId()
            QGuiApplication.processEvents()

    def _refresh_content(self) -> None:
        self._viewport.update()
        self._viewport.repaint()

    def force_visual_refresh(self) -> None:
        self._refresh_content()
        self.update()
        self.repaint()
        QTimer.singleShot(0, self._refresh_content)
        QTimer.singleShot(16, self._refresh_content)

    def set_diagnostic_pattern(self, letter: str = "A") -> None:
        self._diagnostic_active = True
        self._pixmap_source = None
        self._last_png_path = None
        self._last_png_diagnostic = None
        c = (letter.strip().upper() or "A")[0]
        self._diag_letter = c if c.isalpha() else "A"
        self._current_letter = self._diag_letter
        self._color_bar.show()
        self._slm_log(f"[SLM] set_diagnostic_pattern: letter={self._diag_letter}")
        self.force_visual_refresh()

    def set_text_challenge(self, label: str) -> None:
        """Render a text challenge on the SLM (letters, digits, short labels)."""
        self._diagnostic_active = False
        self._color_bar.hide()
        text = (label or "").strip() or " "
        self._slm_log(f"[SLM] set_text_challenge: {text!r}")
        self._current_letter = text
        self._pixmap_source = None
        self._last_png_path = None
        self._last_png_diagnostic = None

        if not _env_truthy("SPECKLE_SLM_TEXT_ONLY"):
            img_path = _find_challenge_png(text)
            if img_path is None:
                roots = _letter_image_search_roots()
                self._last_png_diagnostic = (
                    f"No image for {text!r} (using Qt painter text). Searched: {roots}"
                )
                self._slm_log(f"[SLM] {self._last_png_diagnostic}")
            else:
                pix = QPixmap(img_path)
                if pix.isNull():
                    im = QImage(img_path)
                    if not im.isNull():
                        pix = QPixmap.fromImage(im)
                if pix.isNull():
                    self._last_png_diagnostic = (
                        f"Decode failed, painter text fallback: {img_path}"
                    )
                    self._slm_log(f"[SLM] {self._last_png_diagnostic}")
                else:
                    self._pixmap_source = pix
                    self._last_png_path = img_path

        self.force_visual_refresh()
        self._slm_log("[SLM] content updated (set_text_challenge)")
        self.letter_changed.emit(text)

    def set_letter(self, letter: str) -> None:
        """Backward-compatible alias for single-character letter challenges."""
        self.set_text_challenge(letter)

    def png_load_diagnostic(self) -> str | None:
        return self._last_png_diagnostic

    def last_letter_png_path(self) -> str | None:
        return self._last_png_path

    def set_font_size(self, size: int) -> None:
        self._base_font_size = max(10, size)
        if self._pixmap_source is None and not self._diagnostic_active:
            self._refresh_content()

    def set_colors(self, bg: str = "#000000", fg: str = "#FFFFFF") -> None:
        self._bg_color = bg
        self._fg_color = fg
        self._apply_colors()
        self._refresh_content()

    def set_stretch(self, stretch: bool) -> None:
        self._stretch_to_fill = stretch
        self._refresh_content()

    def load_image(self, path: str) -> bool:
        self._diagnostic_active = False
        self._color_bar.hide()
        pixmap = QPixmap(path)
        if pixmap.isNull():
            im = QImage(path)
            if not im.isNull():
                pixmap = QPixmap.fromImage(im)
        if pixmap.isNull():
            return False
        self._pixmap_source = pixmap
        self._current_letter = ""
        self.force_visual_refresh()
        return True

    def show_on_screen(self, screen: QScreen | None, fullscreen: bool = True) -> None:
        self._ensure_native_window()
        sys_name = platform.system()
        self._slm_log(f"[SLM] Platform: {sys_name}")

        if self.isVisible() and self._is_fullscreen and not _is_darwin():
            self.showNormal()
            QGuiApplication.processEvents()

        geom = screen.geometry() if screen is not None else self.geometry()
        avail = screen.availableGeometry() if screen is not None else geom
        name = screen.name() if screen is not None else "?"
        self._slm_log(
            f"[SLM] Target screen: {name} | geometry x={geom.x()}, y={geom.y()}, "
            f"w={geom.width()}, h={geom.height()}"
        )
        prisc = QGuiApplication.primaryScreen()

        handle = self.windowHandle()
        if handle is not None and screen is not None:
            if _env_truthy("SPECKLE_SLM_NO_SET_SCREEN"):
                self._slm_log(
                    "[SLM] SPECKLE_SLM_NO_SET_SCREEN=1: skipping windowHandle().setScreen()"
                )
            else:
                handle.setScreen(screen)
                QGuiApplication.processEvents()

        if fullscreen:
            if _is_darwin():
                self._slm_log(
                    "[SLM] macOS detected: borderless fake fullscreen "
                    "(showFullScreen() skipped)"
                )
                self.showNormal()
                self.setGeometry(geom)
                self.move(geom.topLeft())
                self.show()
                self._is_fullscreen = True
            else:
                self._slm_log("[SLM] Non-macOS: using showFullScreen()")
                self.showNormal()
                self.setGeometry(geom)
                self.move(geom.topLeft())
                QGuiApplication.processEvents()
                self.showFullScreen()
                self._is_fullscreen = True
        else:
            self._slm_log("[SLM] Windowed placement on selected screen (availableGeometry)")
            self.showNormal()
            self.setGeometry(avail)
            self.move(avail.topLeft())
            self._is_fullscreen = False
            self.show()

        self.raise_()
        self.activateWindow()
        self.repaint()
        self.update()
        for ms in (0, 40, 120, 280, 450):
            QTimer.singleShot(ms, self.force_visual_refresh)

        wg = self.geometry()
        self._slm_log(
            f"[SLM] Window geometry after show: x={wg.x()}, y={wg.y()}, "
            f"w={wg.width()}, h={wg.height()} | visible={self.isVisible()} "
            f"| is_fullscreen_flag={self._is_fullscreen}"
        )
        if prisc is not None and screen is not None and screen == prisc and fullscreen:
            self._slm_log(
                "[SLM] Note: target screen is Qt primary; "
                "if SLM is external, pick the other index in the demo."
            )

    def toggle_fullscreen(self) -> None:
        handle = self.windowHandle()
        screen = handle.screen() if handle is not None else QGuiApplication.primaryScreen()
        self.show_on_screen(screen, fullscreen=not self._is_fullscreen)

    def current_letter(self) -> str:
        return self._current_letter

    def resizeEvent(self, event) -> None:
        self._refresh_content()
        super().resizeEvent(event)

    def showEvent(self, event) -> None:
        self._refresh_content()
        super().showEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key.Key_F11, Qt.Key.Key_F):
            self.toggle_fullscreen()
        elif key == Qt.Key.Key_Escape:
            if self._is_fullscreen:
                self.toggle_fullscreen()
        elif Qt.Key.Key_A <= key <= Qt.Key.Key_Z:
            self.set_letter(chr(key))
        else:
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        self.toggle_fullscreen()
        super().mouseDoubleClickEvent(event)
