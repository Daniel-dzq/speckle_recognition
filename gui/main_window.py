"""
Main control window for Speckle-PUF live demo.

Improved version:
  - Responsive main layout for full-screen use
  - Scrollable left control panel
  - Adaptive prediction panel width and font sizes
  - Better splitter defaults for wide displays
  - Cleaner SLM control / live demo ergonomics
"""

import os
import re
import sys
import glob
import json
import time
import threading
import platform
from typing import Optional
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QTextEdit,
    QGroupBox, QSpinBox, QDoubleSpinBox, QSlider, QFileDialog,
    QSizePolicy, QFrame, QStatusBar, QCheckBox, QSplitter, QScrollArea,
    QApplication, QMessageBox,
)
from PySide6.QtCore import Qt, Slot, QPoint, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QFontMetrics, QGuiApplication

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _demo_trace_ui(msg: str) -> None:
    if os.environ.get("SPECKLE_DEMO_TRACE", "").strip().lower() not in (
        "1", "true", "yes", "on",
    ):
        return
    t = threading.current_thread()
    ts = time.strftime("%H:%M:%S")
    print(
        f"[SPECKLE_DEMO_TRACE {ts} thread={t.name!r} ident={threading.get_ident()}] [UI] "
        f"{msg}",
        flush=True,
    )


from gui.slm_window         import SLMWindow
from gui.camera_worker      import CameraWorker
from gui.camera_scanner     import CameraDeviceEntry, scan_all_cameras
from gui.mv_camera_worker   import MvCameraWorker
from gui.inference_worker   import InferenceWorker
from gui.robot_panel        import RobotPanel
from gui.challenge_widgets  import (
    ChallengePreviewWidget,
    RecognitionResultWidget,
    normalize_label,
)
from gui.prediction_display import PredictionDisplaySmoother
from gui.voice_announcer import (
    PHRASE_TEST,
    PHRASE_WAITING,
    VoiceAnnouncer,
    phrase_for_decision,
)
from gui.challenge_manifest import (
    challenge_inputs_dir,
    load_challenge_manifest,
    manifest_path,
    resolve_manifest_entries,
)
from gui.demo_presentation  import add_card_title, demo_font, style_control_section
from gui.cute_style         import PREMIUM_STYLE
from gui.effects            import make_glow, apply_premium_shadow


DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #3a3a5c;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: bold;
    color: #a0c4ff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
}
QPushButton {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #3a3a5c;
    border-radius: 5px;
    padding: 6px 12px;
    min-height: 28px;
}
QPushButton:hover  { background-color: #0f3460; border-color: #a0c4ff; }
QPushButton:pressed{ background-color: #0a2040; }
QPushButton:disabled { color: #555; border-color: #2a2a3c; }

QPushButton#accent {
    background-color: #0f3460;
    border-color: #4dabf7;
    color: #ffffff;
    font-weight: bold;
}
QPushButton#accent:hover { background-color: #1a4a7a; }

QPushButton#danger {
    background-color: #3c1515;
    border-color: #e06c75;
    color: #ff8080;
}
QPushButton#danger:hover { background-color: #5a2020; }

QComboBox {
    background-color: #16213e;
    border: 1px solid #3a3a5c;
    border-radius: 4px;
    padding: 4px 8px;
    color: #e0e0e0;
    min-height: 26px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background-color: #16213e;
    border: 1px solid #3a3a5c;
    selection-background-color: #0f3460;
}
QLineEdit {
    background-color: #16213e;
    border: 1px solid #3a3a5c;
    border-radius: 4px;
    padding: 4px 8px;
    color: #e0e0e0;
    min-height: 26px;
}
QTextEdit {
    background-color: #0d0d1a;
    border: 1px solid #2a2a3c;
    border-radius: 4px;
    color: #a0b8c0;
    font-family: 'Consolas', monospace;
    font-size: 11px;
}
QLabel#pred_letter {
    color: #4dabf7;
    font-weight: bold;
    background-color: #0d0d1a;
    border: 2px solid #3a3a5c;
    border-radius: 10px;
    qproperty-alignment: AlignCenter;
}
QLabel#pred_smooth {
    color: #51cf66;
    font-weight: bold;
    background-color: #0d0d1a;
    border: 2px solid #2a5c2a;
    border-radius: 10px;
    qproperty-alignment: AlignCenter;
}
QLabel#conf_label {
    color: #ffd43b;
    font-weight: bold;
    qproperty-alignment: AlignCenter;
}
QLabel#status_ok  { color: #51cf66; font-weight: bold; }
QLabel#status_err { color: #e06c75; font-weight: bold; }
QFrame#divider { background-color: #3a3a5c; }
QStatusBar { background-color: #0d0d1a; color: #888; border-top: 1px solid #2a2a3c; }
QSpinBox, QDoubleSpinBox {
    background-color: #16213e;
    border: 1px solid #3a3a5c;
    border-radius: 4px;
    padding: 2px 6px;
    color: #e0e0e0;
    min-height: 26px;
}
QSlider::groove:horizontal {
    background: #2a2a3c;
    height: 4px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #4dabf7;
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}
QSlider::sub-page:horizontal {
    background: #0f3460;
    border-radius: 2px;
}
QSplitter::handle { background-color: #3a3a5c; }
QSplitter::handle:horizontal { width: 3px; }
QCheckBox { color: #e0e0e0; }
QCheckBox::indicator:checked { background-color: #4dabf7; border-radius: 3px; }
QScrollArea {
    border: none;
    background: transparent;
}
"""


FIBER_MODELS_DIR = os.path.join(ROOT, "models", "final_15fibers")
FIBER_MODELS_DIR_LEGACY = os.path.join(ROOT, "results", "fiber_auth", "fiber_models")
LABEL_MAP_PATH = os.path.join(FIBER_MODELS_DIR, "label_map.json")
LOW_CONFIDENCE_THRESHOLD = 0.60

DEFAULT_TEXT_CHALLENGES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "1", "2", "3", "boy_avatar", "girl_avatar",
]
CHALLENGE_INPUTS_DIR = os.path.join(ROOT, "challenge_inputs")
CHALLENGE_IMAGE_DIRS = [
    CHALLENGE_INPUTS_DIR,
    os.path.join(ROOT, "letter_images"),
    os.path.join(ROOT, "challenge_images"),
]


def _env_manual_screenshot_mode() -> bool:
    """True when running automated/manual documentation capture (not normal lab use)."""
    v = os.environ.get("SPECKLE_MANUAL_SCREENSHOT_MODE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def discover_fiber_models(model_dir: str = None):
    """Return {fiber_name: checkpoint_path} for all Fiber*.pth files."""
    result = {}
    search_dirs = []
    if model_dir:
        search_dirs.append(model_dir)
    else:
        search_dirs.extend([FIBER_MODELS_DIR, FIBER_MODELS_DIR_LEGACY])
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        for f in sorted(os.listdir(directory)):
            if f.endswith(".pth") and not f.endswith(".pth.bak"):
                name = os.path.splitext(f)[0]
                if name.startswith("Fiber") and name not in result:
                    result[name] = os.path.join(directory, f)
    return result


def all_fiber_names() -> list[str]:
    return [f"Fiber{i}" for i in range(1, 16)]


def fiber_model_path(fiber_name: str) -> str:
    return os.path.join(FIBER_MODELS_DIR, f"{fiber_name}.pth")


def _fiber_natural_sort_key(name: str) -> tuple:
    m = re.match(r"^Fiber(\d+)$", name, re.I)
    return (int(m.group(1)), name.lower()) if m else (9999, name.lower())


def load_label_map_class_names() -> list[str]:
    """Load challenge labels from models/final_15fibers/label_map.json if present."""
    for path in (LABEL_MAP_PATH, os.path.join(FIBER_MODELS_DIR_LEGACY, "label_map.json")):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            labels = data.get("labels")
            if labels:
                return list(labels)
            idx_map = data.get("index_to_label", {})
            if idx_map:
                return [idx_map[str(i)] for i in range(len(idx_map))]
        except (json.JSONDecodeError, OSError, KeyError):
            pass
    return []


def discover_fibers(video_dir: str):
    fibers = []
    if not os.path.isdir(video_dir):
        return fibers
    for d in sorted(os.listdir(video_dir)):
        if os.path.isdir(os.path.join(video_dir, d)):
            avis = glob.glob(os.path.join(video_dir, d, "*.avi"))
            if avis:
                fibers.append(d)
    return fibers


def fiber_key(name: str) -> str:
    return name.lower().replace(" ", "_")


def discover_checkpoints(ckpt_dir: str):
    result = {}
    if not os.path.isdir(ckpt_dir):
        return result
    for f in sorted(os.listdir(ckpt_dir)):
        if f.endswith("_best.pth"):
            key = f[:-len("_best.pth")]
            result[key] = os.path.join(ckpt_dir, f)
    return result


class CameraLabel(QLabel):
    """Label that scales camera frame to fit available space."""

    _IDLE_PLACEHOLDER = "Waiting for speckle response"
    _IDLE_FONT_MAX = 37
    _IDLE_FONT_MIN = 30

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("camLabel")
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(False)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pixmap = None
        self._show_idle_placeholder()

    def _idle_stylesheet(self, font_px: int) -> str:
        return (
            "QLabel#camLabel {"
            "  background-color: #000000;"
            "  border: 2px solid #3A3A3C;"
            "  border-radius: 14px;"
            "  color: #AEAEB2;"
            f"  font-size: {font_px}px;"
            "  font-weight: 600;"
            "}"
        )

    def _frame_stylesheet(self) -> str:
        return (
            "QLabel#camLabel {"
            "  background-color: #000000;"
            "  border: 2px solid #3A3A3C;"
            "  border-radius: 14px;"
            "}"
        )

    def _fit_idle_font(self) -> int:
        w = max(120, self.width() - 8)
        for fs in range(self._IDLE_FONT_MAX, self._IDLE_FONT_MIN - 1, -2):
            fm = QFontMetrics(demo_font(fs, weight=QFont.DemiBold))
            if fm.horizontalAdvance(self._IDLE_PLACEHOLDER) <= w:
                return fs
        return self._IDLE_FONT_MIN

    def _show_idle_placeholder(self) -> None:
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setWordWrap(False)
        px = self._fit_idle_font() if self.width() > 40 else self._IDLE_FONT_MAX
        self.setStyleSheet(self._idle_stylesheet(px))
        self.setFont(demo_font(px, weight=QFont.DemiBold))
        self.setText(self._IDLE_PLACEHOLDER)

    def set_frame(self, frame: np.ndarray):
        if frame is None or frame.size == 0:
            return
        h, w = int(frame.shape[0]), int(frame.shape[1])
        if frame.ndim == 2:
            gray = np.ascontiguousarray(frame, dtype=np.uint8)
            bpl = int(gray.strides[0])
            img = QImage(gray.data, w, h, bpl, QImage.Format_Grayscale8).copy()
        elif frame.ndim == 3 and frame.shape[2] >= 3:
            bgr = np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            img = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        else:
            return
        self._pixmap = QPixmap.fromImage(img)
        self.setText("")
        self.setStyleSheet(self._frame_stylesheet())
        self._update_display()

    def resizeEvent(self, event):
        if self._pixmap is None:
            self._show_idle_placeholder()
        else:
            self._update_display()
        super().resizeEvent(event)

    def _update_display(self):
        if self._pixmap is not None:
            scaled = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(scaled)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self._root = ROOT
        self._video_dir = os.path.join(ROOT, "video_capture")
        self._ckpt_dir = os.path.join(ROOT, "checkpoints")
        self._fiber_models: dict = {}
        self._active_fiber: str = ""
        self._slm_window = None
        self._camera_worker = None
        self._infer_worker = InferenceWorker(self)
        self._display_smoother = PredictionDisplaySmoother(
            confidence_threshold=LOW_CONFIDENCE_THRESHOLD,
            hold_sec=2.0,
            banner_hold_sec=2.0,
            granted_release_hold_sec=2.0,
        )
        self._recognition_active = False
        self._last_voice_decision = ""
        self._waiting_voice_spoken = False
        self._voice_announcer = VoiceAnnouncer(enabled=True, log_fn=self._log, parent=self)
        self._banner_hide_timer = QTimer(self)
        self._banner_hide_timer.setSingleShot(True)
        self._banner_hide_timer.timeout.connect(self._hide_overlay_banner)
        self._capture_active = False
        self._last_frame = None
        self._fps = 0.0
        self._preferred_slm_screen = 1
        self._manual_feed_timer: Optional[QTimer] = None
        self._manual_feed_active = False
        self._manual_feed_phase = 0
        self._cam_open_logged = False
        self._first_frame_logged = False
        self.current_challenge_label: Optional[str] = None
        self.last_sent_challenge_label: Optional[str] = None
        self._challenge_source = ""
        self._challenge_image_path: Optional[str] = None
        self._challenge_cycle: list[str] = []
        self._challenge_cycle_index = -1
        self._ppt_challenge_entries: list[dict] = []
        self._challenge_image_by_label: dict[str, str] = {}
        self._manifest_loaded = False

        self._top_splitter = None
        self._left_scroll = None
        self._center_panel = None
        self._right_panel = None
        self._log_box = None
        self._cam_ctrl_widgets: list = []

        self._setup_ui()
        self._apply_style()
        self._connect_signals()
        self._infer_worker.start()
        if not self._try_load_challenge_manifest():
            self._challenge_cycle = self._discover_challenge_cycle()
            self._challenge_cycle_index = -1
        self._apply_startup_challenge_ui()
        self._refresh_fiber_list()
        self._refresh_screen_list()
        self._apply_responsive_metrics(force=True)
        self._update_recognition_status_label()
        self._refresh_demo_step_status()
        self._log("Homepage simplified for demo mode.")
        self._log(
            "Speckle-PUF demo ready: select a challenge, send it to the SLM, "
            "start CCD, then click Start Recognition."
        )

    def _setup_ui(self):
        self.setWindowTitle("Speckle-PUF Live Demo")
        self.resize(1480, 900)
        self.setMinimumSize(1180, 720)

        central = QWidget()
        self.setCentralWidget(central)

        self._top_splitter = QSplitter(Qt.Horizontal)
        self._top_splitter.setChildrenCollapsible(False)
        self._top_splitter.setHandleWidth(3)

        left_widget = self._build_left_panel()
        self._left_scroll = QScrollArea()
        self._left_scroll.setWidgetResizable(True)
        self._left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._left_scroll.setWidget(left_widget)
        self._left_scroll.setMinimumWidth(430)
        self._left_scroll.setMaximumWidth(480)
        self._left_scroll.setFrameShape(QFrame.NoFrame)
        self._top_splitter.addWidget(self._left_scroll)

        self._center_panel = self._build_center_panel()
        self._top_splitter.addWidget(self._center_panel)

        self._right_panel = self._build_right_panel()
        self._top_splitter.addWidget(self._right_panel)

        self._top_splitter.setStretchFactor(0, 0)
        self._top_splitter.setStretchFactor(1, 1)
        self._top_splitter.setStretchFactor(2, 0)
        self._top_splitter.setSizes([460, 620, 400])

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 12)
        main_layout.setSpacing(12)
        main_layout.addWidget(self._top_splitter, stretch=1)

        self._log_box = QGroupBox("Log")
        self._log_box.setObjectName("logPanel")
        log_layout = QVBoxLayout(self._log_box)
        log_layout.setContentsMargins(10, 10, 10, 8)
        self._log_text = QTextEdit()
        self._log_text.setObjectName("demoLogText")
        self._log_text.setReadOnly(True)
        self._log_text.setMinimumHeight(110)
        self._log_text.setMaximumHeight(140)
        log_layout.addWidget(self._log_text)
        apply_premium_shadow(self._log_text)
        main_layout.addWidget(self._log_box)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._lbl_device = QLabel("Device: checking...")
        self._lbl_fps = QLabel("FPS: --")
        self._lbl_model = QLabel("Model: none")
        self._status_bar.addWidget(self._lbl_device)
        self._status_bar.addWidget(self._make_sep())
        self._status_bar.addWidget(self._lbl_model)
        self._status_bar.addWidget(self._make_sep())
        self._status_bar.addPermanentWidget(self._lbl_fps)
        self._update_device_label()

    def _make_sep(self):
        f = QFrame()
        f.setFrameShape(QFrame.VLine)
        f.setObjectName("divider")
        return f

    def _discover_challenge_cycle(self) -> list[str]:
        """Build ordered challenge labels for Prev/Next (manifest, label map, images)."""
        seen: set[str] = set()
        out: list[str] = []

        manifest = load_challenge_manifest()
        if manifest:
            for entry in resolve_manifest_entries(manifest):
                key = entry["label"].strip()
                if key and key not in seen:
                    seen.add(key)
                    out.append(key)

        for label in load_label_map_class_names():
            key = label.strip()
            if key and key not in seen:
                seen.add(key)
                out.append(key)
        for label in DEFAULT_TEXT_CHALLENGES:
            key = label.strip()
            if key and key not in seen:
                seen.add(key)
                out.append(key)
        for base in CHALLENGE_IMAGE_DIRS:
            if not os.path.isdir(base):
                continue
            for fname in sorted(os.listdir(base)):
                low = fname.lower()
                if low == "manifest.json":
                    continue
                if not low.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    continue
                stem = os.path.splitext(fname)[0]
                if stem and stem not in seen:
                    seen.add(stem)
                    out.append(stem)
        return out or ["A"]

    def _auth_challenge_label(self) -> str:
        """Challenge used for recognition (set only after successful Send to SLM)."""
        return (self.last_sent_challenge_label or "").strip()

    def _has_challenge_selected(self) -> bool:
        return bool((self.current_challenge_label or "").strip())

    def _apply_startup_challenge_ui(self) -> None:
        """Blank preview and WAITING recognition until the user selects a challenge."""
        self.current_challenge_label = None
        self._challenge_image_path = None
        self._challenge_source = ""
        if self._manifest_loaded:
            self._challenge_preview.set_manifest_ready(len(self._challenge_cycle))
        else:
            self._challenge_preview.clear_challenge()
        self._input_letter.blockSignals(True)
        self._input_letter.setText("")
        self._input_letter.blockSignals(False)
        self._recognition_result.clear_result()
        self._robot_panel.set_challenge_label("")
        self._robot_panel.on_idle()

    def _try_load_challenge_manifest(self) -> bool:
        """Load challenge_inputs/manifest.json into memory without selecting a challenge."""
        manifest = load_challenge_manifest()
        if not manifest:
            return False
        entries = resolve_manifest_entries(manifest)
        if not entries:
            self._log(f"[Challenge] Manifest empty or missing images: {manifest_path()}")
            return False

        self._ppt_challenge_entries = entries
        self._challenge_image_by_label = {e["label"]: e["image"] for e in entries}
        self._challenge_cycle = [e["label"] for e in entries]
        self._challenge_cycle_index = -1
        self._manifest_loaded = True
        self.current_challenge_label = None
        self.last_sent_challenge_label = None
        self._challenge_image_path = None
        self._log(
            f"[Challenge] Challenge set loaded ({len(entries)} items) from "
            f"{challenge_inputs_dir()} — use Prev/Next to select, then Send to SLM"
        )
        return True

    def _reload_challenge_manifest(self) -> None:
        if self._try_load_challenge_manifest():
            self._apply_startup_challenge_ui()
            QMessageBox.information(
                self,
                "Challenge set",
                f"Loaded {len(self._ppt_challenge_entries)} challenges from\n"
                f"{manifest_path()}\n\n"
                "Use Prev/Next to select a challenge, then Send to SLM.",
            )
        else:
            QMessageBox.warning(
                self,
                "Challenge set",
                f"No manifest found at:\n{manifest_path()}\n\n"
                "Run: python scripts/export_ppt_challenges.py --input input.pptx",
            )

    def _bind_collapsible(self, box: QGroupBox, body: QWidget) -> None:
        """Hide advanced section content when collapsed so it cannot block clicks."""
        body.setVisible(box.isChecked())
        box.toggled.connect(body.setVisible)

    def _build_left_panel(self):
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        layout.setContentsMargins(6, 6, 6, 6)

        # ── Demo status checklist ───────────────────────────────────────
        status_box = QGroupBox("")
        style_control_section(status_box)
        status_outer = QVBoxLayout(status_box)
        status_outer.setContentsMargins(4, 4, 4, 4)
        add_card_title(status_outer, "Demo status")
        self._lbl_step_fiber = QLabel("Fiber: checking…")
        self._lbl_step_challenge = QLabel("Challenge: not sent")
        self._lbl_step_ccd = QLabel("CCD: stopped")
        self._lbl_step_recognition = QLabel("Recognition: stopped")
        for lbl in (
            self._lbl_step_fiber,
            self._lbl_step_challenge,
            self._lbl_step_ccd,
            self._lbl_step_recognition,
        ):
            lbl.setObjectName("demoHintLabel")
            lbl.setFont(demo_font(14))
            status_outer.addWidget(lbl)
        layout.addWidget(status_box)

        # ── Connected Fiber ───────────────────────────────────────────
        fiber_box = QGroupBox("")
        style_control_section(fiber_box)
        fiber_outer = QVBoxLayout(fiber_box)
        fiber_outer.setContentsMargins(4, 4, 4, 4)
        add_card_title(fiber_outer, "Connected Fiber")
        fl = QGridLayout()
        fl.setSpacing(8)
        fl.setContentsMargins(4, 0, 4, 4)

        self._combo_fiber = QComboBox()
        self._combo_fiber.setMinimumHeight(44)
        self._combo_fiber.setFont(demo_font(16))
        self._combo_fiber.setToolTip(
            "Select the fiber connected to the setup; FiberN.pth loads automatically."
        )
        self._combo_fiber.currentIndexChanged.connect(self._on_fiber_selected)
        fl.addWidget(self._combo_fiber, 0, 0, 1, 2)

        self._btn_refresh_fiber = QPushButton("Refresh")
        self._btn_refresh_fiber.setFixedWidth(84)
        self._btn_refresh_fiber.setMinimumHeight(44)
        self._btn_refresh_fiber.clicked.connect(self._refresh_fiber_list)
        fl.addWidget(self._btn_refresh_fiber, 0, 2)

        self._lbl_model_status = QLabel("No model loaded")
        self._lbl_model_status.setObjectName("demoHintLabel")
        self._lbl_model_status.setFont(demo_font(15, weight=QFont.DemiBold))
        self._lbl_model_status.setWordWrap(True)
        fl.addWidget(self._lbl_model_status, 1, 0, 1, 3)

        self._lbl_model_path = QLabel("")
        self._lbl_model_path.setObjectName("demoHintLabel")
        self._lbl_model_path.setWordWrap(True)
        self._lbl_model_path.hide()
        fl.addWidget(self._lbl_model_path, 2, 0, 1, 3)

        self._lbl_auth_warning = QLabel("")
        self._lbl_auth_warning.setWordWrap(True)
        self._lbl_auth_warning.setVisible(False)
        fl.addWidget(self._lbl_auth_warning, 3, 0, 1, 3)
        fiber_outer.addLayout(fl)
        layout.addWidget(fiber_box)

        # ── Challenge Input ───────────────────────────────────────────
        self._challenge_preview = ChallengePreviewWidget()
        self._challenge_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self._challenge_preview, stretch=0)

        ch_nav = QHBoxLayout()
        ch_nav.setSpacing(8)
        self._btn_prev = QPushButton("Prev")
        self._btn_next = QPushButton("Next")
        for btn in (self._btn_prev, self._btn_next):
            btn.setMinimumHeight(40)
            btn.setFont(demo_font(15, weight=QFont.DemiBold))
        self._btn_prev.clicked.connect(self._prev_challenge)
        self._btn_next.clicked.connect(self._next_challenge)
        ch_nav.addWidget(self._btn_prev, 1)
        ch_nav.addWidget(self._btn_next, 1)
        layout.addLayout(ch_nav)

        self._btn_send_slm = QPushButton("Send to SLM")
        self._btn_send_slm.setObjectName("primary")
        self._btn_send_slm.setMinimumHeight(46)
        self._btn_send_slm.setFont(demo_font(16, weight=QFont.Bold))
        self._btn_send_slm.clicked.connect(self._send_to_slm)
        layout.addWidget(self._btn_send_slm)

        self._btn_show_slm = QPushButton("Open SLM Window")
        self._btn_show_slm.setMinimumHeight(40)
        self._btn_show_slm.setFont(demo_font(15, weight=QFont.DemiBold))
        self._btn_show_slm.clicked.connect(self._toggle_slm_window)
        layout.addWidget(self._btn_show_slm)

        self._lbl_sent_challenge = QLabel("Sent to SLM: —")
        self._lbl_sent_challenge.setObjectName("demoHintLabel")
        self._lbl_sent_challenge.setFont(demo_font(14, weight=QFont.DemiBold))
        self._lbl_sent_challenge.setWordWrap(True)
        layout.addWidget(self._lbl_sent_challenge)

        # ── CCD Acquisition (homepage minimal) ────────────────────────
        cam_box = QGroupBox("")
        style_control_section(cam_box)
        cam_outer = QVBoxLayout(cam_box)
        cam_outer.setContentsMargins(4, 4, 4, 4)
        add_card_title(cam_outer, "CCD Acquisition")
        cl = QGridLayout()
        cl.setSpacing(8)
        cl.setColumnStretch(1, 1)

        self._btn_scan_cameras = QPushButton("Scan Cameras")
        self._btn_scan_cameras.setMinimumHeight(42)
        self._btn_scan_cameras.setFont(demo_font(15, weight=QFont.DemiBold))
        self._btn_scan_cameras.setToolTip(
            "Detect OpenCV camera indices and MindVision SDK devices."
        )
        self._btn_scan_cameras.clicked.connect(self._scan_cameras)
        cl.addWidget(self._btn_scan_cameras, 0, 0, 1, 2)

        cam_src_lbl = QLabel("Camera Source:")
        cam_src_lbl.setFont(demo_font(14, weight=QFont.DemiBold))
        cl.addWidget(cam_src_lbl, 1, 0)
        self._combo_camera_source = QComboBox()
        self._combo_camera_source.setMinimumHeight(40)
        self._combo_camera_source.addItem("Click Scan Cameras", None)
        cl.addWidget(self._combo_camera_source, 1, 1, 1, 2)

        self._btn_start_cam = QPushButton("Start CCD")
        self._btn_start_cam.setObjectName("primary")
        self._btn_start_cam.setMinimumHeight(46)
        self._btn_start_cam.setFont(demo_font(16, weight=QFont.Bold))
        self._btn_start_cam.clicked.connect(self._start_camera)
        cl.addWidget(self._btn_start_cam, 2, 0)

        self._btn_stop_cam = QPushButton("Stop CCD")
        self._btn_stop_cam.setObjectName("danger")
        self._btn_stop_cam.setMinimumHeight(44)
        self._btn_stop_cam.setEnabled(False)
        self._btn_stop_cam.clicked.connect(self._stop_camera)
        cl.addWidget(self._btn_stop_cam, 2, 1)

        self._lbl_source = QLabel("Camera status: not connected")
        self._lbl_source.setObjectName("demoHintLabel")
        self._lbl_source.setWordWrap(True)
        cl.addWidget(self._lbl_source, 3, 0, 1, 2)
        cam_outer.addLayout(cl)
        layout.addWidget(cam_box)
        self._group_camera_video = cam_box
        self._camera_catalog: list[CameraDeviceEntry] = []

        # ── Offline video input (collapsed, below CCD) ──────────────────
        self._offline_input_box = QGroupBox("Offline Demo / Video Input")
        self._offline_input_box.setCheckable(True)
        self._offline_input_box.setChecked(False)
        self._offline_input_box.setStyleSheet(
            "QGroupBox { font-weight: 600; color: #636366; }"
        )
        offline_outer = QVBoxLayout(self._offline_input_box)
        offline_outer.setContentsMargins(8, 4, 8, 8)
        offline_body = QWidget()
        offline_layout = QVBoxLayout(offline_body)
        offline_layout.setContentsMargins(0, 0, 0, 0)
        offline_hint = QLabel(
            "Load a recorded video when the CCD is unavailable or for offline demo."
        )
        offline_hint.setObjectName("demoHintLabel")
        offline_hint.setWordWrap(True)
        offline_layout.addWidget(offline_hint)
        self._btn_load_video = QPushButton("Load Video File")
        self._btn_load_video.setMinimumHeight(40)
        self._btn_load_video.setToolTip(
            "Play an AVI/MP4 file into the live preview and recognition pipeline."
        )
        self._btn_load_video.clicked.connect(self._load_video_file)
        offline_layout.addWidget(self._btn_load_video)
        offline_outer.addWidget(offline_body)
        self._bind_collapsible(self._offline_input_box, offline_body)
        layout.addWidget(self._offline_input_box)

        # ── Recognition ─────────────────────────────────────────────────
        recog_box = QGroupBox("")
        style_control_section(recog_box)
        recog_outer = QVBoxLayout(recog_box)
        recog_outer.setContentsMargins(4, 4, 4, 4)
        add_card_title(recog_outer, "Recognition")
        rl = QGridLayout()
        rl.setSpacing(8)

        self._btn_start_recognition = QPushButton("Start Recognition")
        self._btn_start_recognition.setObjectName("primary")
        self._btn_start_recognition.setMinimumHeight(46)
        self._btn_start_recognition.setFont(demo_font(16, weight=QFont.Bold))
        self._btn_start_recognition.clicked.connect(self._start_recognition)
        rl.addWidget(self._btn_start_recognition, 0, 0, 1, 2)

        self._btn_stop_recognition = QPushButton("Stop Recognition")
        self._btn_stop_recognition.setObjectName("danger")
        self._btn_stop_recognition.setMinimumHeight(44)
        self._btn_stop_recognition.setEnabled(False)
        self._btn_stop_recognition.clicked.connect(self._stop_recognition)
        rl.addWidget(self._btn_stop_recognition, 1, 0, 1, 2)

        self._chk_voice = QCheckBox("Voice announcement")
        self._chk_voice.setChecked(True)
        self._chk_voice.toggled.connect(self._on_voice_toggled)
        rl.addWidget(self._chk_voice, 2, 0)

        self._btn_test_voice = QPushButton("Test Voice")
        self._btn_test_voice.setMinimumHeight(38)
        self._btn_test_voice.clicked.connect(self._test_voice)
        rl.addWidget(self._btn_test_voice, 2, 1)

        self._lbl_recognition_status = QLabel("Recognition: stopped")
        self._lbl_recognition_status.setObjectName("demoHintLabel")
        self._lbl_recognition_status.setWordWrap(True)
        rl.addWidget(self._lbl_recognition_status, 3, 0, 1, 2)
        recog_outer.addLayout(rl)
        layout.addWidget(recog_box)

        # ── Advanced Settings (collapsed) ─────────────────────────────
        self._advanced_box = QGroupBox("Advanced Settings")
        self._advanced_box.setCheckable(True)
        self._advanced_box.setChecked(False)
        self._advanced_box.setStyleSheet(
            "QGroupBox { font-weight: 600; color: #636366; }"
        )
        adv_outer = QVBoxLayout(self._advanced_box)
        adv_outer.setContentsMargins(8, 4, 8, 8)
        adv_body = QWidget()
        adv_layout = QVBoxLayout(adv_body)
        adv_layout.setContentsMargins(0, 0, 0, 0)
        adv_layout.setSpacing(8)

        self._btn_load_challenge_set = QPushButton("Load Challenge Set")
        self._btn_load_challenge_set.setMinimumHeight(36)
        self._btn_load_challenge_set.setToolTip(
            "Reload challenge_inputs/manifest.json (exported from input.pptx)."
        )
        self._btn_load_challenge_set.clicked.connect(self._reload_challenge_manifest)
        adv_layout.addWidget(self._btn_load_challenge_set)

        cam_adv = QGridLayout()
        cam_adv.setSpacing(6)
        cam_adv.addWidget(QLabel("Resolution:"), 0, 0)
        self._combo_cam_res = QComboBox()
        self._combo_cam_res.setMinimumHeight(36)
        self._combo_cam_res.addItem("Auto (default)", (None, None))
        self._combo_cam_res.addItem("2048×1536", (2048, 1536))
        self._combo_cam_res.addItem("1920×1440", (1920, 1440))
        self._combo_cam_res.addItem("1280×960", (1280, 960))
        self._combo_cam_res.addItem("1024×768", (1024, 768))
        self._combo_cam_res.addItem("640×480", (640, 480))
        self._combo_cam_res.setCurrentIndex(1)
        cam_adv.addWidget(self._combo_cam_res, 0, 1, 1, 2)
        adv_layout.addLayout(cam_adv)

        sl = QGridLayout()
        sl.setSpacing(6)
        sl.addWidget(QLabel("SLM screen:"), 0, 0)
        self._combo_slm_screen = QComboBox()
        self._combo_slm_screen.setMinimumHeight(34)
        sl.addWidget(self._combo_slm_screen, 0, 1)
        self._btn_refresh_screens = QPushButton("Refresh")
        self._btn_refresh_screens.clicked.connect(self._refresh_screen_list)
        sl.addWidget(self._btn_refresh_screens, 0, 2)

        self._chk_slm_fullscreen = QCheckBox("Fullscreen on selected screen")
        self._chk_slm_fullscreen.setChecked(True)
        sl.addWidget(self._chk_slm_fullscreen, 1, 0, 1, 3)

        self._btn_move_slm = QPushButton("Move SLM to Selected Screen")
        self._btn_move_slm.clicked.connect(self._move_slm_to_selected_screen)
        sl.addWidget(self._btn_move_slm, 2, 0, 1, 3)

        self._btn_test_slm = QPushButton("Test SLM Output")
        self._btn_test_slm.clicked.connect(self._test_slm_output)
        sl.addWidget(self._btn_test_slm, 3, 0, 1, 3)

        self._spin_font = QSpinBox()
        self._spin_font.setRange(50, 800)
        self._spin_font.setValue(400)
        self._spin_font.setSingleStep(20)
        self._spin_font.valueChanged.connect(self._on_font_size_changed)
        sl.addWidget(QLabel("Font size:"), 4, 0)
        sl.addWidget(self._spin_font, 4, 1)

        self._chk_slm_stretch = QCheckBox("Stretch to fill")
        self._chk_slm_stretch.setChecked(True)
        self._chk_slm_stretch.toggled.connect(self._on_slm_stretch_toggled)
        sl.addWidget(self._chk_slm_stretch, 4, 2)

        self._lbl_screen_hint = QLabel("Detected displays: checking...")
        self._lbl_screen_hint.setObjectName("demoHintLabel")
        self._lbl_screen_hint.setWordWrap(True)
        sl.addWidget(self._lbl_screen_hint, 5, 0, 1, 3)

        sl.addWidget(QLabel("Challenge type:"), 6, 0)
        self._combo_challenge_type = QComboBox()
        self._combo_challenge_type.addItems(["Text", "Image"])
        self._combo_challenge_type.currentTextChanged.connect(
            self._on_challenge_type_changed
        )
        sl.addWidget(self._combo_challenge_type, 6, 1)

        self._input_letter = QLineEdit("")
        self._input_letter.setMaxLength(64)
        self._input_letter.setPlaceholderText("A, B, 1, boy")
        self._input_letter.textChanged.connect(self._on_challenge_text_changed)
        sl.addWidget(self._input_letter, 6, 2)

        self._btn_load_img = QPushButton("Load Image to SLM")
        self._btn_load_img.clicked.connect(self._load_image_to_slm)
        sl.addWidget(self._btn_load_img, 7, 0, 1, 3)
        adv_layout.addLayout(sl)

        adv_layout.addWidget(self._build_cam_settings_box())

        inf_box = QGroupBox("Inference Settings")
        il = QGridLayout(inf_box)
        il.setSpacing(6)
        il.addWidget(QLabel("Infer every N frames:"), 0, 0)
        self._spin_infer_every = QSpinBox()
        self._spin_infer_every.setRange(1, 30)
        self._spin_infer_every.setValue(4)
        self._spin_infer_every.valueChanged.connect(
            lambda v: self._infer_worker.set_infer_every(v)
        )
        il.addWidget(self._spin_infer_every, 0, 1)
        il.addWidget(QLabel("Vote window:"), 1, 0)
        self._spin_vote = QSpinBox()
        self._spin_vote.setRange(1, 30)
        self._spin_vote.setValue(8)
        self._spin_vote.valueChanged.connect(
            lambda v: self._infer_worker.set_vote_window(v)
        )
        il.addWidget(self._spin_vote, 1, 1)
        self._chk_infer_active = QCheckBox("Recognition active (sync)")
        self._chk_infer_active.setChecked(False)
        self._chk_infer_active.setToolTip(
            "Mirrors Start/Stop Recognition on the homepage."
        )
        self._chk_infer_active.toggled.connect(self._on_infer_active_toggled)
        il.addWidget(self._chk_infer_active, 2, 0, 1, 2)
        adv_layout.addWidget(inf_box)

        adv_outer.addWidget(adv_body)
        self._bind_collapsible(self._advanced_box, adv_body)
        layout.addWidget(self._advanced_box)
        layout.addStretch(1)

        return container

    def _refresh_demo_step_status(self) -> None:
        if not hasattr(self, "_lbl_step_fiber"):
            return
        if self._infer_worker._model is not None and self._active_fiber:
            self._lbl_step_fiber.setText(f"Fiber: loaded ({self._active_fiber})")
        elif self._active_fiber:
            self._lbl_step_fiber.setText(f"Fiber: missing model ({self._active_fiber})")
        else:
            self._lbl_step_fiber.setText("Fiber: not loaded")

        sent = (self.last_sent_challenge_label or "").strip()
        if sent:
            self._lbl_step_challenge.setText(f"Challenge: sent ({sent})")
            self._lbl_sent_challenge.setText(f"Sent to SLM: {sent}")
        else:
            self._lbl_step_challenge.setText("Challenge: not sent")
            self._lbl_sent_challenge.setText("Sent to SLM: —")

        if self._capture_active:
            src = self._lbl_source.text() if hasattr(self, "_lbl_source") else ""
            if src.startswith("File:"):
                self._lbl_step_ccd.setText("CCD: video file active")
            else:
                self._lbl_step_ccd.setText("CCD: running")
        else:
            self._lbl_step_ccd.setText("CCD: stopped")

        if self._recognition_active:
            self._lbl_step_recognition.setText("Recognition: running")
        else:
            self._lbl_step_recognition.setText("Recognition: stopped")

    def _build_center_panel(self) -> QFrame:
        self._cam_card = QFrame()
        self._cam_card.setObjectName("camCard")
        self._cam_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        outer = QVBoxLayout(self._cam_card)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        title = QLabel("Live speckle response")
        title.setObjectName("camTitle")
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(False)
        title.setFont(demo_font(30, bold=True))
        outer.addWidget(title)
        outer.addSpacing(8)

        self._cam_label = CameraLabel()
        self._cam_label.setObjectName("camLabel")
        self._cam_label.setMinimumSize(420, 320)
        self._cam_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._cam_label.setAlignment(Qt.AlignCenter)
        outer.addWidget(self._cam_label, stretch=1)

        self._overlay_banner = QLabel(self._cam_card)
        self._overlay_banner.setObjectName("overlayBanner")
        self._overlay_banner.setAlignment(Qt.AlignCenter)
        self._overlay_banner.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._overlay_banner.hide()

        self._cam_card.setStyleSheet(
            "QFrame#camCard {"
            "  background-color: #FFFFFF;"
            "  border: 1px solid #D1D1D6;"
            "  border-radius: 16px;"
            "}"
        )
        self._cam_glow = make_glow(self._cam_label, color="#3ddc84", radius=0)
        return self._cam_card

    def _build_right_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("resultPanel")
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(8, 8, 8, 8)

        self._recognition_result = RecognitionResultWidget()
        apply_premium_shadow(self._recognition_result)
        layout.addWidget(self._recognition_result, stretch=0)

        self._robot_card = QFrame()
        self._robot_card.setObjectName("robotPanel")
        robot_card_layout = QVBoxLayout(self._robot_card)
        robot_card_layout.setContentsMargins(0, 0, 0, 0)
        robot_card_layout.setSpacing(0)
        self._robot_panel = RobotPanel(self._robot_card)
        self._robot_panel.attach_banner_callback(self._show_overlay_banner)
        robot_card_layout.addWidget(self._robot_panel, stretch=1)
        apply_premium_shadow(self._robot_card)
        layout.addWidget(self._robot_card, stretch=1)

        panel.setMinimumWidth(360)
        panel.setMaximumWidth(420)
        return panel

    def _apply_style(self):
        self.setStyleSheet(PREMIUM_STYLE)

    def _apply_responsive_metrics(self, force: bool = False):
        w = max(1, self.width())
        h = max(1, self.height())

        left_width = max(430, min(480, int(w * 0.30)))
        right_width = max(360, min(420, int(w * 0.26)))
        log_h = max(110, min(130, 120))

        if self._left_scroll is not None:
            self._left_scroll.setMinimumWidth(left_width)
            self._left_scroll.setMaximumWidth(left_width + 16)

        if self._right_panel is not None:
            self._right_panel.setMinimumWidth(right_width)
            self._right_panel.setMaximumWidth(right_width + 16)

        if self._top_splitter is not None and (force or w > 0):
            center_width = max(440, w - left_width - right_width - 56)
            self._top_splitter.setSizes([left_width, center_width, right_width])

        if self._log_text is not None:
            self._log_text.setMinimumHeight(log_h)
            self._log_text.setMaximumHeight(log_h)

        if self._cam_label is not None:
            cam_max_h = max(340, min(540, int(h * 0.52)))
            self._cam_label.setMaximumHeight(cam_max_h)

        if hasattr(self, "_challenge_preview"):
            self._challenge_preview.apply_metrics(h)
        if hasattr(self, "_recognition_result"):
            self._recognition_result.apply_metrics(h)
        if hasattr(self, "_robot_panel"):
            self._robot_panel.apply_metrics(h)

    def resizeEvent(self, event):
        self._apply_responsive_metrics()
        super().resizeEvent(event)

    def _update_device_label(self):
        try:
            import torch
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
                self._lbl_device.setText(f"Device: CUDA ({gpu})")
                self._lbl_device.setObjectName("status_ok")
            else:
                self._lbl_device.setText("Device: CPU (no GPU)")
                self._lbl_device.setStyleSheet("color: #ffd43b;")
        except ImportError:
            self._lbl_device.setText("Device: PyTorch not found")

    def _connect_signals(self):
        self._infer_worker.prediction_ready.connect(self._on_prediction)
        self._infer_worker.model_loaded.connect(self._on_model_loaded)
        self._infer_worker.error.connect(self._on_infer_error)

    def _refresh_fiber_list(self) -> None:
        """Populate connected-fiber combo (Fiber1–Fiber15) and load FiberN.pth."""
        discovered = discover_fiber_models()
        prev = self._combo_fiber.currentText() if hasattr(self, "_combo_fiber") else ""
        self._combo_fiber.blockSignals(True)
        self._combo_fiber.clear()

        names = all_fiber_names()
        self._fiber_models = {}
        for name in names:
            path = discovered.get(name) or fiber_model_path(name)
            self._fiber_models[name] = path
            self._combo_fiber.addItem(name)

        found = [n for n in names if os.path.isfile(self._fiber_models[n])]
        if found:
            self._log(f"Fiber models available: {found}")
        else:
            self._log(f"[WARNING] No Fiber*.pth in {FIBER_MODELS_DIR}")

        env_name = os.environ.get("SPECKLE_DEFAULT_FIBER", "").strip()
        if env_name and self._combo_fiber.findText(env_name) >= 0:
            self._combo_fiber.setCurrentText(env_name)
        elif prev and self._combo_fiber.findText(prev) >= 0:
            self._combo_fiber.setCurrentText(prev)

        self._combo_fiber.blockSignals(False)

        if self._combo_fiber.count() > 0:
            self._active_fiber = ""
            self._on_fiber_selected(self._combo_fiber.currentIndex())

    def _on_fiber_selected(self, index: int) -> None:
        if index < 0:
            return
        fiber = self._combo_fiber.currentText()
        if not fiber:
            return
        path = self._fiber_models.get(fiber) or fiber_model_path(fiber)
        rel = os.path.join("models", "final_15fibers", f"{fiber}.pth")
        if not os.path.isfile(path):
            self._lbl_model_status.setStyleSheet("color: #e06c75;")
            self._lbl_model_status.setText(f"Model file missing: {rel}")
            self._lbl_model.setText("Model: missing")
            self._active_fiber = ""
            self._log(f"[WARNING] Model file missing: {path}")
            self._refresh_demo_step_status()
            return

        self._active_fiber = fiber
        self._lbl_model_status.setStyleSheet("color: #a0c4ff;")
        self._lbl_model_status.setText(f"Loading: {fiber} …")
        self._lbl_model.setText(f"Model: loading {fiber}…")
        self._log(f"Loading connected fiber model {fiber} ({rel}) …")
        ok = self._infer_worker.load_model(path)
        if not ok:
            self._lbl_model_status.setStyleSheet("color: #e06c75;")
            self._lbl_model_status.setText(f"Load failed: {fiber}")
            self._lbl_model.setText("Model: load failed")
            self._log("[ERROR] Model load failed")

    def _describe_screen(self, idx, screen):
        geom = screen.geometry()
        name = screen.name() or f"Screen {idx}"
        primary = QGuiApplication.primaryScreen()
        suffix = " [Primary]" if screen == primary else ""
        return f"{idx}: {name} | {geom.width()}x{geom.height()} @ ({geom.x()},{geom.y()}){suffix}"

    def _log_all_screens(self):
        """Enumerate Qt screens for SLM debugging."""
        screens = QGuiApplication.screens()
        primary = QGuiApplication.primaryScreen()
        self._log(f"[Screen diag] platform={platform.system()} count={len(screens)}")
        if len(screens) <= 1:
            self._log(
                "[Screen diag] Only one display detected by Qt. "
                "If the SLM should appear as a second monitor, use System Settings → Displays (Extend). "
                "A mirrored-only layout may still expose two screens; if count stays 1, replug the cable."
            )
        for idx, s in enumerate(screens):
            g = s.geometry()
            name = s.name() or f"Screen {idx}"
            dpr = s.devicePixelRatio()
            is_pri = s == primary
            self._log(
                f"[Screen {idx}] name={name}, geometry=({g.x()},{g.y()},{g.width()},{g.height()}), "
                f"dpr={dpr}, primary={is_pri}"
            )

    def _refresh_screen_list(self):
        current_data = None
        if hasattr(self, "_combo_slm_screen") and self._combo_slm_screen.count() > 0:
            current_data = self._combo_slm_screen.currentData()

        screens = QGuiApplication.screens()
        self._combo_slm_screen.clear()

        if not screens:
            self._combo_slm_screen.addItem("No screens detected", 0)
            self._lbl_screen_hint.setText("No displays detected by Qt.")
            return

        selected_idx = 0
        for idx, screen in enumerate(screens):
            self._combo_slm_screen.addItem(self._describe_screen(idx, screen), idx)
            if current_data == idx:
                selected_idx = idx

        if current_data is None and self._preferred_slm_screen < len(screens):
            selected_idx = self._preferred_slm_screen

        self._combo_slm_screen.setCurrentIndex(selected_idx)
        self._preferred_slm_screen = self._combo_slm_screen.currentData() or 0
        self._lbl_screen_hint.setText(
            f"{len(screens)} display(s) detected. Pick the output wired to the SLM or projector "
            "(e.g. HDMI). Mirroring alone does not replace fullscreen output on the chosen screen."
        )
        self._log_all_screens()

    def _selected_screen(self):
        screens = QGuiApplication.screens()
        if not screens:
            return None
        idx = self._combo_slm_screen.currentData()
        if idx is None:
            idx = 0
        idx = max(0, min(int(idx), len(screens) - 1))
        self._preferred_slm_screen = idx
        return screens[idx]

    def _ensure_slm_window(self):
        if self._slm_window is None:
            self._slm_window = SLMWindow()
            self._slm_window.diagnostic_log.connect(self._log)
            self._slm_window.set_font_size(self._spin_font.value())
            self._slm_window.set_stretch(self._chk_slm_stretch.isChecked())
        return self._slm_window

    def _show_slm_on_selected_screen(self, *, force_show=True):
        window = self._ensure_slm_window()
        screen = self._selected_screen()
        fullscreen = self._chk_slm_fullscreen.isChecked()
        if force_show or not window.isVisible():
            window.show_on_screen(screen, fullscreen=fullscreen)
        else:
            window.show_on_screen(screen, fullscreen=fullscreen)

        screen_desc = self._combo_slm_screen.currentText() if self._combo_slm_screen.count() else "unknown"
        mode = "fullscreen" if fullscreen else "windowed"
        self._log(f"SLM moved to {screen_desc} ({mode}).")

    def _move_slm_to_selected_screen(self):
        self._show_slm_on_selected_screen(force_show=True)

    @Slot(str)
    def _on_model_loaded(self, msg: str):
        fiber = self._active_fiber or "?"
        self._lbl_model.setText(f"Model: {fiber}")
        self._lbl_model_status.setStyleSheet("color: #51cf66; font-size: 11px;")
        self._lbl_model_status.setText(f"Loaded: {fiber}")
        self._lbl_auth_warning.setVisible(False)
        if fiber and self._combo_fiber.currentText() != fiber:
            self._combo_fiber.blockSignals(True)
            idx = self._combo_fiber.findText(fiber)
            if idx >= 0:
                self._combo_fiber.setCurrentIndex(idx)
            self._combo_fiber.blockSignals(False)
        self._log(f"[MODEL] {msg}")
        self._update_recognition_status_label()
        self._refresh_demo_step_status()

    def _toggle_slm_window(self):
        if self._slm_window is not None and self._slm_window.isVisible():
            self._slm_window.hide()
            self._btn_show_slm.setText("Open SLM Window")
            self._log("SLM window hidden.")
            return

        self._show_slm_on_selected_screen(force_show=True)
        self._btn_show_slm.setText("Hide SLM Window")

    def _test_slm_output(self):
        """Built-in Qt diagnostic (no letter_images)."""
        self._log("[SLM] Test SLM Output — Qt pattern + RGBW bar; look at the selected SLM screen.")
        self._log(
            "[SLM] Reminder: the camera preview in the main window is NOT the SLM output."
        )
        self._log_all_screens()
        win = self._ensure_slm_window()
        lit = (self._input_letter.text().strip().upper() or "A")[:1]
        if not lit.isalpha():
            lit = "A"
        win.set_diagnostic_pattern(lit)
        self._show_slm_on_selected_screen(force_show=True)
        self._btn_show_slm.setText("Hide SLM Window")
        win.raise_()
        win.activateWindow()
        win.repaint()
        win.update()
        win.force_visual_refresh()

    def _is_image_challenge_mode(self) -> bool:
        return self._combo_challenge_type.currentText() == "Image"

    def _on_challenge_type_changed(self, mode: str) -> None:
        is_image = mode == "Image"
        self._spin_font.setEnabled(not is_image)
        self._chk_slm_stretch.setEnabled(True)
        if is_image and self._challenge_image_path:
            self._challenge_preview.set_image_challenge(
                self._challenge_image_path, self.current_challenge_label
            )
        elif not is_image:
            text = self._input_letter.text().strip()
            if text:
                self._challenge_preview.set_text_challenge(text, source="text")

    def _on_challenge_text_changed(self, text: str) -> None:
        if self._is_image_challenge_mode():
            return
        t = text.strip()
        if t:
            self._apply_challenge_label(t, source="text")
        else:
            self.current_challenge_label = None
            self._challenge_image_path = None
            if self._manifest_loaded:
                self._challenge_preview.set_manifest_ready(len(self._challenge_cycle))
            else:
                self._challenge_preview.clear_challenge()

    def _sync_challenge_cycle_index(self, label: str) -> None:
        if label in self._challenge_cycle:
            self._challenge_cycle_index = self._challenge_cycle.index(label)

    def _apply_challenge_label(
        self,
        label: str,
        *,
        source: str,
        image_path: Optional[str] = None,
        send_slm: bool = False,
    ) -> None:
        """Update left preview selection only (not recognition / SLM)."""
        selected = label.strip()
        if not selected:
            return
        self.current_challenge_label = selected
        self._challenge_source = source
        self._challenge_image_path = image_path
        self._sync_challenge_cycle_index(selected)
        self._input_letter.blockSignals(True)
        self._input_letter.setText(selected)
        self._input_letter.blockSignals(False)
        self._log(f"Challenge selected: {selected}")

        if source == "image" and image_path:
            self._combo_challenge_type.blockSignals(True)
            self._combo_challenge_type.setCurrentText("Image")
            self._combo_challenge_type.blockSignals(False)
            self._challenge_preview.set_image_challenge(image_path, selected)
        else:
            self._combo_challenge_type.blockSignals(True)
            self._combo_challenge_type.setCurrentText("Text")
            self._combo_challenge_type.blockSignals(False)
            self._challenge_preview.set_text_challenge(selected, source=source)

        if send_slm:
            self._push_challenge_to_slm()

    def _activate_auth_challenge(self, label: str) -> None:
        """Recognition compares against the challenge successfully sent to the SLM."""
        self.last_sent_challenge_label = label.strip()
        try:
            from gui.gui_diagnostics import GuiDiagnosticsSession

            sess = GuiDiagnosticsSession.get()
            if sess is not None:
                sess.set_auth_challenge(self.last_sent_challenge_label)
        except ImportError:
            pass
        self._display_smoother.reset()
        self._hide_overlay_banner()
        self._recognition_result.set_waiting(self.last_sent_challenge_label)
        self._robot_panel.set_challenge_label(self.last_sent_challenge_label)
        self._robot_panel.on_idle()
        self._update_recognition_status_label()
        self._refresh_demo_step_status()

    def _push_challenge_to_slm(self) -> bool:
        label = (self.current_challenge_label or "").strip()
        if not label:
            return False
        win = self._ensure_slm_window()
        img_path = self._challenge_image_path or self._resolve_challenge_image_path(label)

        if img_path and os.path.isfile(img_path):
            abs_path = os.path.abspath(img_path)
            self._log(f"SLM image path: {abs_path}")
            if not win.load_image(abs_path):
                self._log(f"[ERROR] Could not load image on SLM: {abs_path}")
                return False
        else:
            from gui.slm_window import _find_challenge_png

            resolved = _find_challenge_png(label)
            if resolved and os.path.isfile(resolved):
                abs_path = os.path.abspath(resolved)
                self._log(f"SLM image path: {abs_path}")
                if not win.load_image(abs_path):
                    self._log(f"[ERROR] Could not load image on SLM: {abs_path}")
                    return False
            else:
                win.set_text_challenge(label)
                diag = win.png_load_diagnostic()
                if diag:
                    self._log(f"[SLM] {diag}")
                p = win.last_letter_png_path()
                if p:
                    self._log(f"SLM image path: {p}")

        self._log(f"Challenge sent to SLM: {label}")
        self._show_slm_on_selected_screen(force_show=True)
        self._btn_show_slm.setText("Hide SLM Window")
        win.raise_()
        win.activateWindow()
        win.repaint()
        win.update()
        win.force_visual_refresh()
        return True

    def _send_to_slm(self):
        if not self._has_challenge_selected():
            self._log(
                "No challenge selected. Select a challenge before sending to SLM."
            )
            QMessageBox.information(
                self,
                "SLM",
                "No challenge selected.\n\n"
                "Use Prev/Next to choose a challenge, then click Send to SLM.",
            )
            return

        label = self.current_challenge_label.strip()
        img_path = self._challenge_image_path or self._resolve_challenge_image_path(label)
        if img_path:
            self._challenge_source = "image"
            self._challenge_image_path = img_path
        else:
            self._challenge_source = "text"
            self._challenge_image_path = None

        idx = self._combo_slm_screen.currentData()
        screen = self._selected_screen()
        name = screen.name() if screen else "?"
        self._log(
            f"[SLM] Send pipeline: challenge={label!r}, "
            f"source={self._challenge_source}, screen_index={idx}, screen_name={name!r}"
        )
        if not self._push_challenge_to_slm():
            return
        self._activate_auth_challenge(label)

    def _on_font_size_changed(self, size: int):
        if self._slm_window:
            self._slm_window.set_font_size(size)

    def _on_slm_stretch_toggled(self, checked: bool):
        if self._slm_window:
            self._slm_window.set_stretch(checked)

    def _prev_challenge(self):
        if not self._challenge_cycle:
            self._challenge_cycle = self._discover_challenge_cycle()
        if not self._challenge_cycle:
            return
        n = len(self._challenge_cycle)
        if self._challenge_cycle_index < 0:
            self._challenge_cycle_index = n - 1
            self._log(
                f"Challenge navigation: Prev from none -> last ({self._challenge_cycle[-1]})"
            )
        else:
            self._challenge_cycle_index = (self._challenge_cycle_index - 1) % n
        label = self._challenge_cycle[self._challenge_cycle_index]
        img_path = self._resolve_challenge_image_path(label)
        if img_path:
            self._apply_challenge_label(label, source="image", image_path=img_path)
        else:
            self._apply_challenge_label(label, source="text")

    def _next_challenge(self):
        if not self._challenge_cycle:
            self._challenge_cycle = self._discover_challenge_cycle()
        if not self._challenge_cycle:
            return
        n = len(self._challenge_cycle)
        if self._challenge_cycle_index < 0:
            self._challenge_cycle_index = 0
            self._log(
                f"Challenge navigation: Next from none -> first ({self._challenge_cycle[0]})"
            )
        else:
            self._challenge_cycle_index = (self._challenge_cycle_index + 1) % n
        label = self._challenge_cycle[self._challenge_cycle_index]
        img_path = self._resolve_challenge_image_path(label)
        if img_path:
            self._apply_challenge_label(label, source="image", image_path=img_path)
        else:
            self._apply_challenge_label(label, source="text")

    def _resolve_challenge_image_path(self, label: str) -> Optional[str]:
        from gui.slm_window import _find_challenge_png

        key = (label or "").strip()
        if key in self._challenge_image_by_label:
            path = self._challenge_image_by_label[key]
            if os.path.isfile(path):
                return path
        for entry in self._ppt_challenge_entries:
            if normalize_label(entry.get("label", "")) == normalize_label(key):
                path = entry.get("image", "")
                if path and os.path.isfile(path):
                    return path
        return _find_challenge_png(label)

    def _load_image_to_slm(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image for SLM",
            self._root,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
        )
        if not path:
            return

        label = os.path.splitext(os.path.basename(path))[0]
        self._apply_challenge_label(label, source="image", image_path=path, send_slm=False)
        self._log(f"Challenge selected: {label} (custom image — click Send to SLM)")

    def _build_cam_settings_box(self) -> QGroupBox:
        box = QGroupBox("Camera Settings")
        gl = QGridLayout(box)
        gl.setSpacing(4)
        gl.setContentsMargins(6, 8, 6, 6)
        gl.setColumnStretch(1, 1)

        row = 0

        # ── Auto Exposure ──────────────────────────────────────────────
        self._chk_auto_exp = QCheckBox("Auto Exposure")
        self._chk_auto_exp.setChecked(True)
        self._chk_auto_exp.toggled.connect(self._on_auto_exp_toggled)
        gl.addWidget(self._chk_auto_exp, row, 0, 1, 3)
        self._cam_ctrl_widgets.append(self._chk_auto_exp)
        row += 1

        # ── Exposure spinbox (wide range: log2-s or µs) ────────────────
        lbl_exp = QLabel("Exposure:")
        gl.addWidget(lbl_exp, row, 0)
        self._spin_exposure = QDoubleSpinBox()
        self._spin_exposure.setRange(-13.0, 500000.0)
        self._spin_exposure.setDecimals(1)
        self._spin_exposure.setValue(-5.0)
        self._spin_exposure.setSingleStep(1.0)
        self._spin_exposure.setToolTip(
            "V4L2 / AVFoundation: log₂(s) e.g. -5 ≈ 1/32 s\n"
            "MindVision / DirectShow: absolute µs, e.g. 10000 = 10 ms"
        )
        self._spin_exposure.valueChanged.connect(self._on_exposure_changed)
        gl.addWidget(self._spin_exposure, row, 1, 1, 2)
        self._cam_ctrl_widgets += [lbl_exp, self._spin_exposure]
        row += 1

        # ── Generic slider rows ────────────────────────────────────────
        # (attr_prefix, label, lo, hi, default, display_fn, setter_name)
        slider_specs = [
            ("_gain",       "Gain",       0,   100,   0,   lambda v: str(v),         "set_gain"),
            ("_brightness", "Brightness", -64,  64,   0,   lambda v: str(v),         "set_brightness"),
            ("_contrast",   "Contrast",    0,  100,  50,   lambda v: str(v),         "set_contrast"),
            ("_gamma",      "Gamma",      100, 500,  100,  lambda v: f"{v/100:.2f}", "set_gamma"),
            ("_saturation", "Saturation",  0,  100,  50,   lambda v: str(v),         "set_saturation"),
            ("_sharpness",  "Sharpness",   0,    7,   2,   lambda v: str(v),         "set_sharpness"),
        ]

        def _make_slider_handler(attr, dfn, setter_name):
            val_lbl = getattr(self, attr + "_val_lbl")
            def handler(v):
                val_lbl.setText(dfn(v))
                if self._camera_worker:
                    getattr(self._camera_worker, setter_name)(float(v))
            return handler

        for attr, lbl_text, lo, hi, default, dfn, setter_name in slider_specs:
            lbl = QLabel(f"{lbl_text}:")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(lo, hi)
            slider.setValue(default)
            val_lbl = QLabel(dfn(default))
            val_lbl.setFixedWidth(46)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val_lbl.setStyleSheet("color: #a0c4ff; font-size: 11px;")

            gl.addWidget(lbl,     row, 0)
            gl.addWidget(slider,  row, 1)
            gl.addWidget(val_lbl, row, 2)

            setattr(self, attr + "_slider",  slider)
            setattr(self, attr + "_val_lbl", val_lbl)
            self._cam_ctrl_widgets += [lbl, slider, val_lbl]
            row += 1

        # Wire up signals after all setattr calls
        for attr, _, _, _, _, dfn, setter_name in slider_specs:
            getattr(self, attr + "_slider").valueChanged.connect(
                _make_slider_handler(attr, dfn, setter_name)
            )

        # ── Separator ─────────────────────────────────────────────────
        sep1 = QFrame(); sep1.setFrameShape(QFrame.HLine); sep1.setObjectName("divider")
        gl.addWidget(sep1, row, 0, 1, 3); row += 1

        # ── Auto White Balance ─────────────────────────────────────────
        self._chk_auto_wb = QCheckBox("Auto White Balance")
        self._chk_auto_wb.setChecked(True)
        self._chk_auto_wb.toggled.connect(self._on_auto_wb_toggled)
        gl.addWidget(self._chk_auto_wb, row, 0, 1, 3)
        self._cam_ctrl_widgets.append(self._chk_auto_wb)
        row += 1

        # ── WB Temperature ─────────────────────────────────────────────
        lbl_wbt = QLabel("WB Temp:")
        self._wb_temp_slider = QSlider(Qt.Horizontal)
        self._wb_temp_slider.setRange(2800, 6500)
        self._wb_temp_slider.setSingleStep(100)
        self._wb_temp_slider.setValue(4500)
        self._wb_temp_val_lbl = QLabel("4500 K")
        self._wb_temp_val_lbl.setFixedWidth(52)
        self._wb_temp_val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._wb_temp_val_lbl.setStyleSheet("color: #a0c4ff; font-size: 11px;")

        def _wb_handler(v):
            self._wb_temp_val_lbl.setText(f"{v} K")
            if self._camera_worker:
                self._camera_worker.set_white_balance_temp(float(v))
        self._wb_temp_slider.valueChanged.connect(_wb_handler)

        gl.addWidget(lbl_wbt,               row, 0)
        gl.addWidget(self._wb_temp_slider,  row, 1)
        gl.addWidget(self._wb_temp_val_lbl, row, 2)
        self._cam_ctrl_widgets += [lbl_wbt, self._wb_temp_slider, self._wb_temp_val_lbl]
        row += 1

        # ── Separator ─────────────────────────────────────────────────
        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine); sep2.setObjectName("divider")
        gl.addWidget(sep2, row, 0, 1, 3); row += 1

        # ── Flip ──────────────────────────────────────────────────────
        flip_lbl = QLabel("Flip:")
        self._chk_flip_h = QCheckBox("Horiz")
        self._chk_flip_v = QCheckBox("Vert")
        self._chk_flip_h.toggled.connect(self._on_flip_changed)
        self._chk_flip_v.toggled.connect(self._on_flip_changed)
        gl.addWidget(flip_lbl,        row, 0)
        gl.addWidget(self._chk_flip_h, row, 1)
        gl.addWidget(self._chk_flip_v, row, 2)
        self._cam_ctrl_widgets += [flip_lbl, self._chk_flip_h, self._chk_flip_v]

        self._set_cam_controls_enabled(False)
        return box

    def _set_cam_controls_enabled(self, enabled: bool):
        for w in self._cam_ctrl_widgets:
            w.setEnabled(enabled)
        if enabled:
            # Exposure only editable when auto-exposure is off
            self._spin_exposure.setEnabled(not self._chk_auto_exp.isChecked())
            # WB temp only editable when auto-WB is off
            wb_manual = not self._chk_auto_wb.isChecked()
            self._wb_temp_slider.setEnabled(wb_manual)
            self._wb_temp_val_lbl.setEnabled(wb_manual)

    def _on_auto_exp_toggled(self, checked: bool):
        self._spin_exposure.setEnabled(not checked and self._capture_active)
        if self._camera_worker:
            self._camera_worker.set_auto_exposure(checked)

    def _on_auto_wb_toggled(self, checked: bool):
        manual = not checked and self._capture_active
        self._wb_temp_slider.setEnabled(manual)
        self._wb_temp_val_lbl.setEnabled(manual)
        if self._camera_worker:
            self._camera_worker.set_auto_wb(checked)

    def _on_exposure_changed(self, value: float):
        if self._camera_worker:
            self._camera_worker.set_exposure(value)

    def _on_flip_changed(self):
        if self._camera_worker:
            self._camera_worker.set_flip(
                self._chk_flip_h.isChecked(),
                self._chk_flip_v.isChecked()
            )

    def _reset_camera_session_logs(self) -> None:
        self._cam_open_logged = False
        self._first_frame_logged = False

    @Slot(dict)
    def _on_cam_props(self, props: dict):
        """Sync UI sliders/spinboxes with values reported by the camera."""
        if not self._cam_open_logged:
            backend = props.get("backend", "OpenCV")
            w_px = int(props.get("width", 0) or 0)
            h_px = int(props.get("height", 0) or 0)
            self._log(
                f"Camera open success ({backend})"
                + (f" — reported {w_px}×{h_px}" if w_px and h_px else "")
            )
            self._cam_open_logged = True

        def _nan(v):
            return v is None or v != v  # None or NaN

        def _safe_slider(slider, val_lbl, val, dfn=None):
            if _nan(val):
                return
            iv = int(round(val))
            if slider.minimum() <= iv <= slider.maximum():
                slider.blockSignals(True)
                slider.setValue(iv)
                slider.blockSignals(False)
                if val_lbl and dfn:
                    val_lbl.setText(dfn(iv))

        # Exposure
        exp = props.get("exposure")
        if not _nan(exp):
            self._spin_exposure.blockSignals(True)
            self._spin_exposure.setValue(float(exp))
            self._spin_exposure.blockSignals(False)

        # Auto exposure
        ae = props.get("auto_exposure")
        if not _nan(ae):
            is_auto = float(ae) > 0.5
            self._chk_auto_exp.blockSignals(True)
            self._chk_auto_exp.setChecked(is_auto)
            self._chk_auto_exp.blockSignals(False)
            self._spin_exposure.setEnabled(not is_auto)

        _safe_slider(self._gain_slider,       self._gain_val_lbl,       props.get("gain"),       lambda v: str(v))
        _safe_slider(self._brightness_slider, self._brightness_val_lbl, props.get("brightness"), lambda v: str(v))
        _safe_slider(self._contrast_slider,   self._contrast_val_lbl,   props.get("contrast"),   lambda v: str(v))
        _safe_slider(self._saturation_slider, self._saturation_val_lbl, props.get("saturation"), lambda v: str(v))
        _safe_slider(self._sharpness_slider,  self._sharpness_val_lbl,  props.get("sharpness"),  lambda v: str(v))

        # Gamma: cameras may report 1.0-5.0 (float) or 100-500 (integer)
        gamma = props.get("gamma")
        if not _nan(gamma) and gamma is not None:
            gamma_int = int(round(gamma * 100)) if float(gamma) < 10 else int(round(float(gamma)))
            _safe_slider(self._gamma_slider, self._gamma_val_lbl, gamma_int, lambda v: f"{v/100:.2f}")

        # Auto WB
        awb = props.get("auto_wb")
        if not _nan(awb):
            is_auto = float(awb) > 0.5
            self._chk_auto_wb.blockSignals(True)
            self._chk_auto_wb.setChecked(is_auto)
            self._chk_auto_wb.blockSignals(False)
            manual = not is_auto
            self._wb_temp_slider.setEnabled(manual)
            self._wb_temp_val_lbl.setEnabled(manual)

        # WB temperature
        wbt = props.get("wb_temp")
        if not _nan(wbt) and wbt and float(wbt) > 0:
            _safe_slider(self._wb_temp_slider, self._wb_temp_val_lbl,
                         float(wbt), lambda v: f"{v} K")

        # Update source label with resolution + backend
        w_px = int(props.get("width",  0) or 0)
        h_px = int(props.get("height", 0) or 0)
        backend = props.get("backend", "")
        if w_px and h_px:
            base = self._lbl_source.text().split(" |")[0]
            self._lbl_source.setText(f"{base} | {w_px}×{h_px} | {backend}")

    # ── macOS helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _macos_list_av_devices() -> list:
        """Return device names in AVFoundation index order (macOS only).

        Uses system_profiler for basic camera names, then cross-checks with
        ffmpeg -list_devices if available for the full ordered list.
        Falls back to an empty list on any error.
        """
        names: list[str] = []
        try:
            import subprocess, json

            # Try ffmpeg first — it lists devices in the exact order OpenCV uses
            r = subprocess.run(
                ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", "dummy"],
                capture_output=True, text=True, timeout=5
            )
            output = r.stderr  # ffmpeg prints device list to stderr
            in_video = False
            for line in output.splitlines():
                if "AVFoundation video devices" in line:
                    in_video = True
                    continue
                if "AVFoundation audio devices" in line:
                    break
                if in_video:
                    import re
                    m = re.search(r'\[(\d+)\]\s+(.+)', line)
                    if m:
                        idx, name = int(m.group(1)), m.group(2).strip()
                        while len(names) <= idx:
                            names.append("")
                        names[idx] = name
            if names:
                return names
        except Exception:
            pass

        try:
            import subprocess
            r = subprocess.run(
                ["system_profiler", "SPCameraDataType"],
                capture_output=True, text=True, timeout=5
            )
            for line in r.stdout.splitlines():
                stripped = line.strip()
                if stripped.endswith(":") and len(stripped) > 1:
                    names.append(stripped[:-1])
        except Exception:
            pass
        return names

    def _make_manual_speckle_frame(self) -> np.ndarray:
        """Manual screenshot only: random speckle-like grayscale (numpy), no camera."""
        rng = np.random.default_rng(42 + self._manual_feed_phase)
        h, w = 520, 700
        g = rng.standard_normal((h, w)).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 2.8)
        g = (g - g.min()) / (float(g.max() - g.min()) + 1e-6)
        return (g * 255).astype(np.uint8)

    @Slot()
    def _on_manual_feed_tick(self) -> None:
        if not self._manual_feed_active:
            return
        self._manual_feed_phase += 1
        self._cam_label.set_frame(self._make_manual_speckle_frame())

    def _stop_manual_screenshot_feed(self) -> None:
        if self._manual_feed_timer is not None:
            self._manual_feed_timer.stop()
            try:
                self._manual_feed_timer.timeout.disconnect()
            except Exception:
                pass
            self._manual_feed_timer.deleteLater()
            self._manual_feed_timer = None
        self._manual_feed_active = False

    def _stop_camera_worker_if_any(self) -> None:
        if self._camera_worker is None:
            return
        if self._camera_worker.isRunning():
            try:
                self._camera_worker.frame_ready.disconnect(self._on_frame)
            except Exception:
                pass
            try:
                self._camera_worker.error.disconnect(self._on_cam_error)
            except Exception:
                pass
            try:
                self._camera_worker.fps_updated.disconnect(self._on_fps_update)
            except Exception:
                pass
            try:
                self._camera_worker.props_read.disconnect(self._on_cam_props)
            except Exception:
                pass
            self._camera_worker.stop()
        self._camera_worker = None

    def start_manual_screenshot_feed(self) -> None:
        """Manual screenshot only: show moving speckle-like preview without OpenCV camera."""
        if not _env_manual_screenshot_mode():
            return
        self._stop_manual_screenshot_feed()
        self._stop_camera_worker_if_any()
        self._capture_active = True
        self._btn_start_cam.setEnabled(False)
        self._btn_stop_cam.setEnabled(True)
        self._lbl_source.setText("Manual screenshot video source")
        self._lbl_fps.setText("FPS: 24.0")
        self._fps = 24.0
        self._set_cam_controls_enabled(True)
        self._manual_feed_active = True
        self._manual_feed_timer = QTimer(self)
        self._manual_feed_timer.timeout.connect(self._on_manual_feed_tick)
        self._manual_feed_timer.start(80)
        self._on_manual_feed_tick()
        self._log("[MANUAL SCREENSHOT] Synthetic speckle preview (no CameraWorker; for documentation).")

    def apply_manual_screenshot_model_fallback_ui(self) -> None:
        """Manual screenshot only: Loaded UI when no Fiber*.pth on disk."""
        if not _env_manual_screenshot_mode():
            return
        self._lbl_model.setText("Model: Manual screenshot placeholder")
        self._lbl_model_status.setStyleSheet("color: #51cf66; font-size: 11px;")
        self._lbl_model_status.setText("Loaded: Manual screenshot placeholder")
        self._lbl_model_path.setText("(placeholder)")
        self._log(
            "[MANUAL SCREENSHOT] Placeholder model labels — add results/fiber_auth/fiber_models/Fiber*.pth "
            "for an authentic load screenshot before formal submission."
        )

    def apply_manual_screenshot_file_source_ui(self, basename: str = "manual_demo_video.mp4") -> None:
        """Manual screenshot only: pretend File: source without QFileDialog (no blocking)."""
        if not _env_manual_screenshot_mode():
            return
        self._lbl_source.setText(f"File: {basename}")
        self._log(f"[MANUAL SCREENSHOT] File source label for documentation (no file opened): {basename}")

    def _populate_camera_selector(self, entries: list[CameraDeviceEntry]) -> None:
        self._camera_catalog = list(entries)
        self._combo_camera_source.blockSignals(True)
        self._combo_camera_source.clear()
        self._combo_camera_source.addItem("No camera selected", None)
        for entry in entries:
            self._combo_camera_source.addItem(entry.label, entry)
        if entries:
            prefer_mv = next(
                (i for i, e in enumerate(entries) if e.backend == "mindvision"), None
            )
            self._combo_camera_source.setCurrentIndex(
                1 + (prefer_mv if prefer_mv is not None else 0)
            )
        else:
            self._combo_camera_source.setCurrentIndex(0)
        self._combo_camera_source.blockSignals(False)

    def _selected_camera_entry(self) -> Optional[CameraDeviceEntry]:
        data = self._combo_camera_source.currentData()
        if isinstance(data, CameraDeviceEntry):
            return data
        return None

    def _scan_cameras(self) -> None:
        """Probe OpenCV and MindVision devices on the main thread (macOS permission)."""
        self._btn_scan_cameras.setEnabled(False)
        self._btn_scan_cameras.setText("Scanning…")
        QApplication.processEvents()

        manual = _env_manual_screenshot_mode()
        _skip_auth_backup = None
        indices = (0, 1) if manual else range(6)
        dev_names: list[str] = []

        if manual:
            if sys.platform == "darwin":
                os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")
            self._log(
                "[MANUAL SCREENSHOT] Scanning camera indices 0-1 only (demo uses 0-5)."
            )
        elif sys.platform == "darwin":
            dev_names = self._macos_list_av_devices() or []
            if dev_names:
                self._log("macOS AVFoundation devices detected by OS:")
                for i, name in enumerate(dev_names):
                    if name:
                        self._log(f"  [{i}] {name}")
            else:
                self._log(
                    "macOS: could not enumerate device names "
                    "(install ffmpeg via Homebrew for full device list)."
                )
            _skip_auth_backup = os.environ.pop("OPENCV_AVFOUNDATION_SKIP_AUTH", None)

        entries = scan_all_cameras(
            opencv_indices=indices,
            log_fn=self._log,
            device_names=dev_names,
            include_mindvision=not manual,
        )

        self._btn_scan_cameras.setEnabled(True)
        self._btn_scan_cameras.setText("Scan Cameras")
        self._populate_camera_selector(entries)

        if entries:
            if sys.platform == "darwin" and not manual:
                os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"
            sel = self._selected_camera_entry()
            if sel is not None:
                self._lbl_source.setText(f"Camera status: {sel.label} (not started)")
        else:
            if not manual and sys.platform == "darwin" and _skip_auth_backup is not None:
                os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = _skip_auth_backup
            self._lbl_source.setText("Camera status: not connected")
            if not manual:
                self._log(
                    "Camera may be occupied. Close vendor camera software and scan again."
                )
                msg = (
                    "No cameras found.\n\n"
                    "Close vendor camera software, click Scan Cameras again, "
                    "then select a device and Start CCD."
                )
                if sys.platform == "darwin":
                    msg += (
                        "\n\nOn macOS, grant Camera access in System Settings, "
                        "then quit and restart this app."
                    )
                QMessageBox.warning(self, "No Cameras Found", msg)

    def _start_camera(self) -> None:
        entry = self._selected_camera_entry()
        if entry is None:
            self._log("Start CCD blocked: no camera selected. Scan cameras first.")
            self._lbl_source.setText("Camera status: not connected")
            return
        self._stop_camera()
        self._reset_camera_session_logs()
        if entry.backend == "opencv":
            self._start_opencv_camera(entry)
        elif entry.backend == "mindvision":
            self._start_mindvision_camera(entry)
        else:
            self._log(f"Start CCD blocked: unknown backend {entry.backend!r}.")

    def _start_opencv_camera(self, entry: CameraDeviceEntry) -> None:
        idx = entry.opencv_index
        if idx is None:
            self._log("Start CCD blocked: invalid OpenCV camera entry.")
            return
        self._log(f"Starting OpenCV CCD (camera index {idx}) …")

        if sys.platform == "darwin":
            os.environ.pop("OPENCV_AVFOUNDATION_SKIP_AUTH", None)
            test_cap = cv2.VideoCapture(idx)
            opened = test_cap.isOpened()
            test_cap.release()
            if not opened:
                dev_names = self._macos_list_av_devices()
                self._log(
                    "Camera open failed. Close vendor camera software and try again, "
                    "or select another camera."
                )
                self._log(
                    "Camera may be occupied. Close vendor camera software and scan again."
                )
                hint = ""
                if dev_names:
                    hint = "\n\nDevices macOS sees:\n" + "\n".join(
                        f"  [{i}] {n}" for i, n in enumerate(dev_names) if n
                    )
                QMessageBox.critical(
                    self,
                    "Cannot Open Camera",
                    f"Cannot open camera index {idx}.{hint}\n\n"
                    "Scan cameras again and pick another source.",
                )
                return
            os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

        w, h = self._combo_cam_res.currentData()
        self._camera_worker = CameraWorker(self)
        self._camera_worker.set_camera(idx, width=w, height=h)
        self._camera_worker.set_target_fps(30)
        self._camera_worker.frame_ready.connect(self._on_frame)
        self._camera_worker.error.connect(self._on_cam_error)
        self._camera_worker.fps_updated.connect(self._on_fps_update)
        self._camera_worker.props_read.connect(self._on_cam_props)
        self._camera_worker.start()
        self._capture_active = True
        self._btn_start_cam.setEnabled(False)
        self._btn_stop_cam.setEnabled(True)
        self._lbl_source.setText(f"Camera status: {entry.label}")
        self._set_cam_controls_enabled(True)
        self._log(f"Camera started (OpenCV index {idx})")
        self._update_recognition_status_label()
        self._refresh_demo_step_status()
        _demo_trace_ui(f"Start Camera OpenCV worker started idx={idx}")

    def _start_mindvision_camera(self, entry: CameraDeviceEntry) -> None:
        dev = entry.mv_device
        if dev is None:
            self._log("Start CCD blocked: invalid MindVision camera entry.")
            return
        name = (getattr(dev, "friendly_name", None) or getattr(dev, "product_name", None) or "MindVision")
        self._log(f"Starting MindVision CCD ({name}) …")
        try:
            self._camera_worker = MvCameraWorker(dev, self)
        except Exception as exc:
            self._log(
                "Camera open failed. Close vendor camera software and try again, "
                "or select another camera."
            )
            QMessageBox.critical(self, "MindVision Error", f"Cannot create worker:\n{exc}")
            return

        self._camera_worker.set_target_fps(30)
        self._camera_worker.frame_ready.connect(self._on_frame)
        self._camera_worker.error.connect(self._on_cam_error)
        self._camera_worker.fps_updated.connect(self._on_fps_update)
        self._camera_worker.props_read.connect(self._on_cam_props)
        self._camera_worker.start()
        self._capture_active = True
        self._btn_start_cam.setEnabled(False)
        self._btn_stop_cam.setEnabled(True)
        self._lbl_source.setText(f"Camera status: {entry.label}")
        self._set_cam_controls_enabled(True)
        self._log(f"MindVision camera started: {name}")
        self._update_recognition_status_label()
        self._refresh_demo_step_status()
        _demo_trace_ui(f"MindVision worker started name={name!r}")

    def _load_video_file(self):
        self._stop_camera()
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", self._video_dir,
            "Video Files (*.avi *.mp4 *.mkv *.mov);;All Files (*)"
        )
        if not path:
            return
        self._camera_worker = CameraWorker(self)
        self._camera_worker.set_video_file(path, loop=True)
        self._camera_worker.set_target_fps(30)
        self._camera_worker.frame_ready.connect(self._on_frame)
        self._camera_worker.error.connect(self._on_cam_error)
        self._camera_worker.fps_updated.connect(self._on_fps_update)
        self._camera_worker.props_read.connect(self._on_cam_props)
        self._camera_worker.start()
        self._capture_active = True
        self._btn_start_cam.setEnabled(False)
        self._btn_stop_cam.setEnabled(True)
        name = os.path.basename(path)
        self._lbl_source.setText(f"File: {name}")
        self._set_cam_controls_enabled(True)
        self._log(f"Video file loaded: {path}")
        self._update_recognition_status_label()
        self._refresh_demo_step_status()
        _demo_trace_ui(f"Video file worker started path={path!r}")

    def _recognition_block_reason(self) -> Optional[str]:
        if not self._capture_active:
            return "CCD is not running. Start CCD first."
        if self._infer_worker._model is None:
            return "No model loaded. Select a connected fiber first."
        if not (self.last_sent_challenge_label or "").strip():
            return "No challenge sent. Select a challenge and send it to SLM first."
        return None

    def _sync_infer_checkbox(self, active: bool) -> None:
        self._chk_infer_active.blockSignals(True)
        self._chk_infer_active.setChecked(active)
        self._chk_infer_active.blockSignals(False)

    def _update_recognition_status_label(self, message: str = "") -> None:
        if message:
            self._lbl_recognition_status.setText(message)
            return
        if self._recognition_active:
            self._lbl_recognition_status.setText("Recognition: running")
            return
        blocked = self._recognition_block_reason()
        if blocked:
            self._lbl_recognition_status.setText(
                f"Recognition: waiting — {blocked}"
            )
        else:
            self._lbl_recognition_status.setText("Recognition: stopped")

    def _start_recognition(self) -> None:
        reason = self._recognition_block_reason()
        if reason:
            self._log(f"Start recognition blocked: {reason}")
            self._update_recognition_status_label(
                f"Recognition: blocked — {reason}"
            )
            return
        if self._recognition_active:
            return
        self._recognition_active = True
        self._sync_infer_checkbox(True)
        self._display_smoother.reset()
        self._last_voice_decision = ""
        self._waiting_voice_spoken = False
        self._infer_worker.reset_inference_state()
        self._btn_start_recognition.setEnabled(False)
        self._btn_stop_recognition.setEnabled(True)
        self._update_recognition_status_label()
        self._recognition_result.set_waiting(self._auth_challenge_label())
        self._robot_panel.set_challenge_label(self._auth_challenge_label())
        self._robot_panel.on_idle()
        self._cam_glow.setBlurRadius(0)
        self._hide_overlay_banner()
        self._log("Recognition started.")
        self._refresh_demo_step_status()

    def _stop_recognition(self) -> None:
        if not self._recognition_active:
            return
        self._recognition_active = False
        self._sync_infer_checkbox(False)
        self._infer_worker.reset_inference_state()
        self._display_smoother.reset()
        self._voice_announcer.stop()
        self._last_voice_decision = ""
        self._waiting_voice_spoken = False
        self._btn_start_recognition.setEnabled(True)
        self._btn_stop_recognition.setEnabled(False)
        self._cam_glow.setBlurRadius(0)
        self._robot_panel.on_idle()
        self._hide_overlay_banner()
        self._recognition_result.set_waiting(self._auth_challenge_label())
        self._update_recognition_status_label()
        self._log("Recognition stopped.")
        self._refresh_demo_step_status()

    def _on_infer_active_toggled(self, checked: bool) -> None:
        if checked == self._recognition_active:
            return
        if checked:
            self._start_recognition()
            if not self._recognition_active:
                self._sync_infer_checkbox(False)
        else:
            self._stop_recognition()

    def _on_voice_toggled(self, checked: bool) -> None:
        self._voice_announcer.set_enabled(checked)
        state = "enabled" if checked else "disabled"
        self._log(f"Voice announcement {state}.")

    def _test_voice(self) -> None:
        self._log("Test voice requested.")
        if self._voice_announcer.speak(
            PHRASE_TEST,
            key="test_voice",
            cooldown_sec=0.0,
            force=True,
            ignore_enabled=True,
        ):
            self._log("Voice announcement: test phrase (Chinese).")
        elif not self._voice_announcer.available:
            self._log("Voice test skipped: no speech backend available.")
        else:
            self._log("Voice test failed to start speech process.")

    def _announce_stable_decision(self, snap) -> None:
        if not self._recognition_active or not self._chk_voice.isChecked():
            return
        decision = (snap.decision or "WAITING").strip().upper()
        if decision == "WAITING":
            if self._waiting_voice_spoken:
                return
            if self._voice_announcer.speak(
                PHRASE_WAITING,
                key="waiting",
                cooldown_sec=4.0,
            ):
                self._log("Voice announcement: waiting (Chinese).")
            self._waiting_voice_spoken = True
            return

        self._waiting_voice_spoken = False
        if decision == self._last_voice_decision:
            return

        text = phrase_for_decision(decision)
        if text is None:
            return
        log_labels = {
            "ACCESS GRANTED": "access granted (Chinese)",
            "ACCESS DENIED": "access denied (Chinese)",
            "LOW CONFIDENCE": "low confidence (Chinese)",
        }
        key = decision.lower().replace(" ", "_")
        if self._voice_announcer.speak(text, key=key, cooldown_sec=3.0):
            self._log(f"Voice announcement: {log_labels.get(decision, decision)}")
        self._last_voice_decision = decision

    def _stop_camera(self):
        _demo_trace_ui("Stop Camera")
        self._stop_recognition()
        self._stop_manual_screenshot_feed()
        self._stop_camera_worker_if_any()
        self._reset_camera_session_logs()
        self._capture_active = False
        self._btn_start_cam.setEnabled(True)
        self._btn_stop_cam.setEnabled(False)
        self._cam_label._show_idle_placeholder()
        self._lbl_fps.setText("FPS: --")
        self._set_cam_controls_enabled(False)
        self._robot_panel.on_idle()
        self._recognition_result.set_waiting(self._auth_challenge_label())
        self._cam_glow.setBlurRadius(0)
        self._overlay_banner.hide()
        self._update_recognition_status_label()
        self._refresh_demo_step_status()
        self._log("Camera stopped.")

    @Slot(object)
    def _on_frame(self, frame: np.ndarray):
        _demo_trace_ui(f"_on_frame shape={frame.shape}")
        if frame is not None and getattr(frame, "size", 0) > 0:
            if not self._first_frame_logged:
                self._log(
                    f"First camera frame: shape={tuple(frame.shape)}, dtype={frame.dtype}"
                )
                self._first_frame_logged = True
            try:
                from gui.gui_diagnostics import GuiDiagnosticsSession

                sess = GuiDiagnosticsSession.get()
                if sess is not None:
                    sess.log_raw_frame(frame, tag="live")
            except ImportError:
                pass
        self._last_frame = frame
        self._cam_label.set_frame(frame)
        if self._recognition_active and self._infer_worker._model is not None:
            self._infer_worker.push_frame(frame)
            if self._cam_glow.blurRadius() == 0:
                self._cam_glow.setBlurRadius(28)
                self._robot_panel.on_reading(self._active_fiber)

    @Slot(str)
    def _on_cam_error(self, msg: str):
        self._log(f"[CAMERA ERROR] {msg}")
        low = msg.lower()
        if "cannot open" in low or "open failed" in low or "occupied" in low:
            self._log(
                "Camera open failed. Close vendor camera software and try again, "
                "or select another camera."
            )
            self._log(
                "Camera may be occupied. Close vendor camera software and scan again."
            )
        self._stop_camera()

    @Slot(float)
    def _on_fps_update(self, fps: float):
        self._fps = fps
        self._lbl_fps.setText(f"FPS: {fps:.1f}")

    @Slot(dict)
    def _on_prediction(self, result: dict):
        if not self._recognition_active:
            return
        _demo_trace_ui(
            f"_on_prediction top1={result.get('top1')!r} conf={result.get('confidence')}"
        )
        if os.environ.get("SPECKLE_FORCE_AUTH_STATE", "").strip() and _env_manual_screenshot_mode():
            self._robot_panel.on_prediction(result)
            return

        conf = result.get("confidence")
        smoothed = result.get("smoothed", "?")
        pred_label = str(smoothed).strip() if smoothed else "?"
        conf_f: Optional[float] = None
        if conf is not None:
            try:
                conf_f = float(conf)
            except (TypeError, ValueError):
                conf_f = None

        feed = self._display_smoother.feed(
            challenge=self._auth_challenge_label(),
            raw_label=pred_label,
            raw_confidence=conf_f,
        )
        if not feed.refresh_ui:
            return

        snap = feed.snapshot
        self._recognition_result.apply_snapshot(snap)
        self._announce_stable_decision(snap)

        show_warn = snap.decision == "LOW CONFIDENCE" or (
            snap.match is True
            and snap.confidence is not None
            and snap.confidence < LOW_CONFIDENCE_THRESHOLD
        )
        if show_warn and snap.confidence is not None:
            self._lbl_auth_warning.setText(
                f"Low confidence ({snap.confidence * 100:.0f}%) — verify before granting access"
            )
            self._lbl_auth_warning.setStyleSheet(
                "color: #ff6b6b; font-weight: bold; font-size: 11px; "
                "background-color: #3c1515; border: 1px solid #e06c75; "
                "border-radius: 4px; padding: 4px;"
            )
            self._lbl_auth_warning.setVisible(True)
        else:
            self._lbl_auth_warning.setVisible(False)

        self._robot_panel.set_challenge_label(self._auth_challenge_label())
        self._robot_panel.apply_decision(
            snap.decision,
            predicted=snap.predicted,
            reason=snap.reason,
            result=result,
        )

        if snap.emit_banner:
            self._show_overlay_banner(
                "ACCESS GRANTED",
                "#3ddc84",
                hold_ms=int(self._display_smoother.banner_hold_sec * 1000),
            )
        elif snap.hide_banner:
            self._hide_overlay_banner()

        from PySide6.QtCore import QTimer as _QTimer
        glow_ms = int(max(2000, self._display_smoother.banner_hold_sec * 1000))
        _QTimer.singleShot(glow_ms, lambda: self._cam_glow.setBlurRadius(0))

    def _hide_overlay_banner(self) -> None:
        self._banner_hide_timer.stop()
        self._overlay_banner.hide()

    def _show_overlay_banner(self, text: str, color: str, hold_ms: int = 2400):
        """Show a brief overlay banner centred over the camera feed.

        The banner is parented to the camera *card* (not the pixmap QLabel): a
        child of QLabel that paints a pixmap does not reliably composite
        overlay text or QGraphicsOpacityEffect on macOS; ``grab()`` then omits
        the glyph layer too.
        """
        if not (text or "").strip():
            self._hide_overlay_banner()
            return
        self._banner_hide_timer.stop()
        self._overlay_banner.setText(text)
        self._overlay_banner.setAlignment(Qt.AlignCenter)
        self._overlay_banner.setGraphicsEffect(None)
        margin = 16
        cw = max(1, self._cam_label.width())
        ch = max(1, self._cam_label.height())
        max_w = max(120, cw - 2 * margin)
        max_h = max(48, ch - 2 * margin)

        pad_x, pad_y = 20, 12
        initial_fs = min(40, max(14, cw // 18))
        self._overlay_banner.setMinimumSize(0, 0)

        chosen = False
        for fs in range(initial_fs, 5, -2):
            self._overlay_banner.setWordWrap(False)
            self._overlay_banner.setMinimumWidth(0)
            self._overlay_banner.setMaximumWidth(16777215)
            self._overlay_banner.setStyleSheet(
                f"QLabel#overlayBanner {{ background-color: rgba(31,41,55,220); "
                f"color: {color}; font-weight:800; font-size:{fs}px; "
                f"letter-spacing:0px; border-radius:16px; "
                f"padding:{pad_y}px {pad_x}px; }}"
            )
            self._overlay_banner.adjustSize()
            tw = self._overlay_banner.sizeHint().width()
            th = self._overlay_banner.sizeHint().height()
            if tw > max_w:
                self._overlay_banner.setWordWrap(True)
                self._overlay_banner.setFixedWidth(max_w)
                self._overlay_banner.adjustSize()
                tw = self._overlay_banner.sizeHint().width()
                th = self._overlay_banner.sizeHint().height()
            if th <= max_h:
                chosen = True
                break

        if not chosen:
            self._overlay_banner.setWordWrap(True)
            self._overlay_banner.setFixedWidth(max_w)
            self._overlay_banner.setStyleSheet(
                f"QLabel#overlayBanner {{ background-color: rgba(31,41,55,220); "
                f"color: {color}; font-weight:800; font-size:10px; "
                f"letter-spacing:0px; border-radius:16px; "
                f"padding:{pad_y}px {pad_x}px; }}"
            )
            self._overlay_banner.adjustSize()

        if not self._overlay_banner.wordWrap():
            tw = self._overlay_banner.sizeHint().width()
            self._overlay_banner.setFixedWidth(tw)

        self._overlay_banner.adjustSize()
        bw = self._overlay_banner.sizeHint().width()
        bh = min(self._overlay_banner.sizeHint().height(), max_h)

        top_left = self._cam_label.mapTo(self._cam_card, QPoint(0, 0))
        x = top_left.x() + max(0, (cw - bw) // 2)
        y = top_left.y() + max(0, (ch - bh) // 2)
        self._overlay_banner.setParent(self._cam_card)
        self._overlay_banner.setGeometry(x, y, bw, bh)
        self._overlay_banner.raise_()
        self._overlay_banner.show()
        self._banner_hide_timer.start(max(800, hold_ms))

    @Slot(str)
    def _on_infer_error(self, msg: str):
        self._log(f"[INFERENCE ERROR] {msg}")

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self._log_text.append(f"[{ts}] {msg}")

    def closeEvent(self, event):
        _demo_trace_ui("closeEvent: stopping camera and inference thread")
        self._stop_recognition()
        self._voice_announcer.stop()
        self._stop_camera()
        if self._infer_worker.isRunning():
            self._infer_worker.stop()
        if self._slm_window:
            self._slm_window.close()
        super().closeEvent(event)
