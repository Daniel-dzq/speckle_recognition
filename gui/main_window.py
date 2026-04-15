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
import sys
import glob
import time
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QTextEdit,
    QGroupBox, QSpinBox, QDoubleSpinBox, QSlider, QFileDialog,
    QSizePolicy, QFrame, QStatusBar, QCheckBox, QSplitter, QScrollArea,
    QApplication, QMessageBox,
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap, QFont, QGuiApplication

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gui.slm_window         import SLMWindow
from gui.camera_worker      import CameraWorker
from gui.mv_camera_worker   import MvCameraWorker
from gui.inference_worker   import InferenceWorker
import gui.mvsdk as mvsdk


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


FIBER_MODELS_DIR = os.path.join(ROOT, "results", "fiber_auth", "fiber_models")
LOW_CONFIDENCE_THRESHOLD = 0.40


def discover_fiber_models(model_dir: str = FIBER_MODELS_DIR):
    """Return {fiber_name: checkpoint_path} for all Fiber*.pth files."""
    result = {}
    if not os.path.isdir(model_dir):
        return result
    for f in sorted(os.listdir(model_dir)):
        if f.endswith(".pth"):
            name = os.path.splitext(f)[0]
            result[name] = os.path.join(model_dir, f)
    return result


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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            "background-color: #0d0d1a; border: 1px solid #2a2a3c; border-radius: 6px;"
        )
        self.setText("No camera feed")
        self._pixmap = None

    def set_frame(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        if len(frame.shape) == 2:
            gray = np.ascontiguousarray(frame)
            img = QImage(gray.tobytes(), w, h, gray.strides[0], QImage.Format_Grayscale8)
        else:
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = QImage(rgb.tobytes(), w, h, rgb.strides[0], QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(img)
        self._update_display()

    def resizeEvent(self, event):
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
        self._capture_active = False
        self._last_frame = None
        self._fps = 0.0
        self._preferred_slm_screen = 1

        self._main_splitter = None
        self._top_splitter = None
        self._left_scroll = None
        self._pred_box = None
        self._log_box = None
        self._cam_ctrl_widgets: list = []

        self._setup_ui()
        self._apply_style()
        self._connect_signals()
        self._refresh_fiber_list()
        self._refresh_screen_list()
        self._apply_responsive_metrics(force=True)
        self._log("Speckle-PUF demo ready. Select a fiber to load its authentication model.")

    def _setup_ui(self):
        self.setWindowTitle("Speckle-PUF Live Demo")
        self.resize(1480, 900)
        self.setMinimumSize(1180, 720)

        central = QWidget()
        self.setCentralWidget(central)

        self._main_splitter = QSplitter(Qt.Horizontal)
        self._main_splitter.setChildrenCollapsible(False)
        self._main_splitter.setHandleWidth(3)

        left_widget = self._build_left_panel()
        self._left_scroll = QScrollArea()
        self._left_scroll.setWidgetResizable(True)
        self._left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._left_scroll.setWidget(left_widget)
        self._left_scroll.setMinimumWidth(320)
        self._left_scroll.setMaximumWidth(460)
        self._main_splitter.addWidget(self._left_scroll)

        right_widget = QWidget()
        right_widget.setLayout(self._build_right_panel())
        self._main_splitter.addWidget(right_widget)

        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setSizes([360, 1080])

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 4)
        main_layout.setSpacing(8)
        main_layout.addWidget(self._main_splitter, stretch=1)

        self._log_box = QGroupBox("Log")
        log_layout = QVBoxLayout(self._log_box)
        log_layout.setContentsMargins(6, 6, 6, 6)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMinimumHeight(84)
        self._log_text.setMaximumHeight(150)
        log_layout.addWidget(self._log_text)
        main_layout.addWidget(self._log_box)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._lbl_device = QLabel("Device: checking...")
        self._lbl_fps = QLabel("FPS: --")
        self._lbl_model = QLabel("Fiber: none")
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

    def _build_left_panel(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        layout.setContentsMargins(4, 4, 10, 4)

        fiber_box = QGroupBox("Fiber Authentication")
        fl = QGridLayout(fiber_box)
        fl.setSpacing(6)

        fl.addWidget(QLabel("Authorized fiber:"), 0, 0)
        self._combo_fiber = QComboBox()
        self._combo_fiber.setToolTip(
            "Select the fiber whose model to use.\n"
            "The model is loaded automatically on selection."
        )
        self._combo_fiber.currentIndexChanged.connect(self._on_fiber_selected)
        fl.addWidget(self._combo_fiber, 0, 1)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.setFixedWidth(78)
        btn_refresh.clicked.connect(self._refresh_fiber_list)
        fl.addWidget(btn_refresh, 0, 2)

        self._lbl_model_path = QLabel("")
        self._lbl_model_path.setWordWrap(True)
        self._lbl_model_path.setStyleSheet("color: #666; font-size: 10px;")
        fl.addWidget(self._lbl_model_path, 1, 0, 1, 3)

        self._lbl_model_status = QLabel("No model loaded")
        self._lbl_model_status.setWordWrap(True)
        self._lbl_model_status.setStyleSheet("color: #888; font-size: 11px;")
        fl.addWidget(self._lbl_model_status, 2, 0, 1, 3)

        self._lbl_auth_warning = QLabel("")
        self._lbl_auth_warning.setWordWrap(True)
        self._lbl_auth_warning.setVisible(False)
        fl.addWidget(self._lbl_auth_warning, 3, 0, 1, 3)

        layout.addWidget(fiber_box)

        slm_box = QGroupBox("SLM Output Window")
        sl = QGridLayout(slm_box)
        sl.setSpacing(6)

        self._btn_show_slm = QPushButton("Open SLM Window")
        self._btn_show_slm.setObjectName("accent")
        self._btn_show_slm.clicked.connect(self._toggle_slm_window)
        sl.addWidget(self._btn_show_slm, 0, 0, 1, 3)

        sl.addWidget(QLabel("SLM screen:"), 1, 0)
        self._combo_slm_screen = QComboBox()
        sl.addWidget(self._combo_slm_screen, 1, 1)

        self._btn_refresh_screens = QPushButton("Refresh")
        self._btn_refresh_screens.clicked.connect(self._refresh_screen_list)
        sl.addWidget(self._btn_refresh_screens, 1, 2)

        self._chk_slm_fullscreen = QCheckBox("Fullscreen on selected screen")
        self._chk_slm_fullscreen.setChecked(True)
        sl.addWidget(self._chk_slm_fullscreen, 2, 0, 1, 3)

        self._btn_move_slm = QPushButton("Move SLM to Selected Screen")
        self._btn_move_slm.clicked.connect(self._move_slm_to_selected_screen)
        sl.addWidget(self._btn_move_slm, 3, 0, 1, 3)

        sl.addWidget(QLabel("Letter:"), 4, 0)
        self._input_letter = QLineEdit("A")
        self._input_letter.setMaxLength(1)
        self._input_letter.setFixedWidth(56)
        self._input_letter.textChanged.connect(self._on_letter_input_changed)
        sl.addWidget(self._input_letter, 4, 1)

        self._btn_send_slm = QPushButton("Send to SLM")
        self._btn_send_slm.clicked.connect(self._send_to_slm)
        sl.addWidget(self._btn_send_slm, 4, 2)

        sl.addWidget(QLabel("Font size:"), 5, 0)
        self._spin_font = QSpinBox()
        self._spin_font.setRange(50, 800)
        self._spin_font.setValue(400)
        self._spin_font.setSingleStep(20)
        self._spin_font.valueChanged.connect(self._on_font_size_changed)
        sl.addWidget(self._spin_font, 5, 1, 1, 2)

        sl.addWidget(QLabel("A-Z cycle:"), 6, 0)
        self._btn_prev = QPushButton("◀ Prev")
        self._btn_next = QPushButton("Next ▶")
        self._btn_prev.clicked.connect(self._prev_letter)
        self._btn_next.clicked.connect(self._next_letter)
        sl.addWidget(self._btn_prev, 6, 1)
        sl.addWidget(self._btn_next, 6, 2)

        self._btn_load_img = QPushButton("Load Image to SLM")
        self._btn_load_img.setToolTip(
            "Load a PNG/JPG/BMP image and display it on the SLM window.\n"
            "Use this to display PPT-exported slides or phase-mask patterns."
        )
        self._btn_load_img.clicked.connect(self._load_image_to_slm)
        sl.addWidget(self._btn_load_img, 7, 0, 1, 3)

        self._lbl_screen_hint = QLabel("Detected displays: checking...")
        self._lbl_screen_hint.setStyleSheet("color: #888; font-size: 11px;")
        self._lbl_screen_hint.setWordWrap(True)
        sl.addWidget(self._lbl_screen_hint, 8, 0, 1, 3)
        layout.addWidget(slm_box)

        cam_box = QGroupBox("Camera / Video Source")
        cl = QGridLayout(cam_box)
        cl.setSpacing(6)

        cl.addWidget(QLabel("Camera index:"), 0, 0)
        self._spin_cam_idx = QSpinBox()
        self._spin_cam_idx.setRange(0, 20)
        cl.addWidget(self._spin_cam_idx, 0, 1)

        self._btn_start_cam = QPushButton("Start Camera")
        self._btn_start_cam.setObjectName("accent")
        self._btn_start_cam.clicked.connect(self._start_camera)
        cl.addWidget(self._btn_start_cam, 0, 2)

        self._btn_scan_cam = QPushButton("Scan Available Cameras")
        self._btn_scan_cam.setToolTip(
            "Probe indices 0-9 on the main thread.\n"
            "On macOS this also triggers the camera permission dialog."
        )
        self._btn_scan_cam.clicked.connect(self._scan_cameras)
        cl.addWidget(self._btn_scan_cam, 1, 0, 1, 3)

        cl.addWidget(QLabel("Resolution:"), 2, 0)
        self._combo_cam_res = QComboBox()
        self._combo_cam_res.addItem("Auto (default)", (None, None))
        self._combo_cam_res.addItem("2048×1536", (2048, 1536))
        self._combo_cam_res.addItem("1920×1440", (1920, 1440))
        self._combo_cam_res.addItem("1280×960",  (1280, 960))
        self._combo_cam_res.addItem("1024×768",  (1024, 768))
        self._combo_cam_res.addItem("640×480",   (640, 480))
        self._combo_cam_res.setCurrentIndex(1)
        cl.addWidget(self._combo_cam_res, 2, 1, 1, 2)

        self._btn_stop_cam = QPushButton("Stop")
        self._btn_stop_cam.setObjectName("danger")
        self._btn_stop_cam.setEnabled(False)
        self._btn_stop_cam.clicked.connect(self._stop_camera)
        cl.addWidget(self._btn_stop_cam, 3, 2)

        self._btn_load_video = QPushButton("Load Video File")
        self._btn_load_video.clicked.connect(self._load_video_file)
        cl.addWidget(self._btn_load_video, 3, 0, 1, 2)

        # ── MindVision CCD (HT-UBS300C) via vendor SDK ──────────────────────
        self._btn_start_mv = QPushButton("MindVision CCD (HT-UBS300C)")
        self._btn_start_mv.setObjectName("accent")
        self._btn_start_mv.setToolTip(
            "Connect via MindVision SDK (libmvsdk.dylib).\n"
            "Use this instead of 'Start Camera' for the HT-UBS300C."
        )
        self._btn_start_mv.clicked.connect(self._start_mv_camera)
        cl.addWidget(self._btn_start_mv, 4, 0, 1, 3)

        self._lbl_source = QLabel("No source")
        self._lbl_source.setStyleSheet("color: #888; font-size: 11px;")
        self._lbl_source.setWordWrap(True)
        cl.addWidget(self._lbl_source, 5, 0, 1, 3)
        layout.addWidget(cam_box)

        layout.addWidget(self._build_cam_settings_box())

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

        self._chk_infer_active = QCheckBox("Recognition active")
        self._chk_infer_active.setChecked(True)
        il.addWidget(self._chk_infer_active, 2, 0, 1, 2)
        layout.addWidget(inf_box)

        layout.addStretch()
        return container

    def _build_right_panel(self):
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(4, 4, 4, 4)

        self._top_splitter = QSplitter(Qt.Horizontal)
        self._top_splitter.setChildrenCollapsible(False)
        self._top_splitter.setHandleWidth(3)

        cam_box = QGroupBox("Live Camera Feed")
        cam_layout = QVBoxLayout(cam_box)
        cam_layout.setContentsMargins(6, 6, 6, 6)
        self._cam_label = CameraLabel()
        cam_layout.addWidget(self._cam_label)
        self._top_splitter.addWidget(cam_box)

        self._pred_box = QGroupBox("Recognition Output")
        pred_layout = QVBoxLayout(self._pred_box)
        pred_layout.setSpacing(8)
        pred_layout.setContentsMargins(8, 8, 8, 8)

        lbl_inst = QLabel("Instant")
        lbl_inst.setAlignment(Qt.AlignCenter)
        lbl_inst.setStyleSheet("color: #888; font-size: 11px;")
        pred_layout.addWidget(lbl_inst)

        self._lbl_pred_letter = QLabel("?")
        self._lbl_pred_letter.setObjectName("pred_letter")
        self._lbl_pred_letter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        pred_layout.addWidget(self._lbl_pred_letter)

        self._lbl_conf = QLabel("Confidence: --")
        self._lbl_conf.setObjectName("conf_label")
        pred_layout.addWidget(self._lbl_conf)

        lbl_smooth = QLabel("Smoothed (majority vote)")
        lbl_smooth.setAlignment(Qt.AlignCenter)
        lbl_smooth.setStyleSheet("color: #888; font-size: 11px;")
        pred_layout.addWidget(lbl_smooth)

        self._lbl_pred_smooth = QLabel("?")
        self._lbl_pred_smooth.setObjectName("pred_smooth")
        self._lbl_pred_smooth.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        pred_layout.addWidget(self._lbl_pred_smooth)

        lbl_topk = QLabel("Top-5 candidates:")
        lbl_topk.setStyleSheet("color: #888; font-size: 11px; margin-top: 4px;")
        pred_layout.addWidget(lbl_topk)

        self._lbl_topk = QLabel("--")
        self._lbl_topk.setWordWrap(True)
        self._lbl_topk.setStyleSheet("color: #a0c4ff; font-size: 12px;")
        self._lbl_topk.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        pred_layout.addWidget(self._lbl_topk)

        pred_layout.addStretch()
        self._top_splitter.addWidget(self._pred_box)
        self._top_splitter.setStretchFactor(0, 5)
        self._top_splitter.setStretchFactor(1, 2)
        self._top_splitter.setSizes([900, 320])

        layout.addWidget(self._top_splitter, stretch=1)
        return layout

    def _apply_style(self):
        self.setStyleSheet(DARK_STYLE)

    def _apply_responsive_metrics(self, force: bool = False):
        w = max(1, self.width())
        h = max(1, self.height())

        left_width = max(320, min(420, int(w * 0.24)))
        pred_width = max(280, min(380, int(w * 0.24)))
        log_h = max(84, min(140, int(h * 0.12)))

        if self._left_scroll is not None:
            self._left_scroll.setMinimumWidth(left_width)
            self._left_scroll.setMaximumWidth(left_width + 20)

        if self._pred_box is not None:
            self._pred_box.setMinimumWidth(pred_width)
            self._pred_box.setMaximumWidth(pred_width + 20)

        if self._main_splitter is not None and (force or w > 0):
            right_width = max(600, w - left_width - 60)
            self._main_splitter.setSizes([left_width, right_width])

        if self._top_splitter is not None and (force or w > 0):
            cam_width = max(560, w - left_width - pred_width - 120)
            self._top_splitter.setSizes([cam_width, pred_width])

        if self._log_text is not None:
            self._log_text.setMinimumHeight(log_h)
            self._log_text.setMaximumHeight(log_h + 20)

        instant_size = max(52, min(92, int(min(pred_width, h) * 0.22)))
        smooth_size = max(38, min(68, int(min(pred_width, h) * 0.16)))
        conf_size = max(16, min(24, int(pred_width * 0.06)))
        topk_size = max(11, min(14, int(pred_width * 0.035)))

        self._lbl_pred_letter.setFont(QFont("Segoe UI", instant_size, QFont.Bold))
        self._lbl_pred_smooth.setFont(QFont("Segoe UI", smooth_size, QFont.Bold))
        self._lbl_conf.setFont(QFont("Segoe UI", conf_size, QFont.Bold))
        self._lbl_topk.setFont(QFont("Segoe UI", topk_size))

        self._lbl_pred_letter.setMinimumHeight(max(110, int(h * 0.17)))
        self._lbl_pred_smooth.setMinimumHeight(max(86, int(h * 0.12)))

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

    def _refresh_fiber_list(self):
        self._fiber_models = discover_fiber_models()
        prev = self._combo_fiber.currentText()
        self._combo_fiber.blockSignals(True)
        self._combo_fiber.clear()

        if self._fiber_models:
            for name in self._fiber_models:
                self._combo_fiber.addItem(name)
            self._log(f"Found fiber models: {list(self._fiber_models.keys())}")
        else:
            self._log(f"[WARNING] No fiber models in {FIBER_MODELS_DIR}")
            self._log("  Run: python -u scripts/fiber_auth_eval.py")

        if prev and self._combo_fiber.findText(prev) >= 0:
            self._combo_fiber.setCurrentText(prev)
        self._combo_fiber.blockSignals(False)

        # Load the currently shown fiber (signals were blocked during population)
        if self._combo_fiber.count() > 0:
            self._active_fiber = ""
            self._on_fiber_selected(self._combo_fiber.currentIndex())

    def _on_fiber_selected(self, index: int):
        fiber = self._combo_fiber.currentText()
        if not fiber:
            return
        path = self._fiber_models.get(fiber)
        if not path or not os.path.isfile(path):
            self._lbl_model_status.setStyleSheet("color: #e06c75; font-size: 11px;")
            self._lbl_model_status.setText(f"Model not found for {fiber}")
            self._lbl_model_path.setText("")
            return
        self._active_fiber = fiber
        self._lbl_model_path.setText(os.path.basename(path))
        self._lbl_model_status.setStyleSheet("color: #a0c4ff; font-size: 11px;")
        self._lbl_model_status.setText(f"Loading {fiber} ...")
        self._log(f"Loading model for {fiber} ...")
        ok = self._infer_worker.load_model(path)
        if not ok:
            self._lbl_model_status.setStyleSheet("color: #e06c75; font-size: 11px;")
            self._lbl_model_status.setText("Load failed")

    def _describe_screen(self, idx, screen):
        geom = screen.geometry()
        name = screen.name() or f"Screen {idx}"
        primary = QGuiApplication.primaryScreen()
        suffix = " [Primary]" if screen == primary else ""
        return f"{idx}: {name} | {geom.width()}x{geom.height()} @ ({geom.x()},{geom.y()}){suffix}"

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
            f"Detected {len(screens)} display(s). Select the SLM target here; do not rely only on Windows duplicate/extend buttons."
        )

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
            self._slm_window.set_font_size(self._spin_font.value())
            letter = self._input_letter.text().strip().upper() or "A"
            self._slm_window.set_letter(letter)
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

    def _load_model(self):
        fiber = self._combo_fiber.currentText()
        if fiber:
            self._on_fiber_selected(self._combo_fiber.currentIndex())
        else:
            self._log("[ERROR] No fiber selected")

    @Slot(str)
    def _on_model_loaded(self, msg: str):
        fiber = self._active_fiber or "?"
        self._lbl_model_status.setStyleSheet("color: #51cf66; font-size: 11px;")
        self._lbl_model_status.setText(f"Loaded: {fiber}")
        self._lbl_model.setText(f"Fiber: {fiber}")
        self._lbl_auth_warning.setVisible(False)
        # Sync dropdown to the fiber that actually finished loading
        if fiber and self._combo_fiber.currentText() != fiber:
            self._combo_fiber.blockSignals(True)
            idx = self._combo_fiber.findText(fiber)
            if idx >= 0:
                self._combo_fiber.setCurrentIndex(idx)
            self._combo_fiber.blockSignals(False)
        self._log(f"[MODEL] {msg}")

    def _toggle_slm_window(self):
        if self._slm_window is not None and self._slm_window.isVisible():
            self._slm_window.hide()
            self._btn_show_slm.setText("Open SLM Window")
            self._log("SLM window hidden.")
            return

        self._show_slm_on_selected_screen(force_show=True)
        self._btn_show_slm.setText("Hide SLM Window")

    def _send_to_slm(self):
        letter = self._input_letter.text().strip().upper()
        if not letter:
            return

        self._show_slm_on_selected_screen(force_show=True)
        self._btn_show_slm.setText("Hide SLM Window")
        self._slm_window.set_letter(letter)
        self._log(f"SLM: displaying letter '{letter}'")

    def _on_letter_input_changed(self, text: str):
        if self._slm_window and self._slm_window.isVisible() and text.strip():
            self._slm_window.set_letter(text.strip().upper())

    def _on_font_size_changed(self, size: int):
        if self._slm_window:
            self._slm_window.set_font_size(size)

    def _prev_letter(self):
        current = self._input_letter.text().strip().upper()
        if not current or current < "A" or current > "Z":
            current = "A"
        c = chr(max(ord("A"), ord(current) - 1))
        self._input_letter.setText(c)
        self._send_to_slm()

    def _next_letter(self):
        current = self._input_letter.text().strip().upper()
        if not current or current < "A" or current > "Z":
            current = "A"
        c = chr(min(ord("Z"), ord(current) + 1))
        self._input_letter.setText(c)
        self._send_to_slm()

    def _load_image_to_slm(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image for SLM",
            self._root,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
        )
        if not path:
            return

        self._show_slm_on_selected_screen(force_show=True)
        self._btn_show_slm.setText("Hide SLM Window")
        ok = self._slm_window.load_image(path)
        if ok:
            self._show_slm_on_selected_screen(force_show=True)
            self._log(f"SLM: image loaded -> {os.path.basename(path)}")
        else:
            self._log(f"[ERROR] Could not load image: {path}")

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

    @Slot(dict)
    def _on_cam_props(self, props: dict):
        """Sync UI sliders/spinboxes with values reported by the camera."""

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

    def _scan_cameras(self):
        """Probe camera indices 0-9 on the main thread and update the spin box.

        Running on the main thread is essential on macOS: the first VideoCapture
        call triggers the system camera-permission dialog, which MUST happen on
        the main run loop.  Calling it from a worker thread causes the
        'can not spin main run loop from other thread' error.
        """
        import sys

        self._btn_scan_cam.setEnabled(False)
        self._btn_scan_cam.setText("Scanning…")
        QApplication.processEvents()

        # ── macOS: list all devices the OS knows about ─────────────────
        if sys.platform == "darwin":
            dev_names = self._macos_list_av_devices()
            if dev_names:
                self._log("macOS AVFoundation devices detected by OS:")
                for i, name in enumerate(dev_names):
                    if name:
                        self._log(f"  [{i}] {name}")
            else:
                self._log("macOS: could not enumerate device names "
                          "(install ffmpeg via Homebrew for full device list).")

        # Remove SKIP_AUTH temporarily so the permission dialog can appear if
        # macOS authorization has not been granted yet (or was revoked).
        _skip_auth_backup = None
        if sys.platform == "darwin":
            _skip_auth_backup = os.environ.pop("OPENCV_AVFOUNDATION_SKIP_AUTH", None)

        available = []
        for i in range(12):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available.append((i, w, h))
            cap.release()

        self._btn_scan_cam.setEnabled(True)
        self._btn_scan_cam.setText("Scan Available Cameras")

        if available:
            # Pick the LAST available index: 0 = built-in, higher = external
            best_idx = available[-1][0]
            self._spin_cam_idx.setValue(best_idx)
            dev_names_local = self._macos_list_av_devices() if sys.platform == "darwin" else []
            for idx, w, h in available:
                name = dev_names_local[idx] if idx < len(dev_names_local) else ""
                tag  = f"  ← {name}" if name else ""
                self._log(f"  Camera [{idx}] {w}×{h}{tag}")
            self._log(f"Scan complete. Found {len(available)} camera(s). "
                      f"Selected index {best_idx} (change in Camera index spinner).")
            if sys.platform == "darwin":
                os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"
        else:
            if sys.platform == "darwin" and _skip_auth_backup is not None:
                os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = _skip_auth_backup

            msg = "No cameras found (OpenCV returned 0 devices)."
            if sys.platform == "darwin":
                msg += (
                    "\n\nPossible causes on macOS:"
                    "\n1. Permission not granted yet — grant Cursor access in"
                    "\n   System Settings → Privacy & Security → Camera,"
                    "\n   then QUIT AND RESTART the app (macOS requires a restart"
                    "\n   before newly-granted permissions take effect)."
                    "\n\n2. External CCD not recognised — check if the camera"
                    "\n   appears in System Information → USB or Camera list."
                    "\n   Some industrial cameras require their own SDK driver."
                )
            self._log(f"[CAMERA] {msg}")
            QMessageBox.warning(self, "No Cameras Found", msg)

    def _start_camera(self):
        self._stop_camera()
        idx = self._spin_cam_idx.value()

        # ── macOS camera-permission pre-check (main thread) ───────────────
        # AVFoundation's permission dialog must be triggered from the main
        # run loop.  We open the camera briefly here so macOS can prompt the
        # user, then hand off to the worker thread.  On non-macOS platforms
        # this is a fast no-op.
        import sys
        if sys.platform == "darwin":
            # Remove SKIP_AUTH so the system permission dialog can appear.
            os.environ.pop("OPENCV_AVFOUNDATION_SKIP_AUTH", None)
            test_cap = cv2.VideoCapture(idx)
            opened = test_cap.isOpened()
            test_cap.release()
            if not opened:
                # Try to get device names so user can identify which index to use
                dev_names = self._macos_list_av_devices()
                hint = ""
                if dev_names:
                    hint = "\n\nDevices macOS sees:\n" + "\n".join(
                        f"  [{i}] {n}" for i, n in enumerate(dev_names) if n
                    )
                msg = (
                    f"Cannot open camera device {idx} on macOS.{hint}\n\n"
                    "Checklist:\n"
                    "1. Grant Cursor access:\n"
                    "   System Settings → Privacy & Security → Camera → Cursor ON\n"
                    "2. QUIT and RESTART this app after granting permission\n"
                    "   (macOS only applies new permissions after restart).\n"
                    "3. Click 'Scan Available Cameras' to find the correct index.\n"
                    "4. External industrial CCDs may need their vendor SDK installed."
                )
                self._log(f"[CAMERA ERROR] Cannot open device {idx}. "
                          f"Check permissions and restart. Devices: {dev_names or 'unknown'}")
                QMessageBox.critical(self, "Cannot Open Camera", msg)
                return
            # Permission confirmed — suppress re-authorization in the worker thread.
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
        self._lbl_source.setText(f"Camera device: {idx}")
        self._set_cam_controls_enabled(True)
        self._log(f"Camera started (device {idx})")

    def _start_mv_camera(self):
        """Start the HT-UBS300C via MindVision SDK (bypasses OpenCV entirely)."""
        self._stop_camera()

        # Enumerate MindVision devices
        try:
            mvsdk.sdk_init()
            devices = mvsdk.enumerate_devices()
        except FileNotFoundError as exc:
            QMessageBox.critical(self, "MindVision SDK Not Found", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "MindVision Error", f"Enumeration failed:\n{exc}")
            return

        if not devices:
            import sys as _sys
            if _sys.platform == "darwin":
                hint = (
                    "macOS checklist:\n"
                    "1. Plug in the HT-UBS300C USB cable and wait ~3 s.\n"
                    "2. Click this button again.\n"
                    "3. If still not found, check System Settings → Privacy → Camera."
                )
            else:
                hint = (
                    "Windows checklist:\n"
                    "1. Plug in the HT-UBS300C USB cable and wait ~5 s.\n"
                    "2. Open Device Manager and verify the camera appears\n"
                    "   (should show as 'MindVision USB Camera' or similar).\n"
                    "3. If it shows as 'Unknown Device', install the MindVision\n"
                    "   USB driver from the SDK folder first.\n"
                    "4. Click this button again."
                )
            QMessageBox.warning(
                self, "No MindVision Camera",
                f"No MindVision camera detected.\n\n{hint}"
            )
            return

        # If multiple devices, pick the first (could add a dialog later)
        dev = devices[0]
        name = dev.friendly_name or dev.product_name
        self._log(f"MindVision: found camera '{name}' (SN: {dev.sn})")

        try:
            self._camera_worker = MvCameraWorker(dev, self)
        except Exception as exc:
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
        self._lbl_source.setText(f"MindVision CCD: {name}")
        self._set_cam_controls_enabled(True)
        self._log(f"MindVision camera started: {name}")

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

    def _stop_camera(self):
        if self._camera_worker and self._camera_worker.isRunning():
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
        self._capture_active = False
        self._btn_start_cam.setEnabled(True)
        self._btn_stop_cam.setEnabled(False)
        self._cam_label.setText("No camera feed")
        self._cam_label._pixmap = None
        self._set_cam_controls_enabled(False)
        self._log("Camera stopped.")

    @Slot(object)
    def _on_frame(self, frame: np.ndarray):
        self._last_frame = frame
        self._cam_label.set_frame(frame)
        if self._chk_infer_active.isChecked() and self._infer_worker._model is not None:
            self._infer_worker.push_frame(frame)

    @Slot(str)
    def _on_cam_error(self, msg: str):
        self._log(f"[CAMERA ERROR] {msg}")
        self._stop_camera()

    @Slot(float)
    def _on_fps_update(self, fps: float):
        self._fps = fps
        self._lbl_fps.setText(f"FPS: {fps:.1f}")

    @Slot(dict)
    def _on_prediction(self, result: dict):
        top1 = result.get("top1", "?")
        conf = result.get("confidence", 0.0)
        topk = result.get("topk", [])
        smoothed = result.get("smoothed", "?")

        self._lbl_pred_letter.setText(top1)
        self._lbl_pred_smooth.setText(smoothed)
        self._lbl_conf.setText(f"Confidence: {conf * 100:.1f}%")
        topk_str = "  ".join(f"{cls}({p * 100:.1f}%)" for cls, p in topk)
        self._lbl_topk.setText(topk_str if topk_str else "--")

        if conf < LOW_CONFIDENCE_THRESHOLD:
            self._lbl_auth_warning.setText(
                f"Low confidence ({conf*100:.0f}%) — possible unauthorized fiber or noise"
            )
            self._lbl_auth_warning.setStyleSheet(
                "color: #ff6b6b; font-weight: bold; font-size: 11px; "
                "background-color: #3c1515; border: 1px solid #e06c75; "
                "border-radius: 4px; padding: 4px;"
            )
            self._lbl_auth_warning.setVisible(True)
        else:
            self._lbl_auth_warning.setVisible(False)

    @Slot(str)
    def _on_infer_error(self, msg: str):
        self._log(f"[INFERENCE ERROR] {msg}")

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self._log_text.append(f"[{ts}] {msg}")

    def closeEvent(self, event):
        self._stop_camera()
        if self._infer_worker.isRunning():
            self._infer_worker.stop()
        if self._slm_window:
            self._slm_window.close()
        super().closeEvent(event)
