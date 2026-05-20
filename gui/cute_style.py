"""
PREMIUM_STYLE — Apple HIG / Material-Design-inspired QSS theme.
Import and pass to QMainWindow.setStyleSheet().
"""

PREMIUM_STYLE = """
/* ── Global Canvas ──────────────────────────────────────────────────── */
QMainWindow, QWidget {
    background-color: #F2F2F7;
    color: #1C1C1E;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                 Helvetica, Arial, sans-serif;
    font-size: 14px;
}

/* ── Cards (Group Boxes → borderless floating cards) ────────────────── */
QGroupBox {
    background-color: #FFFFFF;
    border: none;
    border-radius: 16px;
    margin-top: 32px;
    padding: 16px;
}
QGroupBox::title {
    subcontrol-origin:   margin;
    subcontrol-position: top left;
    padding:   0px;
    color:     #8E8E93;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    top: -12px;
}

/* ── Buttons — default ghost style ─────────────────────────────────── */
QPushButton {
    background-color: #F2F2F7;
    color:            #007AFF;
    border:           none;
    border-radius:    10px;
    padding:          10px 16px;
    font-weight:      600;
    font-size:        14px;
}
QPushButton:hover    { background-color: #E5E5EA; }
QPushButton:pressed  { background-color: #D1D1D6; color: #0056B3; }
QPushButton:disabled { color: #C7C7CC;  background-color: #F2F2F7; }

/* ── Primary / CTA buttons ──────────────────────────────────────────── */
QPushButton#primary {
    background-color: #007AFF;
    color:            #FFFFFF;
    border-radius:    10px;
    font-weight:      600;
}
QPushButton#primary:hover   { background-color: #0066D6; }
QPushButton#primary:pressed { background-color: #0052AB; }

/* ── Danger / destructive buttons ───────────────────────────────────── */
QPushButton#danger {
    background-color: #FFF1F0;
    color:            #FF3B30;
    border-radius:    10px;
}
QPushButton#danger:hover { background-color: #FFE0DE; }

/* ── Inputs ─────────────────────────────────────────────────────────── */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #F2F2F7;
    border:           1px solid transparent;
    border-radius:    10px;
    padding:          8px 12px;
    min-height:       28px;
    font-weight:      500;
}
QComboBox:focus, QLineEdit:focus {
    background-color: #FFFFFF;
    border:           2px solid #007AFF;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background-color: #FFFFFF;
    border:           1px solid #E5E5EA;
    border-radius:    10px;
    selection-background-color: #E5F2FF;
    selection-color:  #007AFF;
}

/* ── Checkboxes ─────────────────────────────────────────────────────── */
QCheckBox { color: #3A3A3C; spacing: 8px; }
QCheckBox::indicator {
    width: 20px; height: 20px;
    border-radius: 6px;
    border: 2px solid #C7C7CC;
    background-color: #FFFFFF;
}
QCheckBox::indicator:checked {
    background-color: #007AFF;
    border-color:     #007AFF;
    image: none;
}

/* ── Sliders ─────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {
    background:    #E5E5EA;
    height:        4px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background:    #007AFF;
    width:         18px;
    height:        18px;
    margin:        -7px 0;
    border-radius: 9px;
}
QSlider::sub-page:horizontal {
    background:    #007AFF;
    border-radius: 2px;
}

/* ── Scrollbars — minimal / hidden ──────────────────────────────────── */
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical {
    border: none; background: transparent;
    width: 6px; margin: 0px;
}
QScrollBar::handle:vertical {
    background:    #D1D1D6;
    border-radius: 3px;
    min-height:    24px;
}
QScrollBar::handle:vertical:hover { background: #AEAEB2; }
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical { height: 0px; }

/* ── Splitter ────────────────────────────────────────────────────────── */
QSplitter::handle           { background-color: transparent; }
QSplitter::handle:horizontal { width: 12px; }

/* ── Cards (Robot panel + Camera card) ───────────────────────────────── */
QFrame#robotPanel, QFrame#camCard {
    background-color: #FFFFFF;
    border:           none;
    border-radius:    20px;
}
QLabel#robotStatus { color: #1C1C1E; letter-spacing: 0px; }
QLabel#robotAction { color: #8E8E93; font-size: 14px; }
QLabel#robotConf   { color: #007AFF; font-weight: 700; font-size: 15px; }
QLabel#robotTopK   { color: #8E8E93; font-size: 13px; }

/* ── Overlay banner ──────────────────────────────────────────────────── */
QLabel#overlayBanner {
    color:            #FFFFFF;
    font-weight:      800;
    font-size:        28px;
    letter-spacing:   0px;
    background-color: rgba(28, 28, 30, 0.85);
    border-radius:    20px;
    padding:          16px 28px;
}

/* ── Log text area ───────────────────────────────────────────────────── */
QTextEdit {
    background-color: #FFFFFF;
    border:           none;
    border-radius:    16px;
    color:            #3A3A3C;
    font-family:      'SF Mono', 'Consolas', monospace;
    font-size:        12px;
    padding:          12px;
}

/* ── Status bar ──────────────────────────────────────────────────────── */
QStatusBar {
    background-color: #F2F2F7;
    color:            #8E8E93;
    font-size:        12px;
    border-top:       none;
}
"""

from gui.demo_presentation import DEMO_PRESENTATION_QSS

PREMIUM_STYLE = PREMIUM_STYLE + DEMO_PRESENTATION_QSS
