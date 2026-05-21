"""
Typography and layout helpers for the public Speckle-PUF demo GUI.
"""

from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel, QGroupBox, QVBoxLayout


def demo_font(
    pixel_size: int,
    *,
    bold: bool = False,
    weight: int | None = None,
) -> QFont:
    f = QFont()
    f.setFamilies(
        ["-apple-system", "BlinkMacSystemFont", "Segoe UI", "Helvetica Neue", "Arial"]
    )
    f.setPixelSize(pixel_size)
    if weight is not None:
        f.setWeight(weight)
    elif bold:
        f.setWeight(QFont.Bold)
    return f


def add_card_title(layout: QVBoxLayout, text: str) -> QLabel:
    title = QLabel(text)
    title.setObjectName("demoCardTitle")
    title.setFont(demo_font(19, bold=True))
    layout.addWidget(title)
    return title


def style_demo_card(box: QGroupBox, *, object_name: str = "challengeInputBox") -> None:
    box.setObjectName(object_name)
    extra = ""
    if object_name == "challengeInputBox":
        extra = " margin-bottom: 12px;"
    box.setStyleSheet(
        "QGroupBox {"
        "  background-color: #FFFFFF;"
        "  border: 1px solid #D1D1D6;"
        "  border-radius: 16px;"
        "  margin-top: 0px;"
        f"  padding: 10px 12px 8px 12px;{extra}"
        "}"
        "QGroupBox::title {"
        "  subcontrol-origin: margin;"
        "  subcontrol-position: top left;"
        "  padding: 0px;"
        "  color: transparent;"
        "  font-size: 1px;"
        "}"
    )


def style_control_section(box: QGroupBox) -> None:
    """SLM / challenge control cards below the challenge preview."""
    box.setObjectName("demoControlSection")
    box.setStyleSheet(
        "QGroupBox#demoControlSection {"
        "  background-color: #FFFFFF;"
        "  border: 1px solid #D1D1D6;"
        "  border-radius: 16px;"
        "  margin-top: 4px;"
        "  padding: 12px;"
        "}"
        "QGroupBox#demoControlSection::title {"
        "  color: transparent;"
        "  font-size: 1px;"
        "  padding: 0px;"
        "}"
    )


def row_html(label: str, value: str, *, value_px: int = 22, label_px: int = 17) -> str:
    safe_val = value.replace("&", "&amp;").replace("<", "&lt;")
    return (
        f'<span style="font-size:{label_px}px; color:#636366; font-weight:600;">'
        f"{label}</span> "
        f'<span style="font-size:{value_px}px; font-weight:800; color:#1C1C1E;">'
        f"{safe_val}</span>"
    )


DEMO_PRESENTATION_QSS = """
QGroupBox#challengeInputBox, QGroupBox#recognitionResultBox {
    background-color: #FFFFFF;
    border: 1px solid #C7C7CC;
    border-radius: 16px;
    margin-top: 0px;
}
QGroupBox#challengeInputBox::title,
QGroupBox#recognitionResultBox::title {
    color: transparent;
    font-size: 1px;
    padding: 0px;
}
QLabel#demoCardTitle {
    color: #1C1C1E;
    font-size: 19px;
    font-weight: 700;
    padding: 0px 0px 6px 0px;
    min-height: 24px;
}
QLabel#challengePreview {
    background-color: #0A0A0C;
    border: 2px solid #3A3A3C;
    border-radius: 12px;
    color: #F5F5F7;
}
QLabel#challengeCurrentLabel {
    color: #1C1C1E;
    font-size: 23px;
    font-weight: 700;
}
QLabel#challengeCurrentValue {
    color: #007AFF;
    font-size: 32px;
    font-weight: 800;
}
QLabel#challengeSourceLabel {
    color: #48484A;
    font-size: 19px;
    font-weight: 600;
    padding-top: 2px;
}
QFrame#recognitionResultBody {
    background-color: #F9F9FB;
    border: 1px solid #E5E5EA;
    border-radius: 12px;
    min-height: 240px;
}
QLabel#recognitionLine {
    color: #1C1C1E;
    padding: 2px 0px;
    min-height: 26px;
}
QLabel#recognitionDecision {
    color: #5f9bff;
    font-size: 24px;
    font-weight: 800;
    padding: 6px 4px 2px 4px;
    min-height: 36px;
    max-height: 48px;
}
QLabel#camTitle {
    color: #1C1C1E;
    font-size: 30px;
    font-weight: 700;
    padding-bottom: 8px;
}
QFrame#camCard {
    background-color: #FFFFFF;
    border: 1px solid #D1D1D6;
    border-radius: 18px;
}
QLabel#camLabel {
    background-color: #000000;
    border: 2px solid #3A3A3C;
    border-radius: 14px;
    color: #AEAEB2;
}
QGroupBox#demoControlSection QPushButton#openSlmPrimary {
    background-color: #007AFF;
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    font-size: 23px;
    font-weight: 700;
    min-height: 72px;
    max-height: 76px;
    padding: 16px 18px;
}
QGroupBox#demoControlSection QPushButton#openSlmPrimary:hover {
    background-color: #0066D6;
}
QGroupBox#demoControlSection QPushButton#openSlmPrimary:pressed {
    background-color: #0052AB;
}
QGroupBox#demoControlSection QLabel {
    color: #1C1C1E;
    font-size: 16px;
    font-weight: 700;
}
QGroupBox#demoControlSection QPushButton {
    font-size: 15px;
    font-weight: 600;
    padding: 10px 14px;
    min-height: 38px;
}
QGroupBox#demoControlSection QPushButton#primary {
    font-size: 16px;
    font-weight: 700;
    min-height: 48px;
    padding: 12px 16px;
}
QGroupBox#demoControlSection QComboBox,
QGroupBox#demoControlSection QLineEdit,
QGroupBox#demoControlSection QSpinBox {
    font-size: 15px;
    font-weight: 500;
    min-height: 38px;
    padding: 6px 10px;
}
QGroupBox#demoControlSection QCheckBox {
    font-size: 15px;
    font-weight: 600;
    spacing: 10px;
}
QGroupBox#demoControlSection QCheckBox::indicator {
    width: 20px;
    height: 20px;
}
QGroupBox#demoControlSection QLabel#demoHintLabel {
    font-size: 13px;
    font-weight: 500;
    color: #8E8E93;
}
QGroupBox#logPanel {
    margin-top: 10px;
    padding: 8px;
}
QGroupBox#logPanel::title {
    font-size: 11px;
    color: #8E8E93;
}
QTextEdit#demoLogText {
    font-size: 11px;
    padding: 8px;
}
QLabel#robotStatus {
    font-size: 22px;
    font-weight: 800;
    padding-top: 0px;
}
QLabel#robotAction {
    font-size: 16px;
    padding-top: 2px;
}
QLabel#robotConf {
    font-size: 17px;
    font-weight: 700;
    padding-top: 2px;
}
QLabel#robotTopK {
    font-size: 14px;
}
"""
