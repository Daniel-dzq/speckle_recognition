"""
Challenge preview and recognition result widgets for the live demo GUI.
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QFontMetrics, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui.demo_presentation import (
    add_card_title,
    demo_font,
    row_html,
    style_demo_card,
)


def _normalize_label(label: str) -> str:
    return (label or "").strip()


def labels_match(challenge: str, predicted: str) -> bool:
    """Case-insensitive match for arbitrary challenge / prediction labels."""
    c = _normalize_label(challenge)
    p = _normalize_label(predicted)
    if not c or not p or p in ("?", "—", "-"):
        return False
    return c.lower() == p.lower()


class ChallengePreviewWidget(QGroupBox):
    """Compact left-column preview of the current SLM challenge pattern."""

    PREVIEW_MIN_H = 150
    PREVIEW_MAX_H = 180
    EMPTY_PLACEHOLDER_MAX_PT = 30
    EMPTY_PLACEHOLDER_MIN_PT = 28
    EMPTY_PLACEHOLDER_SIDE_MARGIN = 28
    EMPTY_PLACEHOLDER_TEXT = "No challenge selected"
    TEXT_SYMBOL_MIN_PT = 160
    TEXT_SYMBOL_MAX_PT = 190

    def __init__(self, parent=None):
        super().__init__("", parent)
        style_demo_card(self)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(12, 10, 12, 8)
        add_card_title(layout, "Challenge input")

        self._preview = QLabel("No challenge selected")
        self._preview.setObjectName("challengePreview")
        self._preview.setProperty("empty", True)
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setWordWrap(False)
        self._preview.setMinimumHeight(self.PREVIEW_MIN_H)
        self._preview.setMaximumHeight(self.PREVIEW_MAX_H)
        self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self._preview, stretch=0)

        current_row = QHBoxLayout()
        current_row.setSpacing(6)
        self._lbl_current_prefix = QLabel("Current challenge:")
        self._lbl_current_prefix.setObjectName("challengeCurrentLabel")
        self._lbl_current_prefix.setFont(demo_font(23, bold=True))
        self._lbl_current_value = QLabel("—")
        self._lbl_current_value.setObjectName("challengeCurrentValue")
        self._lbl_current_value.setFont(demo_font(32, weight=QFont.Black))
        current_row.addWidget(self._lbl_current_prefix)
        current_row.addWidget(self._lbl_current_value, stretch=1)
        current_row.addStretch()
        layout.addLayout(current_row)

        self._lbl_source = QLabel("Source: —")
        self._lbl_source.setObjectName("challengeSourceLabel")
        self._lbl_source.setFont(demo_font(19, weight=QFont.DemiBold))
        layout.addWidget(self._lbl_source)
        layout.addSpacing(4)

        self._label = ""
        self._source = ""
        self._pixmap: Optional[QPixmap] = None
        self._show_empty_placeholder()

    def label(self) -> str:
        return self._label

    def source(self) -> str:
        return self._source

    def apply_metrics(self, window_height: int) -> None:
        del window_height
        self._preview.setMinimumHeight(self.PREVIEW_MIN_H)
        self._preview.setMaximumHeight(self.PREVIEW_MAX_H)
        if self._label and self._source == "text":
            self._render_text_preview(self._label)
        elif self._pixmap is not None:
            self._scale_preview_pixmap()

    def _preview_stylesheet(
        self,
        *,
        font_px: int,
        color: str = "#E5E5EA",
        font_weight: int = 700,
        horizontal_padding_px: int = 0,
    ) -> str:
        """Inline QSS so global 14px theme does not override QLabel.setFont()."""
        pad = ""
        if horizontal_padding_px > 0:
            pad = (
                f"  padding-left: {horizontal_padding_px}px;"
                f"  padding-right: {horizontal_padding_px}px;"
            )
        return (
            "QLabel#challengePreview {"
            f"  color: {color};"
            "  background-color: #0A0A0C;"
            f"  font-size: {font_px}px;"
            f"  font-weight: {font_weight};"
            f"{pad}"
            "}"
        )

    def _apply_preview_font(
        self,
        font_px: int,
        *,
        color: str = "#E5E5EA",
        weight: int = QFont.Bold,
    ) -> None:
        self._preview.setStyleSheet(
            self._preview_stylesheet(font_px=font_px, color=color, font_weight=700),
        )
        self._preview.setFont(demo_font(font_px, weight=weight))

    def _fit_empty_placeholder_font(self) -> int:
        """Single-line placeholder; 28–30px, down to 28px if narrow."""
        text = self.EMPTY_PLACEHOLDER_TEXT
        w = max(
            80,
            self._preview.width() - 2 * self.EMPTY_PLACEHOLDER_SIDE_MARGIN,
        )
        for fs in (self.EMPTY_PLACEHOLDER_MIN_PT, self.EMPTY_PLACEHOLDER_MAX_PT):
            fm = QFontMetrics(demo_font(fs, weight=QFont.DemiBold))
            if fm.horizontalAdvance(text) <= w:
                return fs
        return self.EMPTY_PLACEHOLDER_MIN_PT

    def _show_empty_placeholder(self) -> None:
        self._preview.setProperty("empty", True)
        self._preview.setWordWrap(False)
        px = self._fit_empty_placeholder_font() if self._preview.width() > 40 else self.EMPTY_PLACEHOLDER_MIN_PT
        self._preview.setStyleSheet(
            self._preview_stylesheet(
                font_px=px,
                color="#E5E5EA",
                font_weight=600,
                horizontal_padding_px=6,
            ),
        )
        self._preview.setFont(demo_font(px, weight=QFont.DemiBold))
        self._preview.setPixmap(QPixmap())
        self._preview.setText(self.EMPTY_PLACEHOLDER_TEXT)

    def clear_challenge(self) -> None:
        self._label = ""
        self._source = ""
        self._pixmap = None
        self._show_empty_placeholder()
        self._lbl_current_value.setText("—")
        self._lbl_source.setText("Source: —")

    def set_text_challenge(self, label: str, *, source: str = "text") -> None:
        text = _normalize_label(label)
        if not text:
            self.clear_challenge()
            return
        self._label = text
        self._source = source or "text"
        self._pixmap = None
        self._preview.setProperty("empty", False)
        self._preview.setPixmap(QPixmap())
        self._preview.setText("")
        self._render_text_preview(text)
        QTimer.singleShot(0, lambda t=text: self._render_text_preview(t))
        self._lbl_current_value.setText(text)
        self._lbl_source.setText(f"Source: {self._source}")

    def set_image_challenge(self, path: str, label: Optional[str] = None) -> None:
        path = os.path.abspath(os.path.expanduser(path or ""))
        if not path or not os.path.isfile(path):
            self.clear_challenge()
            return
        stem = label or os.path.splitext(os.path.basename(path))[0]
        text = _normalize_label(stem)
        self._label = text
        self._source = "image"
        self._preview.setProperty("empty", False)
        pix = QPixmap(path)
        if pix.isNull():
            self._preview.setPixmap(QPixmap())
            self._preview.setFont(demo_font(16, bold=True))
            self._preview.setText(f"Could not load image\n{os.path.basename(path)}")
        else:
            self._pixmap = pix
            self._preview.setText("")
            self._scale_preview_pixmap()
        self._lbl_current_value.setText(text)
        self._lbl_source.setText(f"Source: image ({os.path.basename(path)})")

    def resizeEvent(self, event):
        if self._pixmap is not None and not self._pixmap.isNull():
            self._scale_preview_pixmap()
        elif self._label and self._source == "text":
            self._render_text_preview(self._label)
        elif self._preview.property("empty"):
            self._show_empty_placeholder()
        super().resizeEvent(event)

    def _scale_preview_pixmap(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        scaled = self._pixmap.scaled(
            self._preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._preview.setPixmap(scaled)

    def _render_text_preview(self, text: str) -> None:
        w = max(120, self._preview.width() - 8)
        h = max(80, self._preview.height() - 12)
        self._preview.setWordWrap(False)
        display = text if len(text) <= 20 else text[:17] + "..."
        chosen = self.TEXT_SYMBOL_MIN_PT
        for fs in range(self.TEXT_SYMBOL_MAX_PT, self.TEXT_SYMBOL_MIN_PT - 1, -4):
            f = demo_font(fs, weight=QFont.Bold)
            fm = QFontMetrics(f)
            if fm.size(Qt.TextSingleLine, display).width() <= w and fm.height() <= h:
                chosen = fs
                break
        self._apply_preview_font(chosen, color="#F5F5F7")
        self._preview.setText(display)


class RecognitionResultWidget(QGroupBox):
    """Right-column challenge–response recognition summary."""

    BODY_MIN_H = 200

    def __init__(self, parent=None):
        super().__init__("", parent)
        style_demo_card(self, object_name="recognitionResultBox")
        self.setMinimumHeight(260)

        outer = QVBoxLayout(self)
        outer.setSpacing(8)
        outer.setContentsMargins(14, 14, 14, 12)
        add_card_title(outer, "Recognition result")

        self._body = QFrame()
        self._body.setObjectName("recognitionResultBody")
        self._body.setMinimumHeight(self.BODY_MIN_H)
        body_layout = QVBoxLayout(self._body)
        body_layout.setSpacing(6)
        body_layout.setContentsMargins(14, 12, 14, 12)

        self._lbl_challenge = QLabel()
        self._lbl_predicted = QLabel()
        self._lbl_confidence = QLabel()
        self._lbl_match = QLabel()
        self._lbl_decision = QLabel("Decision: Waiting")
        self._lbl_decision.setObjectName("recognitionDecision")

        for lbl in (
            self._lbl_challenge,
            self._lbl_predicted,
            self._lbl_confidence,
            self._lbl_match,
        ):
            lbl.setObjectName("recognitionLine")
            lbl.setWordWrap(True)
            lbl.setTextFormat(Qt.RichText)
            lbl.setFont(demo_font(16))
            body_layout.addWidget(lbl)

        body_layout.addSpacing(4)
        self._lbl_decision.setWordWrap(True)
        body_layout.addWidget(self._lbl_decision)
        body_layout.addStretch()
        outer.addWidget(self._body, stretch=1)

        self._decision_style_waiting()

    def apply_metrics(self, window_height: int) -> None:
        body_h = max(self.BODY_MIN_H, min(240, int(window_height * 0.22)))
        self._body.setMinimumHeight(body_h)
        self.setMinimumHeight(body_h + 72)

    def _decision_style_waiting(self) -> None:
        self._lbl_decision.setStyleSheet(
            "color: #5f9bff; font-size: 26px; font-weight: 800;"
        )

    def _decision_style_granted(self) -> None:
        self._lbl_decision.setStyleSheet(
            "color: #248A3D; font-size: 28px; font-weight: 800;"
        )

    def _decision_style_denied(self) -> None:
        self._lbl_decision.setStyleSheet(
            "color: #D70015; font-size: 28px; font-weight: 800;"
        )

    def _decision_style_unknown(self) -> None:
        self._lbl_decision.setStyleSheet(
            "color: #C93400; font-size: 26px; font-weight: 800;"
        )

    def set_waiting(self, challenge_label: str = "") -> None:
        ch = _normalize_label(challenge_label)
        self._lbl_challenge.setText(
            row_html("Challenge:", ch if ch else "—", value_px=24, label_px=17)
        )
        self._lbl_predicted.setText(row_html("Predicted:", "—", value_px=24, label_px=17))
        self._lbl_confidence.setText(row_html("Confidence:", "—", value_px=18, label_px=17))
        self._lbl_match.setText(row_html("Match:", "—", value_px=18, label_px=17))
        self._lbl_decision.setText("Decision: Waiting")
        self._decision_style_waiting()

    def set_prediction(
        self,
        challenge_label: str,
        predicted_label: str,
        confidence: Optional[float],
    ) -> None:
        ch = _normalize_label(challenge_label)
        pred = _normalize_label(predicted_label) or "—"
        self._lbl_challenge.setText(
            row_html("Challenge:", ch if ch else "—", value_px=24, label_px=17)
        )
        self._lbl_predicted.setText(row_html("Predicted:", pred, value_px=26, label_px=17))
        if confidence is None:
            self._lbl_confidence.setText(row_html("Confidence:", "—", value_px=18, label_px=17))
        else:
            self._lbl_confidence.setText(
                row_html("Confidence:", f"{confidence:.2f}", value_px=19, label_px=17)
            )

        if not ch or pred in ("—", "?", "-"):
            match_val = "—"
            match_color = "#636366"
        elif labels_match(ch, pred):
            match_val = "Yes"
            match_color = "#248A3D"
        else:
            match_val = "No"
            match_color = "#D70015"
        self._lbl_match.setText(
            f'<span style="font-size:17px; color:#636366; font-weight:600;">Match:</span> '
            f'<span style="font-size:20px; font-weight:800; color:{match_color};">'
            f"{match_val}</span>"
        )

    def set_decision(self, decision: str) -> None:
        text = _normalize_label(decision) or "Waiting"
        self._lbl_decision.setText(f"Decision: {text}")
        low = text.lower()
        if "granted" in low:
            self._decision_style_granted()
        elif "denied" in low:
            self._decision_style_denied()
        elif "unknown" in low:
            self._decision_style_unknown()
        else:
            self._decision_style_waiting()

    def clear_result(self) -> None:
        self.set_waiting("")
