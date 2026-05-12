"""
RobotCanvas — a cute custom QWidget painted with QPainter.

Public API:
    set_expression(expr)          : 'happy' | 'sad' | 'thinking' | 'scanning' | 'angry'
    set_state_color(color)        : QColor (body tint)
    set_pose_offset(dx, dy, rot)  : driven by external QPropertyAnimation
    set_sweat_alpha(a)            : 0.0–1.0
    set_aura_alpha(a)             : 0.0–1.0

Qt Properties (for QPropertyAnimation):
    dx, dy, rot, sweat
"""
from __future__ import annotations
import math
import random

from PySide6.QtCore    import Qt, QRectF, QPointF, QTimer, Property
from PySide6.QtGui     import (QPainter, QColor, QPen, QBrush,
                                QPainterPath, QRadialGradient)
from PySide6.QtWidgets import QWidget


class RobotCanvas(QWidget):
    EXPRS = ("happy", "sad", "thinking", "scanning", "angry")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(260, 260)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self._expr        = "happy"
        self._body_color  = QColor("#5f9bff")
        self._eye_color   = QColor("#00e5ff")
        self._dx, self._dy, self._rot = 0.0, 0.0, 0.0
        self._blink_phase = 1.0
        self._sweat_alpha = 0.0
        self._scan_phase  = 0.0
        self._aura_alpha  = 0.0
        self._blink_remaining_ms = 0

        self._t_blink = QTimer(self)
        self._t_blink.timeout.connect(self._tick_blink)
        self._t_blink.start(60)

        self._t_scan = QTimer(self)
        self._t_scan.timeout.connect(self._tick_scan)
        self._t_scan.start(33)

    # ── Public setters ────────────────────────────────────────────────

    def set_expression(self, expr: str):
        if expr in self.EXPRS and expr != self._expr:
            self._expr = expr
            self.update()

    def set_state_color(self, color: QColor):
        self._body_color = QColor(color)
        self.update()

    def set_pose_offset(self, dx: float, dy: float, rot_deg: float):
        self._dx, self._dy, self._rot = dx, dy, rot_deg
        self.update()

    def set_sweat_alpha(self, a: float):
        self._sweat_alpha = max(0.0, min(1.0, a))
        self.update()

    def set_aura_alpha(self, a: float):
        self._aura_alpha = max(0.0, min(1.0, a))
        self.update()

    # ── Qt Properties (for QPropertyAnimation) ───────────────────────

    def _get_dx(self): return self._dx
    def _set_dx(self, v): self.set_pose_offset(v, self._dy, self._rot)
    dx = Property(float, _get_dx, _set_dx)

    def _get_dy(self): return self._dy
    def _set_dy(self, v): self.set_pose_offset(self._dx, v, self._rot)
    dy = Property(float, _get_dy, _set_dy)

    def _get_rot(self): return self._rot
    def _set_rot(self, v): self.set_pose_offset(self._dx, self._dy, v)
    rot = Property(float, _get_rot, _set_rot)

    def _get_sweat(self): return self._sweat_alpha
    def _set_sweat(self, v): self.set_sweat_alpha(v)
    sweat = Property(float, _get_sweat, _set_sweat)

    # ── Internal tickers ──────────────────────────────────────────────

    def _tick_blink(self):
        if self._blink_remaining_ms > 0:
            self._blink_remaining_ms -= 60
            self._blink_phase = 0.0 if self._blink_remaining_ms > 60 else 1.0
            self.update()
        else:
            if random.random() < 0.018:
                self._blink_remaining_ms = 220
                self._blink_phase = 0.0
                self.update()

    def _tick_scan(self):
        if self._expr in ("thinking", "scanning"):
            self._scan_phase = (self._scan_phase + 0.08) % (2 * math.pi)
            self.update()

    # ── paintEvent ───────────────────────────────────────────────────

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        w, h = self.width(), self.height()
        cx = w / 2 + self._dx
        cy = h / 2 + self._dy + 8

        p.translate(cx, cy)
        p.rotate(self._rot)
        s = min(w, h) / 320.0

        self._draw_aura(p, s)
        self._draw_shadow(p, s)
        self._draw_body(p, s)
        self._draw_face_panel(p, s)
        self._draw_eyes(p, s)
        self._draw_mouth(p, s)
        self._draw_antenna(p, s)
        self._draw_arms(p, s)
        self._draw_chest_gem(p, s)
        if self._sweat_alpha > 0:
            self._draw_sweat(p, s)

    # ── Drawing primitives ────────────────────────────────────────────

    def _draw_aura(self, p, s):
        if self._aura_alpha <= 0:
            return
        rad = 130 * s
        grad = QRadialGradient(0, 0, rad)
        c = QColor(self._body_color)
        c.setAlphaF(0.55 * self._aura_alpha)
        grad.setColorAt(0.0, c)
        c2 = QColor(c)
        c2.setAlphaF(0.0)
        grad.setColorAt(1.0, c2)
        p.setBrush(QBrush(grad))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(0, 0), rad, rad)

    def _draw_shadow(self, p, s):
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 60))
        p.drawEllipse(QPointF(0, 90 * s), 70 * s, 10 * s)

    def _draw_body(self, p, s):
        body = QPainterPath()
        body.addRoundedRect(QRectF(-72*s, -55*s, 144*s, 130*s), 36*s, 36*s)
        p.setPen(QPen(QColor("#bfc7d4"), 2.5*s))
        p.setBrush(QColor("#f4f6fa"))
        p.drawPath(body)

        stripe = QPainterPath()
        stripe.addRoundedRect(QRectF(-66*s, 28*s, 132*s, 38*s), 18*s, 18*s)
        p.setPen(Qt.NoPen)
        c = QColor(self._body_color)
        c.setAlphaF(0.85)
        p.setBrush(c)
        p.drawPath(stripe)

    def _draw_face_panel(self, p, s):
        face = QPainterPath()
        face.addRoundedRect(QRectF(-54*s, -38*s, 108*s, 56*s), 14*s, 14*s)
        p.setPen(QPen(QColor("#0a1228"), 2*s))
        p.setBrush(QColor("#0d1b35"))
        p.drawPath(face)

    def _draw_eyes(self, p, s):
        eye_color = QColor(self._eye_color)
        if self._expr == "sad":      eye_color = QColor("#ff5566")
        if self._expr == "angry":    eye_color = QColor("#ffaa00")
        if self._expr == "thinking": eye_color = QColor("#ffd54f")

        pen = QPen(eye_color, 5.5*s, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)

        L = QPointF(-22*s, -10*s)
        R = QPointF( 22*s, -10*s)

        if self._blink_phase < 0.5:
            for c in (L, R):
                p.drawLine(c.x()-9*s, c.y(), c.x()+9*s, c.y())
            return

        if self._expr == "happy":
            for c in (L, R):
                p.drawArc(QRectF(c.x()-12*s, c.y()-3*s, 24*s, 16*s), 0*16, 180*16)

        elif self._expr == "sad":
            for c in (L, R):
                p.drawArc(QRectF(c.x()-12*s, c.y()-8*s, 24*s, 16*s), 180*16, 180*16)

        elif self._expr == "angry":
            p.drawLine(L.x()-10*s, L.y()-7*s, L.x()+10*s, L.y()+7*s)
            p.drawLine(R.x()-10*s, R.y()+7*s, R.x()+10*s, R.y()-7*s)

        elif self._expr == "thinking":
            p.setBrush(eye_color)
            p.drawEllipse(L, 5*s, 5*s)
            p.setBrush(Qt.NoBrush)
            p.drawLine(R.x()-9*s, R.y(), R.x()+9*s, R.y())

        elif self._expr == "scanning":
            for c in (L, R):
                p.drawEllipse(c, 8*s, 8*s)
                sx = c.x() + 8*s * math.cos(self._scan_phase)
                sy = c.y() + 8*s * math.sin(self._scan_phase)
                p.setBrush(QColor("#ffffff"))
                p.drawEllipse(QPointF(sx, sy), 2.5*s, 2.5*s)
                p.setBrush(Qt.NoBrush)

    def _draw_mouth(self, p, s):
        pen = QPen(QColor(self._eye_color), 3.5*s, Qt.SolidLine, Qt.RoundCap)
        if self._expr == "sad":   pen.setColor(QColor("#ff5566"))
        if self._expr == "angry": pen.setColor(QColor("#ffaa00"))
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)

        if self._expr == "happy":
            p.drawArc(QRectF(-10*s, 4*s, 20*s, 14*s), 200*16, 140*16)
        elif self._expr == "sad":
            p.drawArc(QRectF(-10*s, 8*s, 20*s, 12*s), 20*16, 140*16)
        else:
            p.drawLine(-8*s, 10*s, 8*s, 10*s)

    def _draw_antenna(self, p, s):
        p.setPen(QPen(QColor("#bfc7d4"), 3*s, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(0, -55*s, 0, -78*s)
        p.setPen(Qt.NoPen)
        glow = QColor(self._body_color)
        glow.setAlphaF(0.6)
        p.setBrush(glow)
        p.drawEllipse(QPointF(0, -82*s), 11*s, 11*s)
        p.setBrush(QColor(self._eye_color))
        p.drawEllipse(QPointF(0, -82*s), 6*s, 6*s)

    def _draw_arms(self, p, s):
        p.setPen(QPen(QColor("#bfc7d4"), 4.5*s, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(-72*s, 5*s, -92*s, 22*s)
        p.setBrush(QColor("#f4f6fa"))
        p.setPen(QPen(QColor("#bfc7d4"), 2*s))
        p.drawEllipse(QPointF(-92*s, 22*s), 9*s, 9*s)

        p.setPen(QPen(QColor("#bfc7d4"), 4.5*s, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(72*s, 5*s, 92*s, 22*s)
        p.setBrush(QColor("#f4f6fa"))
        p.setPen(QPen(QColor("#bfc7d4"), 2*s))
        p.drawEllipse(QPointF(92*s, 22*s), 9*s, 9*s)

    def _draw_chest_gem(self, p, s):
        p.setPen(QPen(QColor("#ffffff"), 2*s))
        p.setBrush(QColor(255, 255, 255, 60))
        p.drawRoundedRect(QRectF(-14*s, 40*s, 28*s, 16*s), 4*s, 4*s)
        p.setBrush(QColor(255, 255, 255, 90))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(14*s, 44*s, 4*s, 8*s), 1*s, 1*s)

    def _draw_sweat(self, p, s):
        c = QColor("#5cb8ff")
        c.setAlphaF(self._sweat_alpha)
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        path = QPainterPath()
        path.moveTo(40*s, -38*s)
        path.cubicTo(48*s, -22*s, 48*s, -10*s, 40*s, -8*s)
        path.cubicTo(32*s, -10*s, 32*s, -22*s, 40*s, -38*s)
        p.drawPath(path)
