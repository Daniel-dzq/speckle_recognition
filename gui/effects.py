"""
Reusable QPropertyAnimation helpers for the cute robot UI.
"""
from PySide6.QtCore    import QPropertyAnimation, QEasingCurve, QTimer
from PySide6.QtGui     import QColor
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QGraphicsOpacityEffect


def make_glow(widget, color="#00e5ff", radius=28):
    """Attach a QGraphicsDropShadowEffect glow to a widget and return the effect."""
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(radius)
    eff.setOffset(0, 0)
    eff.setColor(QColor(color))
    widget.setGraphicsEffect(eff)
    return eff


def pulse_glow(effect, low=12, high=28, period_ms=1800):
    """Continuously pulse a glow effect's blur radius."""
    anim = QPropertyAnimation(effect, b"blurRadius")
    anim.setDuration(period_ms)
    anim.setStartValue(low)
    anim.setKeyValueAt(0.5, high)
    anim.setEndValue(low)
    anim.setLoopCount(-1)
    anim.setEasingCurve(QEasingCurve.InOutSine)
    anim.start()
    return anim


def jump(robot, height=24, duration_ms=520):
    """Make the robot canvas jump up and land."""
    a = QPropertyAnimation(robot, b"dy")
    a.setDuration(duration_ms)
    a.setKeyValueAt(0.0, 0.0)
    a.setKeyValueAt(0.5, -height)
    a.setKeyValueAt(1.0, 0.0)
    a.setEasingCurve(QEasingCurve.OutQuad)
    a.start()
    return a


def joyful_spin(robot, duration_ms=620):
    """Make the robot do a small joyful tilt."""
    a = QPropertyAnimation(robot, b"rot")
    a.setDuration(duration_ms)
    a.setStartValue(0)
    a.setKeyValueAt(0.5, 14)
    a.setEndValue(0)
    a.setEasingCurve(QEasingCurve.OutBack)
    a.start()
    return a


def shiver(robot, magnitude=8, duration_ms=420):
    """Make the robot shiver left-right."""
    a = QPropertyAnimation(robot, b"dx")
    a.setDuration(duration_ms)
    vals = [0, -magnitude, magnitude, -magnitude*0.7,
            magnitude*0.7, -magnitude*0.4, magnitude*0.4, 0]
    for i, v in enumerate(vals):
        a.setKeyValueAt(i / (len(vals) - 1), v)
    a.start()
    return a


def lean_back(robot, distance=28, duration_ms=560):
    """Make the robot lean back (slide right)."""
    a = QPropertyAnimation(robot, b"dx")
    a.setDuration(duration_ms)
    a.setStartValue(0)
    a.setEndValue(distance)
    a.setEasingCurve(QEasingCurve.OutBack)
    a.start()
    return a


def step_forward(robot, distance=24, duration_ms=520):
    """Make the robot step forward (slide left)."""
    a = QPropertyAnimation(robot, b"dx")
    a.setDuration(duration_ms)
    a.setStartValue(0)
    a.setEndValue(-distance)
    a.setEasingCurve(QEasingCurve.OutBack)
    a.start()
    return a


def apply_premium_shadow(widget):
    """Attach a soft Apple-style drop shadow to a card widget.

    Uses very low opacity (alpha=18/255) so the shadow reads as depth
    without looking harsh — consistent with Apple HIG card elevation.
    Returns the effect so callers can keep a reference if needed.
    """
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(24)
    shadow.setXOffset(0)
    shadow.setYOffset(8)
    shadow.setColor(QColor(0, 0, 0, 18))
    widget.setGraphicsEffect(shadow)
    return shadow


def fade_in_scale(label, start_scale=0.7, duration_ms=380):
    """Fade a QLabel in. `start_scale` is kept for call-site compatibility but ignored.

    Shrinking QLabel geometry (old behaviour) clips styled text like \"ACCESS DENIED\".
    """
    _ = start_scale
    label.show()
    eff = label.graphicsEffect()
    if not isinstance(eff, QGraphicsOpacityEffect):
        eff = QGraphicsOpacityEffect(label)
        label.setGraphicsEffect(eff)
    eff.setOpacity(0.0)
    # Parent the animation to `label` so Shiboken does not GC it before
    # the fade runs (unparented animations often never reach opacity 1.0).
    a = QPropertyAnimation(eff, b"opacity", label)
    a.setDuration(duration_ms)
    a.setStartValue(0.0)
    a.setEndValue(1.0)
    a.setEasingCurve(QEasingCurve.OutCubic)
    a.start()
    return a
