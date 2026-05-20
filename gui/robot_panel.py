"""
RobotPanel — replaces the old "Recognition Output" prediction box.

Public methods (call from MainWindow):
    on_idle()                    — neutral standby state
    on_reading(fiber_name)       — scanning eyes + reading blink
    on_prediction(result: dict)  — drives expression + animation from inference output
    on_unauthorized(reason)      — explicit failure trigger
    attach_banner_callback(fn)   — inject banner overlay function from MainWindow
"""
import os

from PySide6.QtCore    import Qt, QTimer, QPropertyAnimation
from PySide6.QtGui     import QColor, QFont, QFontMetrics
from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QSizePolicy

from gui.challenge_widgets import labels_match
from gui.demo_presentation import demo_font
from gui.robot_canvas import RobotCanvas
from gui.effects      import (make_glow, pulse_glow,
                               jump, joyful_spin, shiver,
                               lean_back, step_forward)


COLOR_NEUTRAL = "#5f9bff"
COLOR_READING = "#ffd54f"
COLOR_OK      = "#3ddc84"
COLOR_FAIL    = "#ff5566"

LOW_CONFIDENCE_THRESHOLD = 0.85


def _manual_screenshot_mode_env() -> bool:
    v = os.environ.get("SPECKLE_MANUAL_SCREENSHOT_MODE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


class RobotPanel(QFrame):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("robotPanel")
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        v = QVBoxLayout(self)
        v.setContentsMargins(14, 14, 14, 14)
        v.setSpacing(10)

        # ── Robot canvas ─────────────────────────────────────
        self.robot = RobotCanvas(self)
        self.robot.setMinimumHeight(220)
        v.addWidget(self.robot, stretch=3, alignment=Qt.AlignHCenter)

        # ── Status banner ─────────────────────────────────────
        self.lbl_status = QLabel("STANDBY", self)
        self.lbl_status.setObjectName("robotStatus")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        v.addWidget(self.lbl_status)

        # ── Sub-status (live action text) ─────────────────────
        self.lbl_action = QLabel("Awaiting fiber\u2026", self)
        self.lbl_action.setObjectName("robotAction")
        self.lbl_action.setAlignment(Qt.AlignCenter)
        self.lbl_action.setWordWrap(True)
        self.lbl_action.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        v.addWidget(self.lbl_action)

        # ── Confidence ────────────────────────────────────────
        self.lbl_conf = QLabel("Confidence \u2014", self)
        self.lbl_conf.setObjectName("robotConf")
        self.lbl_conf.setAlignment(Qt.AlignCenter)
        self.lbl_conf.setWordWrap(True)
        self.lbl_conf.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        v.addWidget(self.lbl_conf)

        # ── Top-K candidates ──────────────────────────────────
        self.lbl_topk = QLabel("", self)
        self.lbl_topk.setObjectName("robotTopK")
        self.lbl_topk.setAlignment(Qt.AlignCenter)
        self.lbl_topk.setWordWrap(True)
        self.lbl_topk.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        v.addWidget(self.lbl_topk)

        v.addStretch()

        # ── Glow effect on panel border ───────────────────────
        self._glow      = make_glow(self, COLOR_NEUTRAL, radius=24)
        self._glow_anim = pulse_glow(self._glow, low=12, high=28, period_ms=1800)

        # ── Reading-text blink timer ──────────────────────────
        self._read_blink_timer = QTimer(self)
        self._read_blink_timer.timeout.connect(self._blink_reading_text)
        self._read_blink_phase = True
        self._reading_fiber    = "fiber"

        # ── Animation references (keep alive) ─────────────────
        self._anims: list = []
        self._challenge_label = ""

        self.on_idle()

    def apply_metrics(self, window_height: int) -> None:
        robot_h = max(200, min(280, int(window_height * 0.24)))
        self.robot.setMinimumHeight(robot_h)
        self._fit_status_heading_font()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_status_heading_font()

    def _fit_status_heading_font(self):
        """Shrink the status headline so Latin text (e.g. ACCESS DENIED) stays on one line."""
        text = self.lbl_status.text()
        if not text:
            return
        w_avail = self.lbl_status.width()
        if w_avail < 48:
            w_avail = max(80, self.width() - 32)
        limit = max(48, w_avail - 10)
        base = self.font()
        for fs in range(20, 12, -1):
            f = demo_font(fs, weight=QFont.Black)
            fm = QFontMetrics(f)
            if fm.size(Qt.TextSingleLine, text).width() <= limit:
                self.lbl_status.setFont(f)
                return
        f = QFont(base)
        f.setPixelSize(10)
        f.setWeight(QFont.Black)
        self.lbl_status.setFont(f)

    # ═══════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════

    def on_idle(self):
        self._set_glow(COLOR_NEUTRAL)
        self.robot.set_state_color(QColor(COLOR_NEUTRAL))
        self.robot.set_expression("happy")
        self.robot.set_aura_alpha(0.0)
        self._read_blink_timer.stop()
        self.lbl_status.setText("STANDBY")
        self.lbl_action.setText("Awaiting fiber\u2026")
        self.lbl_conf.setText("Confidence \u2014")
        self.lbl_topk.setText("")
        self._fit_status_heading_font()

    def on_reading(self, fiber_name: str = ""):
        self._set_glow(COLOR_READING)
        self.robot.set_state_color(QColor(COLOR_READING))
        self.robot.set_expression("scanning")
        self.robot.set_aura_alpha(0.5)
        self.lbl_status.setText("READING PUF")
        self._fit_status_heading_font()
        self._reading_fiber    = fiber_name or "fiber"
        self._read_blink_phase = True
        self._read_blink_timer.start(500)
        self._blink_reading_text()

    def manual_screenshot_seed_reading_display(self, fiber_label: str = "Fiber1") -> None:
        """Manual screenshot only: keep READING PUF with static Top-K/conf on real widgets."""
        if not _manual_screenshot_mode_env():
            return
        self.on_reading(fiber_label)
        self._read_blink_timer.stop()
        self.lbl_conf.setText("Confidence  72.3%")
        self.lbl_topk.setText("A 72%  B 18%  C 6%")
        self.lbl_status.setText("READING PUF")
        self._fit_status_heading_font()

    def on_prediction(self, result: dict):
        """Receive InferenceWorker.prediction_ready dict and drive all visuals."""
        # ── Manual screenshot only (documentation; deterministic UI when env is set):
        # SPECKLE_FORCE_AUTH_STATE overrides this panel's visuals for one shot;
        # it does NOT change InferenceWorker or model weights. Unset the env
        # for normal experiments.
        _force = os.environ.get("SPECKLE_FORCE_AUTH_STATE", "").strip().lower()
        if _force and _manual_screenshot_mode_env():
            self._read_blink_timer.stop()
            if _force in ("granted", "grant", "ok", "pass"):
                self.lbl_conf.setText("Confidence  92.0%")
                self.lbl_topk.setText("A 92%  B 3%  C 2%")
                self._do_authorize("A", 0.92)
                return
            if _force in ("denied", "deny", "denied_low", "deny_low", "fail_low"):
                self.lbl_conf.setText("Confidence  40.0%")
                self.lbl_topk.setText("A 40%  B 35%  C 15%")
                self._do_deny(reason="Low confidence (40%)")
                return
            if _force in ("denied_class", "deny_class", "fail_class"):
                self.lbl_conf.setText("Confidence  90.0%")
                self.lbl_topk.setText("B 90%  A 5%  C 3%")
                self._do_deny(reason="Expected class 'A', got 'B'")
                return

        conf      = float(result.get("confidence", 0.0))
        topk      = result.get("topk",       [])
        smoothed  = result.get("smoothed",   "?")
        challenge = self._challenge_label
        pred = str(smoothed).strip() if smoothed else "?"
        if challenge:
            authorized = (conf >= LOW_CONFIDENCE_THRESHOLD) and labels_match(challenge, pred)
        else:
            authorized = False

        self._read_blink_timer.stop()
        self.lbl_conf.setText(f"Confidence  {conf*100:.1f}%")
        self.lbl_topk.setText(
            "  ".join(f"{cls} {p*100:.0f}%" for cls, p in topk[:3])
        )

        if authorized:
            self._do_authorize(smoothed, conf)
        else:
            if not challenge:
                reason = "No SLM challenge sent yet"
            elif conf < LOW_CONFIDENCE_THRESHOLD:
                reason = f"Low confidence ({conf*100:.0f}%)"
            elif not labels_match(challenge, pred):
                reason = f"Expected challenge '{challenge}', got '{pred}'"
            else:
                reason = "Access denied"
            self._do_deny(reason=reason)

    def on_unauthorized(self, reason: str = "Unrecognized fiber"):
        self._read_blink_timer.stop()
        self._do_deny(reason)

    def attach_banner_callback(self, fn):
        """MainWindow injects a callback to show the overlay banner on the CCD feed."""
        self._banner_callback = fn

    def set_challenge_label(self, label: str) -> None:
        """Current SLM challenge label from MainWindow (any class name)."""
        self._challenge_label = (label or "").strip()

    def set_challenge_letter(self, letter: str) -> None:
        """Alias for set_challenge_label (backward compatibility)."""
        self.set_challenge_label(letter)

    def apply_decision(
        self,
        decision: str,
        *,
        predicted: str = "",
        reason: str = "",
        result: dict | None = None,
    ) -> None:
        """Drive robot visuals from MainWindow decision (granted / denied / waiting)."""
        self._read_blink_timer.stop()
        if result:
            conf = float(result.get("confidence", 0.0))
            topk = result.get("topk", [])
            self.lbl_conf.setText(f"Confidence  {conf * 100:.1f}%")
            self.lbl_topk.setText(
                "  ".join(f"{cls} {p * 100:.0f}%" for cls, p in topk[:3])
            )
        low = (decision or "").strip().lower()
        if "granted" in low:
            self._do_authorize(predicted or "?", 1.0)
            return
        if "denied" in low or "unknown" in low:
            self._do_deny(reason or decision or "Access denied")
            return
        self.on_idle()

    # ═══════════════════════════════════════════════════════
    # Internal — state transitions + animations
    # ═══════════════════════════════════════════════════════

    def _do_authorize(self, letter: str, conf: float):
        self._set_glow(COLOR_OK)
        self.robot.set_state_color(QColor(COLOR_OK))
        self.robot.set_expression("happy")
        self.robot.set_aura_alpha(1.0)
        self.lbl_status.setText("ACCESS GRANTED")
        self._fit_status_heading_font()
        self.lbl_action.setText(f"Authorized \u00b7 Decoded '{letter}'")

        # Jump + spin, then step forward
        self._anims = [jump(self.robot, height=24), joyful_spin(self.robot)]
        QTimer.singleShot(620, lambda: self._anims.append(step_forward(self.robot)))

        # Fire banner overlay over camera feed
        if hasattr(self, "_banner_callback"):
            self._banner_callback("ACCESS GRANTED", COLOR_OK)

        # Fade aura down slightly after 1.8 s
        QTimer.singleShot(1800, lambda: self.robot.set_aura_alpha(0.4))

    def _do_deny(self, reason: str):
        self._set_glow(COLOR_FAIL)
        self.robot.set_state_color(QColor(COLOR_FAIL))
        self.robot.set_expression("sad")
        self.robot.set_aura_alpha(0.7)
        self.lbl_status.setText("ACCESS DENIED")
        self._fit_status_heading_font()
        self.lbl_action.setText(reason)

        # Shiver, then lean back, then sweatdrop fade-in
        self._anims = [shiver(self.robot, magnitude=8)]
        QTimer.singleShot(420, lambda: self._anims.append(lean_back(self.robot)))

        sweat_anim = QPropertyAnimation(self.robot, b"sweat", self)
        sweat_anim.setDuration(500)
        sweat_anim.setStartValue(0.0)
        sweat_anim.setEndValue(1.0)
        sweat_anim.start()
        self._anims.append(sweat_anim)

        if hasattr(self, "_banner_callback"):
            self._banner_callback("ACCESS DENIED", COLOR_FAIL)

    def _set_glow(self, color: str):
        self._glow.setColor(QColor(color))

    def _blink_reading_text(self):
        if self._read_blink_phase:
            self.lbl_action.setText(f"Reading PUF from {self._reading_fiber}\u2026")
        else:
            self.lbl_action.setText("")
        self._read_blink_phase = not self._read_blink_phase
