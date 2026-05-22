"""
GUI-only prediction smoothing and stable decision snapshots for the live demo.

Does not alter model inference; buffers recent predictions for display stability.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Optional

from gui.challenge_widgets import labels_match, normalize_label


@dataclass(frozen=True)
class DisplaySnapshot:
    """Stable values shown in the recognition card, robot panel, and overlay."""

    challenge: str
    predicted: str
    confidence: Optional[float]
    match: Optional[bool]
    decision: str
    reason: str
    emit_banner: bool = False
    hide_banner: bool = False


@dataclass
class _FeedResult:
    snapshot: DisplaySnapshot
    refresh_ui: bool


class PredictionDisplaySmoother:
    """
    Smooth live inference for the UI.

    Updates the visible prediction when the same label appears in at least
    ``vote_min`` of the last ``vote_window`` samples, or confidence is very high
    and matches the challenge. Holds a committed decision for ``hold_sec`` before
    switching to a different decision.
    """

    BUFFER_SIZE = 7
    VOTE_WINDOW = 5
    VOTE_MIN_COUNT = 3
    HIGH_CONF_FAST_PATH = 0.85

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.60,
        hold_sec: float = 2.0,
        banner_hold_sec: float = 2.0,
        granted_release_hold_sec: float = 2.0,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.hold_sec = hold_sec
        self.banner_hold_sec = banner_hold_sec
        self.granted_release_hold_sec = granted_release_hold_sec
        self._buffer: Deque[tuple[str, Optional[float], float]] = deque(
            maxlen=self.BUFFER_SIZE
        )
        self._displayed: Optional[DisplaySnapshot] = None
        self._pending: Optional[DisplaySnapshot] = None
        self._pending_since: float = 0.0
        self._banner_granted_until: float = 0.0

    def reset(self) -> None:
        self._buffer.clear()
        self._displayed = None
        self._pending = None
        self._pending_since = 0.0
        self._banner_granted_until = 0.0

    def feed(
        self,
        *,
        challenge: str,
        raw_label: str,
        raw_confidence: Optional[float],
    ) -> _FeedResult:
        now = time.monotonic()
        label = normalize_label(raw_label) or "?"
        self._buffer.append((label, raw_confidence, now))

        stable_label, stable_conf = self._stable_label(challenge)
        if stable_label is None:
            snap = self._waiting_snapshot(challenge)
            snap = self._apply_banner_flags(snap, now)
            if self._displayed and (now - self._pending_since) < self.hold_sec:
                return _FeedResult(self._displayed, refresh_ui=False)
            return self._commit_if_changed(snap, now, force=True)

        candidate = self._snapshot_from_stable(challenge, stable_label, stable_conf)
        candidate = self._apply_banner_flags(candidate, now)
        return self._commit_if_changed(candidate, now)

    def _stable_label(
        self, challenge: str
    ) -> tuple[Optional[str], Optional[float]]:
        if not self._buffer:
            return None, None
        recent = list(self._buffer)[-self.VOTE_WINDOW :]
        labels = [
            normalize_label(lbl)
            for lbl, _, _ in recent
            if lbl and lbl not in ("?", "—", "-")
        ]
        if not labels:
            return None, None

        counts = Counter(labels)
        top_label, top_count = counts.most_common(1)[0]
        if top_count >= self.VOTE_MIN_COUNT:
            confs = [c for lbl, c, _ in recent if normalize_label(lbl) == top_label and c is not None]
            avg_conf = sum(confs) / len(confs) if confs else None
            return top_label, avg_conf

        last_label, last_conf, _ = recent[-1]
        norm = normalize_label(last_label)
        if (
            norm
            and last_conf is not None
            and last_conf >= self.HIGH_CONF_FAST_PATH
            and challenge
            and labels_match(challenge, norm)
        ):
            return norm, last_conf

        return None, None

    def _snapshot_from_stable(
        self,
        challenge: str,
        predicted: str,
        confidence: Optional[float],
    ) -> DisplaySnapshot:
        ch = normalize_label(challenge)
        pred = normalize_label(predicted) or "—"
        if not ch or pred in ("—", "?", "-"):
            return self._waiting_snapshot(challenge)

        conf_f = float(confidence) if confidence is not None else None
        label_match = labels_match(ch, pred)

        if not label_match:
            return DisplaySnapshot(
                challenge=ch,
                predicted=pred,
                confidence=conf_f,
                match=False,
                decision="ACCESS DENIED",
                reason=f"Expected '{ch}', got '{pred}'",
            )

        if conf_f is None:
            return DisplaySnapshot(
                challenge=ch,
                predicted=pred,
                confidence=None,
                match=True,
                decision="WAITING",
                reason="Confidence unavailable",
            )

        if conf_f < self.confidence_threshold:
            return DisplaySnapshot(
                challenge=ch,
                predicted=pred,
                confidence=conf_f,
                match=True,
                decision="LOW CONFIDENCE",
                reason=f"Low confidence ({conf_f * 100:.0f}%)",
            )

        return DisplaySnapshot(
            challenge=ch,
            predicted=pred,
            confidence=conf_f,
            match=True,
            decision="ACCESS GRANTED",
            reason=f"Match on challenge '{ch}'",
        )

    def _waiting_snapshot(self, challenge: str) -> DisplaySnapshot:
        ch = normalize_label(challenge)
        return DisplaySnapshot(
            challenge=ch,
            predicted="—",
            confidence=None,
            match=None,
            decision="WAITING",
            reason="Awaiting stable prediction",
        )

    def _apply_banner_flags(self, snap: DisplaySnapshot, now: float) -> DisplaySnapshot:
        emit = False
        hide = False
        if snap.decision == "ACCESS GRANTED":
            if self._displayed is None or self._displayed.decision != "ACCESS GRANTED":
                emit = True
            self._banner_granted_until = now + self.banner_hold_sec
        elif now >= self._banner_granted_until:
            hide = self._displayed is not None and self._displayed.decision == "ACCESS GRANTED"
        return DisplaySnapshot(
            challenge=snap.challenge,
            predicted=snap.predicted,
            confidence=snap.confidence,
            match=snap.match,
            decision=snap.decision,
            reason=snap.reason,
            emit_banner=emit,
            hide_banner=hide,
        )

    def _commit_if_changed(
        self,
        candidate: DisplaySnapshot,
        now: float,
        *,
        force: bool = False,
    ) -> _FeedResult:
        if self._displayed is None:
            self._displayed = candidate
            self._pending = None
            return _FeedResult(candidate, refresh_ui=True)

        same = (
            candidate.decision == self._displayed.decision
            and candidate.predicted == self._displayed.predicted
            and candidate.match == self._displayed.match
        )
        if same:
            self._displayed = DisplaySnapshot(
                challenge=candidate.challenge,
                predicted=candidate.predicted,
                confidence=candidate.confidence,
                match=candidate.match,
                decision=candidate.decision,
                reason=candidate.reason,
                emit_banner=candidate.emit_banner,
                hide_banner=candidate.hide_banner,
            )
            if candidate.emit_banner or candidate.hide_banner:
                return _FeedResult(self._displayed, refresh_ui=True)
            return _FeedResult(self._displayed, refresh_ui=True)

        if force:
            self._displayed = candidate
            self._pending = None
            return _FeedResult(candidate, refresh_ui=True)

        if self._pending is None or self._pending.decision != candidate.decision:
            self._pending = candidate
            self._pending_since = now
            return _FeedResult(self._displayed, refresh_ui=False)

        required_hold = self.hold_sec
        if (
            self._displayed is not None
            and self._displayed.decision == "ACCESS GRANTED"
            and candidate.decision != "ACCESS GRANTED"
        ):
            required_hold = max(self.hold_sec, self.granted_release_hold_sec)

        if now - self._pending_since >= required_hold:
            self._displayed = candidate
            self._pending = None
            return _FeedResult(candidate, refresh_ui=True)

        return _FeedResult(self._displayed, refresh_ui=False)
