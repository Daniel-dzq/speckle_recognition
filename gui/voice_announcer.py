"""
Non-blocking local text-to-speech for the live demo (no network, no audio files).
"""

from __future__ import annotations

import platform
import re
import shutil
import time
from typing import Callable, Optional

from PySide6.QtCore import QObject, QProcess


class VoiceAnnouncer(QObject):
    """Speak short phrases via the OS TTS backend without blocking the GUI thread."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        log_fn: Optional[Callable[[str], None]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._enabled = enabled
        self._log_fn = log_fn
        self._backend = self._detect_backend()
        self._last_key_time: dict[str, float] = {}
        self._process: Optional[QProcess] = None
        if self._backend is None:
            self._log("Voice backend unavailable; continuing without speech.")

    @property
    def backend_name(self) -> str:
        return self._backend or "none"

    @property
    def available(self) -> bool:
        return self._backend is not None

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def is_enabled(self) -> bool:
        return self._enabled

    def speak(
        self,
        text: str,
        *,
        key: str = "",
        cooldown_sec: float = 3.0,
        force: bool = False,
    ) -> bool:
        """Queue speech if enabled, backend exists, and cooldown allows."""
        phrase = (text or "").strip()
        if not phrase or not self._enabled or self._backend is None:
            return False

        slot_key = key or phrase
        now = time.monotonic()
        if not force and cooldown_sec > 0:
            last = self._last_key_time.get(slot_key, 0.0)
            if now - last < cooldown_sec:
                return False
        self._last_key_time[slot_key] = now

        self.stop()
        proc = QProcess(self)
        proc.finished.connect(self._on_process_finished)
        self._process = proc
        program, args = self._command_for(phrase)
        proc.start(program, args)
        return True

    def _on_process_finished(self) -> None:
        sender = self.sender()
        if sender is self._process:
            self._process = None

    def stop(self) -> None:
        if self._process is None:
            return
        if self._process.state() != QProcess.NotRunning:
            self._process.kill()
            self._process.waitForFinished(200)
        self._process = None

    def _log(self, msg: str) -> None:
        if self._log_fn is not None:
            self._log_fn(msg)

    @staticmethod
    def _detect_backend() -> Optional[str]:
        system = platform.system()
        if system == "Darwin" and shutil.which("say"):
            return "say"
        if system == "Windows":
            return "powershell"
        if shutil.which("spd-say"):
            return "spd-say"
        if shutil.which("espeak"):
            return "espeak"
        return None

    def _command_for(self, text: str) -> tuple[str, list[str]]:
        safe = text.replace('"', "'")
        backend = self._backend or ""
        if backend == "say":
            return "say", [safe]
        if backend == "spd-say":
            return "spd-say", ["-w", safe]
        if backend == "espeak":
            return "espeak", [safe]
        if backend == "powershell":
            escaped = re.sub(r"['`$]", "", safe)
            script = (
                "Add-Type -AssemblyName System.Speech; "
                f"(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
                f".Speak('{escaped}');"
            )
            return "powershell", ["-NoProfile", "-Command", script]
        return "say", [safe]
