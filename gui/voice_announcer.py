"""
Non-blocking local text-to-speech for the live demo (no network, no audio files).
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import time
from typing import Callable, List, Optional

from PySide6.QtCore import QObject, QProcess, QTimer


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
        self._backend, self._program_path = self._detect_backend()
        self._last_key_time: dict[str, float] = {}
        self._qprocesses: List[QProcess] = []
        self._popen_procs: List[subprocess.Popen] = []
        self._prune_timer = QTimer(self)
        self._prune_timer.setInterval(500)
        self._prune_timer.timeout.connect(self._prune_finished)
        self._prune_timer.start()
        if self._backend is None:
            self._log("Voice backend unavailable; continuing without speech.")
            self._log(
                "If speech is expected, check system volume and output device in "
                "System Settings."
            )

    @property
    def backend_name(self) -> str:
        return self._backend or "none"

    @property
    def available(self) -> bool:
        return self._backend is not None and self._program_path is not None

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
        ignore_enabled: bool = False,
    ) -> bool:
        """Queue speech if enabled, backend exists, and cooldown allows."""
        phrase = (text or "").strip()
        if not phrase:
            self._log("Voice speak skipped: empty text.")
            return False
        if not ignore_enabled and not self._enabled:
            self._log("Voice speak skipped: voice announcement disabled.")
            return False
        if self._backend is None or self._program_path is None:
            self._log("Voice speak skipped: backend unavailable.")
            return False

        slot_key = key or phrase
        now = time.monotonic()
        if not force and cooldown_sec > 0:
            last = self._last_key_time.get(slot_key, 0.0)
            if now - last < cooldown_sec:
                return False
        self._last_key_time[slot_key] = now

        self._stop_active()
        started = self._start_speech(phrase)
        if started:
            self._log(f"Voice process started: {self._backend}")
        else:
            self._log("Voice speak failed: process did not start.")
        return started

    def stop(self) -> None:
        self._stop_active()

    def _stop_active(self) -> None:
        for proc in list(self._qprocesses):
            if proc.state() != QProcess.NotRunning:
                proc.kill()
                proc.waitForFinished(300)
        self._qprocesses.clear()
        for popen in list(self._popen_procs):
            if popen.poll() is None:
                try:
                    popen.terminate()
                except OSError:
                    pass
        self._popen_procs.clear()

    def _start_speech(self, phrase: str) -> bool:
        if self._backend == "say_popen":
            return self._start_say_popen(phrase)
        if self._backend == "say":
            return self._start_say_qprocess(phrase)
        if self._backend == "powershell":
            return self._start_powershell(phrase)
        if self._backend in ("spd-say", "espeak"):
            return self._start_simple_qprocess(phrase)
        return False

    def _start_say_popen(self, phrase: str) -> bool:
        path = self._program_path
        if not path:
            return False
        try:
            proc = subprocess.Popen(
                [path, phrase],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            self._log(f"Voice process error: {exc}")
            return False
        self._popen_procs.append(proc)
        return True

    def _start_say_qprocess(self, phrase: str) -> bool:
        path = self._program_path or "say"
        proc = QProcess(self)
        proc.setProgram(path)
        proc.setArguments([phrase])
        proc.finished.connect(lambda *_a, p=proc: self._on_qprocess_finished(p))
        proc.errorOccurred.connect(
            lambda _err, p=proc: self._on_qprocess_error(p)
        )
        proc.start()
        if not proc.waitForStarted(2000):
            self._log(
                f"Voice process error: QProcess failed to start {path!r} "
                f"({proc.errorString()})"
            )
            proc.deleteLater()
            return self._start_say_popen(phrase)
        self._qprocesses.append(proc)
        return True

    def _start_simple_qprocess(self, phrase: str) -> bool:
        path = self._program_path
        if not path:
            return False
        args = [phrase] if self._backend == "espeak" else ["-w", phrase]
        proc = QProcess(self)
        proc.setProgram(path)
        proc.setArguments(args)
        proc.finished.connect(lambda *_a, p=proc: self._on_qprocess_finished(p))
        proc.errorOccurred.connect(
            lambda _err, p=proc: self._on_qprocess_error(p)
        )
        proc.start()
        if not proc.waitForStarted(2000):
            self._log(f"Voice process error: {proc.errorString()}")
            proc.deleteLater()
            return False
        self._qprocesses.append(proc)
        return True

    def _start_powershell(self, phrase: str) -> bool:
        escaped = re.sub(r"['`$]", "", phrase.replace('"', "'"))
        script = (
            "Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
            f".Speak('{escaped}');"
        )
        proc = QProcess(self)
        proc.setProgram("powershell")
        proc.setArguments(["-NoProfile", "-Command", script])
        proc.finished.connect(lambda *_a, p=proc: self._on_qprocess_finished(p))
        proc.errorOccurred.connect(
            lambda _err, p=proc: self._on_qprocess_error(p)
        )
        proc.start()
        if not proc.waitForStarted(3000):
            self._log(f"Voice process error: {proc.errorString()}")
            proc.deleteLater()
            return False
        self._qprocesses.append(proc)
        return True

    def _on_qprocess_finished(self, proc: QProcess) -> None:
        if proc in self._qprocesses:
            self._qprocesses.remove(proc)
        self._log("Voice process finished.")
        proc.deleteLater()

    def _on_qprocess_error(self, proc: QProcess) -> None:
        self._log(f"Voice process error: {proc.errorString()}")
        if proc in self._qprocesses:
            self._qprocesses.remove(proc)

    def _prune_finished(self) -> None:
        self._popen_procs = [p for p in self._popen_procs if p.poll() is None]
        self._qprocesses = [p for p in self._qprocesses if p.state() != QProcess.NotRunning]

    def _log(self, msg: str) -> None:
        if self._log_fn is not None:
            self._log_fn(msg)

    @staticmethod
    def _detect_backend() -> tuple[Optional[str], Optional[str]]:
        system = platform.system()
        say_path = shutil.which("say") or "/usr/bin/say"
        if system == "Darwin" and os.path.isfile(say_path):
            return "say_popen", say_path
        if system == "Darwin" and shutil.which("say"):
            return "say", shutil.which("say")
        if system == "Windows":
            return "powershell", "powershell"
        spd = shutil.which("spd-say")
        if spd:
            return "spd-say", spd
        espeak = shutil.which("espeak")
        if espeak:
            return "espeak", espeak
        return None, None
