"""
Console + file logging configured consistently across experiments.

A minimal coloured formatter is used for the console so that running an
experiment from the terminal feels polished without depending on extra
packages. The file handler writes plain text for easy ``grep``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

_LEVEL_COLORS = {
    logging.DEBUG: "\x1b[38;5;244m",
    logging.INFO: "\x1b[38;5;39m",
    logging.WARNING: "\x1b[38;5;214m",
    logging.ERROR: "\x1b[38;5;203m",
    logging.CRITICAL: "\x1b[1;38;5;196m",
}
_RESET = "\x1b[0m"


class _ColorFormatter(logging.Formatter):
    def __init__(self, use_color: bool):
        super().__init__("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                         datefmt="%H:%M:%S")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        if not self.use_color:
            return base
        color = _LEVEL_COLORS.get(record.levelno, "")
        return f"{color}{base}{_RESET}" if color else base


def configure_logging(
    name: str = "analysis",
    log_file: Union[str, Path, None] = None,
    level: int = logging.INFO,
    color: Optional[bool] = None,
) -> logging.Logger:
    """
    Create (or return) the root analysis logger with stream + optional file handlers.

    Safe to call multiple times; handlers are deduplicated by destination.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    existing = {getattr(h, "_key", None) for h in logger.handlers}

    use_color = sys.stderr.isatty() if color is None else color

    if "stream" not in existing:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(_ColorFormatter(use_color=use_color))
        sh.setLevel(level)
        sh._key = "stream"  # type: ignore[attr-defined]
        logger.addHandler(sh)

    if log_file is not None:
        key = f"file:{Path(log_file).resolve()}"
        if key not in existing:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(_ColorFormatter(use_color=False))
            fh.setLevel(level)
            fh._key = key  # type: ignore[attr-defined]
            logger.addHandler(fh)

    return logger


def get_logger(name: str = "analysis") -> logging.Logger:
    """Return a namespaced sub-logger."""
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
