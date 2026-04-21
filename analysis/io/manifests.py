"""Experiment manifest reader / writer for full reproducibility."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from ..utils.types import Capture


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        return out.decode().strip()
    except Exception:  # pragma: no cover
        return None


def write_manifest(
    path: str | Path,
    *,
    experiment: str,
    config_snapshot: Mapping[str, Any],
    captures: Optional[Iterable[Capture]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Serialise a full manifest (config, provenance, captures) as JSON."""
    manifest = {
        "experiment": experiment,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "config": dict(config_snapshot),
    }
    if captures is not None:
        manifest["captures"] = [c.as_dict() for c in captures]
    if extra:
        manifest["extra"] = dict(extra)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return out


def read_manifest(path: str | Path) -> Mapping[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = ["write_manifest", "read_manifest"]
