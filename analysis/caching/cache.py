"""
Deterministic, version-aware on-disk cache.

Every cache entry is a tiny directory under ``<root>/<bucket>/<key>/`` that
contains ``data.npz`` (the actual payload) and ``meta.json`` (source file
mtime/size + a user-provided ``version`` tag). Stale entries are
automatically invalidated when any of those change.

The cache is deliberately simple — we never import pickle, we never execute
arbitrary code on load, and we never touch entries from other buckets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


def _hash_key(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


class FeatureCache:
    """Per-experiment cache for small numpy arrays with metadata."""

    def __init__(
        self,
        root: str | Path,
        bucket: str = "default",
        enabled: bool = True,
        version: str = "v1",
    ) -> None:
        self.enabled = enabled
        self.version = version
        self.root = Path(root) / bucket
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    # ----- helpers --------------------------------------------------------
    def _entry_dir(self, key: str) -> Path:
        return self.root / _hash_key(key)

    def _file_signature(self, source: Path) -> Dict[str, Any]:
        try:
            stat = source.stat()
            return {"mtime": stat.st_mtime, "size": stat.st_size}
        except FileNotFoundError:
            return {"mtime": None, "size": None}

    # ----- public API -----------------------------------------------------
    def get(
        self,
        key: str,
        source: Optional[Path] = None,
    ) -> Optional[Mapping[str, np.ndarray]]:
        if not self.enabled:
            return None
        entry = self._entry_dir(key)
        data_file = entry / "data.npz"
        meta_file = entry / "meta.json"
        if not data_file.exists() or not meta_file.exists():
            return None
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            return None
        if meta.get("version") != self.version:
            return None
        if source is not None:
            sig = self._file_signature(source)
            cached_sig = meta.get("source", {})
            if cached_sig.get("mtime") != sig["mtime"] or cached_sig.get("size") != sig["size"]:
                return None
        try:
            with np.load(data_file, allow_pickle=False) as npz:
                return {k: npz[k] for k in npz.files}
        except Exception:
            return None

    def put(
        self,
        key: str,
        payload: Mapping[str, np.ndarray],
        source: Optional[Path] = None,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Path, Path]:
        if not self.enabled:
            return Path(), Path()
        entry = self._entry_dir(key)
        entry.mkdir(parents=True, exist_ok=True)
        data_file = entry / "data.npz"
        meta_file = entry / "meta.json"

        arrays = {k: np.asarray(v) for k, v in payload.items()}
        np.savez_compressed(data_file, **arrays)

        full_meta = {
            "version": self.version,
            "key": key,
            "source": self._file_signature(source) if source is not None else None,
            "user": dict(meta or {}),
        }
        meta_file.write_text(json.dumps(full_meta, indent=2, default=str), encoding="utf-8")
        return data_file, meta_file

    def invalidate(self) -> int:
        """Delete all entries in this bucket."""
        if not self.enabled or not self.root.exists():
            return 0
        count = 0
        for sub in self.root.iterdir():
            if sub.is_dir():
                for p in sub.rglob("*"):
                    p.unlink(missing_ok=True)
                sub.rmdir()
                count += 1
        return count


__all__ = ["FeatureCache"]
