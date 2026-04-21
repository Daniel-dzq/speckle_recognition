"""
Configuration loading and lightweight attribute-style access.

The framework is fully config-driven. YAML is the canonical format but JSON
is accepted for convenience. ``ExperimentConfig`` wraps the parsed mapping
and provides both dict-style and attribute-style access so that experiment
code can write ``cfg.dataset.root`` instead of ``cfg["dataset"]["root"]``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Union


class ExperimentConfig(Mapping):
    """
    Immutable-ish wrapper around a parsed YAML/JSON configuration.

    * Supports ``cfg.dataset.root`` attribute access and ``cfg["dataset"]``
      indexing; nested dicts are auto-wrapped.
    * Retains its source path so experiments can resolve relative data roots.
    * Offers ``cfg.get("dataset.root", default)`` dotted-key lookup.
    """

    __slots__ = ("_data", "_path")

    def __init__(self, data: Mapping[str, Any], source_path: Union[str, Path, None] = None):
        if not isinstance(data, Mapping):
            raise TypeError(f"ExperimentConfig expects a mapping, got {type(data).__name__}")
        self._data = dict(data)
        self._path = Path(source_path).resolve() if source_path is not None else None

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, Mapping):
            return ExperimentConfig(value)
        if isinstance(value, list):
            return [ExperimentConfig._wrap(v) for v in value]
        return value

    # Mapping protocol -----------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        val = self._data[key]
        return self._wrap(val)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        return key in self._data

    # Attribute access -----------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"ExperimentConfig has no key {name!r}") from exc

    # Convenience ----------------------------------------------------------
    def get(self, dotted_key: str, default: Any = None) -> Any:
        parts = dotted_key.split(".")
        current: Any = self._data
        for p in parts:
            if isinstance(current, Mapping) and p in current:
                current = current[p]
            else:
                return default
        return self._wrap(current)

    def to_dict(self) -> dict:
        def _deep(value: Any) -> Any:
            if isinstance(value, ExperimentConfig):
                return value.to_dict()
            if isinstance(value, Mapping):
                return {k: _deep(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_deep(v) for v in value]
            return value

        return {k: _deep(v) for k, v in self._data.items()}

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def base_dir(self) -> Path:
        return self._path.parent if self._path is not None else Path.cwd()

    def __repr__(self) -> str:
        keys = ", ".join(list(self._data.keys())[:5])
        return f"ExperimentConfig(keys=[{keys}{', ...' if len(self._data) > 5 else ''}])"


def load_config(path: Union[str, Path]) -> ExperimentConfig:
    """Load a YAML (or JSON) configuration file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PyYAML is required to load YAML configs. Install with `pip install PyYAML`."
            ) from exc
        data = yaml.safe_load(text) or {}
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config extension: {suffix}")
    if not isinstance(data, Mapping):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")
    return ExperimentConfig(data, source_path=p)


def dump_config(cfg: Union[ExperimentConfig, Mapping[str, Any]], out_path: Union[str, Path]) -> Path:
    """Write a configuration snapshot next to experiment results."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.to_dict() if isinstance(cfg, ExperimentConfig) else dict(cfg)
    suffix = out.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyYAML is required for YAML output") from exc
        out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    elif suffix == ".json":
        out.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported config output extension: {suffix}")
    return out


def resolve_path(path: Union[str, Path], base: Union[str, Path, None] = None) -> Path:
    """
    Resolve ``path`` against ``base`` if it is relative. ``~`` is expanded.
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    base_p = Path(base).expanduser().resolve() if base is not None else Path.cwd()
    return (base_p / p).resolve()


__all__ = ["ExperimentConfig", "load_config", "dump_config", "resolve_path"]
