"""
Serialisation and report helpers.

The goal is to produce artefacts that are simultaneously:

* easy for humans to skim (Markdown reports + neat tables)
* easy for downstream scripts to parse (CSV + JSON)
* deterministic with respect to key ordering.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

from ..utils.types import PlotArtifact, ReportArtifact


# ---------------------------------------------------------------------------
# Serialisers
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj).__name__}")


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def write_json(path: Union[str, Path], data: Any, indent: int = 2) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize(data)
    p.write_text(
        json.dumps(sanitized, indent=indent, default=_json_default),
        encoding="utf-8",
    )
    return p


def _csv_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if np.isnan(v) or np.isinf(v):
            return ""
        return f"{v:.6g}"
    return str(v)


def write_csv(
    path: Union[str, Path],
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Optional[Sequence[str]] = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return p
    if fieldnames is None:
        seen: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.append(k)
        fieldnames = seen
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: _csv_cell(row.get(k)) for k in fieldnames})
    return p


def write_markdown(path: Union[str, Path], text: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Markdown builder
# ---------------------------------------------------------------------------


class MarkdownBuilder:
    """
    A small fluent builder for well-formatted Markdown sections.

    Each method returns ``self`` so call chains remain readable. ``None`` and
    non-finite floats render as ``—`` to keep tables looking clean.
    """

    def __init__(self, title: str = "") -> None:
        self._lines: List[str] = []
        if title:
            self.h(1, title)

    # ----- primitives -----------------------------------------------------
    def h(self, level: int, text: str) -> "MarkdownBuilder":
        level = max(1, min(6, level))
        self._lines.append(f"{'#' * level} {text}")
        self._lines.append("")
        return self

    def p(self, text: str) -> "MarkdownBuilder":
        self._lines.append(text)
        self._lines.append("")
        return self

    def bullet(self, items: Iterable[Any]) -> "MarkdownBuilder":
        for i in items:
            self._lines.append(f"- {i}")
        self._lines.append("")
        return self

    def kv(self, mapping: Mapping[str, Any]) -> "MarkdownBuilder":
        for k, v in mapping.items():
            self._lines.append(f"- **{k}**: {self._fmt(v)}")
        self._lines.append("")
        return self

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> "MarkdownBuilder":
        self._lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        self._lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            self._lines.append("| " + " | ".join(self._fmt(c) for c in row) + " |")
        self._lines.append("")
        return self

    def image(self, caption: str, path: str) -> "MarkdownBuilder":
        self._lines.append(f"![{caption}]({path})")
        self._lines.append("")
        return self

    def code(self, code: str, lang: str = "") -> "MarkdownBuilder":
        self._lines.append(f"```{lang}")
        self._lines.append(code)
        self._lines.append("```")
        self._lines.append("")
        return self

    def raw(self, text: str) -> "MarkdownBuilder":
        self._lines.append(text)
        return self

    # ----- output ---------------------------------------------------------
    def to_string(self) -> str:
        return "\n".join(self._lines).rstrip() + "\n"

    def save(self, path: Union[str, Path]) -> Path:
        return write_markdown(path, self.to_string())

    # ----- formatting -----------------------------------------------------
    @staticmethod
    def _fmt(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            if np.isnan(v) or np.isinf(v):
                return "—"
            if abs(v) >= 1000 or (abs(v) > 0 and abs(v) < 1e-3):
                return f"{v:.3g}"
            return f"{v:.3f}"
        if isinstance(v, np.generic):
            return MarkdownBuilder._fmt(v.item())
        return str(v)


# ---------------------------------------------------------------------------
# Experiment report aggregator
# ---------------------------------------------------------------------------


@dataclass
class ExperimentReport:
    """Collect all artefacts generated by an experiment for manifest output."""

    output_dir: Path
    experiment: str
    artifacts: List[Union[PlotArtifact, ReportArtifact]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_artifact(self, artefact: Union[PlotArtifact, ReportArtifact]) -> None:
        self.artifacts.append(artefact)

    def write_summary_json(self, filename: str = "summary.json") -> Path:
        payload = {
            "experiment": self.experiment,
            "artifacts": [self._artefact_dict(a) for a in self.artifacts],
        }
        return write_json(self.output_dir / filename, payload)

    def write_summary_csv(
        self, filename: str = "artifacts.csv"
    ) -> Path:
        rows: List[Mapping[str, Any]] = []
        for a in self.artifacts:
            d = self._artefact_dict(a)
            rows.append({k: v if not isinstance(v, list) else "|".join(str(x) for x in v)
                         for k, v in d.items()})
        return write_csv(self.output_dir / filename, rows)

    @staticmethod
    def _artefact_dict(a: Any) -> Mapping[str, Any]:
        if hasattr(a, "to_dict"):
            return a.to_dict()
        return dict(a)


__all__ = [
    "write_json",
    "write_csv",
    "write_markdown",
    "MarkdownBuilder",
    "ExperimentReport",
]
