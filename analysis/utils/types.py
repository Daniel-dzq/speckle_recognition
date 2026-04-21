"""
Typed data model used across the analysis framework.

These dataclasses provide a single source of truth for the domain entities
(channels, challenges, fibers, captures) and the artefacts produced by each
experiment. They are deliberately small and serialisable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Domain primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Channel:
    """A spectral / modal readout channel (e.g. ``green``, ``red``)."""

    name: str
    wavelength_nm: Optional[float] = None
    role: str = "response"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Channel name must be non-empty")


@dataclass(frozen=True)
class Challenge:
    """A structured-light challenge (letter / pattern / index)."""

    name: str
    kind: str = "letter"
    payload: Optional[str] = None


@dataclass(frozen=True)
class Fiber:
    """A physical fiber under test."""

    fiber_id: str
    length_mm: Optional[float] = None
    length_group: Optional[str] = None
    description: str = ""


@dataclass
class Capture:
    """
    A single recorded response file (video or image) annotated with provenance.

    This is the fundamental unit of ingestion. Experiments iterate captures and
    ask the preprocessing pipeline to produce a feature or frame stack.
    """

    path: Path
    fiber: str
    channel: str
    challenge: str
    condition: Optional[str] = None
    session: Optional[str] = None
    repeat: Optional[int] = None
    length_group: Optional[str] = None
    length_mm: Optional[float] = None
    media_kind: str = "video"
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        return d


@dataclass
class Sample:
    """
    A processed feature sample derived from one (or aggregated) capture(s).

    ``vector`` is a flat feature (e.g. resized intensity image flattened) used
    by downstream metrics. ``image`` is the 2-D representative image kept for
    plotting and profile analysis.
    """

    capture_key: Tuple[str, ...]
    fiber: str
    channel: str
    challenge: str
    condition: Optional[str]
    vector: Any  # numpy array (kept generic to avoid circular imports)
    image: Any  # numpy 2-D array
    metadata: Dict[str, Any] = field(default_factory=dict)

    def key(self, *fields: str) -> Tuple[str, ...]:
        return tuple(str(getattr(self, x)) for x in fields)


# ---------------------------------------------------------------------------
# Run / artefact provenance
# ---------------------------------------------------------------------------


@dataclass
class ExperimentRun:
    """Provenance record for a single experiment invocation."""

    name: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    finished_at: Optional[str] = None
    config_path: Optional[str] = None
    config_snapshot: Optional[Mapping[str, Any]] = None
    output_dir: Optional[str] = None
    git_commit: Optional[str] = None
    host: Optional[str] = None
    seed: Optional[int] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricResult:
    """A named scalar or structured metric produced by an experiment."""

    name: str
    value: Any
    unit: Optional[str] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlotArtifact:
    """A rendered figure plus its backing data CSV path."""

    name: str
    paths: List[str]
    source_csv: Optional[str] = None
    caption: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReportArtifact:
    """A generated textual or tabular output (CSV / JSON / MD)."""

    name: str
    kind: str
    path: str
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = [
    "Channel",
    "Challenge",
    "Fiber",
    "Capture",
    "Sample",
    "ExperimentRun",
    "MetricResult",
    "PlotArtifact",
    "ReportArtifact",
]
