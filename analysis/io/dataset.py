"""
Configurable dataset discovery.

The paper's experimental data is organised in several layout styles. Rather
than hard-coding any single one we provide a small DSL:

* ``domain_fiber_letter`` — ``<root>/<Domain>/<Fiber>/<Letter>.avi``
  (the layout currently used under ``videocapture/``)
* ``length_fiber_repeat`` — ``<root>/<length_group>/<Fiber>/(rep_i/)<Letter>.avi``
  for the length-optimisation experiment where repeats live per fiber.
* ``session_fiber_channel`` — ``<root>/<session>/<Fiber>/<channel>/<Letter>.avi``
  for time-stability acquisitions.
* ``explicit`` — a user-supplied list of dicts in the config.

Every discovered file is returned as a :class:`Capture` with full provenance.
The discovery logic itself is pure; experiments can freely filter the
resulting :class:`DatasetIndex`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional

from ..utils.config import ExperimentConfig, resolve_path
from ..utils.types import Capture

from .video import IMAGE_EXTS, VIDEO_EXTS


# ---------------------------------------------------------------------------
# Layout config
# ---------------------------------------------------------------------------


@dataclass
class DatasetLayout:
    """Parsed + normalised view of ``config.dataset``."""

    root: Path
    layout: str = "domain_fiber_letter"
    fibers: Optional[List[str]] = None
    domains: Optional[List[str]] = None
    channels: Optional[List[str]] = None
    sessions: Optional[List[str]] = None
    length_groups: Optional[List[str]] = None
    fiber_lookup: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    domain_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    extensions: List[str] = field(default_factory=lambda: sorted(VIDEO_EXTS | IMAGE_EXTS))
    files: List[Dict[str, Any]] = field(default_factory=list)
    custom_resolver: Optional[Callable[["DatasetLayout"], List[Capture]]] = None

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any] | ExperimentConfig,
        base_dir: Path | None = None,
    ) -> "DatasetLayout":
        data = cfg.to_dict() if isinstance(cfg, ExperimentConfig) else dict(cfg)
        raw_root = data.get("root")
        if raw_root is None:
            raise ValueError("dataset.root is required in the configuration")
        root = resolve_path(raw_root, base_dir)
        layout = data.get("layout", "domain_fiber_letter")
        return cls(
            root=root,
            layout=layout,
            fibers=data.get("fibers"),
            domains=data.get("domains"),
            channels=data.get("channels"),
            sessions=data.get("sessions"),
            length_groups=data.get("length_groups"),
            fiber_lookup=dict(data.get("fiber_lookup") or {}),
            domain_map=dict(data.get("domain_map") or {}),
            extensions=list(data.get("extensions") or sorted(VIDEO_EXTS | IMAGE_EXTS)),
            files=list(data.get("files") or []),
        )

    def with_custom_resolver(
        self, fn: Callable[["DatasetLayout"], List[Capture]]
    ) -> "DatasetLayout":
        self.custom_resolver = fn
        return self


def default_layout_from_repo(repo_root: Path) -> DatasetLayout:
    """Convenience default matching the existing ``videocapture/`` tree."""
    return DatasetLayout(
        root=Path(repo_root) / "videocapture",
        layout="domain_fiber_letter",
        domains=["Green", "GreenAndRed", "RedChange"],
        fibers=[f"Fiber{i}" for i in range(1, 6)],
        domain_map={
            "Green": {"channel": "green", "condition": "side_green"},
            "GreenAndRed": {"channel": "green_red_fixed", "condition": "dual_fixed"},
            "RedChange": {"channel": "green_red_dynamic", "condition": "dual_dynamic"},
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_media(path: Path, exts: Iterable[str]) -> bool:
    return path.suffix.lower() in {e.lower() for e in exts}


_LETTER_RE = re.compile(r"^([A-Za-z])")


def _normalize_letter(stem: str) -> str:
    m = _LETTER_RE.match(stem)
    return m.group(1).upper() if m else stem


def _iter_dir(path: Path) -> List[Path]:
    if not path.is_dir():
        return []
    return sorted(path.iterdir())


# ---------------------------------------------------------------------------
# Layout resolvers
# ---------------------------------------------------------------------------


def _resolve_domain_fiber_letter(layout: DatasetLayout) -> List[Capture]:
    caps: List[Capture] = []
    root = layout.root
    target_domains = set(layout.domains) if layout.domains else None
    target_fibers = set(layout.fibers) if layout.fibers else None

    for dom_dir in _iter_dir(root):
        if not dom_dir.is_dir():
            continue
        dom = dom_dir.name
        if target_domains is not None and dom not in target_domains:
            continue
        dom_info = layout.domain_map.get(dom, {})
        channel = dom_info.get("channel", dom.lower())
        condition = dom_info.get("condition")
        for fdir in _iter_dir(dom_dir):
            if not fdir.is_dir():
                continue
            fiber = fdir.name
            if target_fibers is not None and fiber not in target_fibers:
                continue
            fiber_info = layout.fiber_lookup.get(fiber, {})
            length_group = fiber_info.get("length_group")
            length_mm = fiber_info.get("length_mm")
            for f in _iter_dir(fdir):
                if not f.is_file() or not _is_media(f, layout.extensions):
                    continue
                letter = _normalize_letter(f.stem)
                caps.append(
                    Capture(
                        path=f,
                        fiber=fiber,
                        channel=channel,
                        challenge=letter,
                        condition=condition,
                        length_group=length_group,
                        length_mm=length_mm,
                        media_kind="image" if _is_media(f, IMAGE_EXTS) else "video",
                        extra={"domain": dom},
                    )
                )
    return caps


def _resolve_length_fiber_repeat(layout: DatasetLayout) -> List[Capture]:
    caps: List[Capture] = []
    root = layout.root
    length_groups = layout.length_groups
    for lg_dir in _iter_dir(root):
        if not lg_dir.is_dir():
            continue
        lg = lg_dir.name
        if length_groups is not None and lg not in length_groups:
            continue
        lg_info = layout.fiber_lookup.get(lg, {})
        length_mm = lg_info.get("length_mm")
        fibers = layout.fibers
        for fdir in _iter_dir(lg_dir):
            if not fdir.is_dir():
                continue
            fiber = fdir.name
            if fibers is not None and fiber not in fibers:
                continue
            repeat_dirs = [p for p in _iter_dir(fdir) if p.is_dir()]
            if repeat_dirs:
                for rep_idx, rep_dir in enumerate(repeat_dirs):
                    for f in _iter_dir(rep_dir):
                        if not f.is_file() or not _is_media(f, layout.extensions):
                            continue
                        caps.append(
                            Capture(
                                path=f,
                                fiber=fiber,
                                channel="green",
                                challenge=_normalize_letter(f.stem),
                                repeat=rep_idx,
                                length_group=lg,
                                length_mm=length_mm,
                                media_kind="image" if _is_media(f, IMAGE_EXTS) else "video",
                                extra={"repeat_dir": rep_dir.name},
                            )
                        )
            else:
                for f in _iter_dir(fdir):
                    if not f.is_file() or not _is_media(f, layout.extensions):
                        continue
                    caps.append(
                        Capture(
                            path=f,
                            fiber=fiber,
                            channel="green",
                            challenge=_normalize_letter(f.stem),
                            repeat=0,
                            length_group=lg,
                            length_mm=length_mm,
                            media_kind="image" if _is_media(f, IMAGE_EXTS) else "video",
                        )
                    )
    return caps


def _resolve_session_fiber_channel(layout: DatasetLayout) -> List[Capture]:
    caps: List[Capture] = []
    root = layout.root
    sessions = layout.sessions
    for session_dir in _iter_dir(root):
        if not session_dir.is_dir():
            continue
        session = session_dir.name
        if sessions is not None and session not in sessions:
            continue
        for fdir in _iter_dir(session_dir):
            if not fdir.is_dir():
                continue
            fiber = fdir.name
            if layout.fibers is not None and fiber not in layout.fibers:
                continue
            fiber_info = layout.fiber_lookup.get(fiber, {})
            for channel_dir in _iter_dir(fdir):
                if not channel_dir.is_dir():
                    continue
                channel = channel_dir.name
                if layout.channels is not None and channel not in layout.channels:
                    continue
                for f in _iter_dir(channel_dir):
                    if not f.is_file() or not _is_media(f, layout.extensions):
                        continue
                    caps.append(
                        Capture(
                            path=f,
                            fiber=fiber,
                            channel=channel,
                            challenge=_normalize_letter(f.stem),
                            session=session,
                            length_group=fiber_info.get("length_group"),
                            length_mm=fiber_info.get("length_mm"),
                            media_kind="image" if _is_media(f, IMAGE_EXTS) else "video",
                        )
                    )
    return caps


def _resolve_explicit_files(layout: DatasetLayout) -> List[Capture]:
    caps: List[Capture] = []
    base = layout.root
    for entry in layout.files:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = base / path
        caps.append(
            Capture(
                path=path.resolve(),
                fiber=str(entry.get("fiber", "unknown")),
                channel=str(entry.get("channel", "green")),
                challenge=str(entry.get("challenge", path.stem.upper())),
                condition=entry.get("condition"),
                session=entry.get("session"),
                repeat=entry.get("repeat"),
                length_group=entry.get("length_group"),
                length_mm=entry.get("length_mm"),
                media_kind="image" if _is_media(path, IMAGE_EXTS) else "video",
                extra={k: v for k, v in entry.items() if k not in {
                    "path", "fiber", "channel", "challenge", "condition",
                    "session", "repeat", "length_group", "length_mm",
                }},
            )
        )
    return caps


_RESOLVERS: Dict[str, Callable[[DatasetLayout], List[Capture]]] = {
    "domain_fiber_letter": _resolve_domain_fiber_letter,
    "length_fiber_repeat": _resolve_length_fiber_repeat,
    "session_fiber_channel": _resolve_session_fiber_channel,
    "explicit": _resolve_explicit_files,
}


def discover_captures(layout: DatasetLayout) -> List[Capture]:
    """Return all captures for a given :class:`DatasetLayout`."""
    if layout.custom_resolver is not None:
        caps = layout.custom_resolver(layout)
    else:
        resolver = _RESOLVERS.get(layout.layout)
        if resolver is None:
            raise ValueError(
                f"Unknown dataset layout: {layout.layout!r}. "
                f"Available: {sorted(_RESOLVERS)}"
            )
        caps = resolver(layout)

    # Deduplicate by absolute path while preserving order.
    seen = set()
    unique: List[Capture] = []
    for c in caps:
        key = str(c.path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# Index abstraction
# ---------------------------------------------------------------------------


class DatasetIndex:
    """
    Queryable view over a list of captures.

    Filtering is non-destructive and returns a new :class:`DatasetIndex`.
    """

    def __init__(self, captures: List[Capture], layout: Optional[DatasetLayout] = None):
        self._captures = list(captures)
        self.layout = layout

    def __len__(self) -> int:
        return len(self._captures)

    def __iter__(self) -> Iterator[Capture]:
        return iter(self._captures)

    def as_list(self) -> List[Capture]:
        return list(self._captures)

    def filter(
        self,
        *,
        fiber: Optional[str] = None,
        channel: Optional[str] = None,
        challenge: Optional[str] = None,
        length_group: Optional[str] = None,
        condition: Optional[str] = None,
        session: Optional[str] = None,
    ) -> "DatasetIndex":
        def keep(c: Capture) -> bool:
            if fiber is not None and c.fiber != fiber:
                return False
            if channel is not None and c.channel != channel:
                return False
            if challenge is not None and c.challenge != challenge:
                return False
            if length_group is not None and c.length_group != length_group:
                return False
            if condition is not None and c.condition != condition:
                return False
            if session is not None and c.session != session:
                return False
            return True

        return DatasetIndex([c for c in self._captures if keep(c)], layout=self.layout)

    def fibers(self) -> List[str]:
        return sorted({c.fiber for c in self._captures})

    def channels(self) -> List[str]:
        return sorted({c.channel for c in self._captures})

    def challenges(self) -> List[str]:
        return sorted({c.challenge for c in self._captures})

    def length_groups(self) -> List[str]:
        return sorted({c.length_group for c in self._captures if c.length_group})

    def conditions(self) -> List[str]:
        return sorted({c.condition for c in self._captures if c.condition})

    def sessions(self) -> List[str]:
        return sorted({c.session for c in self._captures if c.session})

    def summary(self) -> Dict[str, Any]:
        return {
            "n_captures": len(self._captures),
            "n_fibers": len(self.fibers()),
            "n_channels": len(self.channels()),
            "n_challenges": len(self.challenges()),
            "fibers": self.fibers(),
            "channels": self.channels(),
            "challenges": self.challenges(),
            "length_groups": self.length_groups(),
            "conditions": self.conditions(),
        }


__all__ = [
    "DatasetLayout",
    "DatasetIndex",
    "default_layout_from_repo",
    "discover_captures",
]
