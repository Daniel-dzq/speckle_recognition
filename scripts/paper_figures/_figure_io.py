"""Save helpers for paper figures (PNG/PDF/SVG + CSV/JSON/MD)."""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt

DPI_PNG = 600
_ARCHIVE_OLD = False


def set_archive_old(enabled: bool) -> None:
    global _ARCHIVE_OLD
    _ARCHIVE_OLD = enabled


def archive_existing(path: Path) -> None:
    """Move an existing file to a timestamped sibling (only when archiving enabled)."""
    if not _ARCHIVE_OLD or not path.is_file():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = path.with_name(f"{path.stem}_archived_{stamp}{path.suffix}")
    shutil.move(str(path), str(dest))


def save_figure_triplet(fig: plt.Figure, stem: Path) -> List[Path]:
    """Save PNG (600 dpi), PDF, and SVG; overwrite by default."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    tight = {"bbox_inches": "tight", "pad_inches": 0.06}
    for ext, kwargs in (
        ("png", {"dpi": DPI_PNG, **tight}),
        ("pdf", dict(tight)),
        ("svg", dict(tight)),
    ):
        out = stem.with_suffix(f".{ext}")
        archive_existing(out)
        fig.savefig(out, **kwargs)
        written.append(out)
    return written


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    archive_existing(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    archive_existing(path)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    archive_existing(path)
    path.write_text(text, encoding="utf-8")
