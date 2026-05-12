#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect figure exports into one tree for thesis writing::

    paper_assets/
      png/   svg/   pdf/   csv/
      INDEX.csv
      README.md

Source roots (defaults):
  - figures_publication/
  - figures_competition/
  - figures/
Plus optional official tables under results/length_optimization_green/tables/

Run from repository root::

    python scripts/collect_paper_assets.py

Use ``--dry-run`` to print the plan without copying.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_assets"
COLLECT_EXTS = {".png", ".svg", ".pdf", ".csv"}

DEFAULT_SOURCES: List[Tuple[Path, str]] = [
    (ROOT / "figures_publication", "publication"),
    (ROOT / "figures_competition", "competition"),
    (ROOT / "figures", "figures"),
]

# Official analysis tables (if present — results/ is often gitignored locally)
EXTRA_FILES: List[Tuple[Path, str]] = [
    (ROOT / "results" / "length_optimization_green" / "tables" / "per_length_summary.csv", "results_lo_green"),
    (ROOT / "results" / "length_optimization_green" / "tables" / "per_fiber_metrics.csv", "results_lo_green"),
]


def _safe_key(name: str) -> str:
    """Flatten path into a single filename component."""
    s = name.replace("\\", "/")
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:180] if len(s) > 180 else s


def iter_collectable_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in COLLECT_EXTS:
            continue
        yield p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="List actions only")
    ap.add_argument("--clean", action="store_true", help="Remove existing paper_assets/png|svg|pdf|csv and INDEX before copy")
    ap.add_argument("--out", type=Path, default=OUT, help="Output root (default: paper_assets/)")
    args = ap.parse_args()
    out_root: Path = args.out
    dry = args.dry_run
    clean: bool = args.clean

    if clean and not dry and out_root.is_dir():
        for sub in ("png", "svg", "pdf", "csv"):
            p = out_root / sub
            if p.is_dir():
                shutil.rmtree(p)
        for f in ("INDEX.csv", "README.md"):
            fp = out_root / f
            if fp.is_file():
                fp.unlink()

    rows: List[List[str]] = []

    def plan_copy(src: Path, prefix: str, logical_name: str) -> None:
        ext = src.suffix.lower()
        if ext not in COLLECT_EXTS:
            return
        dest_dir = out_root / ext[1:]  # png, svg, pdf, csv
        stem = _safe_key(f"{prefix}__{Path(logical_name).with_suffix('').as_posix()}")
        dest = dest_dir / f"{stem}{ext}"
        rows.append([src.resolve().as_posix(), dest.resolve().as_posix(), prefix, ext[1:]])
        if not dry:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    if not dry:
        for sub in ("png", "svg", "pdf", "csv"):
            (out_root / sub).mkdir(parents=True, exist_ok=True)

    for base, tag in DEFAULT_SOURCES:
        if not base.is_dir():
            continue
        for src in iter_collectable_files(base):
            try:
                rel = src.relative_to(base).as_posix()
            except ValueError:
                rel = src.name
            plan_copy(src, tag, rel)

    for src, tag in EXTRA_FILES:
        if src.is_file():
            plan_copy(src, tag, src.name)

    index = out_root / "INDEX.csv"
    if not dry:
        with open(index, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source_path", "paper_assets_path", "source_bundle", "format"])
            w.writerows(rows)
        readme = out_root / "README.md"
        when = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        readme.write_text(
            f"# Paper assets bundle\n\n"
            f"Generated **{when}** by `python scripts/collect_paper_assets.py`.\n\n"
            f"All **PNG / SVG / PDF / CSV** from `figures_publication/`, `figures_competition/`, "
            f"`figures/` (recursive) are copied into the subfolders here, with a **source prefix** "
            f"on each filename (`publication__…`, `competition__…`, `figures__…`) so names stay unique.\n\n"
            f"Official length tables (if `results/length_optimization_green/tables/` exists) are "
            f"copied into `csv/` as `results_lo_green__*.csv`.\n\n"
            f"**Regenerate** this folder after you refresh figures; it is gitignored by default.\n",
            encoding="utf-8",
        )

    print(f"{'[dry-run] ' if dry else ''}Planned {len(rows)} files → {out_root}")
    if dry and rows:
        for r in rows[:30]:
            print(" ", r[0], "→", r[1])
        if len(rows) > 30:
            print(f"  … and {len(rows) - 30} more")


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    main()
