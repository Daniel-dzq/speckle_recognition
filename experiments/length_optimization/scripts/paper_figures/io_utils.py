"""
Canonical-output archival: timestamped folders under each figure directory.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set


def archive_existing_outputs(output_dir: Path, figure_stem: str) -> Optional[Path]:
    """
    Before overwriting canonical paper outputs, move prior versions to:
        <output_dir>/archive/YYYYMMDD_HHMMSS/

    Matched files (top-level of output_dir only; never README.md or archive/):
      {stem}*.png, {stem}*.pdf, {stem}*.svg, {stem}*_data.csv, {stem}*_meta.json

    Returns path to archive folder if anything was moved, else None.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None

    patterns = (
        f"{figure_stem}*.png",
        f"{figure_stem}*.pdf",
        f"{figure_stem}*.svg",
        f"{figure_stem}*_data.csv",
        f"{figure_stem}*_meta.json",
    )
    to_move: Set[Path] = set()
    for pat in patterns:
        for p in output_dir.glob(pat):
            if not p.is_file():
                continue
            rel = p.relative_to(output_dir)
            if rel.parts and rel.parts[0] == "archive":
                continue
            to_move.add(p)

    if not to_move:
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arch_root = output_dir / "archive"
    arch_dir = arch_root / stamp
    arch_dir.mkdir(parents=True, exist_ok=False)

    manifest: Dict[str, Any] = {
        "archive_stamp": stamp,
        "archive_dir": str(arch_dir.resolve()),
        "figure_stem": figure_stem,
        "moved_files": [],
    }
    for src in sorted(to_move, key=lambda x: x.name):
        dest = arch_dir / src.name
        shutil.move(str(src), str(dest))
        manifest["moved_files"].append({
            "from": src.name,
            "to": str(dest.relative_to(output_dir)),
        })

    man_path = arch_dir / "archive_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return arch_dir
