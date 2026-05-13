#!/usr/bin/env python3
"""Sanity checks for generated paper figures (English-only SVG, artifacts present)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG_PAPER = ROOT / "figures" / "paper"

CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def svg_contains_cjk(path: Path) -> bool:
    try:
        txt = path.read_text(encoding="utf-8", errors="strict")
    except Exception:
        return False
    return CJK_RE.search(txt) is not None


def check_bundle(fig_dir: Path, base: str) -> list[str]:
    errs: list[str] = []
    for ext in ("png", "pdf", "svg"):
        p = fig_dir / f"{base}.{ext}"
        if not p.is_file():
            errs.append(f"missing {p.relative_to(ROOT)}")
    csv_p = fig_dir / f"{base}_data.csv"
    if not csv_p.is_file():
        errs.append(f"missing {csv_p.relative_to(ROOT)}")
    meta_p = fig_dir / f"{base}_meta.json"
    if not meta_p.is_file():
        errs.append(f"missing {meta_p.relative_to(ROOT)}")
    svg = fig_dir / f"{base}.svg"
    if svg.is_file() and svg_contains_cjk(svg):
        errs.append(f"CJK in SVG {svg.relative_to(ROOT)}")
    return errs


def main() -> int:
    if not FIG_PAPER.is_dir():
        print("No figures/paper yet")
        return 0
    all_errs: list[str] = []
    for meta in sorted(FIG_PAPER.rglob("*_meta.json")):
        data = json.loads(meta.read_text(encoding="utf-8"))
        base = meta.name.replace("_meta.json", "")
        fig_dir = meta.parent
        all_errs.extend(check_bundle(fig_dir, base))
    if all_errs:
        print("SANITY FAILURES:")
        for e in all_errs:
            print(" -", e)
        return 1
    print("Sanity checks passed for bundles under figures/paper/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
