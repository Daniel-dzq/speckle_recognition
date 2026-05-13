#!/usr/bin/env python3
"""Sanity checks for generated paper figures (English-only SVG, artifacts present)."""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG_PAPER = ROOT / "figures" / "paper"

CJK_RE = re.compile(r"[\u4e00-\u9fff]")

FIG6_WARNING_EXPECTED = (
    "G/R ratio values are not final until paired red/green raw data are verified."
)

FIG7_SAFE_CLAIM = (
    "The 5×5 matrix supports fiber/device specificity and cross-fiber rejection behavior."
)
FIG7_UNSAFE_CLAIM = (
    "Do not claim final ROC/EER or threshold authentication from this figure."
)


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


def policy_errors(fig_dir: Path, base: str, meta: dict) -> list[str]:
    errs: list[str] = []
    if base == "Fig3_length_optimization":
        if meta.get("length_meaning") != "total_fiber_length_cm":
            errs.append(f"{base}: meta length_meaning must be total_fiber_length_cm")
        if meta.get("optimal_total_fiber_length_cm") != 9:
            errs.append(f"{base}: meta optimal_total_fiber_length_cm must be 9")
        if meta.get("confirmed_by_PI") is not True:
            errs.append(f"{base}: meta confirmed_by_PI must be true")
        csv_p = fig_dir / f"{base}_data.csv"
        if csv_p.is_file():
            with csv_p.open(encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                if "length_meaning" not in (r.fieldnames or []):
                    errs.append(f"{base}: data CSV missing length_meaning column")
                elif "is_selected_optimal" not in (r.fieldnames or []):
                    errs.append(f"{base}: data CSV missing is_selected_optimal column")
                else:
                    rows = list(r)
                    opt = [row for row in rows if row.get("is_selected_optimal", "").lower() in ("true", "1")]
                    if len(opt) != 1:
                        errs.append(f"{base}: expected exactly one is_selected_optimal row, got {len(opt)}")
                    else:
                        tcm = opt[0].get("total_length_cm", "").strip()
                        if tcm and float(tcm) != 9.0:
                            errs.append(f"{base}: selected optimal row total_length_cm must be 9")
    if base == "Fig5_dual_channel":
        if meta.get("manuscript_ready") is not True:
            errs.append(f"{base}: meta manuscript_ready must be true")
        if meta.get("data_validated_by_PI") is not True:
            errs.append(f"{base}: meta data_validated_by_PI must be true")
        if meta.get("source_dataset_status") != "final_or_PI_confirmed":
            errs.append(f"{base}: meta source_dataset_status must be final_or_PI_confirmed")
    if base == "Fig6_common_mode_suppression":
        if meta.get("manuscript_ready") is not False:
            errs.append(f"{base}: meta manuscript_ready must be false (draft)")
        if meta.get("data_status") != "draft":
            errs.append(f"{base}: meta data_status must be draft")
        if meta.get("warning") != FIG6_WARNING_EXPECTED:
            errs.append(f"{base}: meta warning must match expected draft paired-data notice")
    if base == "Fig7_authentication":
        if meta.get("manuscript_ready") is not False:
            errs.append(f"{base}: meta manuscript_ready must be false until PI confirms")
        if meta.get("safe_claim") != FIG7_SAFE_CLAIM:
            errs.append(f"{base}: meta safe_claim must match expected string")
        if meta.get("unsafe_claim") != FIG7_UNSAFE_CLAIM:
            errs.append(f"{base}: meta unsafe_claim must match expected string")
    return errs


def main() -> int:
    if not FIG_PAPER.is_dir():
        print("No figures/paper yet")
        return 0
    all_errs: list[str] = []
    for meta_path in sorted(FIG_PAPER.rglob("*_meta.json")):
        try:
            rel = meta_path.relative_to(FIG_PAPER)
        except ValueError:
            continue
        if len(rel.parts) > 1 and rel.parts[1] == "archive":
            continue
        if "archive" in rel.parts:
            continue
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        base = meta_path.name.replace("_meta.json", "")
        fig_dir = meta_path.parent
        all_errs.extend(check_bundle(fig_dir, base))
        all_errs.extend(policy_errors(fig_dir, base, data))
    if all_errs:
        print("SANITY FAILURES:")
        for e in all_errs:
            print(" -", e)
        return 1
    print("Sanity checks passed for bundles under figures/paper/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
