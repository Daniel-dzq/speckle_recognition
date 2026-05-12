#!/usr/bin/env python3
"""
Read-only repository inventory: sizes, file types, experiment markers.

Writes:
  docs/repository_inventory.md
  docs/repository_inventory.csv
  docs/generated_figures_manifest.csv   (image list under figures/ if present)

Usage:
  python scripts/inventory_repository.py
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# Top-level names treated as historical campaigns (do not move; document only)
HISTORICAL_CAMPAIGNS = frozenset(
    {
        "LengthOptimize",
        "disturbance_sensitivity",
        "fiber_loss",
        "long_term_stability",
        "power_common_mode",
        "Green",
    }
)

IGNORE_TOP_INVENTORY = frozenset({".git", ".venv", "venv", "env", ".eggs"})


def classify_top_level(name: str, is_dir: bool) -> str:
    if name in HISTORICAL_CAMPAIGNS:
        return "historical_campaigns"
    if name in ("gui", "scripts", "analysis", "config", "archive", "docs", "letter_images"):
        return "code_core"
    if name == "results":
        return "experiment_results"
    if name == "figures":
        return "paper_figures"
    if name == "figures_publication":
        return "publication_figures"
    if name in ("videocapture", "video_capture", "videos", "screenshots", "data"):
        return "raw_or_input_data"
    if name in ("checkpoints", "output"):
        return "experiment_results"
    if name in ("analysis_cache", ".analysis_cache", ".cache", ".pytest_cache", ".mypy_cache", ".ruff_cache"):
        return "cache_or_temp"
    if name == "__pycache__" or (name.startswith(".") and name not in (".github",)):
        return "cache_or_temp"
    if name == "experiment_archive":
        return "experiment_archive"
    if name.endswith(".py") or name in ("requirements.txt", "README.md", ".gitignore"):
        return "code_core"
    if is_dir:
        return "unknown"
    return "unknown"


def walk_stats(base: Path) -> Tuple[int, int, float, Optional[float], Counter[str], Dict[str, bool]]:
    """Returns file_count, total_bytes, total_mb, latest_mtime (or None), extensions, flags."""
    if not base.exists():
        return 0, 0, 0.0, None, Counter(), {}

    total_files = 0
    total_bytes = 0
    latest_mtime: Optional[float] = None
    ext_counter: Counter[str] = Counter()
    flags = {
        "has_report_md": False,
        "has_manifest_json": False,
        "has_figures_subdir": False,
        "has_tables_subdir": False,
        "has_pth_pt_ckpt": False,
        "has_video": False,
        "has_numpy_cache": False,
    }

    ignore_names = {"__pycache__", ".DS_Store", ".ipynb_checkpoints"}

    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in ignore_names]
        if "report.md" in filenames:
            flags["has_report_md"] = True
        if "manifest.json" in filenames:
            flags["has_manifest_json"] = True

        for fn in filenames:
            if fn == ".DS_Store":
                continue
            fp = Path(dirpath) / fn
            try:
                st = fp.stat()
            except OSError:
                continue
            total_files += 1
            total_bytes += st.st_size
            if latest_mtime is None or st.st_mtime > latest_mtime:
                latest_mtime = st.st_mtime

            suf = fp.suffix.lower()
            ext_counter[suf or "(no_ext)"]
            if suf in (".pth", ".pt", ".ckpt"):
                flags["has_pth_pt_ckpt"] = True
            if suf in (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"):
                flags["has_video"] = True
            if suf in (".npy", ".npz", ".pkl", ".joblib"):
                flags["has_numpy_cache"] = True

    if (base / "figures").is_dir():
        flags["has_figures_subdir"] = True
    if (base / "tables").is_dir():
        flags["has_tables_subdir"] = True

    total_mb = total_bytes / (1024 * 1024)
    return total_files, total_bytes, total_mb, latest_mtime, ext_counter, flags


def fmt_ext_snapshot(counter: Counter[str], limit: int = 8) -> str:
    items = counter.most_common(limit)
    return ", ".join(f"{ext}:{n}" for ext, n in items)


def fmt_time(ts: Optional[float]) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def scan_results_children() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    res = ROOT / "results"
    if not res.is_dir():
        return rows
    for child in sorted(res.iterdir()):
        if not child.is_dir():
            continue
        fc, tb, mb, mt, exts, fl = walk_stats(child)
        rows.append(
            {
                "path": str(child.relative_to(ROOT)),
                "category": "experiment_results",
                "file_count": fc,
                "total_mb": round(mb, 3),
                "mtime_latest": fmt_time(mt),
                "ext_snapshot": fmt_ext_snapshot(exts),
                **{k: int(v) for k, v in fl.items()},
            }
        )
    return rows


def scan_figures_images() -> List[Dict[str, Any]]:
    """List image-like files under figures/ for manifest."""
    fig = ROOT / "figures"
    rows: List[Dict[str, Any]] = []
    if not fig.is_dir():
        return rows
    image_ext = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".eps", ".tif", ".tiff"}
    likely_script = "scripts/make_paper_figures.py"
    for dirpath, _, filenames in os.walk(fig):
        for fn in filenames:
            suf = Path(fn).suffix.lower()
            if suf not in image_ext:
                continue
            fp = Path(dirpath) / fn
            rel = fp.relative_to(ROOT)
            try:
                st = fp.stat()
            except OSError:
                continue
            note = ""
            rel_s = str(rel).replace("\\", "/")
            if "/softcopyright/" in rel_s:
                likely_script = "scripts/capture_manual_screenshots.py"
                note = "softcopyright UI captures"
            elif "/new_datasets_analysis/" in rel_s:
                likely_script = "scripts/analyze_new_datasets.py"
                note = "extension dataset analysis"
            elif "softcopyright" in rel_s:
                likely_script = "scripts/capture_manual_screenshots.py"
                note = ""
            elif "new_datasets_analysis" in rel_s:
                likely_script = "scripts/analyze_new_datasets.py"
                note = ""

            rows.append(
                {
                    "filename": fn,
                    "relative_path": rel_s,
                    "size_kb": round(st.st_size / 1024, 2),
                    "mtime": fmt_time(st.st_mtime),
                    "likely_source_script": likely_script,
                    "note": note,
                }
            )
    rows.sort(key=lambda r: r["relative_path"])
    return rows


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)

    top_rows: List[Dict[str, Any]] = []
    for p in sorted(ROOT.iterdir(), key=lambda x: x.name.lower()):
        name = p.name
        if name in IGNORE_TOP_INVENTORY:
            continue
        if p.is_dir():
            cat = classify_top_level(name, True)
            fc, tb, mb, mt, exts, fl = walk_stats(p)
            top_rows.append(
                {
                    "path": name + "/",
                    "category": cat,
                    "file_count": fc,
                    "total_mb": round(mb, 3),
                    "mtime_latest": fmt_time(mt),
                    "ext_snapshot": fmt_ext_snapshot(exts),
                    **{k: int(v) for k, v in fl.items()},
                }
            )
        else:
            cat = classify_top_level(name, False)
            if cat == "unknown" and name.endswith(".py"):
                cat = "code_core"
            try:
                st = p.stat()
                sz = st.st_size
                mt = st.st_mtime
            except OSError:
                sz, mt = 0, None
            top_rows.append(
                {
                    "path": name,
                    "category": cat,
                    "file_count": 1,
                    "total_mb": round(sz / (1024 * 1024), 3),
                    "mtime_latest": fmt_time(mt),
                    "ext_snapshot": p.suffix.lower() or "(no_ext)",
                    "has_report_md": 0,
                    "has_manifest_json": 0,
                    "has_figures_subdir": 0,
                    "has_tables_subdir": 0,
                    "has_pth_pt_ckpt": int(name.endswith((".pth", ".pt", ".ckpt"))),
                    "has_video": int(name.endswith((".mp4", ".avi", ".mov"))),
                    "has_numpy_cache": int(name.endswith((".npy", ".npz", ".pkl"))),
                }
            )

    results_detail = scan_results_children()
    fig_manifest = scan_figures_images()

    # Main CSV (top-level + summary)
    fields = [
        "path",
        "category",
        "file_count",
        "total_mb",
        "mtime_latest",
        "ext_snapshot",
        "has_report_md",
        "has_manifest_json",
        "has_figures_subdir",
        "has_tables_subdir",
        "has_pth_pt_ckpt",
        "has_video",
        "has_numpy_cache",
    ]
    write_csv(DOCS / "repository_inventory.csv", fields, top_rows)

    if results_detail:
        with (DOCS / "repository_inventory.csv").open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            for row in results_detail:
                w.writerow(
                    {
                        "path": row["path"],
                        "category": row["category"],
                        "file_count": row["file_count"],
                        "total_mb": row["total_mb"],
                        "mtime_latest": row["mtime_latest"],
                        "ext_snapshot": row["ext_snapshot"],
                        "has_report_md": row["has_report_md"],
                        "has_manifest_json": row["has_manifest_json"],
                        "has_figures_subdir": row["has_figures_subdir"],
                        "has_tables_subdir": row["has_tables_subdir"],
                        "has_pth_pt_ckpt": row["has_pth_pt_ckpt"],
                        "has_video": row["has_video"],
                        "has_numpy_cache": row["has_numpy_cache"],
                    }
                )

    if fig_manifest:
        write_csv(
            DOCS / "generated_figures_manifest.csv",
            ["filename", "relative_path", "size_kb", "mtime", "likely_source_script", "note"],
            fig_manifest,
        )

    # Markdown report
    lines: List[str] = []
    lines.append("# Repository inventory")
    lines.append("")
    lines.append(f"Generated: **{datetime.now().isoformat(timespec='seconds')}** (local time)")
    lines.append("")
    lines.append("This file is **machine-regenerated**. Re-run:")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/inventory_repository.py")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Safety note (historical campaign directories)")
    lines.append("")
    lines.append(
        "The following top-level names are classified as **historical_campaigns**. "
        "**Keep them in place** until you confirm no active script depends on their paths; "
        "do not move into `archive/legacy_campaigns/` without a code audit:"
    )
    lines.append("")
    for n in sorted(HISTORICAL_CAMPAIGNS):
        lines.append(f"- `{n}/`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Category legend")
    lines.append("")
    lines.append("| Tag | Meaning |")
    lines.append("|-----|---------|")
    lines.append("| `code_core` | Application / library / config / docs |")
    lines.append("| `experiment_results` | `results/*`, checkpoints, training outputs |")
    lines.append("| `paper_figures` | Root `figures/` (paper bundle) |")
    lines.append("| `publication_figures` | `figures_publication/` |")
    lines.append("| `software_copyright_materials` | Typically `figures/softcopyright/` (inside `figures/`) |")
    lines.append("| `raw_or_input_data` | Video roots, letter images parent if raw, etc. |")
    lines.append("| `historical_campaigns` | Ad hoc campaign trees — **do not relocate blindly** |")
    lines.append("| `cache_or_temp` | Caches and temp |")
    lines.append("| `experiment_archive` | Local snapshot root (see `scripts/archive_experiment_snapshot.py`) |")
    lines.append("| `unknown` | Review manually |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Top-level scan")
    lines.append("")
    lines.append(
        "| Path | Category | Files | Size (MB) | Latest mtime | report.md | manifest.json | weights | video | cache-like | Top extensions |"
    )
    lines.append("|------|----------|-------|-----------|--------------|-----------|---------------|---------|-------|------------|----------------|")
    for r in top_rows:
        if r["path"].endswith("/"):
            pass
        lines.append(
            f"| `{r['path']}` | {r['category']} | {r['file_count']} | {r['total_mb']} | {r['mtime_latest']} | "
            f"{r['has_report_md']} | {r['has_manifest_json']} | {r['has_pth_pt_ckpt']} | {r['has_video']} | {r['has_numpy_cache']} | {r['ext_snapshot'][:80]} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## `results/` sub-runs (if present)")
    lines.append("")
    if not results_detail:
        lines.append("*No `results/` directory or no subdirectories found.*")
    else:
        lines.append(
            "| Run folder | Files | Size (MB) | Latest mtime | report | manifest | figures dir | tables dir | weights | video | Top extensions |"
        )
        lines.append("|------------|-------|-----------|--------------|--------|----------|-------------|------------|---------|-------|----------------|")
        for row in results_detail:
            lines.append(
                f"| `{row['path']}` | {row['file_count']} | {row['total_mb']} | {row['mtime_latest']} | "
                f"{row['has_report_md']} | {row['has_manifest_json']} | {row['has_figures_subdir']} | {row['has_tables_subdir']} | "
                f"{row['has_pth_pt_ckpt']} | {row['has_video']} | {row['ext_snapshot'][:60]} |"
            )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Figure file manifest")
    lines.append("")
    if fig_manifest:
        lines.append(f"Saved **`docs/generated_figures_manifest.csv`** ({len(fig_manifest)} image files under `figures/`).")
    else:
        lines.append("*No `figures/` tree or no image files found.* See `docs/generated_figures_manifest.csv` after you generate figures.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## See also")
    lines.append("")
    lines.append("- [`output_organization.md`](output_organization.md) — roles of outputs and paper workflow.")
    lines.append("")

    (DOCS / "repository_inventory.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {DOCS / 'repository_inventory.md'}")
    print(f"Wrote {DOCS / 'repository_inventory.csv'}")
    if fig_manifest:
        print(f"Wrote {DOCS / 'generated_figures_manifest.csv'} ({len(fig_manifest)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
