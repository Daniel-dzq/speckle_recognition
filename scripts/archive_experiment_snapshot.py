#!/usr/bin/env python3
"""
Copy a reproducible snapshot of paper-related outputs into experiment_archive/.

Default is DRY-RUN only. Use --apply to copy files.

Usage:
  python scripts/archive_experiment_snapshot.py --tag puf_paper_current
  python scripts/archive_experiment_snapshot.py --tag puf_paper_current --apply
  python scripts/archive_experiment_snapshot.py --tag full --apply --include-videos --include-models

Does NOT delete or move source data. Copy only (unless blocked by size / policy).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

VIDEO_EXT = frozenset({".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"})
MODEL_EXT = frozenset({".pth", ".pt", ".ckpt"})
SKIP_DIR_NAMES = frozenset({"__pycache__", ".ipynb_checkpoints"})

SCRIPT_SNAPSHOT = [
    "scripts/make_paper_figures.py",
    "scripts/make_publication_figures.py",
    "scripts/run_experiment.py",
    "scripts/fiber_auth_eval.py",
    "scripts/capture_manual_screenshots.py",
    "scripts/generate_soft_ware_manual_revision.py",
]


def run_text(cmd: List[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, text=True, timeout=180)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        return f"# error running {' '.join(cmd)}: {e}\n"


def should_copy_results_child(child: Path) -> bool:
    if not child.is_dir():
        return False
    n = child.name
    if n == "fiber_auth":
        return True
    if re.match(r"(?i)fiber\d+$", n):
        return True
    if (child / "report.md").exists():
        return True
    if (child / "manifest.json").exists():
        return True
    return False


def build_results_subdirs() -> List[Path]:
    res = ROOT / "results"
    if not res.is_dir():
        return []
    return sorted([c for c in res.iterdir() if should_copy_results_child(c)], key=lambda p: p.name.lower())


def skip_reason_for_file(
    size: int,
    ext: str,
    *,
    include_videos: bool,
    include_models: bool,
    large_skip_bytes: int,
) -> Optional[str]:
    if ext in VIDEO_EXT and not include_videos:
        return "video_policy_exclude"
    if ext in MODEL_EXT and not include_models:
        return "model_policy_exclude"
    if size >= large_skip_bytes:
        return "large_file_threshold"
    return None


def iter_files_under(src_root: Path) -> List[Path]:
    out: List[Path] = []
    if src_root.is_file():
        return [src_root]
    for dirpath, dirnames, filenames in os.walk(src_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for fn in filenames:
            if fn == ".DS_Store":
                continue
            out.append(Path(dirpath) / fn)
    return out


def plan_copies(
    tasks: List[Tuple[Path, Path]],
    *,
    include_videos: bool,
    include_models: bool,
    large_skip_bytes: int,
) -> Tuple[List[Tuple[Path, Path]], List[Dict[str, str]]]:
    """
    tasks: (source_abs_path, dest_rel_to_archive_root) where source can be file or directory.
    """
    planned: List[Tuple[Path, Path]] = []
    skipped: List[Dict[str, str]] = []

    for src, dest_prefix in tasks:
        if not src.exists():
            continue
        paths = iter_files_under(src)
        for fp in paths:
            try:
                st = fp.stat()
            except OSError:
                continue
            ext = fp.suffix.lower()
            reason = skip_reason_for_file(
                st.st_size,
                ext,
                include_videos=include_videos,
                include_models=include_models,
                large_skip_bytes=large_skip_bytes,
            )
            if src.is_file():
                rel_from_task_src = Path(src.name)
            else:
                rel_from_task_src = fp.relative_to(src)
            dst_rel = dest_prefix / rel_from_task_src
            try:
                rel_repo = str(fp.relative_to(ROOT))
            except ValueError:
                rel_repo = str(fp)
            if reason:
                skipped.append(
                    {
                        "source_path": rel_repo,
                        "reason": reason,
                        "size_mb": f"{st.st_size / (1024 * 1024):.3f}",
                    }
                )
            else:
                planned.append((fp, dst_rel))
    return planned, skipped


def aggregate_by_topdir(planned: List[Tuple[Path, Path]], dest_base: Path) -> List[Dict[str, str]]:
    agg: Dict[str, List[Tuple[Path, Path]]] = {}
    for src, dst_rel in planned:
        top = dst_rel.parts[0] if dst_rel.parts else "root"
        agg.setdefault(top, []).append((src, dst_rel))

    rows: List[Dict[str, str]] = []
    for top in sorted(agg.keys()):
        items = agg[top]
        total = 0
        mt = 0.0
        for src, _ in items:
            try:
                st = src.stat()
                total += st.st_size
                mt = max(mt, st.st_mtime)
            except OSError:
                pass
        note = ""
        if top == "results":
            note = "fiber_auth, fiber*, runs with report.md or manifest.json"
        rows.append(
            {
                "source_path": f"[aggregate:{top}/]",
                "archive_path": str(dest_base / top),
                "file_count": str(len(items)),
                "total_size_mb": f"{total / (1024 * 1024):.3f}",
                "mtime_latest": datetime.fromtimestamp(mt).isoformat(timespec="seconds") if mt else "",
                "note": note,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive paper-related snapshot (copy-only, dry-run by default).")
    parser.add_argument("--tag", required=True, help="Short label for this snapshot folder")
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=ROOT / "experiment_archive",
        help="Root directory for all snapshots (default: experiment_archive/)",
    )
    parser.add_argument("--apply", action="store_true", help="Actually copy files (default: dry-run only).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only (default when --apply is omitted).",
    )
    parser.add_argument("--include-videos", action="store_true", help="Include video files.")
    parser.add_argument("--include-models", action="store_true", help="Include .pth/.pt/.ckpt files.")
    parser.add_argument("--large-skip-mb", type=float, default=200.0, help="Skip files >= this size (MB).")
    args = parser.parse_args()
    if args.apply and args.dry_run:
        parser.error("Use either --apply or --dry-run, not both.")
    dry_run = not args.apply
    large_bytes = int(args.large_skip_mb * 1024 * 1024)

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    dest_base = (args.archive_root / f"{stamp}_{args.tag}").resolve()

    tasks: List[Tuple[Path, Path]] = []

    for rd in build_results_subdirs():
        tasks.append((rd, Path("results") / rd.name))

    for name, rel in (
        ("figures", Path("figures")),
        ("figures_publication", Path("figures_publication")),
    ):
        p = ROOT / name
        if p.is_dir():
            tasks.append((p, rel))

    if (ROOT / "config").is_dir():
        tasks.append((ROOT / "config", Path("configs_snapshot")))

    for dp in [
        ROOT / "docs" / "output_organization.md",
        ROOT / "docs" / "repository_inventory.md",
        ROOT / "docs" / "repository_inventory.csv",
        ROOT / "docs" / "generated_figures_manifest.csv",
        ROOT / "README.md",
    ]:
        if dp.is_file():
            tasks.append((dp, Path("docs") / dp.name))

    for sp in SCRIPT_SNAPSHOT:
        p = ROOT / sp
        if p.is_file():
            tasks.append((p, Path("scripts_snapshot") / Path(sp).name))

    planned, skipped = plan_copies(
        tasks,
        include_videos=args.include_videos,
        include_models=args.include_models,
        large_skip_bytes=large_bytes,
    )

    print("=" * 72)
    print("Experiment snapshot")
    print(f"  Mode:      {'DRY-RUN' if dry_run else 'APPLY (copy)'}")
    print(f"  Dest:      {dest_base}")
    print(f"  Tag:       {args.tag}")
    print(f"  Videos:   {'included' if args.include_videos else 'EXCLUDED (use --include-videos)'}")
    print(f"  Models:   {'included' if args.include_models else 'EXCLUDED (use --include-models)'}")
    print(f"  Large skip if >= {args.large_skip_mb} MB")
    print("=" * 72)

    rsub = build_results_subdirs()
    print(f"\n`results/` subdirs selected: {len(rsub)}")
    for d in rsub:
        print(f"  - {d.relative_to(ROOT)}")
    print(f"\nPlanned files: {len(planned)}")
    print(f"Skipped files: {len(skipped)}")

    by_top: Dict[str, int] = {}
    by_bytes: Dict[str, int] = {}
    for src, dst_rel in planned:
        top = dst_rel.parts[0] if dst_rel.parts else "root"
        by_top[top] = by_top.get(top, 0) + 1
        by_bytes[top] = by_bytes.get(top, 0) + src.stat().st_size
    print("\nPer top-level folder under archive:")
    for top in sorted(by_top.keys()):
        print(f"  {top}: {by_top[top]} files, {by_bytes[top] / (1024 * 1024):.2f} MB")

    readme_lines = [
        f"# Experiment archive snapshot `{dest_base.name}`",
        "",
        f"- Created (local): `{datetime.now().isoformat(timespec='seconds')}`",
        "- **Models excluded by default** (no `.pth`/`.pt`/`.ckpt` unless you pass `--include-models`).",
        "- **Videos excluded by default** (no `.mp4`/`.avi`/… unless you pass `--include-videos`).",
        "",
        "Restore weights from your machine at e.g. `results/fiber_auth/fiber_models/` or `checkpoints/`.",
        "",
        "## Snapshot contents",
        "",
        "- `results/<selected>/` — `fiber_auth`, `fiber*`, and analysis runs with `report.md` or `manifest.json`",
        "- `figures/` — root paper figure bundle (if present)",
        "- `figures_publication/` — journal re-exports (if present)",
        "- `configs_snapshot/` — copy of `config/`",
        "- `docs/` — key documentation + inventory CSV/Markdown if generated",
        "- `scripts_snapshot/` — plotting / eval entry scripts",
        "",
        "## Provenance",
        "",
        "See `GIT_COMMIT.txt`, `GIT_STATUS.txt`, `ENVIRONMENT.txt`, `copied_paths_manifest.csv`, "
        "`copied_paths_manifest_files.csv`, `skipped_large_files.csv`.",
        "",
    ]
    readme_text = "\n".join(readme_lines)

    if dry_run:
        print("\n[DRY-RUN] No files written. Preview README:")
        print("-" * 40)
        print(readme_text)
        print("-" * 40)
        if skipped and len(skipped) <= 40:
            print("\nSkipped (sample):")
            for row in skipped[:40]:
                print(f"  {row['source_path']}: {row['reason']} ({row['size_mb']} MB)")
        elif skipped:
            print(f"\n{len(skipped)} files would be skipped (policy/size).")
        return 0

    dest_base.mkdir(parents=True, exist_ok=True)
    (dest_base / "README.md").write_text(readme_text, encoding="utf-8")
    (dest_base / "GIT_COMMIT.txt").write_text(run_text(["git", "rev-parse", "HEAD"], ROOT), encoding="utf-8")
    (dest_base / "GIT_STATUS.txt").write_text(run_text(["git", "status", "--short"], ROOT), encoding="utf-8")
    env_chunks = [
        run_text([sys.executable, "--version"], ROOT),
        "\n--- pip freeze ---\n",
        run_text([sys.executable, "-m", "pip", "freeze"], ROOT),
    ]
    if shutil.which("conda"):
        env_chunks.append("\n--- conda env export ---\n")
        env_chunks.append(run_text(["conda", "env", "export"], ROOT))
    (dest_base / "ENVIRONMENT.txt").write_text("".join(env_chunks), encoding="utf-8")

    n_ok = 0
    for src, dst_rel in planned:
        dst = dest_base / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
            n_ok += 1
        except OSError as e:
            print(f"[warn] {src} -> {dst}: {e}", file=sys.stderr)

    manifest_rows = aggregate_by_topdir(planned, dest_base)
    per_file_rows: List[Dict[str, str]] = []
    for src, dst_rel in planned:
        st = src.stat()
        per_file_rows.append(
            {
                "source_path": str(src.relative_to(ROOT)),
                "archive_path": str(dest_base / dst_rel),
                "file_count": "1",
                "total_size_mb": f"{st.st_size / (1024 * 1024):.3f}",
                "mtime_latest": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                "note": "",
            }
        )

    fields = ["source_path", "archive_path", "file_count", "total_size_mb", "mtime_latest", "note"]
    with (dest_base / "copied_paths_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in manifest_rows:
            w.writerow(row)

    with (dest_base / "copied_paths_manifest_files.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in per_file_rows:
            w.writerow(row)

    with (dest_base / "skipped_large_files.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source_path", "reason", "size_mb"])
        w.writeheader()
        for row in skipped:
            w.writerow(row)

    print(f"\nCopied {n_ok} files → {dest_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
