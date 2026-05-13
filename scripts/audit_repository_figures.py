#!/usr/bin/env python3
"""
Scan the repository for figures and data-adjacent artifacts; write docs/figure_audit_report.md
and docs/figure_audit_inventory.csv.

Usage (repo root):
    python3 scripts/audit_repository_figures.py
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT_CSV = DOCS / "figure_audit_inventory.csv"

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svg", ".pdf"}
DATA_EXT = {".csv", ".json", ".txt", ".md", ".npy", ".npz"}
SKIP_DIR_NAMES = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
}
SKIP_PATH_PARTS = {
    "results/length_optimization_green/cache",
    "results/green_partial_32",
    ".analysis_cache",
}

# Known figure-related generators (heuristic linkage)
SCRIPT_FIGure_HINTS = [
    ("scripts/make_paper_figures.py", ["figures/fig_", "figures/fig_"]),
    ("scripts/make_publication_figures.py", ["figures_publication/"]),
    ("scripts/generate_competition_figures.py", ["figures_competition/"]),
    ("scripts/analyze_new_datasets.py", ["figures/new_datasets_analysis/"]),
    ("scripts/run_partial_length_analysis.py", ["fig_green_length"]),
]


def has_cjk_in_path(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", s))


def classify_figure_role(rel: str) -> str:
    p = rel.lower()
    if "softcopyright" in p or "screenshot" in p or "archive" in p:
        return "gui_debug_or_archive"
    if "letter_images" in p:
        return "challenge_assets"
    if "figures_competition" in p:
        return "competition_planning"
    if "figures_publication" in p or "paper_assets" in p:
        return "publication_pack"
    if "patent" in p:
        return "patent_diagram"
    if "new_datasets_analysis" in p:
        return "dataset_exploratory"
    if "fig_green_length" in p:
        return "legacy_length_partial_pipeline"
    if "fig_auth" in p or "fig_test_accuracy" in p or "fig_training" in p:
        return "paper_performance"
    if "fig_speckle" in p or "fig_same_fiber" in p or "fig_ncc" in p:
        return "paper_illustration"
    return "other"


def training_vs_paper(role: str, rel: str) -> str:
    if role in ("gui_debug_or_archive", "challenge_assets", "legacy_length_partial_pipeline"):
        return "non_paper_or_legacy"
    if role == "dataset_exploratory":
        return "supporting_exploratory"
    if role == "competition_planning":
        return "planning_parallel_to_paper"
    if "paper/paper" in rel.replace("\\", "/"):
        return "paper_ready_target"
    if role in ("publication_pack", "patent_diagram", "paper_performance", "paper_illustration"):
        return "paper_ready_or_near"
    return "review_needed"


def likely_generator(rel: str) -> str:
    for script, hints in SCRIPT_FIGure_HINTS:
        for h in hints:
            if h.rstrip("/") in rel.replace("\\", "/"):
                return script
    if "paper_assets" in rel:
        return "scripts/collect_paper_assets.py"
    return ""


def likely_source_data(rel: str) -> str:
    r = rel.replace("\\", "/")
    if "length" in r and "green_partial" in r:
        return "results/green_partial_32 OR results/length_optimize_current (legacy)"
    if "competition" in r and "fig4" in r:
        return "results/length_optimization_green/tables/per_length_summary.csv"
    if "auth" in r or "accuracy" in r:
        return "results/fiber_auth/auth_matrix.json; results/*/test_predictions.csv"
    if "new_datasets" in r:
        return "long_term_stability/; disturbance_sensitivity/; power_common_mode/"
    if "patent" in r or "optical_path" in r:
        return "hand-authored / Blender render"
    return "unknown"


def deprecated_hint(rel: str, role: str) -> str:
    if role == "legacy_length_partial_pipeline":
        return "LIKELY_DEPRECATED_VS_length_optimization_green"
    if "green_partial_32" in rel or "length_optimize_current" in rel:
        return "deprecated_pipeline_output"
    return ""


def disposition(rel: str, role: str, training_paper: str, dep: str) -> str:
    if dep or role == "legacy_length_partial_pipeline":
        return "archive_old / do_not_use_for_final_length_fig"
    if role == "gui_debug_or_archive":
        return "supplementary_fig8_or_archive"
    if "figures/paper" in rel.replace("\\", "/"):
        return "canonical_paper_output"
    if training_paper == "paper_ready_or_near":
        return "redraw_or_copy_into_figures_paper"
    return "review"


def iter_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        rel_dir = Path(dirpath).relative_to(ROOT).as_posix()
        if any(part in rel_dir for part in SKIP_PATH_PARTS):
            dirnames[:] = []
            continue
        for fn in filenames:
            p = Path(dirpath) / fn
            ext = p.suffix.lower()
            if ext not in IMAGE_EXT:
                continue
            yield p


def scan_data_scripts():
    data_files = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        rel_dir = Path(dirpath).relative_to(ROOT).as_posix()
        if rel_dir.startswith("results/") and "/cache/" in rel_dir:
            continue
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() not in DATA_EXT:
                continue
            rel = p.relative_to(ROOT).as_posix()
            if any(part in rel for part in SKIP_PATH_PARTS):
                continue
            data_files.append(rel)
    return sorted(data_files)


def git_tracked_files() -> set[str] | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return set(line.strip() for line in out.splitlines() if line.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> None:
    tracked = git_tracked_files()
    rows = []
    by_top = defaultdict(list)

    for p in sorted(iter_files(), key=lambda x: str(x)):
        rel = p.relative_to(ROOT).as_posix()
        top = rel.split("/")[0] if "/" in rel else rel
        by_top[top].append(rel)
        st = p.stat()
        role = classify_figure_role(rel)
        tp = training_vs_paper(role, rel)
        dep = deprecated_hint(rel, role)
        rows.append({
            "path": rel,
            "ext": p.suffix.lower(),
            "size_kb": round(st.st_size / 1024, 1),
            "mtime_utc": datetime.utcfromtimestamp(st.st_mtime).replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "figure_role": role,
            "training_vs_paper": tp,
            "cjk_in_path": str(has_cjk_in_path(rel)),
            "likely_generator_script": likely_generator(rel),
            "likely_source_data": likely_source_data(rel),
            "deprecated_hint": dep,
            "recommended_disposition": disposition(rel, role, tp, dep),
            "git_tracked": "" if tracked is None else str(rel in tracked),
        })

    DOCS.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    data_files = scan_data_scripts()
    # grep-like: scripts mentioning savefig / figures
    plot_scripts = []
    for dirpath, _, filenames in os.walk(ROOT / "scripts"):
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            p = Path(dirpath) / fn
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "savefig" in txt or "save_figure" in txt or "figures/" in txt:
                plot_scripts.append(p.relative_to(ROOT).as_posix())

    lines = [
        "# Repository figure audit",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        "- **Images scanned:** " + str(len(rows)) + " files matching " + ", ".join(sorted(IMAGE_EXT)),
        "- **Inventory CSV:** `docs/figure_audit_inventory.csv`",
        "- **Data-like files (csv/json/txt/md/npy/npz) indexed:** " + str(len(data_files)) + " (excluding heavy cache paths)",
        "- **Plot-related scripts found:** " + str(len(plot_scripts)),
        "",
        "## Top-level directories containing images",
        "",
    ]
    for top in sorted(by_top.keys()):
        lines.append(f"- **{top}/** — {len(by_top[top])} file(s)")
    lines.extend([
        "",
        "## Length optimization data consistency (critical)",
        "",
        "**Canonical final length experiment:** `results/length_optimization_green/tables/per_length_summary.csv`",
        "",
        "- Length groups present: **Fiber8cm, Fiber9cm, Fiber11cm, Fiber13cm, Fiber16cm** (total fiber lengths **8–16 cm** scale, not 5/30/45 cm).",
        "- **Do not mix** with `results/green_partial_32/` or `figures/fig_green_length_*` (regeneration_manifest lists `run_partial_length_analysis`).",
        "",
        "## Legacy / suspicious assets",
        "",
        "| Pattern | Issue | Action |",
        "|---------|-------|--------|",
        "| `figures/fig_green_length_*` | Partial green length pipeline | **Moved to** `figures/archive_old/` |",
        "| `图表与实验结果分析报告.md` | Chinese report in figures tree | **Moved to** `figures/archive_old/` |",
        "| `figures/softcopyright/*` | GUI capture | Supplementary Fig. 8 / demo only |",
        "| `figures_competition/*` | Planning triplets | Parallel to paper; regenerate via `figures/paper/` pipeline |",
        "",
        "## Figure disposition summary",
        "",
    ])
    disp_count = defaultdict(int)
    for r in rows:
        disp_count[r["recommended_disposition"]] += 1
    for k, v in sorted(disp_count.items(), key=lambda x: -x[1]):
        lines.append(f"- **{k}:** {v}")
    lines.extend([
        "",
        "## Plot scripts (heuristic)",
        "",
        "```text",
    ])
    for s in sorted(plot_scripts)[:80]:
        lines.append(s)
    if len(plot_scripts) > 80:
        lines.append(f"... ({len(plot_scripts) - 80} more)")
    lines.extend([
        "```",
        "",
        "## TODO: missing data for full journal story",
        "",
        "- **ROC / EER (fiber-level verification):** unified `test_predictions.csv` supports **letter** scores; dedicated genuine–impostor score exports for **fiber identity** may be missing — extend `fiber_auth` eval if needed.",
        "- **7-day stability curve:** no dedicated time-series CSV located under `results/` — needs experiment log or labeled captures.",
        "- **Surface roughness Rq distributions:** no `data/processed/` roughness table found — needs profilometer export.",
        "- **Known- vs unknown-challenge ROC split:** requires explicit protocol labels in predictions export.",
        "",
        "## Chinese text in figures",
        "",
        "- Raster audit requires OCR; **SVG/PDF** should be scanned before submission (see `scripts/paper_figures/sanity.py`).",
        "- Paths may contain CJK: filter `cjk_in_path=true` in the inventory CSV.",
        "",
    ])

    (DOCS / "figure_audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", DOCS / "figure_audit_report.md")
    print("Wrote", OUT_CSV)


if __name__ == "__main__":
    main()
