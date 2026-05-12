#!/usr/bin/env python3
"""Generate docs/figure_audit_report.csv from paper_assets/INDEX.csv + repository heuristics."""
from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "paper_assets" / "INDEX.csv"
DOCS = ROOT / "docs"
LO_GREEN_CSV = ROOT / "results" / "length_optimization_green" / "tables" / "per_length_summary.csv"
LO_OPTIMAL = ROOT / "results" / "length_optimization_green" / "optimal_length.json"
AUTH_MATRIX = ROOT / "results" / "fiber_auth" / "auth_matrix.json"
METRICS_JSON = ROOT / "figures" / "new_datasets_analysis" / "metrics_summary.json"
LEGACY_LEN = ROOT / "results" / "length_optimize_current" / "summary.json"


def img_size_px(path: Path) -> str:
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            return f"{im.width}x{im.height}"
    except Exception:
        return ""


def sha16(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except Exception:
        return ""


def infer_source_data_type(source_data_file: str, data_status: str) -> str:
    """File-kind hint for CSV column ``source_data_type`` (distinct from verification ``data_status``)."""
    s = (source_data_file or "").lower()
    if not s or s in ("n/a", "none"):
        return "none"
    if "summary" in s and ("proposal" in s or "manuscript" in s):
        return "manuscript_summary"
    if "videocapture" in s or ".avi" in s:
        return "video_frames"
    if ";" in s or " + " in s:
        return "mixed_paths"
    if s.endswith(".csv"):
        return "csv"
    if s.endswith(".json"):
        return "json"
    if ".csv" in s and ".json" in s:
        return "csv+json"
    if "/" in s and not s.endswith((".csv", ".json", ".yaml", ".yml")):
        return "directory+jpeg_or_media"
    if data_status == "generated_without_data":
        return "schematic_or_ui_capture"
    return "other_or_unspecified"


def audit_path(src: Path, bundle: str, fmt: str) -> Tuple[str, str, str, str, str, str, str, str, str, str, str]:
    """
    Returns: lid, section, caption_cand, script, sdata, data_status, is_outdated,
             problem_level, problem_desc, recommended_action, dup_hint
    """
    p = src.as_posix()
    name = src.name

    dup_hint = ""
    if "publication_fig03_length_optimization" in p or "publication_fig04_length_optimization" in p:
        if fmt in ("png", "pdf", "svg"):
            dup_hint = "group_pub_lenopt_fig03_fig04_same_raster" if fmt == "png" else "group_pub_lenopt_fig03_fig04_vector"
        return (
            "publication_length_optimization_4panel",
            "Journal pack — NOT 策划书 Fig 4",
            "Montage (a) + entropy (b) + L2 bars (c) + separate ratio (d); footnote may list optimum text",
            "scripts/make_publication_figures.py::fig03_length_optimization",
            str(LO_GREEN_CSV) if LO_GREEN_CSV.is_file() else "length summary missing",
            "raw_data_verified" if LO_GREEN_CSV.is_file() else "missing_data",
            "no",
            "warning",
            "Four-panel layout and naming collide with 策划书 triple-panel Fig 4; png fig03 == png fig04 (duplicate export).",
            "archive_only",
            dup_hint,
        )
    if "publication_fig04_cross_fiber_auth" in p:
        return (
            "publication_cross_fiber_auth",
            "3.3 recognition performance",
            "Cross-fiber / same-fiber accuracy matrix + scatter + histograms",
            "scripts/make_publication_figures.py::fig04_cross_fiber_auth",
            str(AUTH_MATRIX),
            "raw_data_verified" if AUTH_MATRIX.is_file() else "missing_data",
            "no",
            "OK",
            "Excluded from 策划书 图7/8 per brief; PR-style auth figure.",
            "use_after_minor_style_fix",
            "",
        )
    if "/figures_competition/fig4_length_optimization" in p:
        return (
            "fig4_length_optimization",
            "3.2(1) 策划书 — fiber length optimization",
            "Triple: transmission loss; intra/inter + twin ratio; pixel entropy vs total length",
            "scripts/generate_competition_figures.py::fig4_length_optimization",
            f"{LO_GREEN_CSV}; {LO_OPTIMAL}",
            "raw_data_verified" if LO_GREEN_CSV.is_file() else "missing_data",
            "no",
            serious_if_no_lo(),
            "Uses official per_length_summary.csv; aligns 28.39 / 1.5653 / 6.183 when CSV present.",
            "use_after_minor_style_fix",
            "",
        )
    if "/figures_competition/fig7_dual_channel" in p:
        return (
            "fig7_dual_channel_characterization",
            "3.2(3) 策划书 — dual-channel characterization",
            "(a) red vs green NCC; (b) bend summary; (c) radial proxy",
            "scripts/generate_competition_figures.py::fig7_dual_channel",
            "long_term_stability/ + videocapture + manuscript summaries (see manifest)",
            "mixed",
            "no",
            "warning",
            "Red bar partly summary; (b) summary-only 5%/25%; (c) video channel proxy — not dual-channel experiment CSV.",
            "use_after_minor_style_fix",
            "",
        )
    if "/figures_competition/fig8_common_mode" in p:
        return (
            "fig8_common_mode_suppression",
            "3.2 策划书 — common-mode suppression narrative",
            "CV bars 38.2% vs 4.3%; reinstall NCC +28% story",
            "scripts/generate_competition_figures.py::fig8_common_mode",
            "manuscript summary constants (no power_common_mode CSV in script)",
            "summary_statistics_verified",
            "no",
            "minor",
            "Numerics from proposal; not recomputed from common_mode experiment outputs in-repo.",
            "use_after_minor_style_fix",
            "",
        )
    if "publication_fig05_dual_channel" in p:
        return (
            "publication_dual_channel_metrics_grid",
            "Support / methods — metrics explorer",
            "Power sweeps + NCC violins from metrics_summary.json",
            "scripts/make_publication_figures.py::fig05_dual_channel_robustness",
            str(METRICS_JSON),
            "raw_data_verified" if METRICS_JSON.is_file() else "missing_data",
            "no",
            "warning",
            "Different layout from 策划书 fig7; uses pooled metrics JSON.",
            "archive_only",
            "",
        )
    if "publication_fig02" in p:
        return (
            "publication_speckle_response",
            "2–3 illustrative speckle",
            "Representative speckle panels from videocapture",
            "scripts/make_publication_figures.py::fig02_speckle_response",
            "videocapture (see script LENGTH_IMAGE paths)",
            "raw_data_verified",
            "no",
            "OK",
            "",
            "use_after_minor_style_fix",
            "",
        )
    if "supplementary_fig_s1_ncc_hd" in p or "supplementary_fig_s2_reinstallation" in p:
        return (
            name.rsplit(".", 1)[0],
            "Supplementary",
            "NCC / reinstall robustness from publication script",
            "scripts/make_publication_figures.py",
            str(METRICS_JSON),
            "raw_data_verified" if METRICS_JSON.is_file() else "summary_statistics_verified",
            "no",
            "minor",
            "",
            "use_after_minor_style_fix",
            "",
        )
    if "fig_auth_matrix" in p or "fig_auth_gap" in p or "fig_auth_scores" in p:
        return (
            Path(name).stem,
            "3.3 authentication / performance",
            "Auth matrix, gap, score distributions",
            "scripts/make_paper_figures.py",
            str(AUTH_MATRIX),
            "raw_data_verified" if AUTH_MATRIX.is_file() else "source_unclear",
            "no",
            "OK",
            "Do not place in 3.2; rename if confused with fig7/8.",
            "use_after_minor_style_fix",
            "",
        )
    if "fig_training_curves" in p or "fig_test_accuracy" in p:
        return (
            Path(name).stem,
            "3.3 training & evaluation",
            "Learning curves / accuracy summaries",
            "train_eval.py + make_paper_figures.py",
            "results/fiber*/training_log.csv (typical)",
            "source_unclear",
            "no",
            "minor",
            "Confirm which training run maps to each export.",
            "need_user_confirmation",
            "",
        )
    if "fig_green_length" in p or "fig_green_length" in name:
        return (
            Path(name).stem,
            "3.2 legacy — partial green length pipeline",
            "Entropy / intra-inter / montage from older partial runner",
            "scripts/run_partial_length_analysis.py / regenerate_figures",
            str(LEGACY_LEN)
            if LEGACY_LEN.is_file()
            else "results/length_optimize_current or green_partial",
            "legacy_or_old_version",
            "yes",
            "serious",
            "Risk of 1.5779 / 6.8369 if sourced from length_optimize_current; not LO_GREEN.",
            "archive_only",
            "",
        )
    if "long_term_stability_analysis" in p:
        return (
            "analysis_long_term_stability",
            "Supporting (dataset characterization)",
            "Per-fiber temporal NCC from long_term_stability JPEGs",
            "scripts/analyze_new_datasets.py::fig_long_term",
            str(ROOT / "long_term_stability"),
            "raw_data_verified",
            "no",
            "OK",
            "Green-only flat layout; informs but ≠ dual-channel fig7.",
            "use_as_final",
            "",
        )
    if "disturbance_sensitivity_analysis" in p:
        return (
            "analysis_disturbance_sensitivity",
            "Supporting",
            "Within-fiber NCC bars — pooled disturbance dataset",
            "scripts/analyze_new_datasets.py::fig_disturbance",
            str(ROOT / "disturbance_sensitivity"),
            "raw_data_verified",
            "no",
            "OK",
            "Not micro-bend before/after panel required for 策划书 fig7(b).",
            "archive_only",
            "",
        )
    if "power_common_mode_analysis" in p:
        return (
            "analysis_power_common_mode_folder",
            "Supporting",
            "Inter-intra vs pump tag plots",
            "scripts/analyze_new_datasets.py::fig_power",
            str(ROOT / "power_common_mode"),
            "raw_data_verified",
            "no",
            "minor",
            "Different from 策划书 fig8 summary bars (38.2/4.3%).",
            "archive_only",
            "",
        )
    if "patent/" in p or "figures_patent_" in p:
        return (
            Path(name).stem,
            "Patent figures",
            "Optical path / system schematic assets",
            "manual / scripts under figures/patent",
            str(src),
            "generated_without_data",
            "no",
            "minor",
            "fig3_system_setup candidate: optical_path_clean.png / fig1_system.png — author must verify beam directions vs brief.",
            "need_user_confirmation",
            "",
        )
    if "softcopyright" in p:
        return (
            Path(name).stem,
            "Softcopyright GUI capture",
            "Screenshot",
            "capture_manual_screenshots.py",
            str(src),
            "generated_without_data",
            "no",
            "minor",
            "",
            "do_not_use",
            "",
        )
    if "results_lo_green_" in p or name == "per_length_summary.csv":
        return (
            "data_per_length_summary_green",
            "3.2 data — Fig 4",
            "",
            "analysis/experiments/length_optimization.py",
            str(LO_GREEN_CSV),
            "raw_data_verified",
            "no",
            "OK",
            "",
            "use_as_final",
            "",
        )
    if "per_fiber_metrics" in name:
        return (
            "data_per_fiber_metrics_green",
            "3.2 data — Fig 4 supporting",
            "",
            "analysis/experiments/length_optimization.py",
            str(LO_GREEN_CSV.parent / "per_fiber_metrics.csv"),
            "raw_data_verified",
            "no",
            "OK",
            "",
            "use_as_final",
            "",
        )
    if "competition_manifest" in name or "/figures_competition/manifest.csv" in p:
        return (
            "figures_competition_manifest",
            "provenance",
            "Links fig4/7/8 to data roots",
            "scripts/generate_competition_figures.py",
            str(ROOT / "figures_competition" / "manifest.csv"),
            "raw_data_verified",
            "no",
            "OK",
            "",
            "use_as_final",
            "",
        )
    if "fig_same_fiber" in p or "fig_speckle_examples" in p or "fig_ncc_hd" in p:
        return (
            Path(name).stem,
            "3.x supporting illustration / correlation",
            "",
            "scripts/make_paper_figures.py",
            str(AUTH_MATRIX),
            "source_unclear",
            "no",
            "minor",
            "",
            "need_user_confirmation",
            "",
        )
    # default
    return (
        Path(name).stem if src.suffix else str(src),
        "unmapped",
        "",
        "",
        str(src),
        "source_unclear",
        "no",
        "minor",
        "Classify manually",
        "need_user_confirmation",
        "",
    )


def serious_if_no_lo() -> str:
    return "serious" if not LO_GREEN_CSV.is_file() else "OK"


def main() -> None:
    if not INDEX.is_file():
        print("Missing", INDEX)
        return

    # PNG duplicate hash groups (length optimisation family)
    hmap: Dict[str, List[str]] = defaultdict(list)
    for cand in [
        ROOT / "figures_publication" / "publication_fig03_length_optimization.png",
        ROOT / "figures_publication" / "publication_fig04_length_optimization.png",
        ROOT / "figures_competition" / "fig4_length_optimization.png",
    ]:
        if cand.is_file():
            hmap[sha16(cand)].append(cand.name)

    rows: List[Dict[str, str]] = []
    with open(INDEX, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            src = Path(r["source_path"])
            bundle = r.get("source_bundle", "")
            fmt = (r.get("format") or "").lower() or src.suffix.lower().lstrip(".")
            rel = src.relative_to(ROOT).as_posix() if src.is_file() else r["source_path"]

            sz, fs_kb, mt = "", "", ""
            if src.is_file():
                st = src.stat()
                fs_kb = f"{st.st_size / 1024:.1f}"
                mt = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                if src.suffix.lower() == ".png":
                    sz = img_size_px(src)

            (
                lid,
                section,
                cap,
                script,
                sdata,
                dstatus,
                outdated,
                plevel,
                prob,
                action,
                dup_hint,
            ) = audit_path(src, bundle, fmt)

            is_dup = "no"
            dup_gid = ""
            if src.suffix.lower() == ".png" and src.is_file():
                h = sha16(src)
                files_same = hmap.get(h, [])
                if len(files_same) > 1:
                    if "publication_fig03" in src.name and "publication_fig04" in " ".join(files_same):
                        is_dup = "yes"
                        dup_gid = f"content_duplicate_sha256_{h}_pub03_pub04"
                    elif "fig4_length_optimization" in src.name:
                        dup_gid = f"unique_{h}_competition_fig4"

            if dup_hint.startswith("group_pub"):
                is_dup = "same_figure_formats" if "svg" in dup_hint or fmt != "png" else is_dup
                dup_gid = dup_gid or "publication_fig03_fig04_length_optimization_pair"

            sdt = infer_source_data_type(sdata, dstatus)
            rows.append(
                {
                    "figure_file": src.name if src.suffix else "",
                    "relative_path": rel,
                    "format": fmt,
                    "image_size_px": sz,
                    "file_size_kb": fs_kb,
                    "mtime": mt,
                    "likely_figure_id": lid,
                    "section_in_manuscript": section,
                    "caption_candidate": cap,
                    "source_script": script,
                    "source_data_file": sdata[:600],
                    "source_data_type": sdt,
                    "data_status": dstatus,
                    "is_duplicate": is_dup,
                    "duplicate_group_id": dup_gid,
                    "is_outdated": outdated,
                    "problem_level": plevel,
                    "problem_description": prob,
                    "recommended_action": action,
                }
            )

    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "figure_audit_report.csv"
    fields = list(rows[0].keys()) if rows else []
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("Wrote", out, "n=", len(rows))


if __name__ == "__main__":
    main()
