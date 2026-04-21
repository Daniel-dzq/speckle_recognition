"""
Experiment 3.1 — System setup / acquisition sanity checks.

Builds the run manifest for the acquisition platform:

* validates that every expected (fiber, channel, challenge) triple is present
* reports media kind, frame count and any outliers
* ingests optional power-meter CSV data into the manifest
* emits a Markdown summary + CSVs for the paper appendix.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..io.dataset import DatasetLayout, DatasetIndex, discover_captures
from ..io.video import video_frame_count
from ..reporting.writers import MarkdownBuilder, write_csv, write_json
from ..utils.config import resolve_path
from .base import BaseExperiment, ExperimentContext


def _load_power_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


class SystemSetupExperiment(BaseExperiment):
    name = "system_setup"

    def execute(self, ctx: ExperimentContext) -> None:
        cfg = ctx.config
        logger = ctx.logger
        dataset_cfg = cfg.get("dataset", {})
        layout = DatasetLayout.from_config(dataset_cfg, base_dir=cfg.base_dir)
        logger.info("Dataset root: %s", layout.root)
        captures = discover_captures(layout)
        logger.info("Discovered %d captures", len(captures))
        ctx.captures = captures
        index = DatasetIndex(captures, layout=layout)

        acq = cfg.get("acquisition", {}) or {}
        expected_challenges = list(acq.get("expected_challenges", []) or [])
        expected_fibers = list(acq.get("expected_fibers", []) or [])
        min_frames = int(acq.get("min_frames", 0) or 0)

        # ----- per-capture manifest -----
        rows: List[Dict[str, Any]] = []
        warnings: List[str] = []
        for cap in captures:
            try:
                nf = video_frame_count(cap.path) if cap.media_kind == "video" else 1
            except Exception as exc:
                nf = -1
                warnings.append(f"{cap.path}: frame count failed ({exc})")
            row = {
                "fiber": cap.fiber,
                "channel": cap.channel,
                "challenge": cap.challenge,
                "condition": cap.condition,
                "length_group": cap.length_group,
                "media_kind": cap.media_kind,
                "n_frames": nf,
                "path": str(cap.path),
            }
            if min_frames and nf >= 0 and nf < min_frames:
                warnings.append(
                    f"{cap.path}: only {nf} frames (< min_frames={min_frames})"
                )
            rows.append(row)

        # ----- missing triples -----
        missing: List[str] = []
        missing_ch: Dict[str, List[str]] = {}
        if expected_challenges and expected_fibers:
            for fiber in expected_fibers:
                present = {c.challenge for c in captures if c.fiber == fiber}
                missing_for_fiber = sorted(set(expected_challenges) - present)
                if missing_for_fiber:
                    missing_ch[fiber] = missing_for_fiber
                    missing.extend(f"{fiber}:{ch}" for ch in missing_for_fiber)

        # ----- optional power CSV -----
        power_cfg = cfg.get("power", {}) or {}
        power_entries: List[Dict[str, Any]] = []
        if isinstance(power_cfg, Mapping):
            raw_path = power_cfg.get("csv_path")
        else:
            raw_path = getattr(power_cfg, "get", lambda *_: None)("csv_path")
        if raw_path:
            csv_path = resolve_path(raw_path, cfg.base_dir)
            if csv_path.exists():
                try:
                    power_entries = _load_power_csv(csv_path)
                    logger.info("Loaded %d power measurements from %s",
                                len(power_entries), csv_path)
                except Exception as exc:
                    warnings.append(f"Failed to read power CSV {csv_path}: {exc}")
            else:
                warnings.append(f"Power CSV not found: {csv_path}")

        # ----- outputs -----
        manifest_csv = write_csv(ctx.csv_path("captures_manifest.csv"), rows)
        ctx.add_report("captures_manifest", "csv", manifest_csv,
                       "Full per-capture acquisition manifest")

        summary = {
            "n_captures": len(captures),
            "n_fibers": len(index.fibers()),
            "n_channels": len(index.channels()),
            "n_challenges": len(index.challenges()),
            "fibers": index.fibers(),
            "channels": index.channels(),
            "challenges": index.challenges(),
            "length_groups": index.length_groups(),
            "conditions": index.conditions(),
            "missing": missing_ch,
            "warnings": warnings,
            "power_entries": len(power_entries),
        }
        summary_json = write_json(ctx.json_path("summary.json"), summary)
        ctx.add_report("summary", "json", summary_json, "Acquisition summary")

        # ----- Markdown report -----
        md = MarkdownBuilder("Experiment 3.1 — System Setup / Acquisition")
        md.p("Audit of the experimental dataset on disk: presence of every "
             "expected capture, frame counts, and optional input-power metadata.")
        md.h(2, "Dataset summary")
        md.kv({
            "Captures": len(captures),
            "Fibers": ", ".join(index.fibers()),
            "Channels": ", ".join(index.channels()),
            "Challenges": f"{len(index.challenges())} ({', '.join(index.challenges()[:6])}{'...' if len(index.challenges()) > 6 else ''})",
            "Length groups": ", ".join(index.length_groups()) or "—",
            "Conditions": ", ".join(index.conditions()) or "—",
            "Dataset root": str(layout.root),
        })
        if missing_ch:
            md.h(2, "Missing captures")
            md.table(
                ["Fiber", "Missing challenges"],
                [[f, ", ".join(ch)] for f, ch in sorted(missing_ch.items())],
            )
        else:
            md.h(2, "Missing captures")
            md.p("No gaps detected against the expected challenge/fiber grid.")
        if warnings:
            md.h(2, "Warnings")
            md.bullet(warnings)
        if power_entries:
            md.h(2, "Input-power metadata")
            md.p(f"{len(power_entries)} rows ingested. Fields: "
                 + ", ".join(sorted({k for e in power_entries for k in e.keys()})))

        md.h(2, "Artefacts")
        md.bullet([
            f"`tables/{manifest_csv.name}` — per-capture frame counts",
            f"`summary.json` — compact acquisition summary",
        ])
        md_path = md.save(ctx.output_dir / "report.md")
        ctx.add_report("report", "markdown", md_path)


def run(config) -> ExperimentContext:
    return SystemSetupExperiment(config).run()


__all__ = ["SystemSetupExperiment", "run"]
