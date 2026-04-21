"""
Experiment 3.2 — Fiber length optimisation.

For each candidate fiber length we compute:

* transmission loss (dB) per channel (requires a power CSV)
* intra-class distance (repeated captures of the same fiber)
* inter-class distance (different fibers within the same length group)
* inter/intra ratio
* pixel entropy on a configurable ROI

Then we emit a recommendation that prefers the largest inter/intra ratio
while keeping green-channel loss under a configurable threshold.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from ..caching.cache import FeatureCache
from ..io.dataset import DatasetIndex, DatasetLayout, discover_captures
from ..metrics.basic import shannon_entropy, transmission_loss_db
from ..metrics.group import intra_inter_ratio
from ..metrics.stability import aggregate_mean_std
from ..plotting.charts import grouped_bars, line_with_error
from ..plotting.style import DOMAIN_COLORS
from ..preprocessing.pipeline import PreprocessConfig
from ..reporting.writers import MarkdownBuilder, write_csv, write_json
from ..utils.config import resolve_path
from ._features import CaptureFeature, extract_features
from .base import BaseExperiment, ExperimentContext


def _load_power_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


class LengthOptimizationExperiment(BaseExperiment):
    name = "length_optimization"

    # ----- top-level execution -------------------------------------------
    def execute(self, ctx: ExperimentContext) -> None:
        cfg = ctx.config
        logger = ctx.logger

        # ----- Dataset -----
        layout = DatasetLayout.from_config(cfg.get("dataset", {}),
                                           base_dir=cfg.base_dir)
        captures = discover_captures(layout)
        ctx.captures = captures
        logger.info("Captures discovered: %d", len(captures))
        index = DatasetIndex(captures, layout=layout)

        length_order = list(cfg.get("length_groups_order", []) or index.length_groups())
        if not length_order:
            raise RuntimeError("No length groups present in dataset (check fiber_lookup)")

        # ----- Preprocessing + features -----
        pp_cfg = PreprocessConfig.from_dict(cfg.get("preprocess", {}).to_dict()
                                             if hasattr(cfg.get("preprocess", {}), "to_dict")
                                             else dict(cfg.get("preprocess", {})))
        cache = FeatureCache(
            root=ctx.cache_dir,
            bucket="features",
            enabled=bool(cfg.get("cache", {}).get("enabled", True)),
        )
        features = extract_features(captures, pp_cfg, cache=cache, logger=logger)
        logger.info("Extracted features: %d", len(features))

        # ----- Transmission loss from optional power CSV -----
        power_rows = self._load_power(cfg, logger)

        # ----- Per-fiber metrics -----
        fiber_rows = self._per_fiber_metrics(features, layout, power_rows, cfg, logger)
        fiber_csv = write_csv(ctx.csv_path("per_fiber_metrics.csv"), fiber_rows)
        ctx.add_report("per_fiber_metrics", "csv", fiber_csv,
                       "Per-fiber entropy, intra distance, and loss")

        # ----- Per-length aggregate -----
        length_rows = self._per_length_metrics(features, fiber_rows, length_order)
        length_csv = write_csv(ctx.csv_path("per_length_summary.csv"), length_rows)
        ctx.add_report("per_length_summary", "csv", length_csv,
                       "Per-length-group aggregated metrics")

        # ----- Recommendation -----
        threshold = float(cfg.get("recommendation", {}).get("green_loss_threshold_db", 10.0))
        recommendation = self._recommend(length_rows, threshold)
        rec_json = write_json(ctx.output_dir / "optimal_length.json", recommendation)
        ctx.add_report("optimal_length", "json", rec_json, "Optimal length recommendation")

        # ----- Figures -----
        self._make_figures(ctx, length_rows)

        # ----- Markdown report -----
        self._write_report(ctx, length_rows, recommendation, threshold,
                           has_loss_data=bool(power_rows))

    # ----- helpers -------------------------------------------------------
    def _load_power(self, cfg, logger) -> Dict[str, Dict[str, float]]:
        power_cfg = cfg.get("power", {}) or {}
        raw = power_cfg.get("csv_path") if hasattr(power_cfg, "get") else None
        if not raw:
            return {}
        path = resolve_path(raw, cfg.base_dir)
        if not path.exists():
            logger.warning("Power CSV not found: %s", path)
            return {}
        rows = _load_power_csv(path)
        out: Dict[str, Dict[str, float]] = {}
        for r in rows:
            fiber = r.get("fiber") or r.get("Fiber")
            if not fiber:
                continue
            rec = out.setdefault(str(fiber), {})
            for key in ("p_in_green", "p_out_green", "p_in_red", "p_out_red"):
                if key in r and r[key] not in (None, ""):
                    try:
                        rec[key] = float(r[key])
                    except ValueError:
                        pass
        return out

    def _per_fiber_metrics(
        self,
        features: List[CaptureFeature],
        layout: DatasetLayout,
        power_rows: Mapping[str, Mapping[str, float]],
        cfg,
        logger,
    ) -> List[Dict[str, Any]]:
        entropy_cfg = cfg.get("entropy", {}) or {}
        roi = int(entropy_cfg.get("roi", 0) or 0)

        by_fiber: Dict[str, List[CaptureFeature]] = defaultdict(list)
        for f in features:
            by_fiber[f.capture.fiber].append(f)

        rows: List[Dict[str, Any]] = []
        for fiber, feats in sorted(by_fiber.items()):
            if not feats:
                continue
            cap = feats[0].capture
            length_group = cap.length_group or layout.fiber_lookup.get(fiber, {}).get("length_group")
            length_mm = cap.length_mm or layout.fiber_lookup.get(fiber, {}).get("length_mm")

            # Entropy over mean image (ROI-cropped if requested)
            images = [f.image for f in feats]
            mean_img = np.mean(np.stack(images, axis=0), axis=0)
            if roi and roi < min(mean_img.shape):
                h, w = mean_img.shape
                y0 = (h - roi) // 2
                x0 = (w - roi) // 2
                mean_img_roi = mean_img[y0:y0 + roi, x0:x0 + roi]
            else:
                mean_img_roi = mean_img
            entropy = shannon_entropy(mean_img_roi)

            # Intra class distance: repeated challenges of the same fiber
            vectors = np.stack([f.vector for f in feats], axis=0)
            labels = [fiber] * len(feats)
            sep = intra_inter_ratio(vectors, labels)
            intra = sep["intra"]

            power = power_rows.get(fiber, {})
            green_loss = (
                transmission_loss_db(power.get("p_in_green"), power.get("p_out_green"))
                if power else float("nan")
            )
            red_loss = (
                transmission_loss_db(power.get("p_in_red"), power.get("p_out_red"))
                if power else float("nan")
            )

            rows.append({
                "fiber": fiber,
                "length_group": length_group,
                "length_mm": length_mm,
                "n_captures": len(feats),
                "entropy_bits": entropy,
                "intra_distance": intra,
                "green_loss_dB": green_loss,
                "red_loss_dB": red_loss,
            })

        return rows

    def _per_length_metrics(
        self,
        features: List[CaptureFeature],
        fiber_rows: List[Dict[str, Any]],
        length_order: List[str],
    ) -> List[Dict[str, Any]]:
        by_fiber_row = {r["fiber"]: r for r in fiber_rows}
        by_length: Dict[str, List[CaptureFeature]] = defaultdict(list)
        for f in features:
            lg = f.capture.length_group
            if lg:
                by_length[lg].append(f)

        rows: List[Dict[str, Any]] = []
        for lg in length_order:
            feats = by_length.get(lg, [])
            fiber_ids = sorted({f.capture.fiber for f in feats})
            if not feats:
                rows.append({
                    "length_group": lg,
                    "n_fibers": 0,
                    "entropy_bits_mean": float("nan"),
                    "intra_distance_mean": float("nan"),
                    "inter_distance": float("nan"),
                    "inter_intra_ratio": float("nan"),
                    "green_loss_dB_mean": float("nan"),
                    "red_loss_dB_mean": float("nan"),
                })
                continue

            entropies = [by_fiber_row[fid]["entropy_bits"] for fid in fiber_ids
                         if fid in by_fiber_row]
            intras = [by_fiber_row[fid]["intra_distance"] for fid in fiber_ids
                      if fid in by_fiber_row]
            greens = [by_fiber_row[fid]["green_loss_dB"] for fid in fiber_ids
                      if fid in by_fiber_row]
            reds = [by_fiber_row[fid]["red_loss_dB"] for fid in fiber_ids
                    if fid in by_fiber_row]

            if len(fiber_ids) >= 2:
                vectors = np.stack([f.vector for f in feats], axis=0)
                labels = [f.capture.fiber for f in feats]
                sep = intra_inter_ratio(vectors, labels)
                inter = sep["inter"]
                ratio = sep["ratio"]
            else:
                inter = float("nan")
                ratio = float("nan")

            rows.append({
                "length_group": lg,
                "n_fibers": len(fiber_ids),
                "entropy_bits_mean": aggregate_mean_std(entropies)["mean"],
                "intra_distance_mean": aggregate_mean_std(intras)["mean"],
                "inter_distance": inter,
                "inter_intra_ratio": ratio,
                "green_loss_dB_mean": aggregate_mean_std(greens)["mean"],
                "red_loss_dB_mean": aggregate_mean_std(reds)["mean"],
            })
        return rows

    def _recommend(
        self, length_rows: List[Dict[str, Any]], threshold_db: float
    ) -> Dict[str, Any]:
        candidates: List[Dict[str, Any]] = []
        for r in length_rows:
            ratio = r.get("inter_intra_ratio")
            if ratio is None or (isinstance(ratio, float) and (np.isnan(ratio) or np.isinf(ratio))):
                continue
            green = r.get("green_loss_dB_mean")
            if green is not None and not (isinstance(green, float) and np.isnan(green)) and green > threshold_db:
                continue
            candidates.append({
                "length_group": r["length_group"],
                "ratio": float(ratio),
                "entropy": r.get("entropy_bits_mean"),
                "green_loss": green,
            })
        if not candidates:
            return {
                "recommended_length_group": None,
                "reason": "No length group met the criteria.",
                "threshold_db": threshold_db,
                "candidates": [],
            }
        best = max(candidates, key=lambda c: c["ratio"])
        return {
            "recommended_length_group": best["length_group"],
            "inter_intra_ratio": best["ratio"],
            "entropy_bits": best["entropy"],
            "green_loss_dB": best["green_loss"],
            "threshold_db": threshold_db,
            "reason": (
                "Highest inter/intra ratio among groups whose green-channel "
                f"loss is <= {threshold_db:.1f} dB."
            ),
            "candidates": candidates,
        }

    # ----- figures -------------------------------------------------------
    def _make_figures(self, ctx: ExperimentContext, length_rows: List[Dict[str, Any]]):
        groups = [r["length_group"] for r in length_rows]
        if not groups:
            return

        # Loss vs length
        green = [r["green_loss_dB_mean"] for r in length_rows]
        red = [r["red_loss_dB_mean"] for r in length_rows]
        fig, ax = grouped_bars(
            groups,
            {"Green loss (dB)": green, "Red loss (dB)": red},
            ylabel="Transmission loss (dB)",
            xlabel="Fiber length group",
            title="Transmission loss vs length",
            colors={"Green loss (dB)": DOMAIN_COLORS["Green"],
                    "Red loss (dB)": DOMAIN_COLORS["RedChange"]},
        )
        ctx.add_plot("loss_vs_length", fig, caption="Red/green transmission loss per length group")

        # Separability vs length
        intra = [r["intra_distance_mean"] for r in length_rows]
        inter = [r["inter_distance"] for r in length_rows]
        ratio = [r["inter_intra_ratio"] for r in length_rows]
        fig, ax = line_with_error(
            list(range(len(groups))),
            {
                "Intra distance": (intra, None),
                "Inter distance": (inter, None),
                "Inter / Intra": (ratio, None),
            },
            xlabel="Fiber length group",
            ylabel="Distance (a.u.) / ratio",
            title="Intra / Inter / ratio vs length",
        )
        ax.set_xticks(list(range(len(groups))))
        ax.set_xticklabels(groups)
        ctx.add_plot("separability_vs_length", fig,
                     caption="Intra, inter distance and their ratio per length group")

        # Entropy vs length
        entropies = [r["entropy_bits_mean"] for r in length_rows]
        fig, ax = line_with_error(
            list(range(len(groups))),
            {"Entropy (bits)": (entropies, None)},
            xlabel="Fiber length group",
            ylabel="Pixel entropy (bits)",
            title="Entropy vs length",
        )
        ax.set_xticks(list(range(len(groups))))
        ax.set_xticklabels(groups)
        ctx.add_plot("entropy_vs_length", fig,
                     caption="Mean ROI pixel entropy per length group")

    # ----- report --------------------------------------------------------
    def _write_report(
        self,
        ctx: ExperimentContext,
        length_rows: List[Dict[str, Any]],
        recommendation: Dict[str, Any],
        threshold_db: float,
        *,
        has_loss_data: bool,
    ):
        md = MarkdownBuilder("Experiment 3.2 — Fiber Length Optimisation")
        md.p(
            "Evaluation of candidate fiber lengths on three axes: transmission loss, "
            "intra/inter class separability, and pixel entropy."
        )
        md.h(2, "Per-length summary")
        headers = [
            "Length group", "# Fibers", "Entropy (bits)", "Intra distance", "Inter distance",
            "Inter/Intra", "Green loss (dB)", "Red loss (dB)",
        ]
        md.table(
            headers,
            [
                [
                    r["length_group"], r["n_fibers"], r["entropy_bits_mean"],
                    r["intra_distance_mean"], r["inter_distance"], r["inter_intra_ratio"],
                    r["green_loss_dB_mean"], r["red_loss_dB_mean"],
                ]
                for r in length_rows
            ],
        )

        md.h(2, "Recommended length")
        md.kv(recommendation)
        if not has_loss_data:
            md.p(
                "> Note: no power CSV was provided, so transmission-loss columns "
                "and the loss-based gate on the recommendation are informational only."
            )

        md.h(2, "Figures")
        md.bullet([
            "`figures/loss_vs_length.png`",
            "`figures/separability_vs_length.png`",
            "`figures/entropy_vs_length.png`",
        ])
        md_path = md.save(ctx.output_dir / "report.md")
        ctx.add_report("report", "markdown", md_path)


def run(config) -> ExperimentContext:
    return LengthOptimizationExperiment(config).run()


__all__ = ["LengthOptimizationExperiment", "run"]
