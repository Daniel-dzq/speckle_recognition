"""
Experiment 3.4 — Common-mode suppression evaluation.

Two robustness axes:

* **Power fluctuation** — with synchronised red/green variation, the raw
  green feature drifts whereas the green/red ratio is approximately
  invariant. We compute the coefficient of variation of both.
* **Mechanical reinstall** — after repeated fiber re-installation the raw
  green feature varies more than the ratio feature. We compute the mean
  within-class NCC for both.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..caching.cache import FeatureCache
from ..io.dataset import DatasetLayout, discover_captures
from ..metrics.basic import coefficient_of_variation, pairwise_ncc
from ..metrics.stability import aggregate_mean_std
from ..plotting.charts import grouped_bars
from ..plotting.style import CHANNEL_COLORS
from ..preprocessing.pipeline import PreprocessConfig
from ..reporting.writers import MarkdownBuilder, write_csv, write_json
from ._features import CaptureFeature, extract_features
from .base import BaseExperiment, ExperimentContext


def _ratio_feature(green: np.ndarray, red: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
    g = np.asarray(green, dtype=np.float32)
    r = np.asarray(red, dtype=np.float32)
    return g / np.clip(r + epsilon, epsilon, None)


def _build_green_red_pairs(
    features: List[CaptureFeature],
) -> Dict[Tuple[str, str, str], Dict[str, CaptureFeature]]:
    """
    Group features by (fiber, challenge, condition+session+repeat) where each
    group has 'green' and 'red' counterparts. Dual-channel channel names are
    expected to contain the colour as a substring.
    """
    groups: Dict[Tuple, Dict[str, CaptureFeature]] = defaultdict(dict)
    for f in features:
        c = f.capture
        colour: Optional[str] = None
        if "green" in c.channel.lower():
            colour = "green"
        elif "red" in c.channel.lower():
            colour = "red"
        if colour is None:
            continue
        key = (c.fiber, c.challenge, c.condition or "", c.session or "", c.repeat or 0)
        groups[key][colour] = f
    return groups


class CommonModeExperiment(BaseExperiment):
    name = "common_mode"

    def execute(self, ctx: ExperimentContext) -> None:
        cfg = ctx.config
        logger = ctx.logger

        layout = DatasetLayout.from_config(cfg.get("dataset", {}),
                                           base_dir=cfg.base_dir)
        captures = discover_captures(layout)
        ctx.captures = captures
        pp_cfg = PreprocessConfig.from_dict(
            cfg.get("preprocess", {}).to_dict()
            if hasattr(cfg.get("preprocess", {}), "to_dict")
            else dict(cfg.get("preprocess", {}))
        )
        cache = FeatureCache(ctx.cache_dir, bucket="features",
                             enabled=bool(cfg.get("cache", {}).get("enabled", True)))
        features = extract_features(captures, pp_cfg, cache=cache, logger=logger)
        logger.info("Features: %d", len(features))

        power_conditions = list(cfg.get("power_fluctuation", {}).get("conditions", []) or [])
        reinstall_conditions = list(cfg.get("reinstall", {}).get("conditions", []) or [])

        power_rows, per_fiber_power = self._analyze_power_fluctuation(
            features, power_conditions, logger
        )
        power_csv = write_csv(ctx.csv_path("power_fluctuation.csv"), power_rows)
        ctx.add_report("power_fluctuation", "csv", power_csv,
                       "CV of green feature vs ratio under synchronised variation")

        reinstall_rows = self._analyze_reinstall(
            features, reinstall_conditions, logger
        )
        reinstall_csv = write_csv(ctx.csv_path("reinstall_robustness.csv"), reinstall_rows)
        ctx.add_report("reinstall_robustness", "csv", reinstall_csv,
                       "Within-class NCC of green vs ratio under reinstall")

        self._make_figures(ctx, power_rows, reinstall_rows)

        reduction = aggregate_mean_std([r["reduction_factor"] for r in power_rows
                                        if np.isfinite(r.get("reduction_factor", np.nan))])["mean"]
        gain = aggregate_mean_std([r["delta_ncc"] for r in reinstall_rows
                                   if np.isfinite(r.get("delta_ncc", np.nan))])["mean"]
        summary = {
            "power_fluctuation": {
                "n_fibers": len(power_rows),
                "mean_cv_green": aggregate_mean_std([r["cv_green"] for r in power_rows])["mean"],
                "mean_cv_ratio": aggregate_mean_std([r["cv_ratio"] for r in power_rows])["mean"],
                "mean_reduction_factor": reduction,
            },
            "reinstall": {
                "n_fibers": len(reinstall_rows),
                "mean_within_ncc_green": aggregate_mean_std(
                    [r["within_ncc_green"] for r in reinstall_rows]
                )["mean"],
                "mean_within_ncc_ratio": aggregate_mean_std(
                    [r["within_ncc_ratio"] for r in reinstall_rows]
                )["mean"],
                "mean_ncc_gain": gain,
            },
        }
        s = write_json(ctx.output_dir / "summary.json", summary)
        ctx.add_report("summary", "json", s)

        md = (
            MarkdownBuilder("Experiment 3.4 — Common-mode Suppression")
            .p("Comparing the raw green feature against the green/red ratio feature "
               "under two shared-disturbance conditions.")
            .h(2, "Power fluctuation robustness")
            .table(
                ["Fiber", "CV (green)", "CV (ratio)", "Reduction factor"],
                [[r["fiber"], r["cv_green"], r["cv_ratio"], r["reduction_factor"]]
                 for r in power_rows],
            )
            .h(2, "Mechanical reinstall robustness")
            .table(
                ["Fiber", "NCC (green)", "NCC (ratio)", "Δ NCC"],
                [[r["fiber"], r["within_ncc_green"], r["within_ncc_ratio"], r["delta_ncc"]]
                 for r in reinstall_rows],
            )
            .h(2, "Take-aways")
            .bullet([
                f"Mean CV reduction factor (green / ratio) = {reduction:.2f}"
                if np.isfinite(reduction) else
                "CV reduction factor could not be computed (insufficient data).",
                f"Mean within-class NCC gain after ratio normalisation = {gain:+.3f}"
                if np.isfinite(gain) else
                "Within-class NCC gain could not be computed (insufficient data).",
            ])
            .h(2, "Figures")
            .bullet([
                "`figures/cv_comparison.png`",
                "`figures/reinstall_comparison.png`",
            ])
        )
        report_path = md.save(ctx.output_dir / "report.md")
        ctx.add_report("report", "markdown", report_path)

    # ----- analyses ------------------------------------------------------
    @staticmethod
    def _analyze_power_fluctuation(
        features: List[CaptureFeature],
        power_conditions: List[str],
        logger,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        per_fiber: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"green": [], "ratio": []})
        groups = _build_green_red_pairs(features)
        by_fiber_challenge: Dict[Tuple[str, str], List[Tuple[CaptureFeature, CaptureFeature]]] = defaultdict(list)
        for (fiber, challenge, condition, _sess, _rep), pair in groups.items():
            if power_conditions and condition not in power_conditions:
                continue
            if "green" not in pair or "red" not in pair:
                continue
            by_fiber_challenge[(fiber, challenge)].append((pair["green"], pair["red"]))

        for (fiber, challenge), items in by_fiber_challenge.items():
            if len(items) < 2:
                continue
            greens = np.stack([it[0].vector for it in items], axis=0)
            reds = np.stack([it[1].vector for it in items], axis=0)
            ratios = _ratio_feature(greens, reds)
            per_fiber[fiber]["green"].append(coefficient_of_variation(greens.mean(axis=1)))
            per_fiber[fiber]["ratio"].append(coefficient_of_variation(ratios.mean(axis=1)))

        rows: List[Dict[str, Any]] = []
        for fiber, data in sorted(per_fiber.items()):
            cv_green = aggregate_mean_std(data["green"])["mean"]
            cv_ratio = aggregate_mean_std(data["ratio"])["mean"]
            reduction = cv_green / cv_ratio if cv_ratio and np.isfinite(cv_ratio) and cv_ratio > 0 else float("nan")
            rows.append({
                "fiber": fiber,
                "n_challenges": len(data["green"]),
                "cv_green": cv_green,
                "cv_ratio": cv_ratio,
                "reduction_factor": reduction,
            })
        return rows, per_fiber

    @staticmethod
    def _analyze_reinstall(
        features: List[CaptureFeature],
        reinstall_conditions: List[str],
        logger,
    ) -> List[Dict[str, Any]]:
        if not reinstall_conditions:
            return []
        groups = _build_green_red_pairs(features)
        by_key: Dict[str, Dict[str, List[Tuple[CaptureFeature, CaptureFeature, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for (fiber, challenge, condition, _sess, _rep), pair in groups.items():
            if condition not in reinstall_conditions:
                continue
            if "green" not in pair or "red" not in pair:
                continue
            by_key[fiber][challenge].append((pair["green"], pair["red"], condition))

        def avg_within(vectors: np.ndarray, labels: List[str]) -> float:
            """Mean within-class pairwise NCC."""
            if vectors.shape[0] < 2:
                return float("nan")
            S = pairwise_ncc(vectors)
            labels_arr = np.asarray(labels)
            N = labels_arr.shape[0]
            iu = np.triu_indices(N, k=1)
            mask = labels_arr[iu[0]] == labels_arr[iu[1]]
            vals = S[iu][mask]
            return float(np.mean(vals)) if vals.size else float("nan")

        rows: List[Dict[str, Any]] = []
        for fiber, by_challenge in sorted(by_key.items()):
            green_vecs = []
            ratio_vecs = []
            challenge_labels = []
            for ch, items in by_challenge.items():
                if len(items) < 2:
                    continue
                for g, r, _cond in items:
                    green_vecs.append(g.vector)
                    ratio_vecs.append(_ratio_feature(g.vector, r.vector))
                    challenge_labels.append(ch)
            if not green_vecs:
                continue
            wg = avg_within(np.stack(green_vecs, axis=0), challenge_labels)
            wr = avg_within(np.stack(ratio_vecs, axis=0), challenge_labels)
            rows.append({
                "fiber": fiber,
                "n_samples": len(green_vecs),
                "n_challenges": len(set(challenge_labels)),
                "within_ncc_green": wg,
                "within_ncc_ratio": wr,
                "delta_ncc": wr - wg if np.isfinite(wg) and np.isfinite(wr) else float("nan"),
            })
        return rows

    @staticmethod
    def _make_figures(
        ctx: ExperimentContext,
        power_rows: List[Dict[str, Any]],
        reinstall_rows: List[Dict[str, Any]],
    ):
        if power_rows:
            fibers = [r["fiber"] for r in power_rows]
            fig, _ = grouped_bars(
                fibers,
                {
                    "Green (raw)": [r["cv_green"] for r in power_rows],
                    "Green/Red ratio": [r["cv_ratio"] for r in power_rows],
                },
                xlabel="Fiber",
                ylabel="Coefficient of variation",
                title="Power fluctuation robustness",
                colors={
                    "Green (raw)": CHANNEL_COLORS.get("green"),
                    "Green/Red ratio": CHANNEL_COLORS.get("ratio"),
                },
            )
            ctx.add_plot("cv_comparison", fig,
                         caption="CV of green vs ratio feature under synchronised variation")

        if reinstall_rows:
            fibers = [r["fiber"] for r in reinstall_rows]
            fig, _ = grouped_bars(
                fibers,
                {
                    "Green (raw)": [r["within_ncc_green"] for r in reinstall_rows],
                    "Green/Red ratio": [r["within_ncc_ratio"] for r in reinstall_rows],
                },
                xlabel="Fiber",
                ylabel="Within-class NCC",
                title="Reinstall robustness",
                colors={
                    "Green (raw)": CHANNEL_COLORS.get("green"),
                    "Green/Red ratio": CHANNEL_COLORS.get("ratio"),
                },
            )
            ctx.add_plot("reinstall_comparison", fig,
                         caption="Within-class NCC of green vs ratio under mechanical reinstall")


def run(config) -> ExperimentContext:
    return CommonModeExperiment(config).run()


__all__ = ["CommonModeExperiment", "run"]
