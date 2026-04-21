"""
Experiment 3.3 — Dual-channel characterisation.

Three sub-experiments:

* **Time stability** — repeated captures of the same (fiber, challenge)
  across sessions; report consecutive + first-frame NCC per channel.
* **Perturbation sensitivity** — compare a baseline condition vs a
  perturbed condition, again per channel.
* **Spot / profile comparison** — for each fiber+channel build a mean
  response image, extract the radial profile, fit a Gaussian, summarise
  FWHM / sigma, and render an image panel.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..caching.cache import FeatureCache
from ..io.dataset import DatasetIndex, DatasetLayout, discover_captures
from ..metrics.basic import normalized_cross_correlation
from ..metrics.profile import fit_gaussian_profile, profile_width, radial_intensity_profile
from ..metrics.stability import aggregate_mean_std, temporal_stability_score
from ..plotting.charts import grouped_bars, image_panel
from ..plotting.style import CHANNEL_COLORS
from ..preprocessing.pipeline import PreprocessConfig
from ..reporting.writers import MarkdownBuilder, write_csv, write_json
from ._features import CaptureFeature, extract_features
from .base import BaseExperiment, ExperimentContext


def _group_by(caps, keys: Tuple[str, ...]) -> Dict[Tuple, List]:
    out: Dict[Tuple, List] = defaultdict(list)
    for c in caps:
        out[tuple(getattr(c.capture, k) for k in keys)].append(c)
    return out


def _mean_stack(images: List[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(images, axis=0), axis=0)


class DualChannelExperiment(BaseExperiment):
    name = "dual_channel"

    def execute(self, ctx: ExperimentContext) -> None:
        cfg = ctx.config
        logger = ctx.logger

        ds_cfg = cfg.get("dataset", {})
        layout = DatasetLayout.from_config(ds_cfg, base_dir=cfg.base_dir)
        captures = discover_captures(layout)
        ctx.captures = captures
        logger.info("Captures: %d", len(captures))
        index = DatasetIndex(captures, layout=layout)

        pp_cfg = PreprocessConfig.from_dict(
            cfg.get("preprocess", {}).to_dict()
            if hasattr(cfg.get("preprocess", {}), "to_dict")
            else dict(cfg.get("preprocess", {}))
        )
        cache = FeatureCache(ctx.cache_dir, bucket="features",
                             enabled=bool(cfg.get("cache", {}).get("enabled", True)))
        features = extract_features(captures, pp_cfg, cache=cache, logger=logger)
        logger.info("Features: %d", len(features))

        stability_rows = self._analyze_time_stability(features, logger)
        stability_csv = write_csv(ctx.csv_path("time_stability.csv"), stability_rows)
        ctx.add_report("time_stability", "csv", stability_csv,
                       "Consecutive / first-frame NCC per fiber+channel")

        pert_cfg = cfg.get("perturbation", {}) or {}
        baseline = pert_cfg.get("baseline_condition")
        perturbed = pert_cfg.get("perturbed_condition")
        pert_rows = self._analyze_perturbation(features, baseline, perturbed, logger)
        pert_csv = write_csv(ctx.csv_path("perturbation_sensitivity.csv"), pert_rows)
        ctx.add_report("perturbation_sensitivity", "csv", pert_csv,
                       "Baseline vs perturbed NCC per fiber+channel")

        profile_rows, panel_images, row_labels, col_labels = self._analyze_profiles(features, logger)
        profile_csv = write_csv(ctx.csv_path("profile_summary.csv"), profile_rows)
        ctx.add_report("profile_summary", "csv", profile_csv,
                       "Radial profile widths and Gaussian fits")

        self._make_figures(ctx, stability_rows, pert_rows, profile_rows,
                           panel_images, row_labels, col_labels)

        summary = {
            "stability": {
                "channels": sorted({r["channel"] for r in stability_rows}),
                "mean_consecutive_ncc": aggregate_mean_std(
                    [r["consecutive_ncc"] for r in stability_rows]
                )["mean"],
            },
            "perturbation": {
                "baseline": baseline,
                "perturbed": perturbed,
                "n_pairs": len(pert_rows),
            },
            "profile": {
                "n_rows": len(profile_rows),
            },
        }
        s_path = write_json(ctx.output_dir / "summary.json", summary)
        ctx.add_report("summary", "json", s_path)

        md = (
            MarkdownBuilder("Experiment 3.3 — Dual-channel characterisation")
            .p("Validation of red/green channel behaviour: time stability, "
               "perturbation sensitivity, and output spot profile.")
            .h(2, "Time stability")
            .table(
                ["Fiber", "Channel", "# sessions", "Consecutive NCC", "vs-first NCC"],
                [[r["fiber"], r["channel"], r["n_sessions"],
                  r["consecutive_ncc"], r["vs_first_ncc"]] for r in stability_rows],
            )
            .h(2, "Perturbation sensitivity")
            .table(
                ["Fiber", "Channel", "NCC baseline", "NCC perturbed", "Δ NCC"],
                [[r["fiber"], r["channel"], r["ncc_baseline"], r["ncc_perturbed"],
                  r["delta_ncc"]] for r in pert_rows],
            )
            .h(2, "Profile summary")
            .table(
                ["Fiber", "Channel", "FWHM", "Gaussian σ", "Fit RMSE"],
                [[r["fiber"], r["channel"], r["fwhm"], r["sigma"], r["rmse"]]
                 for r in profile_rows],
            )
            .h(2, "Figures")
            .bullet([
                "`figures/time_stability.png`",
                "`figures/perturbation_sensitivity.png`",
                "`figures/profile_panel.png`",
            ])
        )
        md_path = md.save(ctx.output_dir / "report.md")
        ctx.add_report("report", "markdown", md_path)

    # ----- analyses ------------------------------------------------------
    @staticmethod
    def _analyze_time_stability(features: List[CaptureFeature], logger) -> List[Dict[str, Any]]:
        # Group by (fiber, channel, challenge, repeat key) and order by session/repeat.
        groups = defaultdict(list)
        for f in features:
            key = (f.capture.fiber, f.capture.channel, f.capture.challenge)
            groups[key].append(f)

        rows: List[Dict[str, Any]] = []
        by_fiber_channel: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        by_fiber_channel_first: Dict[Tuple[str, str], List[float]] = defaultdict(list)

        for (fiber, channel, challenge), items in groups.items():
            if len(items) < 2:
                continue
            items = sorted(items, key=lambda f: (f.capture.session or "",
                                                 f.capture.repeat or 0,
                                                 str(f.capture.path)))
            vecs = [f.vector for f in items]
            ts = temporal_stability_score(vecs)
            consec = ts["consecutive_ncc"]
            vs_first = ts["vs_first_ncc"]
            by_fiber_channel[(fiber, channel)].append(consec)
            by_fiber_channel_first[(fiber, channel)].append(vs_first)

        for (fiber, channel), values in by_fiber_channel.items():
            consec_means = aggregate_mean_std(values)
            first_means = aggregate_mean_std(by_fiber_channel_first[(fiber, channel)])
            rows.append({
                "fiber": fiber,
                "channel": channel,
                "n_sessions": consec_means["count"],
                "consecutive_ncc": consec_means["mean"],
                "consecutive_ncc_std": consec_means["std"],
                "vs_first_ncc": first_means["mean"],
            })
        return sorted(rows, key=lambda r: (r["fiber"], r["channel"]))

    @staticmethod
    def _analyze_perturbation(
        features: List[CaptureFeature],
        baseline: Optional[str],
        perturbed: Optional[str],
        logger,
    ) -> List[Dict[str, Any]]:
        if not baseline or not perturbed:
            return []
        by_key: Dict[Tuple[str, str], Dict[str, List[CaptureFeature]]] = defaultdict(
            lambda: {"baseline": [], "perturbed": []}
        )
        for f in features:
            c = f.capture
            if c.condition == baseline:
                by_key[(c.fiber, c.channel)]["baseline"].append(f)
            elif c.condition == perturbed:
                by_key[(c.fiber, c.channel)]["perturbed"].append(f)

        rows: List[Dict[str, Any]] = []
        for (fiber, channel), groups_d in sorted(by_key.items()):
            base = groups_d["baseline"]
            pert = groups_d["perturbed"]
            if len(base) < 1 or len(pert) < 1:
                continue
            by_challenge_base = {b.capture.challenge: b for b in base}
            by_challenge_pert = {p.capture.challenge: p for p in pert}
            # Baseline intra-cluster = mean pairwise NCC across repeats (use challenge repeats)
            if len(base) >= 2:
                base_vecs = np.stack([b.vector for b in base], axis=0)
                nccs = []
                for i in range(len(base_vecs)):
                    for j in range(i + 1, len(base_vecs)):
                        nccs.append(normalized_cross_correlation(base_vecs[i], base_vecs[j]))
                ncc_baseline = float(np.mean(nccs)) if nccs else float("nan")
            else:
                ncc_baseline = float("nan")
            # NCC between matched challenge pairs across conditions
            matched = []
            for ch, b in by_challenge_base.items():
                if ch in by_challenge_pert:
                    matched.append(
                        normalized_cross_correlation(b.vector, by_challenge_pert[ch].vector)
                    )
            ncc_perturbed = float(np.mean(matched)) if matched else float("nan")
            rows.append({
                "fiber": fiber,
                "channel": channel,
                "ncc_baseline": ncc_baseline,
                "ncc_perturbed": ncc_perturbed,
                "delta_ncc": ncc_baseline - ncc_perturbed
                    if (not np.isnan(ncc_baseline) and not np.isnan(ncc_perturbed)) else float("nan"),
                "n_matched_challenges": len(matched),
            })
        return rows

    @staticmethod
    def _analyze_profiles(features: List[CaptureFeature], logger):
        by_key: Dict[Tuple[str, str], List[CaptureFeature]] = defaultdict(list)
        for f in features:
            by_key[(f.capture.fiber, f.capture.channel)].append(f)

        fibers = sorted({f.capture.fiber for f in features})
        channels = sorted({f.capture.channel for f in features})

        rows: List[Dict[str, Any]] = []
        panel_images: List[Optional[np.ndarray]] = []
        for fiber in fibers:
            for channel in channels:
                items = by_key.get((fiber, channel), [])
                if not items:
                    panel_images.append(None)
                    continue
                mean_img = _mean_stack([f.image for f in items])
                panel_images.append(mean_img)
                r, I = radial_intensity_profile(mean_img)
                fwhm = profile_width(r, I, level=0.5)
                fit = fit_gaussian_profile(r, I)
                rows.append({
                    "fiber": fiber,
                    "channel": channel,
                    "n_images": len(items),
                    "fwhm": fwhm,
                    "sigma": fit.get("sigma"),
                    "A": fit.get("A"),
                    "c": fit.get("c"),
                    "rmse": fit.get("rmse"),
                    "fit_success": fit.get("success", False),
                })
        return rows, panel_images, fibers, channels

    # ----- figures -------------------------------------------------------
    @staticmethod
    def _make_figures(
        ctx: ExperimentContext,
        stability_rows: List[Dict[str, Any]],
        pert_rows: List[Dict[str, Any]],
        profile_rows: List[Dict[str, Any]],
        panel_images: List[Optional[np.ndarray]],
        row_labels: List[str],
        col_labels: List[str],
    ):
        # Stability bar chart
        if stability_rows:
            fibers = sorted({r["fiber"] for r in stability_rows})
            channels = sorted({r["channel"] for r in stability_rows})
            data: Dict[str, List[float]] = {}
            for ch in channels:
                vals = []
                for fb in fibers:
                    val = next((r["consecutive_ncc"] for r in stability_rows
                                if r["fiber"] == fb and r["channel"] == ch), float("nan"))
                    vals.append(val)
                data[ch] = vals
            colors = {ch: CHANNEL_COLORS.get(ch) for ch in channels}
            fig, _ = grouped_bars(
                fibers, data,
                xlabel="Fiber", ylabel="Consecutive NCC",
                title="Time stability (consecutive NCC)",
                colors=colors,
            )
            ctx.add_plot("time_stability", fig,
                         caption="Consecutive NCC per fiber and channel")

        # Perturbation grouped bar
        if pert_rows:
            fibers = sorted({r["fiber"] for r in pert_rows})
            by_channel = defaultdict(list)
            for ch in sorted({r["channel"] for r in pert_rows}):
                for fb in fibers:
                    val = next((r["ncc_perturbed"] for r in pert_rows
                                if r["fiber"] == fb and r["channel"] == ch), float("nan"))
                    by_channel[f"{ch} (perturbed)"].append(val)
                for fb in fibers:
                    pass
                before_len = len(by_channel[f"{ch} (perturbed)"])
                base_vals = []
                for fb in fibers:
                    v = next((r["ncc_baseline"] for r in pert_rows
                              if r["fiber"] == fb and r["channel"] == ch), float("nan"))
                    base_vals.append(v)
                by_channel[f"{ch} (baseline)"] = base_vals
            colors = {}
            for k in by_channel:
                ch = k.split(" (")[0]
                colors[k] = CHANNEL_COLORS.get(ch)
            fig, _ = grouped_bars(
                fibers, dict(by_channel),
                xlabel="Fiber", ylabel="NCC",
                title="Perturbation sensitivity (NCC)",
                colors=colors,
            )
            ctx.add_plot("perturbation_sensitivity", fig,
                         caption="Baseline vs perturbed NCC per fiber and channel")

        # Profile image panel
        if panel_images and any(im is not None for im in panel_images):
            # Pad Nones with zero images
            clean = []
            for im in panel_images:
                clean.append(im if im is not None else np.zeros((16, 16)))
            fig, _ = image_panel(
                clean,
                row_labels=row_labels,
                col_labels=col_labels,
                rows=len(row_labels),
                cols=len(col_labels),
                title="Mean response images",
                cmap="magma",
            )
            ctx.add_plot("profile_panel", fig,
                         caption="Representative mean response images (rows=fibers, cols=channels)")


def run(config) -> ExperimentContext:
    return DualChannelExperiment(config).run()


__all__ = ["DualChannelExperiment", "run"]
