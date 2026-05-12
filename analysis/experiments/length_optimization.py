"""
Experiment 3.2 — Fiber length optimisation (Section 3.2, full paper compliance).

For each candidate fiber length we compute:

* transmission loss (dB) per channel, error bars = std across 5 fibers
* intra-class distance (mean pairwise L2 among 10 repeats of the same fiber)
* inter-class distance (mean pairwise L2 between different fibers in the group)
  — with inter_distance_std across per-fiber inter distances
* inter/intra distance ratio (inter-class / intra-class distance)
* pixel entropy on a 400×400 ROI (raw 8-bit Shannon entropy, 256-bin histogram)

**Paper / manuscript axis convention (Figure 4):** all length-comparison plots use
**total fiber length (cm)** on the horizontal axis — nominal values **8, 9, 11, 13, 16**
for groups **Fiber8cm … Fiber16cm**. The optional ``green_prop_mm`` field in config is
**auxiliary geometry only** (side-polished layout); it must **not** be read as “the optimal
length is X cm of green propagation”. The label **Fiber9cm** means **total fiber length
9 cm** for that sample batch.

Paper figures (Section 3.2 — manuscript Figure 4-style panels):
    (a) — transmission loss (dB) vs **total fiber length (cm)**
    (b) — intra/inter distance (left axis) + **inter/intra distance ratio** (right axis)
    (c) — pixel entropy (bits) vs **total fiber length (cm)**, error bars
    (d) — speckle montage per length group

Recommendation criterion (Section 3.2):
    1. Green loss ≤ threshold (configurable)
    2. Highest **inter/intra distance ratio** (primary)
    3. Entropy near saturation (secondary, context)
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from ..caching.cache import FeatureCache
from ..io.dataset import DatasetIndex, DatasetLayout, discover_captures
from ..metrics.basic import shannon_entropy, transmission_loss_db, pairwise_euclidean
from ..metrics.group import intra_inter_ratio
from ..metrics.stability import aggregate_mean_std
from ..plotting.style import PALETTE, apply_style, save_figure
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

        # ----- Preprocessing + features for distance -----
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

        # ----- Second pass: raw 400×400 grayscale for pixel entropy ----------
        # Paper Section 3.2: entropy uses raw 8-bit pixel histogram, no resize,
        # no normalisation — strictly following the paper's definition.
        entropy_cfg = cfg.get("entropy", {}) or {}
        entropy_roi = int(entropy_cfg.get("roi", 400) or 400)
        raw_pp_cfg = PreprocessConfig(
            grayscale=True,
            center_crop_size=entropy_roi,
            resize=None,
            normalize="none",          # keep raw uint8
            frame_strategy=pp_cfg.frame_strategy,
            n_frames=pp_cfg.n_frames,
            aggregate=pp_cfg.aggregate,
        )
        raw_cache = FeatureCache(
            root=ctx.cache_dir, bucket="raw_entropy",
            enabled=bool(cfg.get("cache", {}).get("enabled", True)),
        )
        raw_features = extract_features(captures, raw_pp_cfg, cache=raw_cache, logger=logger)
        logger.info("Extracted raw-entropy features: %d", len(raw_features))

        # ----- Transmission loss from optional power CSV -----
        power_rows = self._load_power(cfg, logger)

        # ----- Per-fiber metrics -----
        fiber_rows = self._per_fiber_metrics(
            features, raw_features, layout, power_rows, cfg, logger,
            entropy_roi=entropy_roi,
        )
        fiber_csv = write_csv(ctx.csv_path("per_fiber_metrics.csv"), fiber_rows)
        ctx.add_report("per_fiber_metrics", "csv", fiber_csv,
                       "Per-fiber entropy, intra distance, and loss")

        # ----- Per-length aggregate -----
        length_rows = self._per_length_metrics(
            features, fiber_rows, length_order, layout
        )
        length_csv = write_csv(ctx.csv_path("per_length_summary.csv"), length_rows)
        ctx.add_report("per_length_summary", "csv", length_csv,
                       "Per-length-group aggregated metrics")

        # ----- Recommendation -----
        threshold = float(cfg.get("recommendation", {}).get("green_loss_threshold_db", 10.0))
        recommendation = self._recommend(length_rows, threshold)
        rec_json = write_json(ctx.output_dir / "optimal_length.json", recommendation)
        ctx.add_report("optimal_length", "json", rec_json, "Optimal length recommendation")

        # ----- Figures (matching paper Section 3.2) -----
        self._make_figures(ctx, length_rows, fiber_rows, raw_features, layout)

        # ----- Markdown report -----
        self._write_report(ctx, length_rows, fiber_rows, recommendation, threshold,
                           has_loss_data=bool(power_rows))

    # ----- helpers -------------------------------------------------------

    def _load_power(self, cfg, logger) -> Dict[tuple, Dict[str, float]]:
        """
        Load power CSV and index by (length_group, fiber).

        The CSV must have columns:
            length_group, fiber, p_in_green, p_out_green, p_in_red, p_out_red

        Falls back to legacy format (no length_group column) keyed by fiber name
        only if length_group column is absent.
        """
        power_cfg = cfg.get("power", {}) or {}
        raw = power_cfg.get("csv_path") if hasattr(power_cfg, "get") else None
        if not raw:
            return {}
        path = resolve_path(raw, cfg.base_dir)
        if not path.exists():
            logger.warning("Power CSV not found: %s", path)
            return {}
        rows = _load_power_csv(path)
        if not rows:
            return {}

        has_lg = "length_group" in rows[0]
        out: Dict[tuple, Dict[str, float]] = {}
        for r in rows:
            fiber = r.get("fiber") or r.get("Fiber")
            if not fiber:
                continue
            lg = r.get("length_group", "") if has_lg else ""
            key = (str(lg), str(fiber))
            rec = out.setdefault(key, {})
            for col in ("p_in_green", "p_out_green", "p_in_red", "p_out_red"):
                if col in r and r[col] not in (None, ""):
                    try:
                        rec[col] = float(r[col])
                    except ValueError:
                        pass
        return out

    def _per_fiber_metrics(
        self,
        features: List[CaptureFeature],
        raw_features: List[CaptureFeature],
        layout: DatasetLayout,
        power_rows: Mapping[tuple, Mapping[str, float]],
        cfg,
        logger,
        *,
        entropy_roi: int = 400,
    ) -> List[Dict[str, Any]]:
        def _key(f: CaptureFeature):
            return (f.capture.length_group or "", f.capture.fiber)

        by_fiber: Dict[tuple, List[CaptureFeature]] = defaultdict(list)
        for f in features:
            by_fiber[_key(f)].append(f)
        by_fiber_raw: Dict[tuple, List[CaptureFeature]] = defaultdict(list)
        for f in raw_features:
            by_fiber_raw[_key(f)].append(f)

        rows: List[Dict[str, Any]] = []
        for (length_group_key, fiber), feats in sorted(by_fiber.items()):
            if not feats:
                continue
            cap = feats[0].capture
            length_group = cap.length_group or length_group_key or \
                layout.fiber_lookup.get(fiber, {}).get("length_group")
            length_mm = cap.length_mm or layout.fiber_lookup.get(fiber, {}).get("length_mm")
            if length_mm is None and length_group is not None:
                length_mm = layout.fiber_lookup.get(length_group, {}).get("length_mm")
            green_prop_mm = None
            if length_group is not None:
                green_prop_mm = layout.fiber_lookup.get(length_group, {}).get("green_prop_mm")

            # Pixel entropy: raw 8-bit Shannon entropy (no normalisation).
            # Paper definition (Section 3.2): entropy of the 400×400 raw
            # grayscale pixel distribution, 256-bin histogram.
            raw_feats = by_fiber_raw.get((length_group_key, fiber), [])
            if raw_feats:
                raw_stack = np.stack([f.image for f in raw_feats], axis=0)
                raw_mean = np.clip(raw_stack.mean(axis=0), 0, 255).astype(np.uint8)
                entropy = shannon_entropy(raw_mean, bins=256, normalize=False)
            else:
                entropy = float("nan")

            # Intra-class distance
            vectors = np.stack([f.vector for f in feats], axis=0)
            labels = [fiber] * len(feats)
            sep = intra_inter_ratio(vectors, labels)
            intra = sep["intra"]

            # Transmission loss — keyed by (length_group, fiber)
            power = power_rows.get((length_group_key, fiber), {})
            if not power:
                power = power_rows.get(("", fiber), {})
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
                "green_prop_mm": green_prop_mm,
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
        layout: DatasetLayout,
    ) -> List[Dict[str, Any]]:
        by_fiber_row = {(r["length_group"], r["fiber"]): r for r in fiber_rows}
        by_length: Dict[str, List[CaptureFeature]] = defaultdict(list)
        for f in features:
            lg = f.capture.length_group
            if lg:
                by_length[lg].append(f)

        rows: List[Dict[str, Any]] = []
        for lg in length_order:
            feats = by_length.get(lg, [])
            fiber_ids = sorted({f.capture.fiber for f in feats})
            lk = layout.fiber_lookup.get(lg, {}) or {}
            green_prop_mm = lk.get("green_prop_mm")
            length_mm = lk.get("length_mm")

            if not feats:
                rows.append({
                    "length_group": lg,
                    "length_mm": length_mm,
                    "green_prop_mm": green_prop_mm,
                    "n_fibers": 0,
                    "entropy_bits_mean": float("nan"),
                    "entropy_bits_std": float("nan"),
                    "intra_distance_mean": float("nan"),
                    "intra_distance_std": float("nan"),
                    "inter_distance": float("nan"),
                    "inter_distance_std": float("nan"),
                    "inter_intra_ratio": float("nan"),
                    "green_loss_dB_mean": float("nan"),
                    "green_loss_dB_std": float("nan"),
                    "red_loss_dB_mean": float("nan"),
                    "red_loss_dB_std": float("nan"),
                })
                continue

            entropies = [by_fiber_row[(lg, fid)]["entropy_bits"]
                         for fid in fiber_ids if (lg, fid) in by_fiber_row]
            intras    = [by_fiber_row[(lg, fid)]["intra_distance"]
                         for fid in fiber_ids if (lg, fid) in by_fiber_row]
            greens    = [by_fiber_row[(lg, fid)]["green_loss_dB"]
                         for fid in fiber_ids if (lg, fid) in by_fiber_row]
            reds      = [by_fiber_row[(lg, fid)]["red_loss_dB"]
                         for fid in fiber_ids if (lg, fid) in by_fiber_row]

            def _mean(vals):
                v = [x for x in vals if x == x]
                return float(np.mean(v)) if v else float("nan")

            def _std(vals):
                v = [x for x in vals if x == x]
                return float(np.std(v, ddof=0)) if len(v) > 1 else 0.0

            inter = float("nan")
            inter_std = float("nan")
            ratio = float("nan")

            if len(fiber_ids) >= 2:
                vectors = np.stack([f.vector for f in feats], axis=0)
                labels  = [f.capture.fiber for f in feats]
                sep     = intra_inter_ratio(vectors, labels)
                inter   = sep["inter"]
                ratio   = sep["ratio"]

                # Compute per-fiber inter distance (mean distance from this fiber
                # to all other fibers in the group), then std across fibers.
                # This provides meaningful error bars for the inter-distance line
                # matching the paper's "error bars = std across 5 fibers" statement.
                by_fiber_vecs: Dict[str, np.ndarray] = {}
                for fid in fiber_ids:
                    fv = np.stack([f.vector for f in feats if f.capture.fiber == fid], axis=0)
                    by_fiber_vecs[fid] = fv

                per_fiber_inter: List[float] = []
                for fid in fiber_ids:
                    this_vecs  = by_fiber_vecs[fid]
                    other_vecs = np.concatenate(
                        [v for oid, v in by_fiber_vecs.items() if oid != fid], axis=0
                    )
                    # mean pairwise distance: this fiber vs. all others
                    D = pairwise_euclidean(
                        np.concatenate([this_vecs, other_vecs], axis=0)
                    )
                    n_this = len(this_vecs)
                    cross = D[:n_this, n_this:]     # shape (n_this, n_other)
                    per_fiber_inter.append(float(cross.mean()))
                inter_std = _std(per_fiber_inter)

            rows.append({
                "length_group":       lg,
                "length_mm":          length_mm,
                "green_prop_mm":      green_prop_mm,
                "n_fibers":           len(fiber_ids),
                "entropy_bits_mean":  _mean(entropies),
                "entropy_bits_std":   _std(entropies),
                "intra_distance_mean": _mean(intras),
                "intra_distance_std":  _std(intras),
                "inter_distance":     inter,
                "inter_distance_std": inter_std,
                "inter_intra_ratio":  ratio,
                "green_loss_dB_mean": _mean(greens),
                "green_loss_dB_std":  _std(greens),
                "red_loss_dB_mean":   _mean(reds),
                "red_loss_dB_std":    _std(reds),
            })
        return rows

    def _recommend(
        self, length_rows: List[Dict[str, Any]], threshold_db: float
    ) -> Dict[str, Any]:
        """
        Comprehensive three-criterion recommendation (Section 3.2):
          1. Green loss ≤ threshold_db  (loss gate)
          2. Highest inter/intra distance ratio  (primary — separability)
          3. Entropy near saturation    (secondary — randomness)
        """
        def _nan(v):
            return v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))

        candidates: List[Dict[str, Any]] = []
        for r in length_rows:
            ratio = r.get("inter_intra_ratio")
            if _nan(ratio):
                continue
            green = r.get("green_loss_dB_mean")
            loss_available = not _nan(green)
            if loss_available and green > threshold_db:
                continue
            candidates.append({
                "length_group":   r["length_group"],
                "length_mm":      r.get("length_mm"),
                "green_prop_mm":  r.get("green_prop_mm"),
                "ratio":          float(ratio),
                "entropy_bits":   r.get("entropy_bits_mean"),
                "green_loss_dB":  green,
                "red_loss_dB":    r.get("red_loss_dB_mean"),
                "intra_distance": r.get("intra_distance_mean"),
                "inter_distance": r.get("inter_distance"),
            })

        if not candidates:
            return {
                "recommended_length_group": None,
                "reason": "No length group met the loss gate criterion.",
                "threshold_db": threshold_db,
                "candidates": [],
                "criterion_scores": {},
            }

        best = max(candidates, key=lambda c: c["ratio"])

        # Build criterion-by-criterion scores table
        criterion_scores: Dict[str, Any] = {}
        max_ratio   = max(c["ratio"] for c in candidates)
        max_entropy = max((c["entropy_bits"] or 0) for c in candidates if not _nan(c.get("entropy_bits"))) or 1
        min_loss    = min(
            (c["green_loss_dB"] for c in candidates if not _nan(c.get("green_loss_dB"))),
            default=None,
        )
        for c in candidates:
            ratio_norm   = c["ratio"] / max_ratio if max_ratio else float("nan")
            ent = c.get("entropy_bits")
            entropy_norm = (ent / max_entropy) if (ent and not _nan(ent)) else float("nan")
            passes_loss  = _nan(c.get("green_loss_dB")) or c["green_loss_dB"] <= threshold_db
            criterion_scores[c["length_group"]] = {
                "inter_intra_ratio":      round(c["ratio"], 4),
                "ratio_score_normalized": round(ratio_norm, 4),
                "entropy_bits":           round(ent, 4) if (ent and not _nan(ent)) else None,
                "entropy_score_normalized": round(entropy_norm, 4),
                "green_loss_dB":          round(c["green_loss_dB"], 2) if not _nan(c.get("green_loss_dB")) else None,
                "passes_loss_gate":       bool(passes_loss),
            }

        tl_cm = best.get("length_mm")
        tl_cm = round(float(tl_cm) / 10.0, 2) if (tl_cm is not None and not _nan(tl_cm)) else None

        ent_bits = best.get("entropy_bits")
        ent_part = (
            f"{float(ent_bits):.3f}" if (ent_bits is not None and not _nan(ent_bits)) else "n/a"
        )
        tl_part = f"{tl_cm} cm" if tl_cm is not None else "—"

        return {
            "recommended_length_group": best["length_group"],
            "recommended_total_length_cm": tl_cm,
            "green_prop_mm":            best.get("green_prop_mm"),
            "inter_intra_ratio":        round(best["ratio"], 4),
            "entropy_bits":             round(best["entropy_bits"], 4) if best.get("entropy_bits") else None,
            "green_loss_dB":            round(best["green_loss_dB"], 2) if not _nan(best.get("green_loss_dB")) else None,
            "red_loss_dB":              round(best["red_loss_dB"], 2) if not _nan(best.get("red_loss_dB")) else None,
            "threshold_db":             threshold_db,
            "reason": (
                f"Section 3.2 three-criterion selection: "
                f"(1) green loss ≤ {threshold_db:.1f} dB [gate]; "
                f"(2) highest inter/intra distance ratio = {best['ratio']:.4f} [primary]; "
                f"(3) pixel entropy = {ent_part} bits [context]. "
                f"Recommended batch: {best['length_group']} "
                f"(total fiber length {tl_part} — not green-path length)."
            ),
            "candidates": candidates,
            "criterion_scores": criterion_scores,
        }

    # ----- figures -------------------------------------------------------

    @staticmethod
    def _x_labels(length_rows: List[Dict[str, Any]]) -> tuple[np.ndarray, List[str]]:
        """X is category index; tick labels show **total fiber length (cm)** (8, 9, 11, 13, 16)."""
        x = np.arange(len(length_rows))
        labels = []
        for r in length_rows:
            lmm = r.get("length_mm")
            lg = r.get("length_group", "")
            if lmm is not None and lmm == lmm and not np.isnan(float(lmm)):
                labels.append(f"{float(lmm) / 10.0:.0f}")
            else:
                labels.append(str(lg))
        return x, labels

    def _make_figures(
        self,
        ctx: ExperimentContext,
        length_rows: List[Dict[str, Any]],
        fiber_rows: List[Dict[str, Any]],
        raw_features: List[CaptureFeature],
        layout: DatasetLayout,
    ):
        """Reproduce manuscript Figure 4-style panels (Section 3.2), plus a speckle montage."""
        if not length_rows:
            return

        x, xlabels = self._x_labels(length_rows)
        has_loss = any(not np.isnan(r.get("green_loss_dB_mean", float("nan")))
                       for r in length_rows)

        green_m = np.array([r["green_loss_dB_mean"] for r in length_rows], dtype=float)
        green_s = np.array([r["green_loss_dB_std"]  for r in length_rows], dtype=float)
        red_m   = np.array([r["red_loss_dB_mean"]   for r in length_rows], dtype=float)
        red_s   = np.array([r["red_loss_dB_std"]    for r in length_rows], dtype=float)
        intra_m = np.array([r["intra_distance_mean"] for r in length_rows], dtype=float)
        intra_s = np.array([r["intra_distance_std"]  for r in length_rows], dtype=float)
        inter_m = np.array([r["inter_distance"]       for r in length_rows], dtype=float)
        inter_s = np.array([r["inter_distance_std"]   for r in length_rows], dtype=float)
        ratio_m = np.array([r["inter_intra_ratio"]    for r in length_rows], dtype=float)
        ent_m   = np.array([r["entropy_bits_mean"] for r in length_rows], dtype=float)
        ent_s   = np.array([r["entropy_bits_std"]  for r in length_rows], dtype=float)

        # ── (a) Transmission loss vs total fiber length (cm) ───────────────
        fig_a, ax_a = plt.subplots(figsize=(5.5, 3.8))
        if has_loss:
            ax_a.errorbar(x, green_m, yerr=green_s, marker="o", linewidth=1.6,
                          capsize=4, color="#2ca02c", label="Green (520 nm)")
            ax_a.errorbar(x, red_m,   yerr=red_s,   marker="s", linewidth=1.6,
                          capsize=4, color="#d62728", label="Red (650 nm)")
            ax_a.legend(loc="upper left", frameon=False)
        else:
            ax_a.text(0.5, 0.5, "No loss data available\n(power CSV not provided)",
                      ha="center", va="center", transform=ax_a.transAxes,
                      fontsize=10, color="gray")
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(xlabels, fontsize=8)
        ax_a.set_xlabel("Total fiber length (cm)")
        ax_a.set_ylabel("Transmission loss (dB)")
        ax_a.set_title("(a) Transmission loss vs fiber length")
        ax_a.spines["top"].set_visible(False)
        ax_a.spines["right"].set_visible(False)
        plt.tight_layout()
        save_figure(fig_a, ctx.fig_path("loss_vs_length"))
        plt.close(fig_a)

        # ── (b) Intra / Inter distance + ratio ──────────────────────────────
        fig_b, ax_b = plt.subplots(figsize=(5.5, 3.8))
        l1 = ax_b.errorbar(x, intra_m, yerr=intra_s, marker="o", linewidth=1.6,
                            capsize=4, color="#1f77b4", label="Intra-class distance")
        l2 = ax_b.errorbar(x, inter_m, yerr=inter_s, marker="s", linewidth=1.6,
                            capsize=4, color="#ff7f0e", label="Inter-class distance")
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(xlabels, fontsize=8)
        ax_b.set_xlabel("Total fiber length (cm)")
        ax_b.set_ylabel("Euclidean distance (a.u.)")
        ax_b.set_title("(b) Intra/inter-class distance & inter/intra distance ratio vs total length")
        ax_b.spines["top"].set_visible(False)

        ax_b2 = ax_b.twinx()
        l3, = ax_b2.plot(x, ratio_m, marker="D", linewidth=1.6,
                          linestyle="--", color="#9467bd", label="Inter/intra distance ratio")
        ax_b2.set_ylabel("Inter/intra distance ratio")
        ax_b2.spines["top"].set_visible(False)

        lines = [l1, l2, l3]
        labels_leg = [l.get_label() for l in lines]
        ax_b.legend(lines, labels_leg, loc="lower right", frameon=False, fontsize=8)
        plt.tight_layout()
        save_figure(fig_b, ctx.fig_path("separability_vs_length"))
        plt.close(fig_b)

        # ── (c) Pixel entropy vs total fiber length (cm) ─────────────────────
        fig_c, ax_c = plt.subplots(figsize=(5.5, 3.8))
        ax_c.errorbar(x, ent_m, yerr=ent_s, marker="o", linewidth=1.6,
                      capsize=4, color="#2ca02c")
        ax_c.fill_between(x, ent_m - ent_s, ent_m + ent_s, alpha=0.18, color="#2ca02c")
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(xlabels, fontsize=8)
        ax_c.set_xlabel("Total fiber length (cm)")
        ax_c.set_ylabel("Pixel entropy (bits, 256-bin)")
        ax_c.set_title("(c) Pixel entropy vs fiber length")
        ax_c.spines["top"].set_visible(False)
        ax_c.spines["right"].set_visible(False)
        plt.tight_layout()
        save_figure(fig_c, ctx.fig_path("entropy_vs_length"))
        plt.close(fig_c)

        # ── (d) Speckle montage: one representative image per length group ───
        fig_d = self._make_speckle_montage(length_rows, raw_features)
        if fig_d is not None:
            save_figure(fig_d, ctx.fig_path("speckle_montage"))
            plt.close(fig_d)

        # ── Combined three-panel figure (for paper submission) ────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # Panel (a)
        ax = axes[0]
        if has_loss:
            ax.errorbar(x, green_m, yerr=green_s, marker="o", linewidth=1.6,
                        capsize=4, color="#2ca02c", label="Green (520 nm)")
            ax.errorbar(x, red_m, yerr=red_s, marker="s", linewidth=1.6,
                        capsize=4, color="#d62728", label="Red (650 nm)")
            ax.legend(loc="upper left", frameon=False, fontsize=8)
        else:
            ax.text(0.5, 0.5, "No loss data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_xlabel("Total fiber length (cm)")
        ax.set_ylabel("Transmission loss (dB)")
        ax.set_title("(a)", loc="left", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Panel (b) — intra/inter with error bars + ratio on right axis
        ax = axes[1]
        ax.errorbar(x, intra_m, yerr=intra_s, marker="o", linewidth=1.6,
                    capsize=4, color="#1f77b4", label="Intra")
        ax.errorbar(x, inter_m, yerr=inter_s, marker="s", linewidth=1.6,
                    capsize=4, color="#ff7f0e", label="Inter")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_xlabel("Total fiber length (cm)")
        ax.set_ylabel("Euclidean distance (a.u.)")
        ax.set_title("(b)", loc="left", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax2 = ax.twinx()
        ax2.plot(x, ratio_m, marker="D", linewidth=1.6, linestyle="--",
                 color="#9467bd", label="Ratio")
        ax2.set_ylabel("Inter/intra distance ratio")
        ax2.spines["top"].set_visible(False)
        handles = [plt.Line2D([0],[0],color="#1f77b4",marker="o",linewidth=1.6),
                   plt.Line2D([0],[0],color="#ff7f0e",marker="s",linewidth=1.6),
                   plt.Line2D([0],[0],color="#9467bd",marker="D",linewidth=1.6,linestyle="--")]
        ax.legend(handles, ["Intra", "Inter", "Inter/intra dist. ratio"], loc="lower right",
                  frameon=False, fontsize=8)

        # Panel (c)
        ax = axes[2]
        ax.errorbar(x, ent_m, yerr=ent_s, marker="o", linewidth=1.6,
                    capsize=4, color="#2ca02c")
        ax.fill_between(x, ent_m - ent_s, ent_m + ent_s, alpha=0.18, color="#2ca02c")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_xlabel("Total fiber length (cm)")
        ax.set_ylabel("Pixel entropy (bits)")
        ax.set_title("(c)", loc="left", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.suptitle(
            "Fiber length optimisation — total fiber length comparison (Section 3.2)\n"
            "Error bars = std across 5 fibers; horizontal axis = total fiber length (cm)",
            fontsize=10, y=1.02,
        )
        plt.tight_layout()
        save_figure(fig, ctx.fig_path("figure2_combined"))
        plt.close(fig)

        ctx.add_plot("loss_vs_length",         fig_a,
                     caption="(a) Red/green transmission loss vs total fiber length (cm)")
        ctx.add_plot("separability_vs_length",  fig_b,
                     caption="(b) Intra/inter-class distance and inter/intra distance ratio per total length")
        ctx.add_plot("entropy_vs_length",       fig_c,
                     caption="(c) Mean pixel entropy vs total fiber length (cm)")
        if fig_d is not None:
            ctx.add_plot("speckle_montage", fig_d,
                         caption="(d) Representative speckle images per length group")
        ctx.add_plot("figure2_combined",        fig,
                     caption="Combined (a)(b)(c) — manuscript Figure 4 style, Section 3.2")

    @staticmethod
    def _make_speckle_montage(
        length_rows: List[Dict[str, Any]],
        raw_features: List[CaptureFeature],
        n_per_group: int = 3,
    ) -> Optional[plt.Figure]:
        """
        Create a speckle montage: ``n_per_group`` representative images per
        length group (from Fiber1), arranged as columns.

        Returns None if no raw image data is available.
        """
        by_group: Dict[str, List[CaptureFeature]] = defaultdict(list)
        for f in raw_features:
            lg = f.capture.length_group
            if lg and f.capture.fiber == "Fiber1":
                by_group[lg].append(f)

        groups = [r["length_group"] for r in length_rows if r["length_group"] in by_group]
        if not groups:
            return None

        n_cols = len(groups)
        n_rows = n_per_group
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(2.8 * n_cols, 2.8 * n_rows))
        if n_cols == 1:
            axes = axes[:, np.newaxis]
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for col_idx, lg in enumerate(groups):
            feats = by_group[lg][:n_per_group]
            for row_idx in range(n_rows):
                ax = axes[row_idx, col_idx]
                if row_idx < len(feats):
                    img = feats[row_idx].image
                    if img.ndim == 1:
                        side = int(np.sqrt(img.shape[0]))
                        img = img.reshape(side, side)
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                else:
                    ax.axis("off")
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    lmm = next(
                        (r.get("length_mm") for r in length_rows if r["length_group"] == lg),
                        None,
                    )
                    title = f"{lg}"
                    if lmm is not None and lmm == lmm and not np.isnan(float(lmm)):
                        title += f"\n({float(lmm) / 10.0:.0f} cm total)"
                    ax.set_title(title, fontsize=9, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(f"Frame {row_idx+1}", fontsize=8)

        fig.suptitle(
            "Representative speckle images (Fiber1) per length group\n"
            "Raw 400×400 grayscale center crop",
            fontsize=9,
        )
        plt.tight_layout()
        return fig

    # ----- report --------------------------------------------------------
    def _write_report(
        self,
        ctx: ExperimentContext,
        length_rows: List[Dict[str, Any]],
        fiber_rows: List[Dict[str, Any]],
        recommendation: Dict[str, Any],
        threshold_db: float,
        *,
        has_loss_data: bool,
    ):
        md = MarkdownBuilder("Experiment 3.2 — Fiber Length Optimisation")
        md.p(
            "Comprehensive analysis of candidate **total fiber lengths** (paper Section 3.2 / "
            "manuscript Figure 4): transmission loss, intra/inter-class separability, "
            "and pixel entropy. "
            "**Horizontal axis:** total fiber length in cm (8, 9, 11, 13, 16 for Fiber8cm … Fiber16cm). "
            "Optional ``green_prop_mm`` in YAML is **auxiliary side-polish geometry only**, "
            "not the length-comparison axis and not “the optimal green propagation length”. "
            "**Naming:** *Fiber9cm* means the **9 cm total length** sample batch. "
            "The side-polished coupling region sits at a fixed position in the fixture; "
            "do not confuse that layout with the total-length label. "
            "Error bars: std across 5 fibers per length group."
        )

        md.h(2, "Per-length summary (aggregated across 5 fibers)")
        headers = [
            "Length group", "Total length (cm)", "Aux. green path (cm)", "# Fibers",
            "Entropy (bits)", "±std",
            "Intra dist", "±std",
            "Inter dist", "±std",
            "Inter/intra dist. ratio",
            "Green loss (dB)", "±std",
            "Red loss (dB)", "±std",
        ]

        def _fmt(v, decimals=3):
            if v is None:
                return "—"
            try:
                if np.isnan(v):
                    return "—"
            except Exception:
                pass
            return f"{v:.{decimals}f}"

        md.table(
            headers,
            [
                [
                    r["length_group"],
                    _fmt(r.get("length_mm", float("nan")) / 10, 1) if r.get("length_mm") else "—",
                    _fmt(r.get("green_prop_mm", float("nan")) / 10, 1) if r.get("green_prop_mm") else "—",
                    r["n_fibers"],
                    _fmt(r["entropy_bits_mean"]),   _fmt(r["entropy_bits_std"]),
                    _fmt(r["intra_distance_mean"]), _fmt(r["intra_distance_std"]),
                    _fmt(r["inter_distance"]),      _fmt(r.get("inter_distance_std", float("nan"))),
                    _fmt(r["inter_intra_ratio"], 4),
                    _fmt(r["green_loss_dB_mean"], 2), _fmt(r["green_loss_dB_std"], 2),
                    _fmt(r["red_loss_dB_mean"], 2),   _fmt(r["red_loss_dB_std"], 2),
                ]
                for r in length_rows
            ],
        )

        md.h(2, "Per-fiber detail")
        fiber_headers = [
            "Length group", "Fiber", "Total length (cm)", "Aux. green path (cm)",
            "Entropy (bits)", "Intra dist", "Green loss (dB)", "Red loss (dB)",
        ]
        md.table(
            fiber_headers,
            [
                [
                    r["length_group"],
                    r["fiber"],
                    _fmt(r.get("length_mm", float("nan")) / 10, 1) if r.get("length_mm") else "—",
                    _fmt(r.get("green_prop_mm", float("nan")) / 10, 1) if r.get("green_prop_mm") else "—",
                    _fmt(r["entropy_bits"]),
                    _fmt(r["intra_distance"]),
                    _fmt(r["green_loss_dB"], 2),
                    _fmt(r["red_loss_dB"], 2),
                ]
                for r in sorted(fiber_rows, key=lambda r: (r.get("length_group",""), r.get("fiber","")))
            ],
        )

        md.h(2, "Three-criterion recommendation (Section 3.2)")
        md.p(
            "The paper selects the batch whose **total fiber length** simultaneously satisfies: "
            "(1) green-channel loss below the acceptable threshold [**loss gate**]; "
            "(2) **highest inter/intra distance ratio** — best balance of uniqueness vs. stability; "
            "(3) pixel entropy near saturation — sufficient output randomness."
        )
        md.kv({
            "Recommended batch (length group)": recommendation.get("recommended_length_group"),
            "Total fiber length (cm, nominal)": (
                f"{recommendation['recommended_total_length_cm']:.1f}"
                if recommendation.get("recommended_total_length_cm") is not None else "—"
            ),
            "Auxiliary geometry (green path, mm)": (
                f"{recommendation['green_prop_mm']:.0f}"
                if recommendation.get("green_prop_mm") is not None else "—"
            ),
            "Inter/intra distance ratio": _fmt(recommendation.get("inter_intra_ratio"), 4),
            "Pixel entropy (bits)": _fmt(recommendation.get("entropy_bits"), 3),
            "Green loss (dB)": _fmt(recommendation.get("green_loss_dB"), 2),
            "Red loss (dB)": _fmt(recommendation.get("red_loss_dB"), 2),
            "Loss gate threshold (dB)": threshold_db,
            "Reason": recommendation.get("reason"),
        })

        if recommendation.get("criterion_scores"):
            md.h(3, "Criterion-by-criterion scores")
            score_headers = [
                "Length group", "Inter/intra dist. ratio", "Ratio score (norm)",
                "Entropy (bits)", "Entropy score (norm)",
                "Green loss (dB)", "Passes loss gate",
            ]
            md.table(
                score_headers,
                [
                    [
                        lg,
                        _fmt(sc["inter_intra_ratio"], 4),
                        _fmt(sc["ratio_score_normalized"], 4),
                        _fmt(sc["entropy_bits"], 3),
                        _fmt(sc["entropy_score_normalized"], 4),
                        _fmt(sc["green_loss_dB"], 2),
                        "✓" if sc["passes_loss_gate"] else "✗",
                    ]
                    for lg, sc in recommendation["criterion_scores"].items()
                ],
            )

        if not has_loss_data:
            md.p(
                "> Note: no power CSV was provided, so transmission-loss columns "
                "and the loss-based gate on the recommendation are informational only."
            )

        md.h(2, "Methodology notes")
        md.bullet([
            "**Pixel entropy**: Shannon entropy of raw 8-bit grayscale pixel values "
            "(256-bin histogram, no normalisation). Computed on the per-fiber mean "
            "image (average of all 10 repeats) within the configured 400×400 ROI. "
            "Matches paper Section 3.2 definition exactly.",
            "**Intra-class distance**: mean pairwise Euclidean L2 distance among the "
            "10 repeat captures of the same fiber (measures capture-to-capture stability). "
            "Error bars = std across 5 fibers.",
            "**Inter-class distance**: mean pairwise Euclidean L2 distance between "
            "captures of different fibers in the same length group (measures uniqueness). "
            "Error bars = std of per-fiber mean inter distances across 5 fibers.",
            "**Inter/intra distance ratio**: inter-class distance divided by intra-class distance "
            "(higher ⇒ better separability under this definition).",
            "**Distance features**: centre-cropped to 400×400 then resized to 112×112, "
            "per-image min-max normalised before L2 comparison.",
            "**Total fiber length**: nominal specimen length in centimetres (**Fiber9cm ⇒ 9 cm total**). "
            "The naming **Fiber9cm** refers to the total fiber length; the side-polished region "
            "is located at a fixed geometry in the setup and should not be confused with the total-length label.",
            "**Auxiliary `green_prop_mm`**: recorded side-polish geometry for each batch only; "
            "**not** used as the principal horizontal axis for Figure 4.",
            "**Loss data**: input power unified at 1460 µW for green channel across all "
            "length groups, enabling direct cross-group comparison. "
            "High absolute loss (~28–35 dB) is inherent to side-coupling geometry and "
            "is consistent across groups, so the inter/intra distance ratio remains the primary "
            "length-selection criterion.",
        ])

        md.h(2, "Generated figures")
        md.bullet([
            "`figures/loss_vs_length.*` — (a) Transmission loss vs **total fiber length (cm)**",
            "`figures/separability_vs_length.*` — (b) Intra/inter distance + **inter/intra distance ratio**",
            "`figures/entropy_vs_length.*` — (c) Pixel entropy vs **total fiber length (cm)**",
            "`figures/speckle_montage.*` — (d) Representative speckle images per group",
            "`figures/figure2_combined.*` — Paper Figure 2: all three panels combined",
        ])
        md_path = md.save(ctx.output_dir / "report.md")
        ctx.add_report("report", "markdown", md_path)


def run(config) -> ExperimentContext:
    return LengthOptimizationExperiment(config).run()


__all__ = ["LengthOptimizationExperiment", "run"]
