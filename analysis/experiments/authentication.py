"""
Experiment 3.5 — Authentication performance.

Supports:

* Multi-class identification with a nearest-template classifier.
* Known-vs-unknown challenge conditioning.
* Verification (genuine vs impostor) scores, ROC, AUC, EER.
* Temporal drift (optional, requires session metadata).
* Hardest-confused fiber pair enumeration.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..caching.cache import FeatureCache
from ..io.dataset import DatasetLayout, discover_captures
from ..metrics.auth import (
    auc_score,
    confusion_matrix,
    equal_error_rate,
    nearest_neighbor_identify,
    roc_curve,
    top_k_accuracy,
)
from ..metrics.basic import normalized_cross_correlation, pairwise_ncc
from ..plotting.charts import heatmap, roc_panel
from ..preprocessing.pipeline import PreprocessConfig
from ..reporting.writers import MarkdownBuilder, write_csv, write_json
from ._features import CaptureFeature, extract_features
from .base import BaseExperiment, ExperimentContext


def _split_enroll_probe(
    features: List[CaptureFeature],
    enroll_ratio: float,
    rng: np.random.Generator,
    strategy: str = "auto",
) -> Tuple[List[CaptureFeature], List[CaptureFeature]]:
    """
    Split features into enrollment (templates) and probe sets.

    Strategies:
        ``stratified``    - per-(fiber, challenge) random split using
                            ``enroll_ratio``. Requires >= 2 samples per cell.
        ``leave_one_out`` - every capture is a probe and the remaining
                            captures form the template pool. Well suited to
                            single-sample-per-cell datasets (e.g. one video
                            per letter per fiber).
        ``auto``          - use stratified whenever at least one cell has
                            > 1 sample, otherwise fall back to leave-one-out.
    """
    groups: Dict[Tuple[str, str], List[CaptureFeature]] = defaultdict(list)
    for f in features:
        groups[(f.capture.fiber, f.capture.challenge)].append(f)

    effective_strategy = strategy
    if strategy == "auto":
        multi_sample = any(len(v) >= 2 for v in groups.values())
        effective_strategy = "stratified" if multi_sample else "leave_one_out"

    if effective_strategy == "leave_one_out":
        return list(features), list(features)

    if effective_strategy != "stratified":
        raise ValueError(f"Unknown split strategy: {strategy!r}")

    enroll: List[CaptureFeature] = []
    probe: List[CaptureFeature] = []
    for items in groups.values():
        if len(items) < 2:
            enroll.extend(items)
            continue
        rng.shuffle(items)
        k = max(1, int(round(len(items) * enroll_ratio)))
        enroll.extend(items[:k])
        probe.extend(items[k:])
    return enroll, probe


class AuthenticationExperiment(BaseExperiment):
    name = "authentication"

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
        if not features:
            raise RuntimeError("No features extracted; check dataset paths.")

        rng = np.random.default_rng(int(cfg.get("seed", 0) or 0))
        split_cfg = cfg.get("split", {}) or {}
        enroll_ratio = float(split_cfg.get("enroll_ratio", 0.5))
        split_strategy = str(split_cfg.get("strategy", "auto"))
        enroll, probes = _split_enroll_probe(
            features, enroll_ratio, rng, strategy=split_strategy,
        )
        leave_one_out = enroll is probes or (
            len(enroll) == len(features) and len(probes) == len(features)
        )
        logger.info(
            "Enroll: %d, probes: %d (mode=%s)",
            len(enroll), len(probes),
            "leave_one_out" if leave_one_out else "stratified",
        )

        # ------------ per-fiber+challenge template set -------------------
        templates: List[CaptureFeature] = enroll
        template_vectors = np.stack([t.vector for t in templates], axis=0)
        template_fibers = [t.capture.fiber for t in templates]
        template_challenges = [t.capture.challenge for t in templates]

        eval_cfg = cfg.get("eval", {}) or {}
        top_k_list = list(eval_cfg.get("top_k", [1, 3, 5]) or [1])

        # ------------ known-challenge identification ---------------------
        known_result = self._identify(
            probes, templates, restrict_challenge=True, top_k_list=top_k_list,
            leave_one_out=leave_one_out,
        ) if eval_cfg.get("known_challenge", True) else None

        # ------------ unknown-challenge identification (fiber only) ------
        unknown_result = self._identify(
            probes, templates, restrict_challenge=False, top_k_list=top_k_list,
            leave_one_out=leave_one_out,
        ) if eval_cfg.get("unknown_challenge", True) else None

        # ------------ confusion matrix (known challenge) -----------------
        if known_result is not None:
            fibers = sorted({t.capture.fiber for t in templates})
            fiber_to_idx = {fb: i for i, fb in enumerate(fibers)}
            y_true = [fiber_to_idx[f] for f in known_result["probe_fibers"]]
            y_pred = [fiber_to_idx[f] for f in known_result["predicted_fibers"]]
            cm = confusion_matrix(y_true, y_pred, len(fibers))
            cm_csv = write_csv(
                ctx.csv_path("confusion_matrix.csv"),
                [dict({"_": fibers[i]}, **{fibers[j]: int(cm[i, j]) for j in range(len(fibers))})
                 for i in range(len(fibers))],
            )
            ctx.add_report("confusion_matrix", "csv", cm_csv,
                           "Row = true fiber, column = predicted fiber")
            fig, _ = heatmap(
                cm, fibers, fibers,
                xlabel="Predicted fiber", ylabel="True fiber",
                cmap="Blues", fmt="d", title="Confusion matrix (known challenge)",
                cbar_label="Count",
            )
            ctx.add_plot("confusion_matrix", fig,
                         caption="Identification confusion matrix (known challenge)")

        # ------------ Verification: genuine vs impostor ------------------
        ver_res = self._verification(probes, templates, leave_one_out=leave_one_out)
        ver_csv = write_csv(ctx.csv_path("verification_scores.csv"),
                            [{"score": s, "label": l} for s, l in
                             zip(ver_res["scores"].tolist(), ver_res["labels"].tolist())])
        ctx.add_report("verification_scores", "csv", ver_csv,
                       "Per-pair NCC scores + genuine/impostor labels")
        if ver_res["n_genuine"] and ver_res["n_impostor"]:
            fig, _ = roc_panel(
                {"All probes": (ver_res["fpr"], ver_res["tpr"], ver_res["auc"])},
                title="Verification ROC",
            )
            ctx.add_plot("roc_verification", fig,
                         caption="ROC curve for the verification task")

        # ------------ Temporal drift (optional) --------------------------
        drift_score = self._temporal_drift(features)

        # ------------ Hardest-confused fiber pairs -----------------------
        hard_pairs = self._hard_pairs(known_result) if known_result else []
        if hard_pairs:
            write_csv(ctx.csv_path("hardest_pairs.csv"), hard_pairs)

        # ------------ Summary --------------------------------------------
        summary = {
            "n_features": len(features),
            "n_enroll": len(enroll),
            "n_probes": len(probes),
            "known_challenge": {
                k: known_result.get(k) for k in
                ("top1_accuracy", "n_probes", *[f"top{kk}_accuracy" for kk in top_k_list])
            } if known_result else None,
            "unknown_challenge": {
                k: unknown_result.get(k) for k in
                ("top1_accuracy", "n_probes", *[f"top{kk}_accuracy" for kk in top_k_list])
            } if unknown_result else None,
            "verification": {
                "auc": ver_res["auc"],
                "eer": ver_res["eer"],
                "eer_threshold": ver_res["eer_threshold"],
                "n_genuine": ver_res["n_genuine"],
                "n_impostor": ver_res["n_impostor"],
            },
            "temporal_drift_mean_ncc": drift_score,
        }
        s_path = write_json(ctx.output_dir / "summary.json", summary)
        ctx.add_report("summary", "json", s_path)

        # ------------ Markdown report ------------------------------------
        md = MarkdownBuilder("Experiment 3.5 — Authentication Performance")
        md.p("Comprehensive NCC-template-based authentication evaluation.")
        if leave_one_out:
            md.p(
                "> **Note:** the dataset contains a single sample per "
                "(fiber, challenge) cell, so leave-one-out mode was used. "
                "Under leave-one-out the `known challenge` configuration "
                "excludes the probe itself, which means — with only one "
                "video per fiber per challenge — there is no genuine "
                "template to match. In this regime, prefer the **unknown "
                "challenge (fiber only)** results for identification "
                "accuracy. For challenge-conditioned identification use "
                "multi-sample acquisitions (e.g. add `session` or "
                "`repeat` axes to the dataset layout)."
            )
        if known_result:
            md.h(2, "Identification — known challenge")
            md.kv({
                "Top-1 accuracy": known_result["top1_accuracy"],
                **{f"Top-{k} accuracy": known_result.get(f"top{k}_accuracy") for k in top_k_list if k > 1},
                "# probes": known_result["n_probes"],
            })
        if unknown_result:
            md.h(2, "Identification — unknown challenge")
            md.kv({
                "Top-1 accuracy (fiber only)": unknown_result["top1_accuracy"],
                **{f"Top-{k} accuracy": unknown_result.get(f"top{k}_accuracy") for k in top_k_list if k > 1},
                "# probes": unknown_result["n_probes"],
            })
        md.h(2, "Verification (genuine vs impostor)")
        md.kv({
            "AUC": ver_res["auc"],
            "EER": ver_res["eer"],
            "Threshold @ EER": ver_res["eer_threshold"],
            "# genuine pairs": ver_res["n_genuine"],
            "# impostor pairs": ver_res["n_impostor"],
        })
        if hard_pairs:
            md.h(2, "Hardest-confused fiber pairs")
            md.table(
                ["Fiber A", "Fiber B", "Mistakes (A→B)", "Mistakes (B→A)"],
                [[p["fiber_a"], p["fiber_b"], p["a_to_b"], p["b_to_a"]] for p in hard_pairs[:10]],
            )
        if drift_score is not None:
            md.h(2, "Temporal robustness")
            md.kv({"Mean NCC across sessions": drift_score})
        md.h(2, "Figures")
        md.bullet([
            "`figures/confusion_matrix.png`",
            "`figures/roc_verification.png`",
        ])
        report_path = md.save(ctx.output_dir / "report.md")
        ctx.add_report("report", "markdown", report_path)

    # ----- analyses ------------------------------------------------------
    @staticmethod
    def _identify(
        probes: List[CaptureFeature],
        templates: List[CaptureFeature],
        *,
        restrict_challenge: bool,
        top_k_list: Sequence[int],
        leave_one_out: bool = False,
    ) -> Dict[str, Any]:
        probe_fibers = [p.capture.fiber for p in probes]
        probe_challenges = [p.capture.challenge for p in probes]
        template_fibers = [t.capture.fiber for t in templates]
        template_ids = [id(t) for t in templates]
        template_vectors = np.stack([t.vector for t in templates], axis=0) if templates else None
        probe_vectors = np.stack([p.vector for p in probes], axis=0) if probes else None
        if probe_vectors is None or template_vectors is None:
            return {"top1_accuracy": float("nan"), "n_probes": 0,
                    "probe_fibers": [], "predicted_fibers": []}

        S = pairwise_ncc(probe_vectors, template_vectors)  # (N_probes, N_templates)
        predicted_fibers: List[str] = []
        correct_topk: Dict[int, int] = {k: 0 for k in top_k_list}
        for i, probe in enumerate(probes):
            if restrict_challenge:
                mask = np.array([t.capture.challenge == probe.capture.challenge for t in templates])
                scores = np.where(mask, S[i], -np.inf)
            else:
                scores = S[i].copy()
            if leave_one_out:
                probe_pid = id(probe)
                self_mask = np.array([tid == probe_pid for tid in template_ids])
                scores = np.where(self_mask, -np.inf, scores)
            # top-k best templates -> vote by fiber
            order = np.argsort(-scores)
            top1_fiber = template_fibers[int(order[0])]
            predicted_fibers.append(top1_fiber)
            for k in top_k_list:
                top_idx = order[:max(1, k)]
                top_fibers = [template_fibers[int(j)] for j in top_idx]
                if probe.capture.fiber in top_fibers:
                    correct_topk[k] += 1

        out = {
            "top1_accuracy": correct_topk[1] / len(probes) if probes else float("nan"),
            "n_probes": len(probes),
            "probe_fibers": probe_fibers,
            "predicted_fibers": predicted_fibers,
        }
        for k in top_k_list:
            out[f"top{k}_accuracy"] = correct_topk[k] / len(probes) if probes else float("nan")
        return out

    @staticmethod
    def _verification(
        probes: List[CaptureFeature],
        templates: List[CaptureFeature],
        *,
        leave_one_out: bool = False,
    ) -> Dict[str, Any]:
        if not probes or not templates:
            return {"scores": np.array([]), "labels": np.array([]),
                    "fpr": np.array([]), "tpr": np.array([]),
                    "auc": float("nan"), "eer": float("nan"),
                    "eer_threshold": float("nan"),
                    "n_genuine": 0, "n_impostor": 0}
        probe_vectors = np.stack([p.vector for p in probes], axis=0)
        template_vectors = np.stack([t.vector for t in templates], axis=0)
        S = pairwise_ncc(probe_vectors, template_vectors)
        scores: List[float] = []
        labels: List[int] = []
        for i, probe in enumerate(probes):
            for j, tpl in enumerate(templates):
                if leave_one_out and probe is tpl:
                    continue
                if probe.capture.challenge != tpl.capture.challenge:
                    continue
                scores.append(float(S[i, j]))
                labels.append(1 if probe.capture.fiber == tpl.capture.fiber else 0)
        scores_arr = np.asarray(scores)
        labels_arr = np.asarray(labels, dtype=np.int64)
        fpr, tpr, thr = roc_curve(scores_arr, labels_arr)
        auc = auc_score(fpr, tpr)
        eer, eer_thr = equal_error_rate(fpr, tpr, thr)
        return {
            "scores": scores_arr,
            "labels": labels_arr,
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc,
            "eer": eer,
            "eer_threshold": eer_thr,
            "n_genuine": int((labels_arr == 1).sum()),
            "n_impostor": int((labels_arr == 0).sum()),
        }

    @staticmethod
    def _temporal_drift(features: List[CaptureFeature]) -> Optional[float]:
        by_session = defaultdict(list)
        for f in features:
            if f.capture.session:
                by_session[f.capture.session].append(f)
        if len(by_session) < 2:
            return None
        sessions = sorted(by_session.keys())
        ref = by_session[sessions[0]]
        scores: List[float] = []
        for session in sessions[1:]:
            for ref_feat in ref:
                matched = next(
                    (f for f in by_session[session]
                     if f.capture.fiber == ref_feat.capture.fiber
                     and f.capture.challenge == ref_feat.capture.challenge),
                    None,
                )
                if matched is None:
                    continue
                scores.append(normalized_cross_correlation(ref_feat.vector, matched.vector))
        return float(np.mean(scores)) if scores else None

    @staticmethod
    def _hard_pairs(known_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        probe_fibers = known_result.get("probe_fibers", [])
        pred_fibers = known_result.get("predicted_fibers", [])
        unique = sorted(set(probe_fibers))
        pair_counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {"a_to_b": 0, "b_to_a": 0})
        for true, pred in zip(probe_fibers, pred_fibers):
            if true == pred:
                continue
            a, b = sorted([true, pred])
            if true == a:
                pair_counts[(a, b)]["a_to_b"] += 1
            else:
                pair_counts[(a, b)]["b_to_a"] += 1
        # include all unique pairs so the table is informative even for zero-confusions
        rows: List[Dict[str, Any]] = []
        for i, a in enumerate(unique):
            for b in unique[i + 1:]:
                counts = pair_counts.get((a, b), {"a_to_b": 0, "b_to_a": 0})
                rows.append({"fiber_a": a, "fiber_b": b, **counts})
        rows.sort(key=lambda r: (r["a_to_b"] + r["b_to_a"]), reverse=True)
        return rows


def run(config) -> ExperimentContext:
    return AuthenticationExperiment(config).run()


__all__ = ["AuthenticationExperiment", "run"]
