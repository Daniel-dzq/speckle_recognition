"""
Experiment 3.6 — Demo / live authentication.

Two modes:

* ``offline_script`` — a scripted sequence of (fiber, challenge, probe) steps
  is replayed against the enrolled templates. Ideal for CI testing the
  end-to-end authentication flow, and for generating a rich markdown log that
  can be shown on the demo screen.
* ``gui`` — launch the live demo GUI (``scripts/launch_demo.py``). This
  delegates to the existing real-time interface, which already integrates the
  CCD camera. If PySide6 is unavailable we fall back cleanly to offline mode.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..caching.cache import FeatureCache
from ..io.dataset import DatasetLayout, discover_captures
from ..metrics.basic import pairwise_ncc
from ..preprocessing.pipeline import PreprocessConfig
from ..reporting.writers import MarkdownBuilder, write_csv, write_json
from ..utils.types import Capture
from ._features import CaptureFeature, extract_features
from .base import BaseExperiment, ExperimentContext


@dataclass
class DemoSession:
    step: int
    claimed_fiber: str
    claimed_challenge: str
    best_match_fiber: str
    best_match_challenge: str
    best_score: float
    target_score: float
    decision: str
    notes: str = ""


def _authenticate(
    probe: CaptureFeature,
    templates: List[CaptureFeature],
    claimed_fiber: str,
    expected_challenge: str,
    threshold: float,
) -> DemoSession:
    if not templates:
        return DemoSession(0, claimed_fiber, expected_challenge, "", "", float("nan"),
                           float("nan"), "no_templates")
    template_vectors = np.stack([t.vector for t in templates], axis=0)
    S = pairwise_ncc(probe.vector[None, :], template_vectors)[0]
    best_idx = int(np.argmax(S))
    best_score = float(S[best_idx])
    best_fiber = templates[best_idx].capture.fiber
    best_challenge = templates[best_idx].capture.challenge

    # Score against the claimed (fiber, challenge) cell when it exists
    target_idx = next(
        (i for i, t in enumerate(templates)
         if t.capture.fiber == claimed_fiber and t.capture.challenge == expected_challenge),
        None,
    )
    target_score = float(S[target_idx]) if target_idx is not None else float("nan")

    decision = "accepted" if (
        best_fiber == claimed_fiber
        and best_challenge == expected_challenge
        and best_score >= threshold
    ) else "rejected"
    return DemoSession(
        step=0,
        claimed_fiber=claimed_fiber,
        claimed_challenge=expected_challenge,
        best_match_fiber=best_fiber,
        best_match_challenge=best_challenge,
        best_score=best_score,
        target_score=target_score,
        decision=decision,
    )


class DemoExperiment(BaseExperiment):
    """
    Driver for the 3.6 demo experiment.

    The offline mode is fully non-interactive; the gui mode hands off to
    ``scripts.launch_demo.main`` if PySide6 is available.
    """

    name = "demo"

    def execute(self, ctx: ExperimentContext) -> None:
        cfg = ctx.config
        logger = ctx.logger

        mode = str(cfg.get("mode", "offline_script"))
        if mode == "gui":
            self._launch_gui(ctx)
            return

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
            raise RuntimeError("No features extracted; cannot run demo.")

        threshold = float(cfg.get("threshold", 0.6))
        script = cfg.get("script", []) or []
        if hasattr(script, "to_dict"):
            script = [s.to_dict() if hasattr(s, "to_dict") else dict(s) for s in script]
        if not script:
            # Default: one pass per (fiber, challenge) in the dataset.
            script = [{"fiber": f.capture.fiber, "challenge": f.capture.challenge}
                      for f in features]

        # Group features by (fiber, challenge).
        groups: Dict[Tuple[str, str], List[CaptureFeature]] = defaultdict(list)
        for f in features:
            groups[(f.capture.fiber, f.capture.challenge)].append(f)

        # Build enrollment set by using the first sample of each cell and probing with the rest.
        enrollment_pool: List[CaptureFeature] = []
        probe_pool: Dict[Tuple[str, str], List[CaptureFeature]] = defaultdict(list)
        for key, items in groups.items():
            enrollment_pool.append(items[0])
            probe_pool[key].extend(items[1:] if len(items) > 1 else items)

        sessions: List[DemoSession] = []
        for step_i, step in enumerate(script, start=1):
            fiber = step.get("fiber")
            challenge = step.get("challenge")
            if not fiber or not challenge:
                logger.warning("Skipping malformed script step: %s", step)
                continue
            probes = probe_pool.get((fiber, challenge))
            if not probes:
                logger.warning("No probe available for %s / %s — skipping", fiber, challenge)
                continue
            probe = probes[0]
            session = _authenticate(probe, enrollment_pool, fiber, challenge, threshold)
            session.step = step_i
            sessions.append(session)
            logger.info(
                "step=%d claimed=%s/%s best=%s/%s score=%.3f decision=%s",
                step_i, fiber, challenge, session.best_match_fiber,
                session.best_match_challenge, session.best_score, session.decision,
            )

        sessions_csv = write_csv(
            ctx.csv_path("sessions.csv"),
            [s.__dict__ for s in sessions],
        )
        ctx.add_report("sessions", "csv", sessions_csv,
                       "Per-step decision log for the scripted demo")

        summary = {
            "mode": mode,
            "threshold": threshold,
            "n_steps": len(sessions),
            "n_accepted": sum(1 for s in sessions if s.decision == "accepted"),
            "n_rejected": sum(1 for s in sessions if s.decision == "rejected"),
        }
        s_path = write_json(ctx.output_dir / "summary.json", summary)
        ctx.add_report("summary", "json", s_path)

        md = MarkdownBuilder("Experiment 3.6 — Demo / live authentication")
        md.p(f"Scripted demo authentication run. Threshold: **{threshold:.3f}**.")
        accepted = summary["n_accepted"]
        md.kv({
            "Steps executed": len(sessions),
            "Accepted": accepted,
            "Rejected": summary["n_rejected"],
            "Accept rate": f"{accepted / max(1, len(sessions)):.1%}",
        })
        md.h(2, "Session log")
        md.table(
            ["#", "Claimed", "Best match", "Best score", "Target score", "Decision"],
            [[s.step,
              f"{s.claimed_fiber}/{s.claimed_challenge}",
              f"{s.best_match_fiber}/{s.best_match_challenge}",
              s.best_score, s.target_score, s.decision]
             for s in sessions],
        )
        md.h(2, "Launching the live GUI")
        md.code(
            "python scripts/run_demo.py --config config/demo.yaml --mode gui",
            lang="bash",
        )
        md_path = md.save(ctx.output_dir / "report.md")
        ctx.add_report("report", "markdown", md_path)

    # ----- GUI hand-off --------------------------------------------------
    def _launch_gui(self, ctx: ExperimentContext) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root))
        try:
            from scripts.launch_demo import main as launch_demo  # type: ignore
        except Exception as exc:
            ctx.logger.error("Could not import GUI launcher: %s", exc)
            raise
        ctx.logger.info("Delegating to scripts/launch_demo.main ...")
        launch_demo()


def run(config) -> ExperimentContext:
    return DemoExperiment(config).run()


__all__ = ["DemoExperiment", "run"]
