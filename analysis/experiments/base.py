"""
Base class + shared context for every experiment.

An :class:`ExperimentContext` is created once per run and carries:

* the parsed :class:`ExperimentConfig`
* paths for figures, tables, cache
* the per-run logger
* the list of artefacts produced during the run

Subclasses only implement :meth:`BaseExperiment.execute`; provenance
(manifest, summary, config snapshot) is handled by the base ``run`` method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from ..io.manifests import write_manifest
from ..plotting.style import apply_style, save_figure
from ..reporting.writers import ExperimentReport
from ..utils.config import ExperimentConfig, dump_config, resolve_path
from ..utils.logging_utils import configure_logging
from ..utils.seed import seed_everything
from ..utils.types import ExperimentRun, PlotArtifact, ReportArtifact


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class ExperimentContext:
    """Runtime context shared across the execution of a single experiment."""

    config: ExperimentConfig
    output_dir: Path
    figures_dir: Path
    tables_dir: Path
    cache_dir: Path
    logger: Any
    run: ExperimentRun
    report: ExperimentReport
    seed: Optional[int] = None
    captures: Optional[Sequence] = None
    extras: dict = field(default_factory=dict)

    # ----- artefact helpers ----------------------------------------------
    def fig_path(self, stem: str) -> Path:
        return self.figures_dir / stem

    def csv_path(self, name: str) -> Path:
        return self.tables_dir / name

    def json_path(self, name: str) -> Path:
        return self.output_dir / name

    def add_plot(
        self,
        name: str,
        fig,
        *,
        csv_path: Optional[Path] = None,
        caption: str = "",
        formats: Sequence[str] = ("png", "pdf", "svg"),
    ) -> PlotArtifact:
        saved_paths = save_figure(fig, self.figures_dir / name, formats=formats)
        art = PlotArtifact(
            name=name,
            paths=[str(p) for p in saved_paths],
            source_csv=str(csv_path) if csv_path is not None else None,
            caption=caption,
        )
        self.report.add_artifact(art)
        return art

    def add_report(
        self,
        name: str,
        kind: str,
        path: Path,
        description: str = "",
    ) -> ReportArtifact:
        art = ReportArtifact(
            name=name,
            kind=kind,
            path=str(path),
            description=description,
        )
        self.report.add_artifact(art)
        return art


# ---------------------------------------------------------------------------
# Base experiment
# ---------------------------------------------------------------------------


class BaseExperiment:
    """
    Base class that handles:

        * output directory creation
        * seeding / matplotlib styling
        * logger initialisation
        * config snapshot + manifest
        * invocation of the subclass :meth:`execute`.
    """

    name: str = "base"

    def __init__(self, config: ExperimentConfig):
        if not isinstance(config, ExperimentConfig):
            raise TypeError("config must be an ExperimentConfig")
        self.config = config

    # ----- main entry -----------------------------------------------------
    def run(self) -> ExperimentContext:
        ctx = self._prepare_context()
        try:
            self.execute(ctx)
        except Exception as exc:
            ctx.logger.exception("Experiment %s failed: %s", self.name, exc)
            raise
        finally:
            self._finalize_context(ctx)
        return ctx

    # ----- lifecycle ------------------------------------------------------
    def _prepare_context(self) -> ExperimentContext:
        cfg = self.config
        output_cfg = cfg.get("output", {}) or {}
        output_root = resolve_path(
            output_cfg.get("root", "results"), self._output_base(cfg)
        )
        output_name = output_cfg.get("name", self.name)
        out_dir = output_root / output_name
        figures_dir = out_dir / "figures"
        tables_dir = out_dir / "tables"
        cache_dir = out_dir / "cache"
        for d in (out_dir, figures_dir, tables_dir, cache_dir):
            d.mkdir(parents=True, exist_ok=True)

        logger = configure_logging(
            name=f"analysis.{self.name}",
            log_file=out_dir / "run.log",
        )

        seed = cfg.get("seed", 0)
        applied_seed = seed_everything(seed)

        apply_style()

        snapshot_path = out_dir / "config_snapshot.yaml"
        try:
            dump_config(cfg, snapshot_path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not dump config snapshot: %s", exc)

        run = ExperimentRun(
            name=self.name,
            config_path=str(cfg.path) if cfg.path else None,
            config_snapshot=cfg.to_dict(),
            output_dir=str(out_dir),
            seed=applied_seed,
        )
        report = ExperimentReport(output_dir=out_dir, experiment=self.name)
        return ExperimentContext(
            config=cfg,
            output_dir=out_dir,
            figures_dir=figures_dir,
            tables_dir=tables_dir,
            cache_dir=cache_dir,
            logger=logger,
            run=run,
            report=report,
            seed=applied_seed,
        )

    def _finalize_context(self, ctx: ExperimentContext) -> None:
        ctx.run.finished_at = datetime.now().isoformat(timespec="seconds")
        manifest = write_manifest(
            ctx.output_dir / "manifest.json",
            experiment=self.name,
            config_snapshot=ctx.config.to_dict(),
            captures=ctx.captures,
            extra={
                "run": ctx.run.to_dict(),
                "artefacts": [
                    (a.to_dict() if hasattr(a, "to_dict") else dict(a))
                    for a in ctx.report.artifacts
                ],
            },
        )
        ctx.add_report("manifest", "manifest", manifest, "Run manifest + provenance")
        ctx.report.write_summary_json()
        ctx.report.write_summary_csv()
        ctx.logger.info("Done. Output: %s", ctx.output_dir)

    # ----- to implement ---------------------------------------------------
    def execute(self, ctx: ExperimentContext) -> None:  # pragma: no cover
        raise NotImplementedError

    # ----- path helpers --------------------------------------------------
    @staticmethod
    def _output_base(cfg: ExperimentConfig) -> Path:
        """
        Resolve the base directory used for the ``output.root`` field.

        Convention: if the configuration lives inside a ``config/`` folder at
        the repository root, the repository root itself is used as the base.
        Otherwise the config's own directory is used. This lets users write
        ``output.root: results`` in YAML files under ``config/`` and still
        land in the canonical ``<repo>/results/`` directory.
        """
        base = cfg.base_dir
        if base.name == "config" and base.parent.exists():
            return base.parent
        return base


__all__ = ["ExperimentContext", "BaseExperiment"]
