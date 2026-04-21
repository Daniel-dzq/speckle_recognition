#!/usr/bin/env python3
"""
Unified experiment runner.

Examples:
    python scripts/run_experiment.py length_optimization \
        --config config/length_optimization.yaml

    python scripts/run_experiment.py authentication \
        --config config/authentication.yaml --output.name auth_smoke

Options following ``--`` are parsed as dotted-key overrides::

    python scripts/run_experiment.py demo --config config/demo.yaml -- mode=gui
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.experiments import EXPERIMENT_REGISTRY
from analysis.utils.config import ExperimentConfig, load_config


def _apply_overrides(data: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply ``dotted.key=value`` CLI overrides to the already-loaded config."""
    for ov in overrides:
        if "=" not in ov:
            raise SystemExit(f"Bad override (expected key=value): {ov!r}")
        key, value = ov.split("=", 1)
        parts = key.split(".")
        node = data
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = _coerce(value)
    return data


def _coerce(raw: str):
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none", "~"}:
        return None
    try:
        if "." in raw or "e" in raw.lower():
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified runner for the speckle-PUF analysis framework.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("experiment", choices=sorted(EXPERIMENT_REGISTRY.keys()),
                   help="Experiment to run.")
    p.add_argument("--config", required=True, type=Path,
                   help="Path to a YAML (or JSON) configuration file.")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="key.path=value",
                   help="Inline override (may be given multiple times).")
    return p


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    overrides = list(args.overrides)
    for tok in unknown:
        if tok.startswith("--"):
            overrides.append(tok[2:])

    cfg = load_config(args.config)
    data = cfg.to_dict()
    if overrides:
        data = _apply_overrides(data, overrides)
    cfg = ExperimentConfig(data, source_path=args.config)

    experiment_cls = EXPERIMENT_REGISTRY[args.experiment]
    experiment = experiment_cls(cfg)
    ctx = experiment.run()
    print(f"\nArtefacts written to: {ctx.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
