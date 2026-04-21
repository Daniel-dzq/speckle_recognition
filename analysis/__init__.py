"""
Speckle-PUF experimental analysis framework.

A modular, publication-ready toolkit for orchestrating the experiments and
figures associated with the dual-channel optical PUF paper:

    * 3.1 System setup / acquisition sanity checks
    * 3.2 Fiber length optimization
    * 3.3 Dual-channel characterization
    * 3.4 Common-mode suppression evaluation
    * 3.5 Authentication performance
    * 3.6 Live demo / interactive authentication

The framework is intentionally decoupled from the training stack in the
repository root. It reuses the existing data on disk through a configurable
``DatasetLayout`` and produces structured, reproducible artefacts under
``results/<experiment_name>/``.

Entry points (CLI):

    python scripts/run_experiment.py <name> --config config/<name>.yaml

Or one of the per-experiment convenience scripts under ``scripts/``.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
