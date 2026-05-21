#!/usr/bin/env python3
"""
Regenerate length-optimization summaries and Figure 2 artifacts.

Calls experiments/length_optimization/scripts/ (optional full pipeline) and
figures/generate_fig2_length_optimization.py. Syncs summary tables/JSON/MD only
into outputs/physical_characterization/length_optimization/ (no figure copies).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = _SCRIPT_DIR.parent
for _p in (str(ROOT), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from physical_characterization_common import (  # noqa: E402
    ROOT,
    load_fig2_module,
    promote_fig2_regen_to_standard,
    run_length_pipeline,
    sync_length_summaries,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Length optimization + Fig2 regeneration (summary sync, no duplicated figures in outputs/)."
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Run full length_optimization_green experiment (slow; reads raw Green JPGs).",
    )
    parser.add_argument(
        "--skip-fig2",
        action="store_true",
        help="Skip figures/generate_fig2_length_optimization.py.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Copy Fig2_length_optimization_regen.* to Fig2_length_optimization.* in figures/.",
    )
    args = parser.parse_args()

    if args.run_pipeline:
        print("Running length optimization pipeline...")
        run_length_pipeline()

    if not args.skip_fig2:
        print("Generating Figure 2 (writes figures/Fig2_length_optimization_regen.*)...")
        mod = load_fig2_module()
        mod.main()
        if args.overwrite:
            promoted = promote_fig2_regen_to_standard()
            for p in promoted:
                print(f"Promoted {p.relative_to(ROOT)}")

    synced = sync_length_summaries()
    for p in synced:
        print(f"Synced summary {p.relative_to(ROOT)}")
    print("Done. Figures: figures/Fig2_length_optimization_regen.* (use --overwrite for standard names).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
