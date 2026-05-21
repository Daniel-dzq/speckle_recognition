#!/usr/bin/env python3
"""
Summarize red/green transmission loss from experiments/fiber_loss/data/.

Writes outputs/physical_characterization/fiber_loss/fiber_loss_summary.csv using the
same Excel aggregation as figures/generate_fig2_length_optimization.py, or extracts
loss columns from the existing Fig2 data summary when Excel is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = _SCRIPT_DIR.parent
for _p in (str(ROOT), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from physical_characterization_common import (  # noqa: E402
    EXP_LOSS,
    FIGURES_DIR,
    PC_LOSS,
    ROOT,
    load_fig2_module,
    sync_fiber_loss_summary_csv,
)


def _write_loss_csv_from_xlsx() -> Path:
    mod = load_fig2_module()
    xl_agg, _, _ = mod.aggregate_power_loss_from_xlsx_dir(EXP_LOSS / "data")
    cols = [
        "total_length_cm",
        "green_loss_dB",
        "green_loss_dB_std",
        "red_loss_dB",
        "red_loss_dB_std",
    ]
    PC_LOSS.mkdir(parents=True, exist_ok=True)
    dst = PC_LOSS / "fiber_loss_summary.csv"
    rows = []
    for cm in sorted(xl_agg.keys()):
        v = xl_agg[cm]
        rows.append(
            {
                "total_length_cm": cm,
                "green_loss_dB": v.get("green_loss_mean", ""),
                "green_loss_dB_std": v.get("green_loss_std", ""),
                "red_loss_dB": v.get("red_loss_mean", ""),
                "red_loss_dB_std": v.get("red_loss_std", ""),
            }
        )
    with dst.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in rows:
            out = {}
            for c in cols:
                val = row[c]
                if isinstance(val, float) and not math.isfinite(val):
                    out[c] = ""
                else:
                    out[c] = val
            w.writerow(out)
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize fiber transmission loss (Excel or Fig2 CSV)."
    )
    parser.add_argument(
        "--from-fig2-csv",
        action="store_true",
        help="Extract loss columns from figures/Fig2_length_optimization_data_summary.csv only.",
    )
    args = parser.parse_args()

    if args.from_fig2_csv:
        dst = sync_fiber_loss_summary_csv()
        print(f"Wrote {dst.relative_to(ROOT)} from Fig2 data summary.")
        return 0

    if any((EXP_LOSS / "data").glob("*.xlsx")):
        try:
            dst = _write_loss_csv_from_xlsx()
            print(f"Wrote {dst.relative_to(ROOT)} from experiments/fiber_loss/data/*.xlsx.")
            return 0
        except Exception as exc:
            print(f"Excel aggregation failed ({exc}); falling back to Fig2 CSV.")

    summary = FIGURES_DIR / "Fig2_length_optimization_data_summary.csv"
    if not summary.is_file():
        print(
            "No xlsx aggregation and no Fig2_length_optimization_data_summary.csv. "
            "Run: python scripts/run_length_optimization.py",
            file=sys.stderr,
        )
        return 1
    dst = sync_fiber_loss_summary_csv()
    print(f"Wrote {dst.relative_to(ROOT)} from Fig2 data summary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
