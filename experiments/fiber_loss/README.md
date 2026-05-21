# Fiber loss (transmission) experiment

## 1. Purpose

Measure **red and green channel transmission loss (dB)** vs total fiber length using bench power-meter sweeps (Excel workbooks per length).

## 2. Input data (`data/`)

- `fiber8cm.xlsx`, `fiber9cm.xlsx`, `fiber11cm.xlsx`, `Fiber13cm.xlsx`, `fiber16cm.xlsx` — Input/Output power columns per channel (Lab instrument export format).

Optional (not in source repo): `power_loss.csv` — aggregated CSV; Fig2 and length optimization read Excel when CSV is missing.

## 3. Processed outputs (`outputs/`)

No dedicated pipeline output folder. Loss values appear in:

- `experiments/length_optimization/outputs/length_optimization_green/tables/per_length_summary.csv`
- `experiments/length_optimization/outputs/fig3/Fig3_length_optimization_data.csv`
- `figures/Fig2_length_optimization_data_summary.csv`

## 4. Paper / report support

- **Figure 2(a)** — transmission loss vs total fiber length (green/red).
- **Section 3.2** — loss gate for length recommendation.

## 5. Scripts (`scripts/`)

**No standalone `analyze_fiber_loss.py`** in the repository. Loss aggregation is implemented inside:

- `figures/generate_fig2_length_optimization.py` (`aggregate_power_loss_from_xlsx_dir`, optional CSV merge)
- `analysis/experiments/length_optimization.py` (during full length optimization run)

## 6. Reproduce

Loss panels are regenerated as part of Fig2:

```bash
python figures/generate_fig2_length_optimization.py
```

Full length study (re-embeds loss into tables):

```bash
python experiments/length_optimization/scripts/run_length_optimization.py \
  --config experiments/length_optimization/scripts/length_optimization_green.yaml
```

**Expected outputs:** updated loss columns in `per_length_summary.csv` and Fig2/Fig3 artifacts.

## 7. Notes / limitations

- Original `fiber_loss/` at repo root is unchanged; only `data/*.xlsx` are copied here.
- `power_loss.csv` was not present in the development tree; Excel-only loss is supported.
- To add a CSV summary, place `data/power_loss.csv` (same schema as development config comments) and re-run Fig2 or length optimization.
