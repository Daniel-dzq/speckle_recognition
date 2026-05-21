# Physical characterization (summary index)

Lightweight **tables and metrics only** — no duplicated figures.

## Run from project root

```bash
python scripts/run_fiber_loss_analysis.py
python scripts/run_length_optimization.py
python scripts/run_long_term_stability.py
python scripts/run_disturbance_sensitivity.py
python scripts/run_all_physical_characterization.py
```

## Layout

| Subfolder | Summary files here | Figures (canonical paths) |
|-----------|-------------------|---------------------------|
| `length_optimization/` | `summary.json`, `optimal_length.json`, `Fig2_*.{csv,md}` | `figures/Fig2_length_optimization.png` · `experiments/length_optimization/outputs/` |
| `fiber_loss/` | `fiber_loss_summary.csv` | Loss in Fig. 2 · raw Excel: `experiments/fiber_loss/data/` |
| `long_term_stability/` | `metrics_summary.json` | `experiments/long_term_stability/outputs/figures/` |
| `disturbance_sensitivity/` | `metrics_summary.json` | `experiments/disturbance_sensitivity/outputs/figures/` |

**Recognition:** `outputs/final_15fiber_training/` · **Auth matrix figure:** `figures/auth_matrix_15x15.png`

Detailed scripts: `experiments/*/scripts/` · Overview: `physical_characterization_summary.md`
