# Building `release_minimal/`

Copy-only workflow: **never delete or modify** the parent repository. Write only under `release_minimal/`.

## Core package (unchanged decisions)

Include: `final_fiber_dataset.py`, `unified_dataset.py`, `train_eval.py`, `models.py`, training/GUI/PPT scripts, `challenge_inputs/`, `input.pptx`, `data/recognition_dataset/`, `models/final_15fibers/`, `outputs/final_15fiber_training/`, `figures/` (Fig2 + auth matrix), MindVision SDK (`gui/libmvsdk.dylib`, `gui/win_sdk/`), split reports, `docs/`.

## Experiment datasets (final paper data)

| Source | Destination |
|--------|-------------|
| `LengthOptimize/` | `experiments/length_optimization/data/` |
| `fiber_loss/` | `experiments/fiber_loss/data/` |
| `long_term_stability/` | `experiments/long_term_stability/data/` |
| `disturbance_sensitivity/` | `experiments/disturbance_sensitivity/data/` |

Also copy processed outputs:

- `results/length_optimization_green/` → `experiments/length_optimization/outputs/length_optimization_green/`
- `figures/paper/Fig3_length_optimization/` (no `archive/`) → `experiments/length_optimization/outputs/fig3/`
- `figures/new_datasets_analysis/*disturbance*` → `experiments/disturbance_sensitivity/outputs/figures/`
- `figures/new_datasets_analysis/*long_term*` → `experiments/long_term_stability/outputs/figures/`

## Scripts to copy

| Experiment | Scripts |
|------------|---------|
| length_optimization | `scripts/run_experiment.py`, `scripts/run_length_optimization.py`, `config/length_optimization_green.yaml` (patch paths), `scripts/paper_figures/plot_fig3_length_optimization.py` |
| fiber_loss | (none standalone — Fig2 + length pipeline) |
| long_term_stability | `scripts/analyze_new_datasets.py` → `analyze_long_term_stability.py` (patched) |
| disturbance_sensitivity | `scripts/analyze_new_datasets.py` → `analyze_disturbance_sensitivity.py` (patched) |

Shared: `analysis/` → `release_minimal/analysis/`.

Patch `figures/generate_fig2_length_optimization.py` to read `experiments/length_optimization/` and `experiments/fiber_loss/data/`.

## rsync excludes

```
--exclude __pycache__ --exclude '*.pyc' --exclude .DS_Store
--exclude .cache --exclude '*.tmp' --exclude '*.log'
```

## Post-build checks

1. `du -sh experiments/*`
2. `python3 figures/generate_fig2_length_optimization.py`
3. Write/update `experiments/**/README.md` and `RELEASE_EXPERIMENTS_REPORT.md`

Do **not** commit or push unless explicitly requested.
