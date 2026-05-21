# Physical characterization experiments

This directory holds **final experiment data and reproduction scripts** for the paper/report sections on fiber geometry, loss, stability, and robustness.

| Folder | Purpose |
|--------|---------|
| `length_optimization/` | Fiber length selection; speckle intra/inter metrics and Shannon entropy vs total length |
| `fiber_loss/` | Red/green transmission loss measurements (Excel power sweeps) |
| `long_term_stability/` | Reliability over repeated / long-term still captures per fiber |
| `disturbance_sensitivity/` | Robustness under mechanical/environmental perturbation |

Each experiment follows:

```
<experiment>/
  data/       # Raw captures (local-only; not tracked on GitHub)
  outputs/    # Processed tables, metrics; large plot PNGs may be local-only
  scripts/    # Detailed implementation (called by root scripts/)
  README.md
```

Raw JPG captures for length optimization, long-term stability, and disturbance sensitivity stay on disk locally but are listed in `.gitignore`. GitHub tracks scripts, READMEs, CSV/JSON/MD summaries, and final paper figures under `figures/paper/`.

## Root entry points (recommended)

From the project root:

```bash
python scripts/run_fiber_loss_analysis.py
python scripts/run_length_optimization.py
python scripts/run_long_term_stability.py
python scripts/run_disturbance_sensitivity.py
python scripts/run_all_physical_characterization.py
```

## Where outputs go

| Type | Location |
|------|----------|
| Paper-ready Fig. 2 | `figures/paper/Fig2_length_optimization/Fig2_length_optimization.png` |
| Experiment plots | `experiments/<name>/outputs/figures/` |
| Summary tables/JSON/MD only | `outputs/physical_characterization/<name>/` |

Shared framework: `analysis/` at project root.

See per-folder READMEs and `outputs/physical_characterization/physical_characterization_summary.md`.
