# Figures

Publication figures live only under `figures/paper/`. Each figure has its own folder with PNG, PDF, SVG, data summary CSV, and report MD.

## Layout

```
figures/paper/
  Fig2_length_optimization/
  Fig3_authentication/
  Fig4_challenge_speckle/
  Fig5_stability/
  Fig6_disturbance/
```

## Generate

From repository root:

```bash
python scripts/paper_figures/generate_fig2_length_optimization.py
python scripts/paper_figures/generate_fig3_auth_performance.py
python scripts/paper_figures/generate_fig4_challenge_speckle_examples.py
python scripts/paper_figures/generate_fig5_stability.py
python scripts/paper_figures/generate_fig6_disturbance.py
```

Or:

```bash
python scripts/paper_figures/generate_all_paper_figures.py
```

Generators overwrite final outputs in place. Use `--archive-old` only if you need timestamped backups.

## Summaries (no figures)

Tabular physical-characterization summaries: `outputs/physical_characterization/` (CSV/JSON/MD only, no PNG/PDF/SVG).
