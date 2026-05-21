# Length optimization experiment

## 1. Purpose

Select the optimal **total fiber length** (nominal 8–16 cm groups) for green-channel speckle PUF characterization. Metrics: transmission loss (from fiber-loss data), intra/inter-class L2 distances, inter/intra ratio, and 400×400 ROI Shannon entropy. **Result: 9 cm (`Fiber9cm`)** recommended (highest inter/intra ratio under loss gate).

## 2. Input data (`data/`)

- `data/Green/Fiber{L}cm/Fiber{N}/<1..10>.JPG` — green still captures (5 fibers × 10 repeats per length group: 8, 9, 11, 13, 16 cm).

## 3. Processed outputs (`outputs/`)

- `outputs/length_optimization_green/` — full analysis run: `tables/per_length_summary.csv`, `optimal_length.json`, `report.md`, `summary.json`, `figures/`, `manifest.json`.
- `outputs/fig3/` — publication Fig3 bundle: `Fig3_length_optimization.{png,pdf,svg}`, `Fig3_length_optimization_data.csv`, `Fig3_length_optimization_meta.json`.

Bundled composite Figure 2 (uses this experiment + `fiber_loss`): `figures/Fig2_length_optimization.{png,pdf,svg}`.

## 4. Paper / report support

- **Section 3.2** — fiber length optimization methodology.
- **Figure 2** — combined length + loss panels (panel data from Fig3 CSV + fiber loss).
- **Figure 3 / manuscript Figure 4-style** — per-length loss, distances, entropy, montage (`outputs/fig3/`).

## 5. Scripts (`scripts/`)

| Script | Role |
|--------|------|
| `run_length_optimization.py` | Wrapper → `run_experiment.py length_optimization` |
| `run_experiment.py` | Unified runner (uses `analysis/`) |
| `length_optimization_green.yaml` | Experiment config (release-relative paths) |
| `paper_figures/plot_fig3_length_optimization.py` | Regenerate Fig3 panels from `per_length_summary.csv` |

Fig2 (cross-experiment): `figures/generate_fig2_length_optimization.py` at release root.

## 6. Reproduce

From `release_minimal/`:

```bash
# Recompute metrics from raw Green JPGs (~minutes; uses feature cache under outputs/)
python experiments/length_optimization/scripts/run_length_optimization.py \
  --config experiments/length_optimization/scripts/length_optimization_green.yaml

# Fig3 panels only (fast; uses existing tables)
python experiments/length_optimization/scripts/paper_figures/plot_fig3_length_optimization.py

# Fig2 composite (needs fiber_loss data + outputs above)
python figures/generate_fig2_length_optimization.py
```

**Expected outputs:** updated `outputs/length_optimization_green/`, `outputs/fig3/`, and `figures/Fig2_length_optimization_regen.*` (script default regen prefix).

## 7. Notes / limitations

- Config `power.csv_path` may point to optional `fiber_loss/data/power_loss.csv`; if absent, loss is merged from Excel in `experiments/fiber_loss/data/`.
- `LengthOptimize/` in the source repo is **not modified**; only a copy under `data/Green/`.
- Re-running the full pipeline overwrites cache under `outputs/length_optimization_green/cache/` (safe to delete to force refresh).
- Requires `analysis/` package and extras in `requirements.txt` (PyYAML, pandas, scipy, openpyxl).
