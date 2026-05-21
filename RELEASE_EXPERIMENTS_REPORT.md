# Release experiment build report

Generated when extending `release_minimal/` with physical characterization experiments. Original repository folders were **not** modified.

## Copy status

| Source (repo root) | Release path | Copied | Size in release |
|--------------------|--------------|--------|-----------------|
| `LengthOptimize/` | `experiments/length_optimization/data/` | **Yes** | ~130 MB (part of 199 MB folder) |
| `fiber_loss/` | `experiments/fiber_loss/data/` | **Yes** | ~60 KB |
| `long_term_stability/` | `experiments/long_term_stability/data/` | **Yes** | ~75 MB |
| `disturbance_sensitivity/` | `experiments/disturbance_sensitivity/data/` | **Yes** | ~30 MB |

**Total `experiments/`:** ~305 MB (excluding recognition dataset).

No raw videos (`.avi`/`.mp4`) were found in these folders.

## Per-experiment detail

### length_optimization

| Item | Status |
|------|--------|
| Data copied | Yes — `data/Green/Fiber{8,9,11,13,16}cm/...` |
| Scripts copied | Yes — `run_length_optimization.py`, `run_experiment.py`, `length_optimization_green.yaml`, `paper_figures/plot_fig3_length_optimization.py` (+ `paper_figures/` helpers) |
| Outputs copied | Yes — `outputs/length_optimization_green/`, `outputs/fig3/` |
| Release-relative paths | Yes — config + `plot_fig3` + shared `analysis/` |
| Reproduce | `python experiments/length_optimization/scripts/run_length_optimization.py --config experiments/length_optimization/scripts/length_optimization_green.yaml` |
| Unresolved | None |

### fiber_loss

| Item | Status |
|------|--------|
| Data copied | Yes — five `*cm.xlsx` files |
| Scripts copied | **No standalone script** — loss via `figures/generate_fig2_length_optimization.py` and length optimization pipeline |
| Outputs copied | Embedded in length_optimization / Fig2 CSVs |
| Reproduce | `python figures/generate_fig2_length_optimization.py` |
| Unresolved | `power_loss.csv` not in source repo (Excel-only path works) |

### long_term_stability

| Item | Status |
|------|--------|
| Data copied | Yes — `Fiber1`…`Fiber15` JPG folders |
| Scripts copied | Yes — `analyze_long_term_stability.py` |
| Outputs copied | Yes — prior `figures/new_datasets_analysis` plots + `metrics_summary.json` |
| Release-relative paths | Yes |
| Reproduce | `python experiments/long_term_stability/scripts/analyze_long_term_stability.py` |
| Unresolved | None |

### disturbance_sensitivity

| Item | Status |
|------|--------|
| Data copied | Yes — `Fiber*` JPG folders |
| Scripts copied | Yes — `analyze_disturbance_sensitivity.py` |
| Outputs copied | Yes — prior analysis PNG/PDF + `metrics_summary.json` |
| Release-relative paths | Yes |
| Reproduce | `python experiments/disturbance_sensitivity/scripts/analyze_disturbance_sensitivity.py` |
| Unresolved | None |

## Shared components

- `analysis/` package copied to release root (~1.1 MB).
- `config/length_optimization_green.yaml` copied to `release_minimal/config/` (reference; active config in `experiments/length_optimization/scripts/`).

## README files

| File | Created |
|------|---------|
| `experiments/README.md` | Yes |
| `experiments/length_optimization/README.md` | Yes |
| `experiments/fiber_loss/README.md` | Yes |
| `experiments/long_term_stability/README.md` | Yes |
| `experiments/disturbance_sensitivity/README.md` | Yes |

## Figure 2

| Check | Result |
|-------|--------|
| `figures/generate_fig2_length_optimization.py` patched | **Yes** — uses `experiments/length_optimization/` and `experiments/fiber_loss/data/` |
| Regeneration smoke test | **Passed** (`python3 figures/generate_fig2_length_optimization.py`) |

## Exclusions applied

`__pycache__/`, `*.pyc`, `.DS_Store`, `.cache/`, `*.tmp`, `*.log`, Fig3 `archive/`.

## Not committed / not pushed

Per project policy, `release_minimal/` remains local; no git commit or push performed.
