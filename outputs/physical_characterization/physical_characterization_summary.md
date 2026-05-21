# Physical characterization summary

Index of physical experiments: **summary data live here**; **figures live under `figures/` and `experiments/*/outputs/`** (not duplicated in this folder).

---

## 1. Length optimization

| Item | Path |
|------|------|
| **Purpose** | Select optimal total fiber length (loss, intra/inter distances, inter/intra ratio, entropy). |
| **Input data** | `experiments/length_optimization/data/Green/` |
| **Scripts** | `experiments/length_optimization/scripts/run_length_optimization.py`, `paper_figures/plot_fig3_length_optimization.py`; `figures/generate_fig2_length_optimization.py` |
| **Detailed outputs** | `experiments/length_optimization/outputs/` (`length_optimization_green/`, `fig3/`) |
| **Summary files (here)** | `outputs/physical_characterization/length_optimization/` |

**Figures (not copied here):**

- Final **Fig. 2:** `figures/Fig2_length_optimization.png` (`.pdf`, `.svg`)
- **Fig. 3:** `experiments/length_optimization/outputs/fig3/Fig3_length_optimization.png`

**Summary artifacts:** `summary.json`, `optimal_length.json`, `Fig2_length_optimization_data_summary.csv`, `Fig2_length_optimization_report.md`

**Main conclusion:** **Fiber9cm** (9.0 cm total length), inter/intra ratio **1.565** — see `optimal_length.json`.

---

## 2. Fiber loss

| Item | Path |
|------|------|
| **Purpose** | Red/green transmission loss vs length (Fig. 2 panel a). |
| **Input data** | `experiments/fiber_loss/data/*.xlsx` |
| **Scripts** | `figures/generate_fig2_length_optimization.py`; length-optimization pipeline |
| **Detailed outputs** | Integrated in length optimization + Fig. 2 (no `experiments/fiber_loss/outputs/`) |
| **Summary files (here)** | `fiber_loss/fiber_loss_summary.csv` |

**Figures:** loss panel in `figures/Fig2_length_optimization.png`

**Note:** `power_loss.csv` absent; Excel path works.

---

## 3. Long-term stability

| Item | Path |
|------|------|
| **Purpose** | Temporal NCC stability over repeated captures. |
| **Input data** | `experiments/long_term_stability/data/` |
| **Scripts** | `experiments/long_term_stability/scripts/analyze_long_term_stability.py` |
| **Detailed outputs** | `experiments/long_term_stability/outputs/` |
| **Summary files (here)** | `long_term_stability/metrics_summary.json` |

**Figures:** `experiments/long_term_stability/outputs/figures/long_term_stability_analysis.png` (`.pdf`)

---

## 4. Disturbance sensitivity

| Item | Path |
|------|------|
| **Purpose** | Within-fiber consistency and discriminability under perturbation. |
| **Input data** | `experiments/disturbance_sensitivity/data/` |
| **Scripts** | `experiments/disturbance_sensitivity/scripts/analyze_disturbance_sensitivity.py` |
| **Detailed outputs** | `experiments/disturbance_sensitivity/outputs/` |
| **Summary files (here)** | `disturbance_sensitivity/metrics_summary.json` |

**Figures:** `experiments/disturbance_sensitivity/outputs/figures/disturbance_sensitivity_analysis.png` (`.pdf`)

---

## Recognition / authentication (separate)

| Item | Path |
|------|------|
| **Results** | `outputs/final_15fiber_training/` |
| **Matrix figure** | `figures/auth_matrix_15x15.png` or `outputs/final_15fiber_training/auth_matrix_15x15.png` |

---

## Notes

1. **Fig. 2 regeneration** writes `figures/Fig2_length_optimization_regen.*` by default; canonical files are `figures/Fig2_length_optimization.*`.
2. **Fiber loss** has no standalone output tree; see `fiber_loss/README.md`.
3. This folder intentionally excludes PNG/PDF/SVG copies to avoid duplication with `figures/` and `experiments/`.
