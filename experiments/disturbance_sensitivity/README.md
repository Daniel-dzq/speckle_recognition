# Disturbance sensitivity experiment

## 1. Purpose

Evaluate **robustness under perturbation** — whether speckle patterns remain consistent within each fiber while staying separable across fibers (within-fiber NCC, pooled intra/inter L2, inter/intra ratio).

## 2. Input data (`data/`)

- `data/Fiber<N>/<repeat>.JPG` — still captures after controlled disturbance / handling (flat fiber layout).

## 3. Processed outputs (`outputs/`)

- `outputs/figures/disturbance_sensitivity_analysis.{png,pdf}` — bar chart of within-fiber NCC + text summary of discriminability metrics.
- `outputs/metrics_summary.json` — pooled intra/inter L2 and per-fiber NCC.

## 4. Paper / report support

- **Physical characterization / robustness** — disturbance and perturbation sensitivity subsection.

## 5. Scripts (`scripts/`)

| Script | Role |
|--------|------|
| `analyze_disturbance_sensitivity.py` | Feature extraction + disturbance plots (patched for release paths) |

Depends on `release_minimal/analysis/`.

## 6. Reproduce

```bash
python experiments/disturbance_sensitivity/scripts/analyze_disturbance_sensitivity.py
```

**Expected outputs:** `outputs/figures/disturbance_sensitivity_analysis.png`, `.pdf`, and `outputs/metrics_summary.json`.

## 7. Notes / limitations

- Source folder `disturbance_sensitivity/` at repo root is not modified.
- Derived from `scripts/analyze_new_datasets.py` in the development repo (disturbance-only entry point here).
- Feature cache under `outputs/.analysis_cache/`.
