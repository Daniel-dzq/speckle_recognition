# Long-term stability experiment

## 1. Purpose

Evaluate **speckle reliability over repeated acquisitions** on the same fibers (temporal NCC: adjacent-frame and vs-first-frame consistency).

## 2. Input data (`data/`)

- `data/Fiber<N>/<repeat>.JPG` — flat layout, one folder per fiber (15 fibers), numbered still frames.

## 3. Processed outputs (`outputs/`)

- `outputs/figures/long_term_stability_analysis.{png,pdf}` — bundled publication plots.
- `outputs/metrics_summary.json` — per-fiber consecutive / vs-first NCC (subset of full multi-dataset summary).

## 4. Paper / report support

- **Physical characterization / performance analysis** — long-term stability and reliability subsection.
- Companion to disturbance study; uses same analysis metric stack as length optimization (NCC, L2 distances).

## 5. Scripts (`scripts/`)

| Script | Role |
|--------|------|
| `analyze_long_term_stability.py` | Feature extraction + stability plots (patched for `data/` and `outputs/`) |

Depends on `release_minimal/analysis/`.

## 6. Reproduce

```bash
python experiments/long_term_stability/scripts/analyze_long_term_stability.py
```

**Expected outputs:** `outputs/figures/long_term_stability_analysis.png`, `.pdf`, and `outputs/metrics_summary.json`.

## 7. Notes / limitations

- Source folder `long_term_stability/` at repo root is not modified.
- Feature cache: `outputs/.analysis_cache/` (excluded from Git via `.gitignore` patterns).
- Does not include `power_common_mode` data (separate experiment, not shipped in this release).
