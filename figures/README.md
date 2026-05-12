# Figures directory (`figures/`)

## What this folder is

1. **This directory is the consolidated “paper figures” bundle** produced mainly by
   `scripts/make_paper_figures.py`. It collects plots that summarize **training /
   authentication** results (e.g. heatmaps, curves) into one flat or lightly nested tree.
2. **It is not the same as a single `analysis/` run.** A one-off experiment with full
   provenance lives under `results/<run_name>/` (see `docs/output_organization.md`).

## Subfolders you may see

| Path | Role |
|------|------|
| (root of `figures/`) | Flat `fig_*` outputs from `make_paper_figures.py` |
| `figures/softcopyright/` | GUI screenshots for legal / registration materials — **not** main paper science figures |
| `figures/new_datasets_analysis/` | Extension / extra-dataset plots from `scripts/analyze_new_datasets.py` |
| `figures/patent/` | Optional ad hoc assets (if you create them) |

## If you use an image from here in a paper

Record in your **experiment archive** (see `scripts/archive_experiment_snapshot.py`) or in supplementary notes:

- Path to **`results/fiber_auth/auth_matrix.json`** (or equivalent) used by the plotting script  
- Paths to **per-fiber** `results/` training logs or checkpoints  
- **Git commit** (`git rev-parse HEAD`)  
- **Script name** (`make_paper_figures.py`, etc.)

## Machine-readable listing

After running `python scripts/inventory_repository.py`, see **`docs/generated_figures_manifest.csv`**
for a table of image files found under `figures/` (if any).
