# Unified `paper_assets/` folder

The repository keeps figures in three places during normal work:

| Location | Role |
|----------|------|
| `figures/` | Classic `fig_*` training / auth plots and subfolders (`new_datasets_analysis/`, etc.) |
| `figures_publication/` | High-DPI publication exports from `make_publication_figures.py` |
| `figures_competition/` | Planning-document figures from `generate_competition_figures.py` |

For **the same material in one place**, sorted **by file type**:

```bash
python scripts/collect_paper_assets.py --clean
```

`--clean` removes the previous `png/`, `svg/`, `pdf/`, `csv/`, `INDEX.csv`, and `README.md` under `paper_assets/` so no stale copies remain. Omit `--clean` only if you intentionally merge manually.

This creates **`paper_assets/`** (gitignored) with:

- `paper_assets/png/`
- `paper_assets/svg/`
- `paper_assets/pdf/`
- `paper_assets/csv/`
- `paper_assets/INDEX.csv` — full source → destination map
- `paper_assets/README.md` — generation timestamp

Filenames are prefixed (`publication__…`, `competition__…`, `figures__…`, `results_lo_green__…`) so nothing is overwritten when flattened.

If `results/length_optimization_green/tables/` exists locally, `per_length_summary.csv` and `per_fiber_metrics.csv` are also copied into `csv/`.

Use `python scripts/collect_paper_assets.py --dry-run` to preview.
