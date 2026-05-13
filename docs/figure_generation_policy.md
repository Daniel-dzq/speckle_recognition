# Figure generation policy (`figures/paper/`)

## Canonical output directories

Each figure has **one** fixed directory under `figures/paper/` (no new random or run-stamped top-level folders):

| Figure | Directory |
|--------|-----------|
| Fig. 3 | `figures/paper/Fig3_length_optimization/` |
| Fig. 5 | `figures/paper/Fig5_dual_channel/` |
| Fig. 6 | `figures/paper/Fig6_common_mode_suppression/` |
| Fig. 7 | `figures/paper/Fig7_authentication/` |

Scripts live under `scripts/paper_figures/` and must write only into these canonical paths (plus `archive/` inside them).

## Stable filenames

For stem `Fig3_length_optimization`, `Fig5_dual_channel`, etc., the standard bundle is:

- `{stem}.png` — high-DPI raster
- `{stem}.pdf`
- `{stem}.svg`
- `{stem}_data.csv` — table written by the plot script
- `{stem}_meta.json` — provenance + status (merged with `save_figure_bundle`)

Auxiliary exports (e.g. Fig. 7 ROC CSV) keep predictable names under the **same** directory; they are **not** moved by the archive helper unless they match the glob patterns for that stem.

## Timestamp archive before overwrite

Before replacing canonical bundle files, scripts call `archive_existing_outputs(output_dir, figure_stem)` from `scripts/paper_figures/io_utils.py`. It:

1. Finds existing top-level files matching `{stem}*.png`, `{stem}*.pdf`, `{stem}*.svg`, `{stem}*_data.csv`, `{stem}*_meta.json`.
2. Creates `figures/paper/<FigureDir>/archive/YYYYMMDD_HHMMSS/`.
3. **Moves** matched files into that folder (does **not** copy README.md or unrelated names).
4. Writes `archive_manifest.json` inside the timestamp folder listing moved files and the archive stamp.

## No new random directory rule

Do not create sibling folders such as `Fig3_length_optimization_run2/` or date-only roots under `figures/paper/`. All runs overwrite the canonical directory after archiving the previous bundle.

## Restoring an archived version

1. Open `figures/paper/<FigureName>/archive/<YYYYMMDD_HHMMSS>/`.
2. Read `archive_manifest.json` for the list of files.
3. Copy (or move) the needed `.png`, `.pdf`, `.svg`, `*_data.csv`, `*_meta.json` back to `figures/paper/<FigureName>/`, **or** adjust your manuscript to point at the archive copy temporarily.

## Figure readiness (manuscript vs draft)

| Figure | Manuscript readiness | Notes |
|--------|---------------------|--------|
| **Fig. 3** | PI-confirmed optimum | `optimal_total_fiber_length_cm = 9`, `length_meaning = total_fiber_length_cm` |
| **Fig. 5** | **Final / PI-confirmed** | `manuscript_ready: true`, `source_dataset_status: final_or_PI_confirmed`, `data_validated_by_PI: true` |
| **Fig. 6** | **Draft** | `manuscript_ready: false` until verified paired red/green data recomputes η; search continues under `power_common_mode/` |
| **Fig. 7** | Per meta JSON | As generated from repository eval exports |

## Sanity check

After regeneration:

```bash
python3 scripts/paper_figures/sanity.py
```

This checks bundle completeness, skips `archive/` metadata, flags CJK in SVG text, and validates key PI/draft flags for Figs. 3, 5, and 6.
