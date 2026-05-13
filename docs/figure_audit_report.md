# Repository figure audit

Generated UTC: 2026-05-13T07:14:30.367597+00:00

## Scope

- **Images scanned:** 1595 files matching .jpeg, .jpg, .pdf, .png, .svg, .tif, .tiff
- **Inventory CSV:** `docs/figure_audit_inventory.csv`
- **Data-like files (csv/json/txt/md/npy/npz) indexed:** 232 (excluding heavy cache paths)
- **Plot-related scripts found:** 25
- **Regenerated journal pack:** run `python3 scripts/generate_all_paper_figures.py` → outputs under `figures/paper/Fig3_*` … `Fig7_*` with PNG/PDF/SVG + `*_data.csv` + `*_meta.json`.

## Top-level directories containing images

- **LengthOptimize/** — 250 file(s)
- **disturbance_sensitivity/** — 75 file(s)
- **experiment_archive/** — 228 file(s)
- **figures/** — 56 file(s)
- **figures_competition/** — 9 file(s)
- **figures_publication/** — 24 file(s)
- **letter_images/** — 26 file(s)
- **long_term_stability/** — 195 file(s)
- **paper_assets/** — 85 file(s)
- **power_common_mode/** — 600 file(s)
- **results/** — 47 file(s)

## Length optimization data consistency (critical)

**Canonical final length experiment:** `results/length_optimization_green/tables/per_length_summary.csv`

- Length groups present: **Fiber8cm, Fiber9cm, Fiber11cm, Fiber13cm, Fiber16cm** (total fiber lengths **8–16 cm** scale, not 5/30/45 cm).
- **Do not mix** with `results/green_partial_32/` or `figures/fig_green_length_*` (regeneration_manifest lists `run_partial_length_analysis`).

## Legacy / suspicious assets

| Pattern | Issue | Action |
|---------|-------|--------|
| `figures/fig_green_length_*` | Partial green length pipeline | **Moved to** `figures/archive_old/legacy_green_partial_32/` |
| `图表与实验结果分析报告.md` | Chinese report in figures tree | **Moved to** `figures/archive_old/` |
| `figures/softcopyright/*` | GUI capture | Supplementary Fig. 8 / demo only |
| `figures_competition/*` | Planning triplets | Parallel to paper; regenerate via `figures/paper/` pipeline |

## Figure disposition summary

- **review:** 1196
- **supplementary_fig8_or_archive:** 228
- **redraw_or_copy_into_figures_paper:** 127
- **archive_old / do_not_use_for_final_length_fig:** 40
- **canonical_paper_output:** 4

## Plot scripts (heuristic)

```text
scripts/analyze_new_datasets.py
scripts/archive_experiment_snapshot.py
scripts/audit_repository_figures.py
scripts/collect_paper_assets.py
scripts/evaluate_cross_fiber.py
scripts/generate_all_paper_figures.py
scripts/generate_competition_figures.py
scripts/generate_figure_audit_report.py
scripts/generate_soft_ware_manual_revision.py
scripts/generate_softcopyright_figures_pptx.py
scripts/install_user_manual_screenshots.py
scripts/inventory_repository.py
scripts/make_paper_figures.py
scripts/make_publication_figures.py
scripts/paper_figures/plot_fig3_length_optimization.py
scripts/paper_figures/plot_fig5_dual_channel.py
scripts/paper_figures/plot_fig6_common_mode.py
scripts/paper_figures/plot_fig7_authentication.py
scripts/paper_figures/sanity.py
scripts/paper_figures/style.py
scripts/plot_style.py
scripts/regenerate_figures_and_report.py
scripts/run_partial_length_analysis.py
scripts/train_fiber.py
scripts/train_unified.py
```

## TODO: missing data for full journal story

- **ROC / EER (fiber-level verification):** unified `test_predictions.csv` supports **letter** scores; dedicated genuine–impostor score exports for **fiber identity** may be missing — extend `fiber_auth` eval if needed.
- **7-day stability curve:** no dedicated time-series CSV located under `results/` — needs experiment log or labeled captures.
- **Surface roughness Rq distributions:** no `data/processed/` roughness table found — needs profilometer export.
- **Known- vs unknown-challenge ROC split:** requires explicit protocol labels in predictions export.

## Chinese text in figures

- Raster audit requires OCR; **SVG/PDF** should be scanned before submission (see `scripts/paper_figures/sanity.py`).
- Paths may contain CJK: filter `cjk_in_path=true` in the inventory CSV.
