# Duplicate and near-duplicate figure report

Method: SHA-256 of raster PNGs for exact duplicates; format triplets (PNG/SVG/PDF) classified from `paper_assets/INDEX.csv` and `figures_competition/manifest.csv`. No image pixels were edited.

---

## Group A — `content_duplicate_sha256_26f20a68696b2851_pub03_pub04`

| duplicate_group_id | files | are_formats_of_same_figure | are_old_new_versions | which_one_to_use | which_ones_to_archive |
|--------------------|-------|----------------------------|----------------------|------------------|------------------------|
| `content_duplicate_sha256_26f20a68696b2851_pub03_pub04` | `figures_publication/publication_fig03_length_optimization.png`<br>`figures_publication/publication_fig04_length_optimization.png` | No (both PNG; true duplicate bytes) | Same export, misleading dual name | **Use neither** for 策划书 Fig. 4; if journal pack needs a montage, keep **one** PNG and delete the duplicate name in a future cleanup (not done in this audit) | Retain one filename for journal continuity; drop the second alias to avoid reviewer confusion |

**Note:** For the same pair, `.svg` and `.pdf` hashes **differ** between `fig03` and `fig04` filenames, but they are the same **four-panel length-optimization family** from `make_publication_figures.py::fig03_length_optimization`. Treat as **naming duplicate**, not two different analyses.

---

## Group B — `publication_fig03_fig04_length_optimization_pair` (format triplets)

| Files | are_formats_of_same_figure | Notes |
|-------|----------------------------|-------|
| `publication_fig03_length_optimization.{png,svg,pdf}` | **yes** | Same publication export stem |
| `publication_fig04_length_optimization.{png,svg,pdf}` | **yes** | Same family; PNG duplicate of fig03 as in Group A |

**which_one_to_use:** For **final manuscript Fig. 4**, use **`figures_competition/fig4_length_optimization`** triple only.  
**which_ones_to_archive:** `publication_fig03/04_length_optimization` as **journal/legacy montage**, not §3.2(1).

---

## Group C — competition `fig4` vs publication length family (different content)

| Files | SHA / content | Conclusion |
|-------|---------------|------------|
| `figures_competition/fig4_length_optimization.png` | **≠** publication pair | **Different raster** — triple-panel 策划书 figure |
| `figures_publication/publication_fig03_length_optimization.png` | Matches fig04 publication PNG | **Four-panel** figure |

**which_one_to_use for §3.2(1):** `figures_competition/fig4_length_optimization.{png,svg,pdf}`.  
**which_ones_to_archive:** publication `fig03/04_length_optimization` as alternate layout / wrong panel count for current brief.

---

## Group D — standard multi-format clones (not counted duplicates)

All other entries where the same logical figure appears as PNG + SVG + PDF are tagged **`same_figure_formats`** in `figure_audit_report.csv` when applicable — **expected** for publication workflow.

---

## Group E — recognition-performance set (unique content, thematic cluster)

`fig_auth_matrix`, `fig_auth_gap`, `fig_auth_scores`, `fig_same_fiber_per_domain`, `fig_test_accuracy_summary`, `publication_fig04_cross_fiber_auth` share **`results/fiber_auth/auth_matrix.json`** as primary numeric source but are **not** pixel duplicates of each other.
