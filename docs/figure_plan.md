# Journal figure plan — dual-channel optical PUF (MM-POF)

English-only exports target `figures/paper/` (see `scripts/generate_all_paper_figures.py` and [Figure generation policy](figure_generation_policy.md)).

| Fig. | Topic | Primary data / assets | Status |
|------|-------|------------------------|--------|
| **1** | Concept: dual-channel PUF, red reference vs green challenge, authentication flow | Patent block diagram (`README_source_patent_fig1_block_diagram.png`); future vector composite | **Source assets staged** — final vector TBD |
| **2** | Experimental setup: end-face red, green + SLM + side polish, CMOS/CCD | `figures/paper/Fig2_setup/`: **`Fig2_optical_setup_9cm.blend`** + `*_render.png` / `*_polish_closeup.png` + composed **`Fig2_optical_setup_9cm.{png,pdf,svg}`** + **`Fig2_optical_setup_9cm_semieditable.pptx`** (`scripts/paper_figures/blender_fig2_setup/` + `compose_fig2_blender_final.py`) | **Blender semi-3D + overlays** — 9 cm POF **5 + 1 + 3 cm** |
| **3** | Length optimization: loss; intra/inter + ratio; entropy | `results/length_optimization_green/tables/per_length_summary.csv` | **Auto-generated** (`Fig3_length_optimization/`) — **PI-confirmed** optimum **9 cm total fiber length** (see metadata / audit) |
| **4** | Surface / entropy source: polish, microscope, roughness, speckle diversity | Microscope + profilometer exports **missing** | **TODO** — see audit report |
| **5** | Dual-channel behaviour: stability, perturbation, speckle, profiles | `figures/new_datasets_analysis/metrics_summary.json` + `videocapture/` | **Auto-generated** (`Fig5_dual_channel/`) — **PI-confirmed valid** for manuscript (`manuscript_ready`, `data_validated_by_PI`) |
| **6** | Common-mode suppression: CV, η, reinstall narrative | `power_common_mode/` JPEGs + summary constants | **Draft** — paired red/green fluctuation data not fully verified; do not treat plot as final until verified pairs drive η (see `Fig6_*_meta.json`) |
| **7** | Authentication: fiber matrix, letter CM, ROC, score histograms | `results/fiber_auth/auth_matrix.json`, `results/unified/test_predictions.csv` | **Auto-generated** (`Fig7_authentication/`) |
| **8** | Demo / two-factor workflow | GUI capture (`README_source_demo_access_granted.png` if present) | **Partial** |

## Fig. 3 — total fiber length (PI policy)

- **Axis and cohort:** Fig. 3 uses **total fiber length (cm)**. The `length_mm` column in `per_length_summary.csv` is **total fiber length in millimetres** (cm = mm/10), **not** “green-only propagation distance.” The separate `green_prop_mm` column describes green path within the fiber for bookkeeping only.
- **Experimentally selected optimum:** **9 cm total fiber length** (**PI-confirmed**). The vertical guide lines and `is_selected_optimal` in the exported CSV mark this choice.
- **Not for the final length figure:** Legacy partial-length or mixed-era sweeps (e.g. narratives around **30 cm**, **5 / 30 / 45 cm**, or treating an ambiguous **8–16 cm** band as the *final* optimization conclusion without the PI’s 9 cm choice). Those datasets remain **archived / audit-only** — see `docs/figure_audit_report.md` and `figures/archive_old/`.

## Length cohort audit (current manuscript)

The regeneration cohort uses **`length_optimization_green`**: lengths **8, 9, 11, 13, 16 cm** (**total fiber**). Do **not** combine with `green_partial_32` / `run_partial_length_analysis` outputs (archived under `figures/archive_old/legacy_green_partial_32/`).

## Chance baselines

- **Letter classification:** 1/26 ≈ **3.85%** (annotated in Fig. 7).
- **Fiber closed-set 5-way:** 1/5 = **20%** if evaluating as uniform random — use narrative carefully; `auth_matrix.json` already encodes empirical impostor scores.

## Supplementary

Place extended ROC splits, additional domains, raw montages under `figures/paper/supplementary/` with matching CSV provenance when generated.
