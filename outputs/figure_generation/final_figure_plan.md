# Final figure plan (optical PUF paper)

All figures use **real project data only**. Scripts live under `scripts/paper_figures/`. Canonical publication assets go to `figures/paper/<figure_name>/`.

---

## Figure 1 — System concept and authentication workflow

| Item | Content |
|------|---------|
| Title | Optical PUF challenge–response authentication workflow |
| Panels | (a) Schematic: SLM challenge → dual-channel fiber → speckle camera → CNN → decision; (b) optional GUI screenshot |
| Data source | Schematic: `experiments/length_optimization/scripts/paper_figures/build_fig2_optical_setup.py`; screenshot: **missing** (capture from demo) |
| Output | `figures/paper/Fig1_system_workflow/` |
| Caption draft | Challenge patterns are displayed on an SLM and coupled into a dual-channel fiber PUF; speckle responses are captured and classified by a fiber-specific model to grant or deny access. |
| Status | **schematic-only** / needs missing data for screenshot panel |

---

## Figure 2 — Fiber length optimization

| Item | Content |
|------|---------|
| Title | Fiber length optimization selects 9 cm operating point |
| Panels | (a) Red/green transmission loss vs length; (b) intra/inter distance and ratio; (c) Shannon entropy vs length |
| Data source | `experiments/length_optimization/`, `experiments/fiber_loss/data/`, `figures/paper/Fig2_length_optimization/Fig2_length_optimization_data_summary.csv` |
| Output | `figures/paper/Fig2_length_optimization/Fig2_length_optimization.{png,pdf,svg}` |
| Script | `scripts/paper_figures/generate_fig2_length_optimization.py` |
| Caption draft | Performance metrics and channel loss versus fiber length; 9 cm maximizes discriminability and entropy while balancing transmission. |
| Status | **ready** |

---

## Figure 3 — 15-fiber authentication performance

| Item | Content |
|------|---------|
| Title | Fifteen-fiber PUF authentication and challenge recognition |
| Panels | (a) Per-fiber test accuracy; (b) 15×15 auth matrix; (c) diagonal vs off-diagonal distribution; (d) summary statistics |
| Data source | `outputs/final_15fiber_training/summary_15fibers.csv`, `auth_matrix_15x15.csv` |
| Output | `figures/paper/Fig3_authentication/` |
| Script | `scripts/paper_figures/generate_fig3_auth_performance.py` |
| Caption draft | Fiber-specific models achieve high test accuracy and near-unity diagonal authentication, while off-diagonal cross-fiber scores cluster at the 12.5% eight-class chance level. |
| Status | **ready** |

---

## Figure 4 — Challenge and speckle examples

| Item | Content |
|------|---------|
| Title | Challenge patterns and Fiber1 speckle responses |
| Panels | (a) Eight challenge inputs; (b) middle-frame speckles from Fiber1 videos (NCC in CSV/report only) |
| Data source | `challenge_inputs/`, `data/recognition_dataset/GreenAndRed/Fiber1/*.avi` |
| Output | `figures/paper/Fig4_challenge_speckle/` |
| Script | `scripts/paper_figures/generate_fig4_challenge_speckle_examples.py` |
| Caption draft | Distinct SLM challenges elicit distinct speckle patterns from real Fiber1 recordings. |
| Status | **ready** (2-panel gallery) |

---

## Figure 5 — Long-term stability

| Item | Content |
|------|---------|
| Title | Long-term speckle stability across repeated captures |
| Panels | (a) Per-fiber consecutive and vs-first NCC; (b) example drift vs acquisition index — **optional, from experiment analyzer** |
| Data source | `outputs/physical_characterization/long_term_stability/metrics_summary.json` |
| Output | `figures/paper/Fig5_stability/` |
| Script | `scripts/paper_figures/generate_fig5_stability.py` |
| Caption draft | Repeat captures maintain high adjacent-frame correlation; slow drift vs the first acquisition quantifies long-term stability per fiber. |
| Status | **ready** (summary panel); **needs missing data** for classifier accuracy vs time |

---

## Figure 6 — Disturbance sensitivity

| Item | Content |
|------|---------|
| Title | Speckle consistency under disturbance |
| Panels | (a) Within-fiber mean NCC; (b) pooled intra/inter L2 text summary |
| Data source | `outputs/physical_characterization/disturbance_sensitivity/metrics_summary.json` |
| Output | `figures/paper/Fig6_disturbance/` |
| Script | `scripts/paper_figures/generate_fig6_disturbance.py` |
| Caption draft | Under repeated disturbance captures, within-fiber speckle correlation and pooled inter/intra separability characterize robustness. |
| Status | **ready** (available metrics); **needs missing data** for accuracy vs disturbance level |

---

## Figure 7 — Confusion matrix examples

| Item | Content |
|------|---------|
| Title | Per-fiber challenge confusion (examples) |
| Panels | (a) Best fiber CM image; (b) worst fiber CM image; (c) average per-class accuracy |
| Data source | `outputs/final_15fiber_training/Fiber*/confusion_matrix.png`, `metrics.json` |
| Output | `figures/paper/Fig7_confusion/` (planned) |
| Script | Not implemented (numeric CM missing) |
| Caption draft | Representative confusion matrices highlight class-wise errors for the lowest- and highest-performing fibers. |
| Status | **needs missing data** for numeric heatmaps; image collage possible |

---

## Batch command

```bash
python scripts/paper_figures/generate_fig3_auth_performance.py
python scripts/paper_figures/generate_fig4_challenge_speckle_examples.py
python scripts/paper_figures/generate_all_paper_figures.py --skip-fig2
```

Style: white background, 600 dpi PNG, PDF/SVG vector, panel labels a–d, colorblind-friendly palette (`analysis/plotting/style.py`).
