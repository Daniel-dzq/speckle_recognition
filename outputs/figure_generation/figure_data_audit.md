# Figure data audit

Audit date: project release tree. All paths are relative to repository root.

Legend: **Available** = real files present; **Generate now** = sufficient data for scripted figure; **Missing** = panel or figure cannot be completed without additional real data.

---

## Figure 1 — System concept and authentication workflow

| Field | Detail |
|-------|--------|
| Scientific question | How does challenge → SLM → dual-channel PUF → speckle → model → access decision work? |
| Data sources | `gui/` (live demo only); `experiments/length_optimization/scripts/paper_figures/build_fig2_optical_setup.py` (schematic); no checked-in GUI screenshot |
| Real & available? | Workflow is real; **no committed GUI screenshot**; schematic scripts exist |
| Processing | Manual schematic and/or screen capture from `python scripts/launch_demo.py` |
| Output path | `figures/paper/Fig1_system_workflow/` (not created) |
| Generate now? | **Schematic-only** — yes as diagram; quantitative panels **no** |
| Missing data | Real GUI screenshot for composite figure (optional) |

---

## Figure 2 — Fiber length optimization

| Field | Detail |
|-------|--------|
| Scientific question | Why was 9 cm fiber length selected? |
| Data sources | `experiments/length_optimization/outputs/fig3/Fig3_length_optimization_data.csv`; `experiments/fiber_loss/data/*.xlsx`; `figures/paper/Fig2_length_optimization/Fig2_length_optimization_data_summary.csv`; `outputs/physical_characterization/length_optimization/` |
| Real & available? | **Yes** — five lengths (8, 9, 11, 13, 16 cm), loss and metrics tables |
| Processing | `scripts/paper_figures/generate_fig2_length_optimization.py` |
| Output path | `figures/paper/Fig2_length_optimization/` |
| Generate now? | **Yes** (requires `pandas`, `openpyxl`) |
| Missing data | None for core panels (a) loss vs length, (b) intra/inter/ratio, (c) entropy |

---

## Figure 3 — 15-fiber recognition / authentication performance

| Field | Detail |
|-------|--------|
| Scientific question | Does each fiber model authenticate its own speckles and reject others? |
| Data sources | `outputs/final_15fiber_training/summary_15fibers.csv`; `auth_matrix_15x15.csv`; `auth_matrix_report.md`; `Fiber*/metrics.json` |
| Real & available? | **Yes** — 15 fibers, 8 classes, full 15×15 matrix |
| Processing | `scripts/paper_figures/generate_fig3_auth_performance.py` |
| Output path | `figures/paper/Fig3_authentication/` |
| Generate now? | **Yes** |
| Missing data | None |

---

## Figure 4 — Challenge classes and speckle responses

| Field | Detail |
|-------|--------|
| Scientific question | Do different challenges produce distinguishable speckle patterns? |
| Data sources | `challenge_inputs/{A,B,C,1,2,3,boy,girl}.png`; `challenge_inputs/manifest.json`; `data/recognition_dataset/GreenAndRed/Fiber1/{a,b,c,1,2,3,boy,girl}.avi` |
| Real & available? | **Yes** — all 8 PNGs and 8 AVIs present |
| Processing | Middle-frame extraction + optional NCC matrix from frames |
| Output path | `figures/paper/Fig4_challenge_speckle/` |
| Generate now? | **Yes** |
| Missing data | Precomputed feature-space matrix from training pipeline not stored separately (NCC from frames used instead) |

---

## Figure 5 — Long-term stability

| Field | Detail |
|-------|--------|
| Scientific question | Are speckle features stable across repeated acquisitions over time? |
| Data sources | `experiments/long_term_stability/data/` (JPEG); `experiments/long_term_stability/outputs/metrics_summary.json`; `outputs/physical_characterization/long_term_stability/metrics_summary.json` |
| Real & available? | **Yes** — 195 captures, per-fiber NCC summaries |
| Processing | Summary bars from JSON; full time-series panel needs `analyze_long_term_stability.py` on JPEG data |
| Output path | `figures/paper/Fig5_stability/`; experiment figures under `experiments/long_term_stability/outputs/figures/` |
| Generate now? | **Partial** — aggregate panel (a) from JSON **yes**; per-index drift curve in paper bundle **missing in JSON alone** |
| Missing data | Model accuracy / confidence vs time (not measured in stability experiment); panel (b) example drift curve not in summary JSON |

---

## Figure 6 — Disturbance sensitivity / robustness

| Field | Detail |
|-------|--------|
| Scientific question | How robust are speckles under physical disturbance? |
| Data sources | `experiments/disturbance_sensitivity/data/`; `outputs/physical_characterization/disturbance_sensitivity/metrics_summary.json` |
| Real & available? | **Yes** — 75 captures, within-fiber NCC and pooled L2 metrics |
| Processing | `scripts/paper_figures/generate_fig6_disturbance.py` or experiment analyzer |
| Output path | `figures/paper/Fig6_disturbance/` |
| Generate now? | **Partial** — within-fiber NCC bars **yes**; accuracy vs disturbance level **no** (no labeled level sweep in data) |
| Missing data | Graded disturbance-level axis; classifier accuracy under disturbance |

---

## Figure 7 — Confusion matrix examples

| Field | Detail |
|-------|--------|
| Scientific question | Which challenge classes are confused per fiber? |
| Data sources | `outputs/final_15fiber_training/Fiber*/confusion_matrix.png` (15 files); `Fiber*/metrics.json` (`per_class_test_accuracy` only) |
| Real & available? | **Partial** — PNG images yes; numeric full confusion matrix **not** in JSON |
| Processing | Compose best/worst PNGs or re-derive from saved predictions (not in repo) |
| Output path | `figures/paper/Fig7_confusion/` (not created) |
| Generate now? | **Image panels only**; numeric heatmap from raw counts **missing data** |
| Missing data | Serialized confusion counts / y_true,y_pred arrays |

---

## Summary table

| Figure | Can generate now | Status |
|--------|------------------|--------|
| Fig 1 | Schematic only | needs missing data (GUI screenshot) for photo panel |
| Fig 2 | Yes | ready |
| Fig 3 | Yes | ready |
| Fig 4 | Yes | ready |
| Fig 5 | Partial | summary JSON ready; time-series panel needs analyzer on JPEGs |
| Fig 6 | Partial | NCC summary ready; disturbance-level accuracy missing |
| Fig 7 | Partial | PNG assets only |
