# Verified figure results — manuscript summary

Concise guidance for paper text based on the current `figures/paper/` pipeline. **Figures were not regenerated** for this document; content reflects **on-disk** metadata, scripts, and exports in this repository.

Paths are relative to the repository root.

---

## Fig. 1 — Concept / block diagram

| Topic | Detail |
|---|---|
| **Status** | **Placeholder** (source assets staged; final vector figure TBD) |
| **Data / outputs** | Staged sources per `docs/figure_plan.md` (e.g. patent-style block diagram); no canonical `figures/paper/Fig1_*` bundle in the auto pipeline yet. |
| **Supports** | High-level dual-channel PUF and authentication *narrative* only once final art exists. |
| **Safe wording** | “Schematic illustrates… (not to scale)” after final figure is placed. |
| **Avoid** | Treating staged bitmaps as publication-final without vector cleanup. |

---

## Fig. 2 — Experimental setup

| Topic | Detail |
|---|---|
| **Status** | **Placeholder** (render / photo staged; journal layout TBD) |
| **Data / outputs** | Source render per `docs/figure_plan.md`; no `figures/paper/Fig2_*` automation bundle. |
| **Supports** | Reader orientation for optics and capture geometry. |
| **Safe wording** | Descriptions that match the **final** labeled diagram actually in the PDF. |
| **Avoid** | Specs or geometry not visible in the final figure. |

---

## Fig. 3 — Total fiber length optimization

| Topic | Detail |
|---|---|
| **Status** | **PI-confirmed** (`confirmed_by_PI`, `optimal_total_fiber_length_cm: 9`, `length_meaning: total_fiber_length_cm` in `figures/paper/Fig3_length_optimization/Fig3_length_optimization_meta.json`) |
| **Data / outputs** | **Sources:** `results/length_optimization_green/tables/per_length_summary.csv` (optional: `results/length_optimization_green/optimal_length.json`). **Exports:** `figures/paper/Fig3_length_optimization/Fig3_length_optimization.{png,pdf,svg}`, `Fig3_length_optimization_data.csv`, `Fig3_length_optimization_meta.json`. |
| **Supports** | Choosing operating **total fiber length** from a controlled sweep (**8, 9, 11, 13, 16 cm** total fiber) using: **(a)** green/red **transmission loss (dB)** vs total length; **(b)** **intra-** vs **inter-class** mean **L₂** distances (ROI) and **inter/intra ratio**; **(c)** **Shannon entropy (bits)** with ±1σ band. The figure marks **9 cm total fiber length** as the selected optimum (vertical guides + `is_selected_optimal` in CSV). |
| **Safe wording** | “**Total fiber length** was swept; **9 cm total fiber length** was taken as the experimental optimum (PI-confirmed).” “Axes report **total** fiber length, not green-only propagation distance.” “At the chosen length, loss, class separability metrics, and ROI entropy support the manuscript operating point.” **Do not** cite legacy **30 cm** or **5 / 30 / 45 cm** cohorts as **final** optimization results; those are **audit/archived only** (see `docs/figure_audit_report.md`). |
| **Avoid** | Calling **green_prop_mm** (bookkeeping column in the table) the optimization axis. Framing **only** “8–16 cm” as the conclusion **without** naming the **9 cm** PI choice. Any mixing with archived partial-length runs. |

---

## Fig. 4 — Surface / entropy source (roughness, diversity)

| Topic | Detail |
|---|---|
| **Status** | **Placeholder** / **TODO** (profilometer / extended assets missing per figure plan) |
| **Data / outputs** | None in `figures/paper/` automation; see `docs/figure_audit_report.md`. |
| **Supports** | — |
| **Safe wording** | None until data exist. |
| **Avoid** | Quantitative roughness or diversity claims without tabulated measurements. |

---

## Fig. 5 — Dual-channel characteristics

| Topic | Detail |
|---|---|
| **Status** | **PI-confirmed** (`manuscript_ready: true`, `source_dataset_status: final_or_PI_confirmed`, `data_validated_by_PI: true` in `figures/paper/Fig5_dual_channel/Fig5_dual_channel_meta.json`) |
| **Data / outputs** | **Sources:** `figures/new_datasets_analysis/metrics_summary.json`; **panel (c)** uses middle-frame proxies from `videocapture/Green/Fiber1/A.avi` (green channel) and `videocapture/GreenAndRed/Fiber1/A.avi` (red sampled from R channel). **Exports:** `figures/paper/Fig5_dual_channel/Fig5_dual_channel.{png,pdf,svg}`, `Fig5_dual_channel_data.csv`, `Fig5_dual_channel_meta.json`. |
| **Supports** | **(a)** Temporal **stability** per fiber: adjacent-frame **NCC** (`long_term_stability.per_fiber`). **(b)** **Disturbance** sensitivity: **within-fiber mean NCC** (`disturbance_sensitivity.within_fiber_mean_ncc`). **(c)** **Morphology:** normalized **radial mean intensity** vs radius (pixels) for **green vs red** from representative frames—comparing speckle profiles between channels. |
| **Safe wording** | “Dual-channel captures: **green (~challenge-sensitive / speckle-bearing)** vs **red (~reference)** behaviour, consistent with **red as common reference** and **green as challenge response**, as summarized in the PI-validated metrics export.” Tie bars to **NCC-based** stability and disturbance summaries; tie panel (c) to **representative** single-frame radial profiles (not full statistical study of all videos unless expanded). |
| **Avoid** | Implying panel (c) is exhaustive over all fibers/times without regenerating from a defined frame protocol. Dropping the distinction that **red** is used as **reference** and **green** carries **challenge-dependent** speckle in this setup. |

---

## Fig. 6 — Common-mode suppression (η, CV)

| Topic | Detail |
|---|---|
| **Status** | **Draft** (`manuscript_ready: false`; `reason` and `paired_data_search_note` in `figures/paper/Fig6_common_mode_suppression/Fig6_common_mode_suppression_meta.json`) |
| **Data / outputs** | **Sources:** `power_common_mode/` (e.g. pooled P90 JPEGs for recomputed **raw green** CV); η summary bar can reflect **manuscript/summary constants** (see `conflict_note` in meta). **Exports:** `figures/paper/Fig6_common_mode_suppression/Fig6_common_mode_suppression.{png,pdf,svg}`, `Fig6_common_mode_suppression_data.csv`, `Fig6_common_mode_suppression_meta.json`. |
| **Supports** | *Illustrative only until* **time-aligned paired red/green** power-fluctuation data are **verified** and η recomputed from them. |
| **Safe wording** | “**Draft** analysis; **raw green** CV recomputed from the pooled images; **η (G/R) quantitative comparison deferred** until verified paired channels are available.” |
| **Avoid** | **Final** claims that **common-mode suppression** or **G/R CV reduction** is **experimentally proven** at manuscript level **unless** η and denominators are **recomputed from verified paired red/green** captures. Treat the **4.3% η CV** bar (manuscript summary) as **provisional** relative to the diagnostic pooled G/R statistic noted in metadata/CSV. |

---

## Fig. 7 — Authentication (fiber matrix + letter protocol)

| Topic | Detail |
|---|---|
| **Status** | **Draft** (pipeline outputs present; **no** PI confirmation flags in `Fig7_authentication_meta.json`; metadata notes missing **7-day stability** time series) |
| **Data / outputs** | **Sources:** `results/fiber_auth/auth_matrix.json`, `results/unified/test_predictions.csv`. **Exports:** `figures/paper/Fig7_authentication/Fig7_authentication.{png,pdf,svg}`, `Fig7_authentication_data.csv`, `Fig7_authentication_meta.json`, plus `Fig7_authentication_fiber_matrix.csv`, `Fig7_authentication_letter_confusion.csv`, `Fig7_authentication_roc_curve.csv`. |
| **Supports** | **Separate concerns:** **(a)** **Fiber/device** closed-set matrix from `auth_matrix` (% scores). **(b–d)** **Letter readout** on the **unified** export: confusion matrix, ROC/AUC on **correct vs incorrect** using reported **confidence**, and score histograms. ROC here is **not** a dedicated fiber-level verification ROC unless you add matching exports. |
| **Safe wording** | “**Letter challenge** classification performance on `test_predictions.csv` (chance baseline **1/26**).” “**Fiber authentication** matrix from **enrolled vs predicted** fiber scores (`auth_matrix.json`); describe protocol exactly as used to build that matrix.” Acknowledge **metadata note** if discussing long-run stability. |
| **Avoid** | Conflating **letter** ROC/confusion with **fiber** impostor–genuine verification unless `fiber_auth` provides aligned scores. Claiming **7-day** stability curves **from this repository** without importing that data. |

---

## Fig. 8 — Demo / workflow

| Topic | Detail |
|---|---|
| **Status** | **Placeholder** / **Partial** (GUI capture supplementary) |
| **Data / outputs** | Per `docs/figure_plan.md` (e.g. demo screenshot if present); not part of the core verified metrics pipeline. |
| **Supports** | Qualitative two-factor / GUI story only. |
| **Safe wording** | Screenshots as **illustrative UI**, not quantitative proof. |
| **Avoid** | Metrics or security claims grounded only in demo images. |

---

## Immediate manuscript replacements

Use these to retire vague or superseded placeholder text:

- **Length / operating point:** Replace any undated “length scan” or ambiguous optimum with: **total fiber length** was optimized over **8–16 cm**; **9 cm total fiber length** is the **PI-confirmed** choice; figures and table cite `per_length_summary.csv` / Fig. 3 bundle. **Do not** reintroduce **30 cm** or **5 / 30 / 45 cm** plots as the **final** optimization story.
- **Dual-channel behaviour:** Replace generic “we observe two channels” claims with the **PI-validated** summaries in Fig. 5: **stability NCC**, **disturbance NCC**, and **representative green vs red radial profiles**, with explicit **reference (red) vs challenge-sensitive (green)** language.
- **Common-mode / η:** Replace any **definitive** sentence on **η CV suppression** or **G/R fluctuation cancellation** with **draft / pending paired verification** language until Fig. 6 metadata flips to final.
- **Authentication:** Split prose into **fiber matrix (closed-set %)** vs **letter protocol (26-class, ROC on confidence)** so readers do not mix **device identity** with **challenge readout** metrics.
- **Missing artifacts:** Any paragraph promising **roughness histograms**, **7-day curves**, or **full-video radial statistics** should be cut or tagged **future work** until Fig. 4 / Fig. 7 notes are resolved.

For regeneration and archival rules, see `docs/figure_generation_policy.md` and `docs/figure_plan.md`.
