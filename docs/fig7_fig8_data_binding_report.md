# Fig. 7 / Fig. 8 — data binding audit

This report records what exists under the repository roots, what was computable for `scripts/generate_competition_figures.py`, and the **verification tier** for each panel. All figure text is generated in **English** only.

---

## Fig. 7(a) — time stability (intra-class correlation)

### `long_term_stability/` (green channel speckle JPEGs)

- **Present:** yes — `long_term_stability/Fiber1` … `Fiber15`, each with numbered `*.JPG` repeats (e.g. `1.JPG` … `13.JPG`).
- **Layout:** `flat_fiber_repeat` — numeric stems, increasing `Capture.repeat` order (see `analysis/io/dataset.py`).
- **Channel:** folder does **not** separate red vs green; paired with `extract_features(..., grayscale=True)` → **single-channel processed green speckle** (same pipeline as `scripts/analyze_new_datasets.py`).
- **Metric:** per-fiber `temporal_stability_score` → **mean consecutive-frame NCC** in embedding space after preprocess (crop 400 → resize 112 → min–max), **not** raw bitmap NCC.
- **Aggregation for figure:** across fibers, **mean ± population standard deviation** of those per-fiber means.
- **Red 1-hour / 5-minute sampling:** **not found** as a separate red-only `long_term_stability` tree or metadata in-repo; red temporal bars **cannot** be tied to this folder.
- **Status:** **partial_raw_data** — green = **raw_data_verified** (computed from `long_term_stability`); red = **summary_statistics_verified** (manuscript constant **0.94**), **not** masked as raw.

---

## Fig. 7(b) — perturbation / bending sensitivity

### `disturbance_sensitivity/`

- **Present:** yes — `Fiber1` … `Fiber15`, repeats `1.JPG` … `5.JPG` (layout `flat_fiber_repeat`).
- **Explicit before/after bend labels:** **not** found in filenames or sidecar metadata.
- **Computation used:** for each `FiberK` present in **both** `long_term_stability` and `disturbance_sensitivity`, compute per-fiber **consecutive NCC** (same embedding pipeline) for each dataset, then  
  **correlation decrease (%)** = `(1 - NCC_dist / NCC_LT) × 100` when `NCC_LT > 0`.
- **Green (plotted):** **mean of the 15 paired values** (~**30.4%** in this workspace with Anaconda run) → **raw_data_verified** (paired cross-dataset proxy; not a literal labeled “before/after bend” time series).
- **Red:** no red channel disturb folder → **summary_statistics_verified** **5%** for the bar height (manuscript).
- **5% / 25% manuscript check:** recomputed green mean **~30%**, not **25%** — noted as **methodology difference** (paired LT vs disturb) in `data_fig7_dual_channel_characterization.csv`.

---

## Fig. 7(c) — mode / speckle comparison

### Still frames from `videocapture/`

- **Proxy images used (when available):**  
  - “Green / side” reference: `videocapture/Green/Fiber1/A.avi` — **B / G / R** channels from **middle frame**.  
  - “Red / end-face” reference: `videocapture/GreenAndRed/Fiber1/A.avi` — **R channel** as red dominance; **G channel** as green leakage in dual-illumination capture (labeled honestly in CSV).
- **Dedicated high-res stills** named “red_speckle.tif / green_speckle.tif” — **not** located in-repo.
- **Radial profiles:** built from the **same frames**; normalized to peak 1 for overlay.
- **Status:** **proxy_image** — real video frames, **not** a calibrated still-photo pair; manifest flag **missing_dedicated_raw_stills**.

---

## Fig. 8(a) — power fluctuation (CV)

### `power_common_mode/` and recomputation

- **Images:** `power_common_mode/<FiberK>/<Pxx>/<1–5>.JPG`, RGB `uint8`.
- **Definitions tested in audit:**
  - **Pooled P90-only** per-image **mean green (G channel)** across **all** captures: **CV ≈ 39.2%** (close to manuscript **38.2%**).
  - Per-image **η ≈ mean(G)/mean(R)** on the **same pool**: **CV ≫ 4.3%** (typically **O(10²)%** in the same pool) — **does not reproduce** manuscript η-CV with this simple scalar definition.
- **Plotted policy:** bar 1 = **recomputed raw green CV** (P90 pooled); bar 2 = **manuscript η CV = 4.3%** (**summary_statistics_verified**) with explicit **conflict note** in CSV/manifest.
- **`metrics_summary.json`:** contains **L2 class-separation metrics per power**, **not** intensity CV or η CV — useful for Fig. 5–style explorer plots, **not** for direct 38.2 / 4.3 extraction.

---

## Fig. 8(b) — mechanical reinstallation robustness

### Raw correlation for “raw green intensity feature” vs “η feature”

- **Searched:** `long_term_stability`, `disturbance_sensitivity`, `power_common_mode`, `figures/new_datasets_analysis/metrics_summary.json` — **no** pre-computed side-by-side intra-class NCC bars for raw-mean-intensity vs η-ratio feature suitable for this two-bar figure.
- **Plotted policy:** two schematic summary bars (+**28%** relative uplift on η) = **summary_statistics_verified**; **no** synthetic scatter of N replicates.

---

## Summary table

| Panel | Dominant source | Verification tier |
|-------|-----------------|-------------------|
| 7(a) green | `long_term_stability` + feature pipeline | **raw_data_verified** |
| 7(a) red | manuscript constant | **summary_statistics_verified** |
| 7(b) green | paired LT vs `disturbance_sensitivity` | **raw_data_verified** (proxy pairing) |
| 7(b) red | manuscript | **summary_statistics_verified** |
| 7(c) | `videocapture` middle frames | **proxy_image** |
| 8(a) green | `power_common_mode` P90 pool | **raw_data_verified** (recomputed) |
| 8(a) η | manuscript | **summary_statistics_verified** (conflict documented) |
| 8(b) | manuscript | **summary_statistics_verified** |

---

## Regeneration snapshot (this workspace)

Command: `python scripts/generate_competition_figures.py --fig 7 8` (Anaconda Python with project `analysis` stack).

| Quantity | Value |
|----------|-------|
| Fig. 7(a) green mean consecutive NCC | ≈ **0.928** (15 fibers); std ≈ **0.064** |
| Fig. 7(b) green mean correlation decrease | ≈ **30.35%** (paired LT vs disturb); red bar **5%** (summary) |
| Fig. 8(a) raw green CV (P90 pooled, `n = 75`) | ≈ **39.16%** (`raw_data_verified`) |
| Fig. 8(a) η CV on figure | **4.3%** (`summary_statistics_verified`; diagnostic pooled scalar η CV ≈ **85.8%** — see `data_fig8_common_mode_suppression.csv`) |
| Fig. 8(b) NCC bars | **0.72** vs **0.9216** (+28% summary) |
