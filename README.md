# Speckle Recognition: Dual-Channel Optical PUF Authentication

## 1. Project overview

This repository is a minimal, reproducible release of a **dual-channel multimode polymer optical fiber physical unclonable function (PUF)** for challenge–response authentication.

The system combines:

- **Programmable challenge inputs** exported from a PowerPoint deck and displayed on a spatial light modulator (SLM)
- **Dual-channel optical excitation** (red reference / end-face and green programmable / side illumination)
- **Speckle response** captured by a camera
- **Per-fiber recognition models** (one trained classifier per authorized fiber)
- **Challenge-prediction matching** for access control (predicted label must match the sent challenge)

**Final recognition experiment (15 fibers, 8 challenge classes):**

| Metric | Value |
|--------|--------|
| Fibers | 15 (one model each) |
| Challenge classes | 8 |
| Same-fiber authentication (diagonal of 15×15 matrix) | ~**98.0%** mean |
| Cross-fiber (off-diagonal) | ~**12.7%** mean |
| Random 8-class baseline | **12.5%** |

All reported numbers are computed from tracked summary CSVs under `outputs/final_15fiber_training/`; see Section 8.

---

## 2. Physical authentication principle

Authentication is **physical**: the fiber microstructure maps a challenge to a speckle pattern that is hard to clone.

1. A **challenge image** is shown on the SLM (programmable green channel / side illumination).
2. Light propagates through a **specific fiber**; scattering produces a **fiber-specific speckle** field on the camera.
3. The speckle depends on the fiber’s **physical microstructure** (geometry, inclusions, coupling), not on software alone.
4. A model **trained only for the authorized fiber** decodes the challenge class from short video clips of the speckle.
5. If the **predicted label matches** the challenge that was sent, access is granted.
6. If a **wrong fiber** is presented, the authorized model’s predictions are near **chance** (~12.5% for 8 classes), so cross-fiber impersonation fails.

**Dual-channel layout (conceptual):**

- **Red channel** — reference / end-face illumination; stabilizes alignment and context for the recorded field.
- **Green channel** — programmable / side-illumination path carrying the **challenge pattern** from the SLM.
- **Speckle on the sensor** — joint response used for recognition; **PUF uniqueness** comes from per-fiber multimode interference and scattering.

This release documents measured recognition and physical-characterization results; it does not claim perfect security against all attack models.

---

## 3. Repository structure

```
.
├── README.md
├── requirements.txt
├── input.pptx
├── challenge_inputs/
├── data/
├── models/
├── outputs/
├── figures/
├── experiments/
├── gui/
├── scripts/
│   └── paper_figures/
├── docs/
├── analysis/
├── final_fiber_dataset.py
├── unified_dataset.py
├── train_eval.py
└── models.py
```

| Path | Role |
|------|------|
| `challenge_inputs/` | Exported SLM challenge PNGs and manifest |
| `data/recognition_dataset/` | Local recognition videos (**not** in Git) |
| `models/final_15fibers/` | Final per-fiber weights (`.pth` local only); `label_map.json` tracked |
| `outputs/final_15fiber_training/` | Training logs, per-fiber metrics, 15×15 auth matrix |
| `outputs/physical_characterization/` | Summary CSV/JSON/MD for physical experiments (no duplicate figures) |
| `figures/paper/` | Final paper-ready figures (PNG/PDF/SVG + sidecar CSV/MD) |
| `experiments/` | Physical-characterization datasets, scripts, and detailed outputs |
| `gui/` | Live demo GUI (PySide6 + optional MindVision camera) |
| `scripts/` | User-facing training, export, physical runs, demo launcher |
| `scripts/paper_figures/` | Regenerate figures from real data files |
| `docs/` | Experiment logic, GUI notes, build/release notes |
| `analysis/` | Shared plotting/metrics helpers used by figure scripts |

Root Python modules (`final_fiber_dataset.py`, `unified_dataset.py`, `train_eval.py`, `models.py`) implement dataset discovery, splits, training, and evaluation.

---

## 4. Installation

**Python virtual environment (recommended):**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Conda (if you already use the project environment):**

```bash
conda activate recognition
pip install -r requirements.txt
```

**Main dependencies:**

| Component | Use |
|-----------|-----|
| **PyTorch** | Training and inference |
| **OpenCV** | Video I/O and image processing |
| **PySide6** | GUI demo |
| **pandas / openpyxl** | Some physical-characterization tables (e.g. length optimization, fiber loss) |

**Hardware camera (optional for live demo):**

- **MindVision** industrial camera SDK is required for live capture in `gui/`.
- SDK binaries (`gui/libmvsdk.dylib`, `gui/win_sdk/`, etc.) are **local/external** and are **not** tracked in Git.
- Install vendor drivers and place libraries per `docs/gui_demo.md`.

---

## 5. Challenge inputs

- **Source deck:** `input.pptx` (project root)
- **Exported images:** `challenge_inputs/` (PNG per class)

**Final eight classes:**

`1`, `2`, `3`, `a`, `b`, `c`, `boy`, `girl`

The GUI may show **A / B / C** while model labels in `label_map.json` are lowercase **`a` / `b` / `c`**. Comparison is **case-insensitive**.

**Regenerate from PowerPoint:**

```bash
python scripts/export_ppt_challenges.py \
  --input input.pptx \
  --output_dir challenge_inputs \
  --force
```

---

## 6. Dataset

Place the recognition dataset locally (not supplied in Git):

```
data/recognition_dataset/
  GreenAndRed/
    Fiber1/ … Fiber15/
  RedChange/
    Fiber1/ … Fiber15/
```

Each fiber folder under each domain contains **eight videos**:

`1.avi`, `2.avi`, `3.avi`, `a.avi`, `b.avi`, `c.avi`, `boy.avi`, `girl.avi`

| Quantity | Count |
|----------|--------|
| Domains | 2 (`GreenAndRed`, `RedChange`) |
| Fibers per domain | 15 |
| Classes per fiber | 8 |
| **Total videos** | **240** (2 × 15 × 8) |

Videos are **ignored by Git** due to size. Copy your dataset under `data/recognition_dataset/` before training or GUI inference that reads files from disk.

See `data/README.md` for layout notes.

---

## 7. Training final 15-fiber models

**One model per fiber** — each `FiberN.pth` sees only that fiber’s clips from both domains.

```bash
python scripts/train_final_15fibers.py \
  --data_root data/recognition_dataset \
  --domains GreenAndRed RedChange \
  --fibers Fiber1 Fiber2 Fiber3 Fiber4 Fiber5 Fiber6 Fiber7 Fiber8 Fiber9 Fiber10 Fiber11 Fiber12 Fiber13 Fiber14 Fiber15 \
  --output_dir outputs/final_15fiber_training \
  --models_dir models/final_15fibers \
  --clip_len 16 \
  --input_mode gray \
  --epochs 30 \
  --batch_size 8 \
  --device auto \
  --split_strategy uniform_temporal \
  --no_tqdm \
  --run_auth_matrix
```

**Split:** `uniform_temporal` assigns clip-level train/val/test segments within each video (70% / 15% / 15% of clips), not a full-video holdout — required because there is only **one video per class per domain**.

**Typical clip counts per fiber (final run):** 176 train / 32 val / 64 test (from 16 source videos × temporal tiling).

**Labels:** 8 classes; `models/final_15fibers/label_map.json` is tracked. **`.pth` weights are not in Git** — train locally or copy weights into `models/final_15fibers/`.

---

## 8. Evaluation and authentication matrix

After training with `--run_auth_matrix`, key artifacts under `outputs/final_15fiber_training/`:

| File | Description |
|------|-------------|
| `summary_15fibers.csv` / `summary_15fibers.md` | Per-fiber train/val/test accuracy and clip counts |
| `auth_matrix_15x15.csv` | Each row = model fiber, columns = test clips from each fiber |
| `auth_matrix_15x15.png` | Heatmap visualization |
| `auth_matrix_report.md` | Diagonal/off-diagonal statistics |

**Reported final results (from tracked summaries):**

- **15/15** fibers trained successfully
- Mean per-fiber **test accuracy** ~**98%**
- Auth matrix **same-fiber (diagonal) mean** ~**98.02%**
- Auth matrix **cross-fiber (off-diagonal) mean** ~**12.69%**
- **Random baseline** (8 classes) **12.5%**

**Interpretation:**

- **High diagonal** — authorized fiber + correct model → challenge recognized
- **Off-diagonal near 12.5%** — wrong fiber does not fool the authorized model

---

## 9. GUI demo

Layout (conceptual):

- **Left** — challenge selection / SLM control
- **Center** — live or file-based speckle view
- **Right** — recognition result and robot access decision

**Launch:**

```bash
python scripts/launch_demo.py
```

**Flow:**

1. Choose authorized model **Fiber1–Fiber15**
2. Select a challenge image (from `challenge_inputs/` or UI)
3. Send pattern to SLM (hardware) or use recorded/video path per setup
4. Model predicts class + confidence
5. GUI compares **normalized challenge** vs **prediction** (case-insensitive)
6. **Confidence-aware smoothing** — low confidence does not assert strong “access granted”

**Requirements:**

- `models/final_15fibers/label_map.json` (tracked)
- `models/final_15fibers/FiberN.pth` locally (not in Git)
- Live camera: MindVision SDK installed (see Section 4)

Details: `docs/gui_demo.md`.

---

## 10. Physical-characterization experiments

Under `experiments/`:

| Directory | Purpose |
|-----------|---------|
| `length_optimization/` | Speckle metrics vs fiber length; **9 cm** selected for final setup |
| `fiber_loss/` | Red/green transmission loss sweeps |
| `long_term_stability/` | Repeated captures / stability over time |
| `disturbance_sensitivity/` | Response under mechanical or environmental disturbance |

**Root wrappers (recommended entry points):**

```bash
python scripts/run_fiber_loss_analysis.py
python scripts/run_length_optimization.py
python scripts/run_long_term_stability.py
python scripts/run_disturbance_sensitivity.py
python scripts/run_all_physical_characterization.py
```

**Outputs:**

- `experiments/<name>/outputs/` — detailed per-run files, plots, caches
- `outputs/physical_characterization/` — **summary index only** (CSV/JSON/MD synced from experiments)
- `figures/paper/` — final figure assets for the manuscript

Raw image folders under `experiments/*/data/` are typically **local-only** (gitignored). See `outputs/physical_characterization/physical_characterization_summary.md`.

---

## 11. Paper figures

Final figures live under `figures/paper/`:

| Folder | Topic |
|--------|--------|
| `Fig2_length_optimization/` | Length sweep and selected 9 cm |
| `Fig3_authentication/` | 15-fiber auth performance and matrix |
| `Fig4_challenge_speckle/` | Challenge PNGs + example speckle frames |
| `Fig5_stability/` | Long-term stability summary |
| `Fig6_disturbance/` | Disturbance sensitivity summary |

**Regenerate (values from real CSV/JSON/video/image inputs — no synthetic placeholders):**

```bash
python scripts/paper_figures/generate_fig2_length_optimization.py
python scripts/paper_figures/generate_fig3_auth_performance.py
python scripts/paper_figures/generate_fig4_challenge_speckle_examples.py
python scripts/paper_figures/generate_fig5_stability.py
python scripts/paper_figures/generate_fig6_disturbance.py
python scripts/paper_figures/generate_all_paper_figures.py --skip-fig2
```

**Notes:**

- Fig2 needs local length-optimization data under `experiments/length_optimization/data/` when running the full pipeline.
- Fig5/Fig6 are **supplementary-style** summaries from available aggregate metrics unless extended time-series/sweep data are added later.

See `figures/README.md` and `outputs/figure_generation/` for data audits and figure plans.

---

## 12. What is not tracked in Git

**Not tracked (local, large, or vendor-specific):**

- `data/recognition_dataset/` — raw `.avi` recognition videos
- `models/final_15fibers/*.pth` — trained weights
- Raw experiment image trees (e.g. `experiments/length_optimization/data/`, stability/disturbance raw folders when gitignored)
- MindVision SDK binaries (`gui/libmvsdk.dylib`, `gui/win_sdk/`)
- Cache directories, logs, `__pycache__`, `.DS_Store`

**Tracked:**

- Source code, `scripts/`, `gui/*.py`, `analysis/`
- `docs/`, `requirements.txt`, `.gitignore`
- `challenge_inputs/`, `input.pptx`, `label_map.json`
- Summary **CSV / JSON / MD** under `outputs/`
- `figures/paper/` final assets and `scripts/paper_figures/`
- Figure-generation notes under `outputs/figure_generation/`

---

## 13. Reproducibility checklist

- [ ] Create venv / conda env and `pip install -r requirements.txt`
- [ ] Copy `data/recognition_dataset/` (240 videos) locally
- [ ] Place `models/final_15fibers/*.pth` (if using GUI without retraining)
- [ ] Export challenges: `scripts/export_ppt_challenges.py` (Section 5)
- [ ] Train: `scripts/train_final_15fibers.py` with `--run_auth_matrix` (Section 7)
- [ ] Verify `outputs/final_15fiber_training/summary_15fibers.csv` and auth matrix (Section 8)
- [ ] Optional: `scripts/launch_demo.py` with SDK if using live camera (Section 9)
- [ ] Optional: physical characterization wrappers (Section 10)
- [ ] Optional: regenerate paper figures (Section 11)

---

## 14. Key results

| Item | Value |
|------|--------|
| Fibers | 15 |
| Classes | 8 |
| Labels | `1`, `2`, `3`, `a`, `b`, `c`, `boy`, `girl` |
| Same-fiber mean (auth diagonal) | 98.02% |
| Cross-fiber mean (off-diagonal) | 12.69% |
| Random baseline | 12.5% |
| Selected fiber length (physical study) | 9 cm |

Extended tables: `docs/results_summary.md`, `outputs/final_15fiber_training/auth_matrix_report.md`.

---

## 15. Notes and limitations

- **Temporal clip split** (`uniform_temporal`) is used instead of a video-level holdout because each class has only **one video per domain**; the same `video_id` can contribute clips to both train and test with **disjoint frame ranges** (leakage checks are in the training script).
- **Stability / disturbance figures** reflect **available summary metrics**; they are not full multi-day time-series panels unless additional data are added.
- **Hardware reproducibility** depends on MindVision driver/SDK versions and camera calibration.
- **Model weights and raw videos** are intentionally **out of Git**; use external storage or train locally.
- Legacy typo `gril` is mapped to `girl` in dataset code if such files appear.

---

## 16. Citation / project status

If you use this code or results in academic work, cite the associated manuscript or technical report when it is available. This tree is a **minimal public release** aligned with the final 15-fiber optical PUF experiment; contact the repository maintainers for dataset or weight distribution if needed.

---

## Quick reference

```bash
python scripts/train_final_15fibers.py --help
python scripts/export_ppt_challenges.py --help
python scripts/launch_demo.py
```

Build and packaging notes: `docs/BUILD_RELEASE.md`.
