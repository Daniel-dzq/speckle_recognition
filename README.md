# Optical PUF — Minimal Reproducible Release

Minimal package for the final multimode-fiber speckle challenge–response experiment, trained models, evaluation results, Figure 2 assets, and live GUI demo.

## 1. Project overview

Fifteen physically distinct PMMA fiber channels each have a dedicated neural classifier. A spatial light modulator (SLM) presents a challenge pattern; the camera records the output speckle field. The authorized fiber model must predict the same label as the challenge for **Access Granted**.

## 2. Challenge–response authentication

1. SLM displays challenge (from `challenge_inputs/` or manual text).
2. Light couples into the fiber; speckle is recorded.
3. The selected `FiberN` model outputs a class and confidence.
4. Labels are normalized (letters case-insensitive); match → granted, mismatch or low confidence → denied/unknown.

See `docs/experiment_logic.md`.

## 3. Dataset structure

```
data/recognition_dataset/
  GreenAndRed/Fiber1 … Fiber15/
  RedChange/Fiber1 … Fiber15/
```

240 videos, ~49 GB. **Not tracked in Git** — see `data/README.md`.

## 4. Challenge labels (8 classes)

`1`, `2`, `3`, `a`, `b`, `c`, `boy`, `girl`

`models/final_15fibers/label_map.json` — `num_classes: 8`.

## 5. Training (15 per-fiber models)

```bash
pip install -r requirements.txt

python scripts/train_final_15fibers.py \
  --data_root data/recognition_dataset \
  --domains GreenAndRed RedChange \
  --fibers Fiber1 Fiber2 Fiber3 Fiber4 Fiber5 Fiber6 Fiber7 Fiber8 Fiber9 Fiber10 Fiber11 Fiber12 Fiber13 Fiber14 Fiber15 \
  --output_dir outputs/final_15fiber_training_reproduce \
  --models_dir models/final_15fibers_reproduce \
  --clip_len 16 --input_mode gray --epochs 30 --batch_size 8 --device auto \
  --split_strategy uniform_temporal --no_tqdm --run_auth_matrix
```

Pre-trained weights are already in `models/final_15fibers/Fiber1.pth` … `Fiber15.pth` (~641 MB total).

## 6. 15×15 authentication matrix

Cross-evaluation: each fiber model on every fiber’s test clips.

- Diagonal (same-fiber) mean: **98.0%**
- Off-diagonal mean: **12.7%**
- Eight-class chance: **12.5%**

Outputs: `outputs/final_15fiber_training/auth_matrix_15x15.csv` and `.png`.

## Output organization

| Path | Contents |
|------|----------|
| `outputs/final_15fiber_training/` | Final **recognition and authentication** results (15-fiber training summaries, per-fiber metrics, 15×15 auth matrix) |
| `outputs/physical_characterization/` | **Physical-characterization summary index** (CSV/JSON/MD only; figures under `figures/` and `experiments/*/outputs/`) |
| `experiments/` | Raw/processed experiment **data**, full analysis trees, and **reproduction scripts** |
| `figures/` | Paper-ready **final figures** (Fig. 2, auth matrix) and regeneration scripts |

See `outputs/physical_characterization/physical_characterization_summary.md` for experiment-level file lists.

## Running physical-characterization analyses

User-facing entry points under `scripts/` (implementation details remain in `experiments/*/scripts/`):

```bash
python scripts/run_fiber_loss_analysis.py
python scripts/run_length_optimization.py
python scripts/run_long_term_stability.py
python scripts/run_disturbance_sensitivity.py
python scripts/run_all_physical_characterization.py
```

Useful flags:

- `run_length_optimization.py --run-pipeline` — full length optimization from raw Green JPGs (slow)
- `run_length_optimization.py --overwrite` — promote regenerated Fig2 assets to standard names under `figures/paper/Fig2_length_optimization/`
- `run_*_stability.py --skip-analysis` — sync summaries only if experiment outputs already exist

**Figures:** `figures/paper/` (see `figures/README.md`) · **Summaries only:** `outputs/physical_characterization/` (no duplicated PNG/PDF/SVG there)

## 7. GUI demo

```bash
python scripts/launch_demo.py
```

Loads `challenge_inputs/manifest.json` and `models/final_15fibers/`. Details: `docs/gui_demo.md`.

**Hardware camera (optional):** MindVision SDK binaries are **not tracked on GitHub**. Install the vendor SDK and place local files as needed:

- **macOS:** `gui/libmvsdk.dylib`
- **Windows:** `gui/win_sdk/` plus the vendor camera driver

**Physical experiment raw images** under `experiments/*/data/` are **local-only** (not in Git). Processed CSV/JSON/MD summaries and paper figures under `figures/paper/` are tracked.

## 8. Reproducing PPT challenge inputs

```bash
python scripts/export_ppt_challenges.py \
  --input input.pptx \
  --output_dir challenge_inputs
```

Source deck: `input.pptx` (project root).

## 9. Paper figures

Canonical publication figures live under `figures/paper/<FigName>/`. Generate with:

```bash
python scripts/paper_figures/generate_fig2_length_optimization.py
python scripts/paper_figures/generate_fig3_auth_performance.py
python scripts/paper_figures/generate_fig4_challenge_speckle_examples.py
python scripts/paper_figures/generate_all_paper_figures.py
```

Example paths:

- `figures/paper/Fig2_length_optimization/Fig2_length_optimization.png`
- `figures/paper/Fig3_authentication/Fig3_authentication.png`
- `figures/paper/Fig4_challenge_speckle/Fig4_challenge_speckle.png`

Fig2 reads `experiments/length_optimization/` and `experiments/fiber_loss/data/` (requires `pandas`, `openpyxl`).

## 9b. Additional experiment datasets

Besides the final **15-fiber recognition** dataset (`data/recognition_dataset/`), this release includes physical characterization data under `experiments/`:

| Dataset | Folder | Role |
|---------|--------|------|
| Length optimization | `experiments/length_optimization/` | Fiber length selection; speckle metrics vs total length |
| Fiber loss | `experiments/fiber_loss/` | Red/green transmission loss (Excel sweeps) |
| Long-term stability | `experiments/long_term_stability/` | Reliability over repeated captures |
| Disturbance sensitivity | `experiments/disturbance_sensitivity/` | Robustness under perturbation |

These support the **physical characterization and performance analysis** sections of the report/paper (Section 3.2, Fig. 2–3, stability/robustness panels). Overview: `experiments/README.md`.

## 10. Key final results

| Result | Value |
|--------|--------|
| Fibers trained | **15/15** success |
| Mean test accuracy | **~98.2%** |
| Auth diagonal | **98.0%** |
| Auth off-diagonal | **12.7%** |
| Random baseline | **12.5%** |

`docs/results_summary.md` — full tables and paths.

## 11. Notes and limitations

- **Split:** `uniform_temporal` clip-level split (70/15/15); same `video_id` may appear in train and test with disjoint frame ranges.
- **Limited videos:** one video per class per domain; not a video-level holdout.
- **`gril` typo:** not in the current dataset; code maps `gril` → `girl` if found.
- **Large assets:** do not commit `data/recognition_dataset/` or `models/final_15fibers/*.pth` to normal Git (use LFS, ZIP, or external storage).

## 12. File tree (depth 3)

```
speckle_recognition-main/
├── README.md
├── requirements.txt
├── challenge_inputs/
├── data/recognition_dataset/          # local only (gitignored)
├── models/final_15fibers/             # .pth local only; label_map.json tracked
├── outputs/
│   ├── final_15fiber_training/        # recognition + authentication
│   └── physical_characterization/     # physical experiment summaries
├── experiments/                         # data, scripts, full outputs
├── figures/                             # paper-ready figures
├── scripts/
├── gui/
├── analysis/
└── docs/
```

## Quick checks

```bash
python scripts/train_final_15fibers.py --help
python scripts/export_ppt_challenges.py --help
```

Original development repository is unchanged; this folder is a self-contained copy.
