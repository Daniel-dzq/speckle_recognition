# Results summary

Final training used `uniform_temporal` clip splits, 8 classes, 30 epochs per fiber, gray 16-frame clips.

## Per-fiber training

| Metric | Value |
|--------|--------|
| Fibers completed | **15 / 15** (`status=ok`) |
| Classes | **8** |
| Mean test accuracy | **~98.2%** |
| Lowest test accuracy | **Fiber10 ~92.2%** |

Details: `outputs/final_15fiber_training/summary_15fibers.csv`

## 15×15 authentication matrix

| Metric | Value |
|--------|--------|
| Same-fiber (diagonal) mean | **98.0%** |
| Cross-fiber (off-diagonal) mean | **12.7%** |
| Random baseline (8 classes) | **12.5%** |

Files:

- `outputs/final_15fiber_training/auth_matrix_15x15.csv`
- `outputs/final_15fiber_training/auth_matrix_15x15.png`
- `figures/auth_matrix_15x15.png` (copy for figures section)
- `outputs/final_15fiber_training/auth_matrix_report.md`

## Interpretation

High diagonal accuracy means each fiber model recognizes its own speckle data. Off-diagonal near 12.5% means models do not generalize across fibers, as expected for a PUF-style identity.

## Per-fiber artifacts

For each `FiberN/` under `outputs/final_15fiber_training/`:

- `metrics.json`
- `confusion_matrix.png`
- `training_log.csv`
- `split_report.md` / `split_report.json`
