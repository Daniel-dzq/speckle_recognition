# Archive

This directory contains files that are no longer part of the active workflow
but are preserved for historical context, reproducibility of older results,
or occasional reference. Nothing inside `archive/` is imported by the main
codebase.

Feel free to restore any file to the project root (or the appropriate
module) if you need to revive an old experiment.

## Contents

### Legacy training scripts

| File | Superseded by | Notes |
| ---- | ------------- | ----- |
| `train_model.py` | `scripts/main.py`, `scripts/train_fiber.py` | Single-frame CNN training entry point (older API) |
| `run_pipeline.py` | `scripts/train_all_fibers.py` | Multi-stage orchestration wrapper |
| `train_video_cad_model.py` | `scripts/train_unified.py` | Random-forest CAD baseline (exploratory) |
| `train_video_cad_model_fixed.py` | `scripts/train_unified.py` | Iteration of the CAD baseline |
| `train_video_cad_model_final.py` | `scripts/train_unified.py` | Final iteration of the CAD baseline |

### Legacy GUI / utilities

| File | Superseded by | Notes |
| ---- | ------------- | ----- |
| `deep_learning_gui.py` | `gui/main_window.py` (PySide6) | Tkinter all-in-one GUI |
| `visualize_features.py` | `scripts/make_paper_figures.py`, `analysis/plotting/` | Standalone feature visualiser |
| `video_screenshot.py` | `gui/camera_worker.py`, `analysis/io/video.py` | One-shot frame grabber |

### One-off debug scripts

| File | Notes |
| ---- | ----- |
| `test_extract.py` | Ad-hoc ffmpeg extraction sanity check |
| `test_ffmpeg.py` | Ad-hoc ffmpeg sanity check |

### Non-source assets

| File | Notes |
| ---- | ----- |
| `paper_draft.docx` | Local paper draft (Chinese filename removed; content preserved). |
| `figures.zip` | Snapshot of publication figures — regenerate with `python scripts/make_paper_figures.py`. |
