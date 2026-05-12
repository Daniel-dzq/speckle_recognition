# Speckle recognition for multimode-fiber optical PUFs

Research code for **26-class letter challenge–response** over fiber speckle: patterns are shown on a spatial light modulator (SLM), light propagates through a multimode fiber, speckle is imaged with a camera, and a **ResNet18 + temporal pooling** model predicts the letter. Models trained on one fiber **do not** generalize to other fibers, which supports fiber-specific PUF-style authentication.

The repo is **training stack + PySide6 live demo + config-driven experiment package** under `analysis/`. Source, UI strings, and comments are **English-only**. Large binaries (weights, raw video, generated figures, Word documents) are **not** tracked; regenerate them locally or use `.gitignore` as the reference list.

## Requirements

- Python **3.9+**
- PyTorch **1.13+** (CUDA optional; MPS on Apple Silicon)
- OpenCV, NumPy, PyYAML
- **PySide6** for the demo GUI and the results dashboard

Install dependencies:

```bash
git clone https://github.com/Daniel-dzq/speckle_recognition.git
cd speckle_recognition
pip install -r requirements.txt
```

Sanity check:

```bash
python scripts/env_check.py
```

## Repository layout

| Path | Role |
|------|------|
| `models.py`, `dataset.py`, `unified_dataset.py`, `train_eval.py` | Core model, datasets, training/eval |
| `scripts/` | CLI entry points (train, eval, predict, figures, experiments) |
| `gui/` | Live demo (`MainWindow`, `SLMWindow`, `RobotPanel`, camera workers, MindVision ctypes wrapper + bundled SDK where applicable) |
| `analysis/` | Reusable experiment pipeline: I/O, caching, metrics, plotting, reporting |
| `config/` | YAML for `analysis` experiments |
| `letter_images/` | Pre-rendered A–Z PNGs for SLM display |
| `docs/` | Extra notes (`experiments.md`, `gui_tutorial.md`, legacy usage) |
| `archive/` | Superseded scripts; see `archive/README.md` |

**Not committed** (typical): `videocapture/` or `video_capture/`, `results/`, `checkpoints/`, `figures/`, `*.pth`, local Word/PDF drafts, scratch dataset trees. See `.gitignore`.

## Quick start: live demo

Trained weights are expected under `results/fiber_auth/fiber_models/` (e.g. after `fiber_auth_eval`). Then:

```bash
python scripts/launch_demo.py
```

The window title is **Speckle-PUF Live Demo**. Use **Fiber Authentication** to pick a model, **Camera / Video Source** for OpenCV or MindVision capture, **SLM Output Window** to drive the external display routed to the SLM, and **Inference Settings** to tune frame rate and voting.

Optional **automated PNG grabs** for documentation (needs a display; uses deterministic UI hooks):

```bash
python scripts/capture_manual_screenshots.py --auto --native
```

Outputs default to `figures/softcopyright/` (that tree is ignored when committed).

## Training and PUF evaluation

Place per-letter videos under a domain layout such as:

`videocapture/<Green|GreenAndRed|RedChange>/FiberN/<A–Z>.avi`

Then, for example:

```bash
# One fiber
python scripts/train_fiber.py --fiber fiber1

# All fibers in batch
python scripts/train_all_fibers.py

# 5×5 train-per-fiber, test-on-all-fiber authentication matrix
python scripts/fiber_auth_eval.py
```

Unified multi-domain training and cross-fiber tests live under `scripts/train_unified.py`, `scripts/evaluate_unified.py`, and `scripts/evaluate_cross_fiber.py`. Domain ablations: `scripts/diagnose_domains.py`, `scripts/run_all_fiber_ablations.py`.

## Analysis framework

Config-driven runs write under `results/<run_name>/` (also ignored by git unless you change policy):

```bash
python scripts/run_experiment.py system_setup --config config/system_setup.yaml
```

Wrappers exist as `scripts/run_<name>.py`. Browse outputs:

```bash
python scripts/launch_dashboard.py
```

Details: `docs/experiments.md`.

## Publication figures

From existing `results/` and data paths configured in the plotting scripts:

```bash
python scripts/make_paper_figures.py
```

Extended journal-style exports may use `scripts/make_publication_figures.py` when those dependencies and data paths are available locally.

## Hardware notes

- **MindVision / HuaTengVision** USB cameras: use the MindVision path in the GUI; macOS bundles `gui/libmvsdk.dylib`, Windows uses `gui/win_sdk/`.
- **SLM** appears as a normal display output; select it under **SLM screen**. The center preview is **not** the SLM surface.

## License

MIT License
