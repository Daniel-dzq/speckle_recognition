# Dual-Channel Multimode Fiber Optical PUF — Speckle Recognition System

A deep-learning-based optical Physical Unclonable Function (PUF) system that uses speckle patterns from multimode plastic optical fibers for **26-letter challenge–response authentication** and **fiber-specific identity verification**.

The system encodes letters (A–Z) onto a Spatial Light Modulator (SLM), injects the patterned light into a prepared multimode fiber, and captures the resulting speckle output with a camera. A CNN classifier then decodes the letter from the speckle image. Because each fiber produces a unique speckle transformation due to its internal random microstructure, the same model trained on one fiber **fails** on a different fiber — this asymmetry is the basis for PUF authentication.

---

## Table of Contents

- [System Overview](#system-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Layout](#data-layout)
- [Training Workflows](#training-workflows)
  - [Single-fiber training](#1-single-fiber-training)
  - [Multi-fiber batch training](#2-multi-fiber-batch-training)
  - [Fiber-PUF authentication evaluation](#3-fiber-puf-authentication-evaluation)
  - [Unified multi-domain training](#4-unified-multi-domain-training)
  - [Domain ablation study](#5-domain-ablation-study)
- [Inference](#inference)
- [Live Demo GUI](#live-demo-gui)
- [Publication Figure Generation](#publication-figure-generation)
- [Authentication Results](#authentication-results)
- [Model Architecture](#model-architecture)
- [Hardware Support](#hardware-support)
- [FAQ](#faq)

---

## System Overview

```
                        Multimode Plastic
  SLM (letter A-Z)  -->  Optical Fiber   -->  CMOS Camera  -->  Speckle Video
                        (random scatter)
                                                                     |
                                                                     v
                                                          +--------------------+
                                                          |  ResNet18 + Pool   |
                                                          |  (fiber-specific)  |
                                                          +--------------------+
                                                                     |
                                                                     v
                                                           Predicted letter
                                                           + confidence score
```

**Three illumination domains** are supported:

| Domain | Directory | Description |
|--------|-----------|-------------|
| Green only | `videocapture/Green/` | Side-illumination green laser only |
| Green + Red (fixed) | `videocapture/GreenAndRed/` | Green side + red end-face at fixed power |
| Green + Red (dynamic) | `videocapture/RedChange/` | Green side + red end-face with power sweep |

**Five fiber samples** (Fiber1–Fiber5) are used for PUF evaluation. Each fiber has 26 letter videos per domain, totalling 390 videos.

---

## Repository Structure

```
speckle_recognition/
|
|-- scripts/
|   |-- train_fiber.py              Single-fiber model training
|   |-- train_all_fibers.py         Batch training for all fibers
|   |-- fiber_auth_eval.py          Fiber-PUF authentication evaluation (5x5 matrix)
|   |-- diagnose_domains.py         Domain ablation study
|   |-- run_all_fiber_ablations.py  Run ablation across all fibers
|   |-- evaluate_cross_fiber.py     Cross-fiber generalization test
|   |-- export_letter_images.py     Export SLM letter images from PowerPoint
|   |-- launch_demo.py              Launch the live demo GUI
|   |-- env_check.py                Environment and dependency check
|   |-- make_paper_figures.py       Generate publication-quality figures
|   +-- plot_style.py               Shared plotting style for figures
|
|-- gui/
|   |-- main_window.py              Main GUI window (PySide6)
|   |-- slm_window.py               SLM output display window
|   |-- camera_worker.py            OpenCV camera capture thread
|   |-- mv_camera_worker.py         MindVision SDK camera thread
|   |-- inference_worker.py         Real-time CNN inference thread
|   |-- mvsdk.py                    MindVision Python ctypes wrapper
|   |-- __init__.py
|   |-- libmvsdk.dylib              MindVision SDK (macOS arm64)
|   +-- win_sdk/                    MindVision SDK DLLs + drivers (Windows x64)
|
|-- models.py                       CNN architectures (CNNPoolModel, R3DModel, SimpleCNN)
|-- dataset.py                      Video-clip dataset with temporal split
|-- unified_dataset.py              Multi-domain unified dataset with caching
|-- train_eval.py                   Training loop, evaluation, metrics export
|-- train_unified.py                Unified multi-domain training entry point
|-- evaluate_unified.py             Unified model evaluation with domain/fiber breakdown
|-- predict.py                      Frame-level inference with majority vote
|-- train_model.py                  Frame-level SimpleCNN training
|-- run_pipeline.py                 End-to-end extraction + training pipeline
|-- deep_learning_gui.py            Legacy Tkinter GUI
|-- visualize_features.py           Feature visualization utilities
|
|-- letter_images/                  Pre-rendered SLM letter PNGs (A-Z, Calibri Bold)
|-- figures/                        Publication figures (PNG 600dpi + PDF + SVG)
|-- requirements.txt                Python dependencies
|-- README.md                       This file
|-- GUI_TUTORIAL.md                 Step-by-step live demo tutorial
+-- USAGE.md                        Additional usage notes
```

**Directories created at runtime (not committed):**

```
videocapture/                       Raw video data (3 domains x 5 fibers x 26 letters)
results/                            Training outputs, metrics, predictions
  |-- fiber1/ ... fiber5/           Per-fiber single-domain results
  |-- fiber_auth/                   5x5 authentication matrix + per-fiber models
  |-- unified/                      Unified multi-domain evaluation
  +-- cross_fiber/                  Cross-fiber generalization results
checkpoints/                        Model checkpoints (.pth)
.cache/                             Decoded frame cache for fast reloading
```

---

## Installation

### Requirements

- Python 3.9+
- PyTorch 1.13+
- OpenCV
- PySide6 (for GUI)

### Setup

```bash
git clone https://github.com/Daniel-dzq/speckle_recognition.git
cd speckle_recognition
pip install -r requirements.txt
```

### GPU acceleration (optional but recommended)

```bash
# NVIDIA CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon — MPS is automatic, no special install needed
pip install torch torchvision
```

### Verify setup

```bash
python scripts/env_check.py
```

---

## Data Layout

Video files are organized by domain, fiber, and letter:

```
videocapture/
|-- Green/
|   |-- Fiber1/
|   |   |-- A.avi
|   |   |-- B.avi
|   |   +-- ... Z.avi
|   |-- Fiber2/
|   +-- ... Fiber5/
|-- GreenAndRed/
|   |-- Fiber1/ ... Fiber5/
+-- RedChange/
    |-- Fiber1/ ... Fiber5/
```

Each `.avi` file contains ~200 frames of speckle video captured while a specific letter pattern is displayed on the SLM. The file name (without extension) is the class label.

---

## Training Workflows

### 1. Single-fiber training

Train a video-clip model (ResNet18 + temporal average pooling) for one fiber. Uses a strict temporal split (70/15/15) within each video to prevent frame-level data leakage.

```bash
python scripts/train_fiber.py --fiber fiber1
python scripts/train_fiber.py --fiber fiber1 --model_type cnn_pool --clip_len 16 --stride 8 --epochs 50 --lr 1e-4
```

Outputs are saved to `results/<fiber>/` and `checkpoints/<fiber>_best.pth`.

| Output file | Content |
|-------------|---------|
| `best_model.pth` | Best model checkpoint (by validation accuracy) |
| `confusion_matrix.png` | Test set confusion matrix |
| `training_log.csv` | Per-epoch loss and accuracy |
| `per_class_metrics.csv` | Per-class precision, recall, F1 |
| `test_predictions.csv` | Per-clip predictions with confidence |
| `classification_report.txt` | Full classification report |
| `metrics.json` | Summary metrics |

### 2. Multi-fiber batch training

Train models for all five fibers sequentially:

```bash
python scripts/train_all_fibers.py
python scripts/train_all_fibers.py --epochs 50 --lr 1e-4
python scripts/train_all_fibers.py --only Fiber1 Fiber2
python scripts/train_all_fibers.py --skip Fiber3
```

### 3. Fiber-PUF authentication evaluation

This is the core experiment for the PUF paper. It trains one model per fiber (using all three illumination domains with temporal splitting), then evaluates each model against all five fibers to produce a 5x5 authentication matrix.

```bash
python scripts/fiber_auth_eval.py
```

Outputs in `results/fiber_auth/`:

| Output file | Content |
|-------------|---------|
| `auth_matrix.csv` | 5x5 accuracy matrix |
| `auth_matrix.json` | Full results including per-domain breakdown |
| `auth_summary.txt` | Human-readable summary |
| `fiber_models/Fiber1.pth` ... `Fiber5.pth` | Per-fiber trained models |

Expected result pattern:

- **Diagonal (authorized)**: high accuracy (>92%) — the fiber's own model recognizes its letters
- **Off-diagonal (unauthorized)**: near-chance accuracy (~4%) — other fibers cannot be decoded
- **Authentication gap**: >90 percentage points

### 4. Unified multi-domain training

Train a single model on all fibers and domains simultaneously. Supports multiple split strategies:

```bash
# Deploy mode: temporal split within each video, all fibers
python train_unified.py --split_mode deploy --epochs 20

# Cross-fiber mode: hold out entire fibers for testing
python train_unified.py --split_mode cross_fiber --epochs 20

# Evaluate a trained unified model
python evaluate_unified.py --checkpoint results/unified/best_model.pth
```

### 5. Domain ablation study

Test the effect of different illumination domains on recognition accuracy:

```bash
# Single fiber
python scripts/diagnose_domains.py --fiber Fiber1

# All fibers
python scripts/run_all_fiber_ablations.py
```

This compares three conditions: Green only, Green + GreenAndRed, and all three domains combined.

---

## Inference

### Command-line prediction

```bash
python predict.py --model checkpoints/fiber1_best.pth --test-dir screenshots/A
python predict.py --model checkpoints/fiber1_best.pth --test-dir screenshots/A --ground-truth A --top-k 3
```

### Programmatic usage

```python
import torch
from models import CNNPoolModel

model = CNNPoolModel(num_classes=26)
checkpoint = torch.load("checkpoints/fiber1_best.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

---

## Live Demo GUI

The PySide6-based GUI provides real-time fiber-PUF authentication:

```bash
python scripts/launch_demo.py
```

Features:
- **Fiber selector**: choose Fiber1–Fiber5, auto-loads the corresponding model from `results/fiber_auth/fiber_models/`
- **SLM display**: shows letter patterns on a second monitor (Calibri Bold, matching data collection)
- **Live camera feed**: capture speckle images from a MindVision CCD or any USB camera
- **Real-time prediction**: displays predicted letter and confidence with low-confidence warning
- **Camera controls**: exposure, gain, flip, auto-exposure

The GUI expects trained models at `results/fiber_auth/fiber_models/FiberX.pth`. Run `scripts/fiber_auth_eval.py` first to generate them.

### Sharing with colleagues (no re-training needed)

To run the demo on another machine, share these files:

| Files | Purpose |
|-------|---------|
| `gui/` | GUI code + bundled camera SDK |
| `scripts/launch_demo.py` | Entry point |
| `results/fiber_auth/fiber_models/*.pth` | Trained models (5 files, ~43 MB each) |
| `models.py` | Model architecture definition |
| `letter_images/` | SLM letter PNGs |
| `requirements.txt` | Dependencies |

The `videocapture/`, `results/` (other than fiber_models), and `checkpoints/` directories are **not needed** for the demo.

See [GUI_TUTORIAL.md](GUI_TUTORIAL.md) for a detailed walkthrough.

---

## Publication Figure Generation

Generate all paper figures from existing experimental results:

```bash
python scripts/make_paper_figures.py
```

This produces 8 figures in `figures/` (PNG at 600 dpi, PDF vector, and SVG), using a shared style defined in `scripts/plot_style.py`:

| Figure | File | Description |
|--------|------|-------------|
| Auth heatmap | `fig_auth_matrix` | 5x5 fiber authentication accuracy matrix |
| Per-domain bars | `fig_same_fiber_per_domain` | Same-fiber accuracy across 3 illumination domains |
| Auth gap | `fig_auth_gap` | Authorized vs unauthorized accuracy comparison |
| Score distributions | `fig_auth_scores` | Confidence and cross-fiber accuracy histograms |
| Speckle examples | `fig_speckle_examples` | Representative speckle images across fibers and domains |
| NCC / HD | `fig_ncc_hd` | Normalized cross-correlation and Hamming distance distributions |
| Training curves | `fig_training_curves` | Loss and validation accuracy during training |
| Accuracy summary | `fig_test_accuracy_summary` | Per-fiber test accuracy bar chart |

All figures use a consistent colorblind-friendly palette and journal-quality formatting.

---

## Authentication Results

### 5x5 Fiber Authentication Matrix

Each row is a fiber-specific model; each column is test data from a fiber. Diagonal entries are authorized (same-fiber) accuracy; off-diagonal entries are unauthorized (cross-fiber attack) accuracy.

| Model \ Data | Fiber1 | Fiber2 | Fiber3 | Fiber4 | Fiber5 |
|:-------------|-------:|-------:|-------:|-------:|-------:|
| **Fiber1** | **97.4** | 5.1 | 5.3 | 2.5 | 1.4 |
| **Fiber2** | 3.1 | **95.3** | 3.4 | 3.7 | 4.6 |
| **Fiber3** | 4.8 | 3.8 | **92.9** | 5.9 | 4.7 |
| **Fiber4** | 3.3 | 3.8 | 4.3 | **98.7** | 2.3 |
| **Fiber5** | 4.3 | 5.4 | 5.0 | 2.6 | **93.6** |

| Metric | Value |
|--------|-------|
| Authorized (same-fiber) average | **95.6%** |
| Unauthorized (cross-fiber) average | **4.0%** |
| Authentication gap | **91.6 pp** |
| Chance level (1/26) | 3.85% |

### Per-Domain Robustness

Same-fiber accuracy under each illumination domain:

| Fiber | Green only | Green+Red (fixed) | Green+Red (sweep) |
|-------|----------:|-------------------:|-------------------:|
| Fiber1 | 96.2% | 100.0% | 96.2% |
| Fiber2 | 98.5% | 100.0% | 82.7% |
| Fiber3 | 97.7% | 88.5% | 92.3% |
| Fiber4 | 100.0% | 100.0% | 96.2% |
| Fiber5 | 92.3% | 88.5% | 100.0% |

### NCC / HD Uniqueness Metrics

Computed from 800 genuine (intra-fiber) and 800 impostor (inter-fiber) speckle frame pairs:

| Metric | Intra-fiber (genuine) | Inter-fiber (impostor) |
|--------|----------------------:|----------------------:|
| NCC | 0.990 +/- 0.016 | 0.878 +/- 0.023 |
| HD | 0.022 +/- 0.014 | 0.105 +/- 0.020 |

---

## Model Architecture

### CNNPoolModel (primary model)

Used by `scripts/train_fiber.py` and `scripts/fiber_auth_eval.py`.

```
Input video clip: (batch, T, C, H, W)
  -> ResNet-18 backbone (ImageNet pretrained, final FC removed)
  -> Per-frame features: (batch * T, 512)
  -> Reshape: (batch, T, 512)
  -> Temporal average pooling: (batch, 512)
  -> Dropout(0.3)
  -> Linear(512, 26)
```

### R3DModel (alternative)

3D ResNet-18 for spatio-temporal feature extraction. Use with `--model_type r3d`.

### SimpleCNN (legacy)

Used by `train_model.py` and `deep_learning_gui.py` for single-frame classification.

### Temporal Split Strategy

To prevent data leakage, frames within each video are split by time:

```
Frame index:  1 .......... 0.70*N .......... 0.85*N .......... N
              [    Train    ][    Val    ][    Test    ]
```

Clips are constructed from consecutive frames within each split. No clip straddles split boundaries.

---

## Hardware Support

### MindVision CCD Camera

The system supports MindVision / HuaTengVision USB cameras (e.g., HT-UBS300C) on macOS and Windows.

**macOS (Apple Silicon):** `gui/libmvsdk.dylib` is bundled and ad-hoc signed. Plug in the camera and select "MindVision CCD" in the GUI.

**Windows (x64):** DLLs are bundled in `gui/win_sdk/`. First-time USB driver setup:
1. Plug in the camera
2. Open Device Manager
3. Right-click the camera, select Update Driver -> Browse -> point to `gui/win_sdk/drivers/`

### Other Cameras

Any UVC-compatible USB camera or webcam works via the OpenCV camera button. No SDK needed.

---

## FAQ

**Q: How do I reproduce the 5x5 authentication matrix?**

Place your video data in `videocapture/{Green,GreenAndRed,RedChange}/Fiber{1-5}/`, then run:
```bash
python scripts/fiber_auth_eval.py
```

**Q: GPU not detected?**
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

**Q: Training accuracy stays low?**
- Ensure each video has at least 150 frames
- Try `--lr 1e-4`
- Verify videos contain actual speckle patterns

**Q: Can I use custom class names instead of A-Z?**

Yes. Class labels are derived from video filenames automatically.

**Q: How do I add a new fiber?**

1. Record 26 letter videos under each domain directory
2. Name the fiber folder (e.g., `Fiber6`)
3. Re-run `scripts/fiber_auth_eval.py`

---

## License

MIT License
