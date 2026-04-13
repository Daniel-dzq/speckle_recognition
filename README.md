# Speckle-PUF Recognition

A deep learning toolkit for classifying optical speckle patterns produced by
**SLM-encoded letters** transmitted through **multimode / plastic optical fibers**.
The system can decode letter encodings or identify fiber identity from speckle
images that are visually indistinguishable to the human eye.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training Workflows](#training-workflows)
  - [Single-fiber decode (recommended)](#1-single-fiber-decode-recommended)
  - [End-to-end pipeline](#2-end-to-end-pipeline)
  - [Multi-fiber training (all fibers)](#3-multi-fiber-training-all-fibers)
  - [Video-clip model (R3D / CNN-Pool)](#4-video-clip-model-r3d--cnn-pool)
- [Inference](#inference)
- [Live Demo GUI](#live-demo-gui)
- [Legacy GUI (Tkinter)](#legacy-gui-tkinter)
- [Cross-Fiber Evaluation](#cross-fiber-evaluation)
- [Model Architecture](#model-architecture)
- [MindVision CCD Camera Support](#mindvision-ccd-camera-support)
- [Results](#results)
- [FAQ](#faq)

---

## Overview

```
SLM encodes letter A  -->  Fiber  -->  Camera captures speckle video
SLM encodes letter B  -->  Fiber  -->  ...
         ...
SLM encodes letter Z  -->  Fiber  -->  ...

                           +----------------------+
  speckle frames  ------>  |  SimpleCNN  /  R3D   |  ----->  predicted letter
                           +----------------------+
```

**Key insight**: Although all speckle patterns look like random noise, each
letter encoding subtly shifts the spatial frequency statistics of the output
speckle. A CNN can reliably extract these high-dimensional statistical
differences that are invisible to human observers.

---

## Repository Structure

```
speckle_recognition/
|-- scripts/
|   |-- train_fiber.py          # Train a single fiber (video-clip model)
|   |-- train_all_fibers.py     # Train all fibers sequentially
|   |-- evaluate_cross_fiber.py # Cross-fiber generalization evaluation
|   |-- launch_demo.py          # Launch live demo GUI (PySide6)
|   +-- env_check.py            # Environment & dependency check
|
|-- gui/
|   |-- main_window.py          # Live demo main window (PySide6)
|   |-- camera_worker.py        # OpenCV camera capture thread
|   |-- mv_camera_worker.py     # MindVision SDK camera capture thread
|   |-- mvsdk.py                # Python ctypes wrapper for MindVision SDK
|   |-- inference_worker.py     # Real-time inference thread
|   |-- slm_window.py           # SLM output window
|   |-- libmvsdk.dylib          # MindVision SDK library (macOS arm64)
|   +-- win_sdk/                # MindVision SDK DLLs (Windows x64, bundled)
|       |-- MVCAMSDK_X64.dll
|       |-- MVImageProcess_X64.DLL
|       |-- hAcqHuaTengVision*_X64.dll
|       |-- Usb*Camera*.Interface
|       +-- drivers/            # Windows USB driver (.inf/.sys)
|
|-- dataset.py              # Video-clip dataset with temporal split
|-- models.py               # CNNPoolModel and R3DModel
|-- train_eval.py           # Training loop, evaluation, output saving
|-- predict.py              # Frame-level inference + majority vote
|-- train_model.py          # Frame-level CNN training (SimpleCNN)
|-- run_pipeline.py         # End-to-end: extraction -> training -> evaluation
|-- deep_learning_gui.py    # Legacy Tkinter GUI application
|-- requirements.txt        # Python dependencies
|-- README.md               # This file
|-- GUI_TUTORIAL.md         # Step-by-step live demo tutorial
|
|-- checkpoints/            # Saved model weights (.pth)
|-- results/                # Per-fiber training results and metrics
|   |-- fiber1/
|   |   |-- best_model.pth
|   |   |-- confusion_matrix.png
|   |   |-- training_log.csv
|   |   |-- per_class_metrics.csv
|   |   +-- test_predictions.csv
|   +-- ...
+-- video_capture/          # Raw video data (not committed)
    |-- fiber1/
    |   |-- A.avi
    |   +-- ...
    +-- ...
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Daniel-dzq/speckle_recognition.git
cd speckle_recognition
pip install -r requirements.txt

# 2. Place your letter videos in video_capture/<fiber_name>/
#    (name them A.avi, B.avi, ... or A.mp4, B.mp4, ...)

# 3. Train a single fiber
python scripts/train_fiber.py --fiber fiber1

# 4. Train all fibers at once
python scripts/train_all_fibers.py --epochs 50 --lr 1e-4

# 5. Launch the live demo GUI
python scripts/launch_demo.py
```

---

## Installation

### Prerequisites

- Python 3.9 or newer
- pip or conda

### CPU-only (no GPU required)

```bash
pip install -r requirements.txt
```

### GPU-accelerated (recommended for faster training)

```bash
# NVIDIA CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon (M1-M4) — MPS acceleration is automatic:
pip install torch torchvision
```

> On Apple Silicon Macs, PyTorch automatically uses the Metal Performance Shaders
> (MPS) backend for GPU acceleration. No special installation is needed.

### Verify your setup

```bash
python scripts/env_check.py
```

Expected output:

```
============================================================
  Speckle-PUF Environment Check
============================================================
  Python       : 3.11.x
  PyTorch      : 2.x.x
  Device       : MPS (Apple Silicon GPU)   # or CUDA / CPU
  OpenCV       : 4.x.x
  PySide6      : 6.x.x
  MindVision   : [OK] 1 camera(s) detected
============================================================
```

---

## Data Preparation

### Option A --- GUI (easiest)

```bash
python deep_learning_gui.py
```

Go to the **Video Extraction** tab, select your video folder and output folder,
set the frame count, and click **Start Extraction**.

### Option B --- `run_pipeline.py` (automated)

Place videos in `video_capture/` named `A.avi`, `B.avi`, ..., then:

```bash
python run_pipeline.py --frames 300
```

### Option C --- Manual with ffmpeg

```bash
ffmpeg -i A.avi -vf "fps=10,format=gray" -frames:v 300 screenshots/A/frame_%05d.png
```

### Required layout after extraction

```
screenshots/
|-- A/
|   |-- frame_00001.png
|   |-- frame_00002.png
|   +-- ...  (>= 300 frames recommended)
|-- B/
+-- Z/
```

> The folder name **is** the class label. Keep names consistent with the
> letter encodings used during recording.

---

## Training Workflows

### 1. Single-fiber decode (recommended)

Trains a video-clip model (ResNet18 + temporal pooling) for one fiber.
Uses a strict **temporal split** (70% train / 15% val / 15% test) to prevent
data leakage between splits from the same video.

```bash
# Basic usage
python scripts/train_fiber.py --fiber fiber1

# Custom hyperparameters
python scripts/train_fiber.py \
    --fiber fiber1 \
    --model_type cnn_pool \
    --clip_len 16 \
    --stride 8 \
    --img_size 224 \
    --epochs 50 \
    --lr 1e-4 \
    --batch_size 8 \
    --patience 10
```

**Outputs:**

| File | Description |
|------|-------------|
| `results/<fiber>/best_model.pth` | Best checkpoint (by val accuracy) |
| `results/<fiber>/confusion_matrix.png` | Confusion matrix on the test set |
| `results/<fiber>/training_log.csv` | Per-epoch loss and accuracy |
| `results/<fiber>/per_class_metrics.csv` | Precision / recall / F1 per class |
| `results/<fiber>/test_predictions.csv` | Per-clip test predictions |
| `results/<fiber>/classification_report.txt` | Full sklearn classification report |
| `results/<fiber>/loss_acc_curve.png` | Training curves plot |
| `results/<fiber>/metrics.json` | Summary metrics and sanity-check warnings |
| `checkpoints/<fiber>_best.pth` | Copy of best model for live demo |

**Training log example:**

```
================================================================================
  Speckle-PUF  |  Training fiber: fiber1
================================================================================
  Device       : mps
  GPU          : Apple Silicon GPU (MPS)
  Model type   : cnn_pool
  Epochs       : 50

 * Epoch 001/050 | train_loss=3.2541  train_acc= 12.35% | val_loss=3.1024  val_acc= 10.80%
   Epoch 002/050 | train_loss=2.8103  train_acc= 28.14% | val_loss=2.6892  val_acc= 25.62%
 * Epoch 003/050 | train_loss=2.1456  train_acc= 51.23% | val_loss=1.8721  val_acc= 49.35%
   ...
================================================================================
  Training complete. Best validation accuracy: 94.20%
================================================================================

================================================================================
  Test results:  loss = 0.2103,  accuracy = 93.87%
================================================================================
```

---

### 2. End-to-end pipeline

Runs frame extraction + training + evaluation in one command.

```bash
python run_pipeline.py
python run_pipeline.py --frames 400 --epochs 80 --lr 5e-4
python run_pipeline.py --skip-extract   # if frames already exist
```

---

### 3. Multi-fiber training (all fibers)

Trains a separate video-clip model for each fiber sequentially.

```bash
python scripts/train_all_fibers.py
python scripts/train_all_fibers.py --epochs 50 --lr 1e-4
python scripts/train_all_fibers.py --skip Fiber3 Fiber4
python scripts/train_all_fibers.py --only Fiber1 Fiber2
```

**Expected data structure:**

```
video_capture/
|-- fiber1/
|   |-- A.avi
|   |-- B.avi
|   +-- ...
|-- fiber2/
+-- fiber5/
```

**Summary table example:**

```
================================================================================
  Training Summary  (total: 1243s)
================================================================================
  Fiber                 Status    Val Acc   Test Acc      Time
  ------------------------------------------------------------
  fiber1                ok         94.50%     93.20%      245s
  fiber2                ok         96.10%     95.40%      251s
  fiber3                ok         92.30%     91.80%      249s
```

---

### 4. Video-clip model (R3D / CNN-Pool)

```bash
# CNN + temporal average pooling (default, more stable)
python main.py --data_dir video_capture --model_type cnn_pool --clip_len 16 --epochs 30

# 3D ResNet (R3D-18)
python main.py --data_dir video_capture --model_type r3d --clip_len 16 --epochs 30
```

| Model | Strength | Best for |
|-------|----------|----------|
| `SimpleCNN` (train_model.py) | Fast, stable, uses ImageNet pretraining | Small datasets, quick experiments |
| `cnn_pool` (scripts/train_fiber.py) | Temporal aggregation, ImageNet pretraining | Multi-video per class |
| `r3d` | Rich spatio-temporal features | Large datasets with temporal dynamics |

---

## Inference

```bash
# Predict all frames in a folder + majority vote
python predict.py --model model.pth --test-dir screenshots/A

# With ground truth
python predict.py --model model.pth --test-dir screenshots/A --ground-truth A

# Top-3 predictions per frame
python predict.py --model model.pth --test-dir screenshots/A --ground-truth A --top-k 3
```

**Example output:**

```
=================================================================
  Filename                       Prediction   Confidence
-----------------------------------------------------------------
  frame_00256.png                A             98.72%
  frame_00257.png                A             97.44%
  frame_00258.png                B             61.23%
=================================================================

Video-level majority vote: A
Frame-level counts: {'A': 43, 'B': 2}

Ground truth : A
Frame accuracy: 95.6%
Video result : CORRECT  (vote=A, truth=A)
```

---

## Live Demo GUI

The PySide6-based live demo is the primary interactive interface. It provides:

- **SLM output** --- display letters or phase patterns on a second monitor
- **Live CCD feed** --- capture speckle images from a MindVision CCD or any webcam
- **Real-time recognition** --- run the trained model on live frames with smoothed predictions
- **Camera controls** --- exposure, gain, flip, auto-exposure, etc.

```bash
python scripts/launch_demo.py
```

See **[GUI_TUTORIAL.md](GUI_TUTORIAL.md)** for a full step-by-step walkthrough.

### Sharing pre-trained models with colleagues

To let others use the demo without training, share these files:

| What to share | Purpose |
|---------------|---------|
| Entire `gui/` folder | GUI code + bundled SDK libraries |
| `scripts/launch_demo.py` | Entry point |
| `checkpoints/*.pth` | Trained model weights |
| `models.py`, `dataset.py` | Required by inference |
| `requirements.txt` | Dependency list |

Large folders like `video_capture/` and `results/` are **not needed** for the demo.

---

## Legacy GUI (Tkinter)

The original Tkinter-based GUI is still available for frame extraction, training,
prediction, and feature visualization:

```bash
python deep_learning_gui.py
```

---

## Cross-Fiber Evaluation

Evaluates how well a model trained on one fiber generalizes to unseen fibers.

```bash
python scripts/evaluate_cross_fiber.py
```

**Outputs:**

```
results/cross_fiber/
|-- cross_fiber_heatmap.png    # accuracy heatmap: train_fiber x test_fiber
|-- cross_fiber_accuracy.csv   # raw accuracy matrix
|-- cross_fiber_results.json   # detailed per-class results
+-- summary.md                 # markdown summary table
```

---

## Model Architecture

### SimpleCNN (used by `train_model.py` and `deep_learning_gui.py`)

```
Input (1 x H x W)
  -> Conv2d(1->32, 3x3) + BN + ReLU + MaxPool(2)
  -> Conv2d(32->64, 3x3) + BN + ReLU + MaxPool(2)
  -> Conv2d(64->128, 3x3) + BN + ReLU + MaxPool(2)
  -> Conv2d(128->256, 3x3) + BN + ReLU + MaxPool(2)
  -> Flatten
  -> Linear(256*(H/16)^2, 512) + ReLU + Dropout(0.5)
  -> Linear(512, num_classes)
```

### CNNPoolModel (used by `scripts/train_fiber.py`)

```
Input clip (B, T, C, H, W)
  -> ResNet18 backbone (ImageNet pretrained, FC removed)
  -> Per-frame 512-dim features  (B*T, 512)
  -> Reshape to (B, T, 512)
  -> Temporal average pooling  -> (B, 512)
  -> Dropout(0.3)
  -> Linear(512, num_classes)
```

### Temporal Split (no data leakage)

```
One video per class, N frames total:

Frame  1  ...  0.70N    ->  train
Frame 0.70N ... 0.85N   ->  val
Frame 0.85N ... N       ->  test

Clips from the same time window never appear in both train and test.
```

---

## MindVision CCD Camera Support

The project includes native support for MindVision / HuaTengVision USB cameras
(e.g., HT-UBS300C) on both macOS and Windows.

### macOS (Apple Silicon)

The SDK library `libmvsdk.dylib` is pre-bundled in `gui/` and ad-hoc signed.
No additional setup is needed --- plug in the camera and click "MindVision CCD"
in the GUI.

### Windows (x64)

All required DLLs are pre-bundled in `gui/win_sdk/`:

| File | Purpose |
|------|---------|
| `MVCAMSDK_X64.dll` | Main SDK library (11 MB) |
| `MVImageProcess_X64.DLL` | ISP image processing |
| `hAcqHuaTengVision*_X64.dll` | Hardware abstraction layer |
| `Usb2/Usb3Camera*.Interface` | USB interface plugins |
| `drivers/MvU2Camera.*` | USB driver (install once per machine) |

**First-time USB driver setup:**

1. Plug in the camera.
2. Open **Device Manager**.
3. Right-click the camera (may show as "Unknown Device").
4. Select **Update Driver** -> **Browse my computer** -> point to `gui/win_sdk/drivers/`.
5. After driver installation, the camera appears as a MindVision device.

> On Windows, the HT-UBS300C also registers as a DirectShow device after driver
> installation. This means the standard **Start Camera** button (OpenCV path)
> often works without the MindVision SDK button. Use whichever works for your setup.

### Other cameras

Any UVC-compatible USB camera or webcam works out of the box via the
**Start Camera** button (OpenCV). No SDK is needed.

---

## Results

Example results on a 5-fiber, 26-letter dataset:

| Fiber | Best Val Acc | Test Acc |
|-------|-------------|----------|
| fiber1 | 94.50% | 93.20% |
| fiber2 | 96.10% | 95.40% |
| fiber3 | 92.30% | 91.80% |
| fiber4 | 95.70% | 94.90% |
| fiber5 | 93.80% | 93.10% |
| **Average** | **94.48%** | **93.68%** |

---

## FAQ

**Q: GPU not detected?**

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.backends.mps.is_available())"
```

- NVIDIA: reinstall CUDA PyTorch (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`)
- Apple Silicon: MPS is used automatically with PyTorch >= 1.13

**Q: Training accuracy stays below 20%?**
- Ensure each class has at least 300 frames
- Try a lower learning rate: `--lr 1e-4`
- Check that images contain actual speckle patterns (not black/white frames)

**Q: Val accuracy is high but test accuracy drops significantly?**
- Speckle drifts over time (temperature, mechanical vibration). Lower patience: `--patience 5`
- Collect recordings at different times for a more robust test set

**Q: MindVision camera not detected on macOS?**
- Check System Settings -> Privacy & Security -> Camera
- Run `codesign --force --sign - gui/libmvsdk.dylib` if you get a code signature error
- Wait 3 seconds after plugging in, then try again

**Q: MindVision camera not detected on Windows?**
- Install the USB driver from `gui/win_sdk/drivers/` via Device Manager
- Verify the camera appears in Device Manager as a MindVision device
- Check that `gui/win_sdk/MVCAMSDK_X64.dll` exists

**Q: Can I use a different camera?**
- Any UVC-compatible USB camera works via the "Start Camera" (OpenCV) button
- For cameras with proprietary SDKs, write a new worker class following the pattern in `gui/mv_camera_worker.py`

**Q: Can I use my own class names (not A-Z)?**
- Yes. Class names are read directly from folder names or video filenames

---

## License

MIT License --- see [LICENSE](LICENSE) for details.
