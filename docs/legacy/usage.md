# Speckle-PUF Recognition — Legacy Usage Guide

> **Legacy notice.** This document describes the **pre-refactor** workflow
> centred on the Tkinter GUI (`deep_learning_gui.py`) and the frame-level
> `train_model.py` CLI. Both files now live under [`archive/`](../../archive)
> for historical reference. The active workflow is documented in the top-level
> [`README.md`](../../README.md) (training / inference / live demo) and in
> [`docs/experiments.md`](../experiments.md) (paper-section experiment
> framework). The notes below are preserved verbatim only for context.

> **Project background**: This project implements speckle-PUF recognition for
> SLM-encoded + multimode/plastic-fiber speckle experiments.
> The input is speckle video frames captured after transmission through an
> optical fiber. The goal is to decode the letter encoding or identify the
> fiber identity from patterns that are visually indistinguishable to the
> human eye.

## Table of Contents

1. [Install Dependencies](#1-install-dependencies)
2. [Data Directory Structure](#2-data-directory-structure)
3. [Full Workflow](#3-full-workflow)
4. [CLI Training Script — `train_model.py`](#4-cli-training-script--train_modelpy)
5. [CLI Inference Script — `predict.py`](#5-cli-inference-script--predictpy)
6. [GUI — `deep_learning_gui.py`](#6-gui--deep_learning_guipy)
7. [Key Design Decisions](#7-key-design-decisions)
8. [FAQ](#8-faq)
9. [Known Limitations](#9-known-limitations)

---

## 1. Install Dependencies

```bash
pip install torch torchvision numpy Pillow scikit-learn matplotlib joblib tqdm imageio-ffmpeg
```

| Package | Min version | Purpose |
|---------|-------------|---------|
| torch | 2.0.0 | Deep learning framework |
| torchvision | 0.15.0 | Image transforms |
| numpy | 1.24.0 | Numerical computing |
| Pillow | 9.0.0 | Image I/O |
| scikit-learn | 1.2.0 | RF model / evaluation metrics |
| matplotlib | 3.6.0 | Confusion matrix / visualization |
| joblib | 1.2.0 | RF model serialization |
| tqdm | 4.65.0 | Training progress bar (optional but recommended) |
| imageio-ffmpeg | 0.4.8 | GUI video frame extraction |

**Install GPU-accelerated PyTorch (recommended):**

```bash
# Uninstall CPU version first
pip uninstall torch torchvision

# Install CUDA 12.1 build (adjust to your actual CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 2. Data Directory Structure

### Flat structure (single fiber, most common)

Use the GUI "Video Extraction" tab or `ffmpeg` manually to extract frames into
the following layout:

```
screenshots/              <- DATA_DIR
├── A/                    <- frames for letter A
│   ├── frame_00001.png
│   ├── frame_00002.png
│   └── ...  (>= 300 frames recommended)
├── B/
│   └── ...
└── Z/
    └── ...
```

> **Naming convention**: the folder name is the class label. If your videos
> are named `A.avi`, `B.avi`, etc., the extracted frames automatically land
> in folders named `A`, `B`, etc. — no manual renaming required.

### Multi-fiber structure (fiber_id / cross-fiber decode)

```
data/                     <- DATA_DIR
├── fiber_01/
│   ├── A/
│   │   ├── frame_00001.png
│   │   └── ...
│   ├── B/
│   └── ...
├── fiber_02/
│   ├── A/
│   └── ...
└── fiber_05/
    └── ...
```

The code **auto-detects** the directory depth — no parameter changes required.

---

## 3. Full Workflow

```
Step 1  Prepare videos
        Place one video per letter (A.avi, B.avi, ...) in a single folder
                │
                ▼
Step 2  Extract frames (choose one method)
        Method A: python deep_learning_gui.py  ->  Video Extraction tab
        Method B: ffmpeg manual extraction
                │
                ▼
Step 3  Train model (choose one method)
        Method A (recommended): python train_model.py   <- includes full test metrics
        Method B: python deep_learning_gui.py  ->  Model Training tab
                │
                ▼
Step 4  Run inference (choose one method)
        Method A (recommended): python predict.py --model model.pth --test-dir screenshots/A
        Method B: python deep_learning_gui.py  ->  Model Prediction tab
                │
                ▼
Step 5  Analyze feature distribution
        python deep_learning_gui.py  ->  Feature Visualization tab
```

---

## 4. CLI Training Script — `train_model.py`

### Basic usage

```bash
# Default: use ./screenshots directory, decode task
python train_model.py

# Specify data directory
python train_model.py --data screenshots

# Fiber identification task (multi-fiber structure)
python train_model.py --data data --task fiber_id --output model_fiber.pth

# Tune hyperparameters
python train_model.py --epochs 100 --lr 5e-4 --batch 64 --patience 15
```

### All arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | `./screenshots` | Data directory (supports flat and multi-fiber structures) |
| `--output` | `./model.pth` | Model save path |
| `--task` | `decode` | Task type: `decode` (classify letter) / `fiber_id` (identify fiber) |
| `--epochs` | `50` | Max training epochs |
| `--lr` | `1e-3` | Initial learning rate (CosineAnnealingLR decay) |
| `--batch` | `32` | Batch size |
| `--image-size` | `128` | Input image size (must be a multiple of 16) |
| `--patience` | `10` | Early stopping patience |
| `--seed` | `42` | Random seed for reproducibility |

### Training output example

```
[Training] device=cuda  classes=26  epochs=50  lr=0.001  image_size=128
================================================================================
* Epoch 001/050 | loss 3.2541 | train  12.35% | val  10.80% | lr 9.97e-04
  Epoch 002/050 | loss 2.8103 | train  28.14% | val  25.62% | lr 9.88e-04
* Epoch 003/050 | loss 2.1456 | train  51.23% | val  49.35% | lr 9.75e-04
  ...
================================================================================
[Training complete] Best val accuracy: 94.20%

[Test set]  loss 0.2103   acc 93.87%

Classification report:
              precision    recall  f1-score   support
           A     0.9512    0.9487    0.9499        15
           B     0.9333    0.9600    0.9465        15
           ...

[Saved] model  -> ./model.pth
[Saved] confusion matrix -> ./confusion_matrix.png
```

### Temporal split strategy

```
Frames per class (letter/fiber) are sorted chronologically then split:

frame_00001.png ─┐
frame_00002.png  │  First 70%   ->  train  (~210 frames/class)
      ...        │
frame_00210.png ─┘
frame_00211.png ─┐
      ...        │  Middle 15%  ->  val    (~45 frames/class)
frame_00255.png ─┘
frame_00256.png ─┐
      ...        │  Last 15%    ->  test   (~45 frames/class)
frame_00300.png ─┘

Frames from the same video never appear in both train and test -> no clip leakage
```

---

## 5. CLI Inference Script — `predict.py`

### Basic usage

```bash
# Predict all frames in a folder, output per-frame results + majority vote
python predict.py --model model.pth --test-dir screenshots/A

# Provide ground truth to compute frame-level / video-level accuracy
python predict.py --model model.pth --test-dir screenshots/A --ground-truth A

# Show Top-3 predictions
python predict.py --model model.pth --test-dir screenshots/A --top-k 3

# Predict a single image
python predict.py --model model.pth --test-dir screenshots/A/frame_00001.png
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--model` | ✓ | Path to model checkpoint (`.pth`) |
| `--test-dir` | ✓ | Directory of test images or single image path |
| `--ground-truth` | optional | True class label; used to compute accuracy |
| `--top-k` | optional | Show top-K predictions (default 1) |

### Output example

```
Found 45 images
=================================================================
  Filename                       Prediction   Confidence
-----------------------------------------------------------------
  frame_00256.png                A             98.72%
  frame_00257.png                A             97.44%
  frame_00258.png                B             61.23%   <- low confidence
  frame_00259.png                A             99.01%
  ...
=================================================================

Video-level majority vote: A
Frame-level counts:        {'A': 43, 'B': 2}

Ground truth : A
Frame acc    : 95.6%
Video result : CORRECT  (vote=A, truth=A)
```

---

## 6. GUI — `deep_learning_gui.py`

```bash
python deep_learning_gui.py
```

### Main window layout

```
┌────────────────────────────────────────────────────────────────────┐
│  Speckle-PUF Deep Learning Suite                         [_][□][X] │
├────────────────────────────────────────────────────────────────────┤
│  [Video Extraction]  [Model Training]  [Model Prediction]          │
│  [Feature Visualization]                                           │
├────────────────────────────────────────────────────────────────────┤
│                        Tab content                                  │
├────────────────────────────────────────────────────────────────────┤
│  Ready                                  GPU: Available / Unavailable│
└────────────────────────────────────────────────────────────────────┘
```

---

### Tab 1 — Video Extraction

**Purpose**: Batch-extract grayscale frames from videos; output is organized
into subfolders named after each video file.

**Steps**:
1. Select **Video folder** (supports `.avi` `.mp4` `.mkv` `.mov`)
2. Select **Output folder** (this becomes your DATA_DIR for training)
3. Set **Frames per video** (>= 300 recommended so each split has enough samples)
4. Click **Start Extraction**

**Output structure** (example: A.avi, B.avi):

```
output_folder/
├── A/
│   ├── frame_00001.png
│   └── ...  (300 frames)
├── B/
└── ...
```

---

### Tab 2 — Model Training

**CNN parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Image size | 128 | Resize input frames; must be a multiple of 16 (64/128/256 all valid) |
| Batch size | 32 | Samples per training step |
| Epochs | 50 | Max epochs |
| Learning rate | 0.001 | Initial LR; CosineAnnealingLR decay applied automatically |
| Train ratio | 0.8 | First 80% of frames per class -> train; last 20% -> val (temporal split) |
| ES patience | 8 | Stop if val accuracy does not improve for N consecutive epochs |

> **Note on Train ratio**: This is a *per-class temporal split*, not a
> class-level split. All classes appear in both train and val sets.

**Training log format**:
```
Epoch 5/50 | Loss 0.8234 | Train 78.32% | Val 75.18% | LR 9.51e-04
```

---

### Tab 3 — Model Prediction

1. Click **Browse...** to load a `.pth` model file
2. Select **Batch (folder)** (predict all frames in a directory) or **Single image**
3. Click **Start Prediction**
4. Results table shows per-frame predictions and confidence; summary shows class counts

> For accuracy-evaluated batch inference, the CLI `predict.py` is recommended.

---

### Tab 4 — Feature Visualization

Visualizes speckle feature distributions to assess class separability.

**Visualization types**:
- **3D Scatter**: X=mean brightness, Y=std deviation, Z=edge density
- **2D Projections**: pairwise feature scatter plots
- **Distance Heatmap**: Euclidean distance matrix between class centroids; darker = more separable

**Steps**:
1. Select data folder (same structure as training data)
2. Set group size (extract features every N frames; default 5)
3. Click **Generate Visualization**
4. Use **Prev / Next** to switch between charts; **Save current figure** to export

---

## 7. Key Design Decisions

### Why temporal split instead of random split?

**Old approach (wrong)**: randomly assign entire class folders to train or val.
Result: val classes are never seen during training → val accuracy is near
random and completely unreliable.

**New approach (correct)**: split each class's frames chronologically.

```
Class A (300 frames):
  [  1 ~ 210 ] -> train   early frames of class A
  [211 ~ 255 ] -> val     middle frames of class A
  [256 ~ 300 ] -> test    late frames of class A

Class B (300 frames):
  [  1 ~ 210 ] -> train   early frames of class B
  ...

-> All 26 classes have samples in train / val / test
-> Val accuracy honestly reflects generalization to unseen frames
```

### Multi-fiber task matrix

| Task | Data structure | Command |
|------|---------------|---------|
| decode (current) | `screenshots/A/` | `python train_model.py --task decode` |
| decode (multi-fiber) | `data/fiber_01/A/` | `python train_model.py --data data --task decode` |
| fiber_id (multi-fiber) | `data/fiber_01/A/` | `python train_model.py --data data --task fiber_id` |

The code auto-detects directory depth. No code modification required.

### Checkpoint format

The `.pth` checkpoint saved by `train_model.py` contains:

```python
{
    "model_state_dict": ...,   # model weights
    "class_names":      [...], # class name list, e.g. ['A','B',...,'Z']
    "num_classes":      26,
    "image_size":       128,
    "task":             "decode",
    "best_val_acc":     94.20,
    "test_acc":         93.87,
    "history":          {...},  # training curves
}
```

---

## 8. FAQ

### Q1: Wrong data directory — classes don't match?

Check folder names: the program uses **folder names directly as class labels**.
Make sure your video files are named `A.avi`, `B.avi`, etc. so that extracted
frames land in folders `A`, `B`, etc.

### Q2: GPU shows unavailable?

```bash
# Check
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch (CUDA 12.1)
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Q3: Training accuracy very low (< 20%)?

1. **Too few frames**: fewer than 100 frames per class leaves very little for
   training after splitting. Extract at least 300 frames per video.
2. **Learning rate too high**: try `--lr 1e-4`
3. **Corrupted frames**: verify that extracted frames are clear speckle images,
   not all-black or all-white.

### Q4: Val accuracy high but test accuracy low?

- This indicates strong temporal correlation in the speckle — the fiber's
  speckle pattern drifts over time (temperature, mechanical stability).
- Lower `--patience` to 5 to prevent overfitting to the val window.

### Q5: No `tqdm` progress bar?

```bash
pip install tqdm
```

Without tqdm, training continues normally — one summary line is printed per epoch.

### Q6: Where is `confusion_matrix.png` saved?

In the same directory as the model output file (`--output`); defaults to the
project root.

### Q7: `visualize_features.py` or `train_video_cad_model.py` raise `cv2` errors?

These legacy scripts depend on OpenCV:

```bash
pip install opencv-python
```

---

## 9. Known Limitations

| Limitation | Details |
|------------|---------|
| **Temporal correlation** | Adjacent frames are highly correlated. True evaluation requires **completely independent recordings** (different sessions / temperatures). |
| **RF evaluation only indicative** | RF mode extracts one feature vector per video; with one video per class, meaningful train/test split is not possible. |
| **Data augmentation** | Horizontal/vertical flip augmentation (`RandomHorizontalFlip`) may or may not be appropriate for speckle, depending on the optical setup. Comment it out in `train_model.py` if unsure. |
| **Frame-level vs. video-level** | The current classifier operates frame-by-frame. For real PUF use, **majority voting** across multiple frames is strongly recommended. |
| **Multi-fiber path untested** | The `fiber_id` task code path is implemented but has not been validated with real multi-fiber data. |

---

*Document version: 2.1*
*Last updated: April 2026*
