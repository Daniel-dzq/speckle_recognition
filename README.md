# Speckle recognition for multimode-fiber optical PUFs

**Complete handbook for this repository.** If you have never used the project before, read sections **1 → 6** in order, then jump to **training**, **live demo**, or **analysis** depending on your goal.

A separate **Chinese software-description Word document** (for copyright registration) is **not** part of this Git repository; this `README` is the **English** master guide for code, data layout, and reproducible workflows.

---

## Table of contents

1. [What this software does (in plain language)](#1-what-this-software-does-in-plain-language)
2. [Glossary](#2-glossary)
3. [Who should read which section](#3-who-should-read-which-section)
4. [Requirements](#4-requirements)
5. [Installation (step by step)](#5-installation-step-by-step)
6. [Verify your environment](#6-verify-your-environment)
7. [Repository map](#7-repository-map)
8. [Data layout (videos and labels)](#8-data-layout-videos-and-labels)
9. [Models (what the neural networks are)](#9-models-what-the-neural-networks-are)
10. [Training workflows](#10-training-workflows)
11. [PUF authentication evaluation (5×5 matrix)](#11-puf-authentication-evaluation-5--5-matrix)
12. [Inference from the command line](#12-inference-from-the-command-line)
13. [Live demo GUI (full operating guide)](#13-live-demo-gui-full-operating-guide)
14. [Experiment framework (`analysis/`)](#14-experiment-framework-analysis)
15. [Figures and where files are written](#15-figures-and-where-files-are-written)
16. [Automation: screenshots and dashboards](#16-automation-screenshots-and-dashboards)
17. [Sharing the demo on another computer](#17-sharing-the-demo-on-another-computer)
18. [Troubleshooting](#18-troubleshooting)
19. [Frequently asked questions](#19-frequently-asked-questions)
20. [Related documents in `docs/`](#20-related-documents-in-docs)
21. [License](#21-license)

---

## 1. What this software does (in plain language)

1. A **letter** (A–Z) is shown on a **spatial light modulator (SLM)**—treated as a normal computer **monitor output** in this software.
2. Light goes through a **multimode optical fiber**. The fiber scrambles the light into a **speckle** pattern.
3. A **camera** records that speckle pattern.
4. A **deep learning model** looks at short **video clips** of speckle and predicts **which letter** was sent.
5. Because each fiber scrambles light differently, a model trained on **fiber A** usually fails on **fiber B**. That asymmetry is what people use for **PUF-like authentication** experiments.

This repository gives you:

- **Training and evaluation code** (PyTorch).
- A **desktop GUI** (PySide6) for live camera + SLM + recognition.
- An **`analysis/` framework**: YAML configs, structured outputs under `results/<run>/`, figures, tables, reports.

---

## 2. Glossary

| Term | Meaning here |
|------|----------------|
| **PUF** | Physical Unclonable Function—hardware that behaves uniquely; this project **simulates** the idea using different fibers. |
| **Challenge** | The letter (or image) you display on the SLM. |
| **Response** | The speckle video/image the camera sees. |
| **Fiber1 … Fiber5** | Example fiber IDs used in the code and configs. Your folder names can differ if you adapt the layouts. |
| **Domain** | Illumination condition (e.g. green only, green + red). Folders: `Green`, `GreenAndRed`, `RedChange`. |
| **Temporal split** | Frames inside each video are split in **time** (train / val / test) so the same physical frames are not reused incorrectly. |

---

## 3. Who should read which section

| Your goal | Read |
|-----------|------|
| Run the live lab demo with a camera | [§13 Live demo GUI](#13-live-demo-gui-full-operating-guide), [§17 Sharing](#17-sharing-the-demo-on-another-computer) |
| Train from your own videos | [§8 Data layout](#8-data-layout-videos-and-labels), [§10 Training](#10-training-workflows) |
| Reproduce the authentication matrix | [§11 PUF evaluation](#11-puf-authentication-evaluation-5--5-matrix) |
| Run paper-style analysis experiments | [§14 Experiment framework](#14-experiment-framework-analysis), `docs/experiments.md` |
| Understand output folders / archiving | [§15 Figures](#15-figures-and-where-files-are-written), `docs/output_organization.md` |

---

## 4. Requirements

### Software

| Component | Minimum / notes |
|-----------|------------------|
| **Python** | 3.9+ (3.10+ recommended) |
| **PyTorch** | See `requirements.txt` (torch ≥ 2.0 in the pinned file; adjust for your CUDA) |
| **OpenCV** | For training I/O and webcam path |
| **PySide6** | Live demo + dashboards |
| **PyYAML, scipy, pandas** | `analysis/` experiments |

### Hardware (optional but typical for the live demo)

- A **second display** (or projector) for the SLM output.
- A **camera**: MindVision **HT-UBS300C** (bundled SDK paths in `gui/`) or any **UVC / OpenCV** camera.
- **GPU**: optional; Apple Silicon can use **MPS**; NVIDIA can use **CUDA**.

### Disk

- **Training data** (`videocapture/…`) is **large** and is **not** committed to Git.
- **Checkpoints** (`*.pth`) are **ignored** by Git—generate or copy them locally.

---

## 5. Installation (step by step)

### 5.1 Clone

```bash
git clone https://github.com/Daniel-dzq/speckle_recognition.git
cd speckle_recognition
```

### 5.2 Virtual environment (strongly recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows cmd/PowerShell
```

### 5.3 Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5.4 (Optional) CUDA PyTorch

If you use NVIDIA GPUs, install a CUDA build of PyTorch **instead of** the CPU wheels—for example (check [PyTorch install page](https://pytorch.org) for the exact index URL):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 5.5 MindVision on Windows (one-time driver)

If you use the HT-UBS300C on Windows and the device shows as unknown in Device Manager:

1. Plug in the camera.
2. **Device Manager** → camera → **Update driver** → **Browse** → point to `gui/win_sdk/drivers/`.

macOS uses the bundled `gui/libmvsdk.dylib`; no driver wizard in the repo.

---

## 6. Verify your environment

From the **repository root**:

```bash
python scripts/env_check.py
```

You should see Python, OpenCV, PyTorch, PySide6, and (if hardware is present) MindVision detection. Fix any missing package before training or GUI work.

---

## 7. Repository map

| Path | Purpose |
|------|---------|
| `models.py` | `CNNPoolModel` (default), `R3DModel`, etc. |
| `dataset.py` | Video-clip dataset, temporal splits. |
| `unified_dataset.py` | Multi-domain dataset + caching helpers. |
| `train_eval.py` | Training loop, metrics, exports. |
| `scripts/` | **All runnable entry points** (see §10–§16). |
| `gui/` | `MainWindow`, `SLMWindow`, `RobotPanel`, camera workers, MindVision wrapper, bundled SDK files. |
| `analysis/` | Config-driven experiments: I/O, preprocessing, metrics, plotting, reporting. |
| `config/` | YAML configs for each experiment type. |
| `letter_images/` | Pre-rendered **A–Z** PNGs for SLM display (Calibri-style assets). |
| `docs/` | Longer references (`experiments.md`, `gui_tutorial.md`, `output_organization.md`). |
| `archive/` | Legacy scripts; read `archive/README.md` if you need old CLIs. |

**Not in Git (see `.gitignore`):** `videocapture/`, `results/`, `checkpoints/`, `figures/`, `*.pth`, local Word/PDF, scratch dataset trees.

---

## 8. Data layout (videos and labels)

Training scripts expect video files organized by **illumination domain**, **fiber**, and **letter filename**:

```text
videocapture/
├── Green/
│   ├── Fiber1/
│   │   ├── A.avi
│   │   ├── B.avi
│   │   └── … Z.avi
│   └── Fiber2/ … Fiber5/
├── GreenAndRed/
│   └── Fiber1/ …
└── RedChange/
    └── Fiber1/ …
```

Rules:

- The **file name without extension** is the **class label** (usually `A` … `Z`).
- Each `.avi` (or other supported video) should contain enough frames (order **hundreds** per file for stable splits—see training docs in code comments if you hit edge cases).

Some configs use `video_capture/` instead of `videocapture/` for ad hoc captures—the idea is the same: a root with domain/fiber/letter videos.

---

## 9. Models (what the neural networks are)

### Default: `CNNPoolModel` (`--model_type cnn_pool`)

1. **ResNet18** backbone (ImageNet pretrained) extracts a **512-D** vector **per frame**.
2. **Temporal average pooling** over the clip → one **512-D** vector.
3. **Dropout** + **linear** layer → **26 outputs** (letters A–Z).

### Alternative: `R3DModel` (`--model_type r3d`)

3D ResNet—can work but often needs **more data** to beat the pooling baseline on small speckle datasets.

---

## 10. Training workflows

Always run commands from the **repository root** unless a script documents otherwise.

### 10.1 Single fiber

```bash
python scripts/train_fiber.py --fiber fiber1
```

Common optional flags (see `train_fiber.py --help`):

- `--model_type cnn_pool` or `r3d`
- `--clip_len`, `--stride`, `--epochs`, `--lr`

Outputs typically go under `results/<fiber>/` with confusion matrices, logs, CSV metrics (exact paths depend on script version—see console summary after a run).

### 10.2 Train all five fibers sequentially

```bash
python scripts/train_all_fibers.py
python scripts/train_all_fibers.py --epochs 50 --lr 1e-4
python scripts/train_all_fibers.py --only Fiber1 Fiber2
```

### 10.3 Unified multi-domain training

```bash
python scripts/train_unified.py --split_mode deploy --epochs 20
python scripts/train_unified.py --split_mode cross_fiber --epochs 20
```

Evaluate:

```bash
python scripts/evaluate_unified.py --checkpoint results/unified/best_model.pth
```

### 10.4 Domain ablation

```bash
python scripts/diagnose_domains.py --fiber Fiber1
python scripts/run_all_fiber_ablations.py
```

---

## 11. PUF authentication evaluation (5×5 matrix)

This trains **one model per fiber** using the multi-domain setup and evaluates **each model on every fiber** → a **5×5 accuracy matrix** (authorized = diagonal, unauthorized = off-diagonal).

**Prerequisite:** videos under `videocapture/` (or the paths your config expects).

```bash
python scripts/fiber_auth_eval.py
```

Important outputs (default layout):

```text
results/fiber_auth/
├── auth_matrix.csv
├── auth_matrix.json
├── auth_summary.txt
└── fiber_models/
    ├── Fiber1.pth
    ├── …
    └── Fiber5.pth
```

The **live demo** loads weights from:

```text
results/fiber_auth/fiber_models/Fiber*.pth
```

If that folder is empty, run §11 first or copy compatible `.pth` files there.

---

## 12. Inference from the command line

Batch prediction on still folders or frames:

```bash
python scripts/predict.py --model checkpoints/fiber1_best.pth --test-dir path/to/A_folder
python scripts/predict.py --model checkpoints/fiber1_best.pth --test-dir path/to/folder --ground-truth A --top-k 3
```

Paths depend on where **you** saved weights; `fiber_auth_eval` uses `results/fiber_auth/fiber_models/` filenames.

---

## 13. Live demo GUI (full operating guide)

### 13.1 Launch

```bash
python scripts/launch_demo.py
```

Window title: **Speckle-PUF Live Demo**. A banner prints in the terminal (Python version, platform, OpenCV, MindVision status).

### 13.2 Layout (what you see)

- **Left column (scrollable):** grouped controls.
- **Center-right:** large **camera preview** (this is **not** the SLM surface).
- **Right strip:** **RobotPanel** (status, READING / GRANTED / DENIED, confidence, Top-K).
- **Bottom:** **Log** text area.
- **Status bar:** **Device** (CUDA / MPS / CPU) | **Fiber: …** | **FPS**.

### 13.3 Fiber Authentication (models)

**Group title:** `Fiber Authentication`.

| Control | What it does |
|---------|----------------|
| **Authorized fiber** (dropdown) | Lists every `Fiber*.pth` found in `results/fiber_auth/fiber_models/`. |
| **Refresh** | Rescans that directory. |

**There is no separate “Load model” button in the flow:** choosing a fiber triggers loading in the inference worker. On success, the status line shows **`Loaded: FiberN`** in green.

**If the dropdown is empty**

1. Run `python scripts/fiber_auth_eval.py` **or**
2. Copy correctly trained `Fiber*.pth` files into `results/fiber_auth/fiber_models/`, then click **Refresh**.

### 13.4 SLM Output Window

| Control | What it does |
|---------|----------------|
| **Open SLM Window** / **Hide** | Shows or hides the SLM window. |
| **SLM screen** | Chooses **which Qt screen** receives the SLM window (pick the HDMI/DP routed to the SLM). |
| **Refresh** | Re-enumerates displays after cable changes. |
| **Fullscreen on selected screen** | Full-screens on the chosen display. |
| **Move SLM to Selected Screen** | Forces geometry onto the selected screen—use this instead of guessing with OS mirroring alone. |
| **Test SLM Output** | Built-in test pattern to verify the video path (look at the **SLM monitor**, not the center preview). |
| **Letter** | Type **A–Z** (one letter). |
| **Send to SLM** | Pushes the letter to the SLM window using `letter_images/` PNGs when applicable. |
| **Font size** | Larger/smaller glyph on the SLM. |
| **◀ Prev / Next ▶** | Cycle A–Z. |
| **Stretch to fill (no letterbox)** | Fill vs preserve aspect. |
| **Load Image to SLM** | PNG/JPG/BMP custom pattern. |

**Critical concept:** the **big center preview** is the **camera**. The **SLM** is whatever **monitor** you selected. The software draws via the **graphics API**, not by flashing firmware inside a specific SLM vendor SDK.

### 13.5 Camera / Video Source

**Option A — MindVision HT-UBS300C (recommended on macOS for this camera)**

Click **MindVision CCD (HT-UBS300C)**. Enumeration runs automatically.

**Option B — OpenCV camera**

Set **Camera index**, optional **Resolution**, then **Scan Available Cameras** (probes indices; on macOS may trigger **system camera permission**), then **Start Camera**.

**Option C — File**

Use **Load Video File** for offline demos (AVI/MP4/MKV/MOV typically).

**Stop** ends any active source.

### 13.6 Camera Settings

Sliders and ticks for exposure, gain, color (if applicable), flips. Disabled until a source is **actively capturing** (implementation detail: greyed out when idle).

### 13.7 Inference Settings

| Control | Meaning |
|---------|---------|
| **Infer every N frames** | Run the network once every N frames (save CPU/GPU). |
| **Vote window** | Smoothing: majority vote over recent predictions—use a **larger** window for steadier public demos. |
| **Recognition active** | **Unchecked** → preview only, no inference loop. **Checked** → predictions update. |

### 13.8 Reading the RobotPanel

Depending on model output and thresholds, you will see states such as **STANDBY**, **READING PUF**, **ACCESS GRANTED**, **ACCESS DENIED**, plus **confidence** and **Top-K** text. Exact strings are defined in `gui/robot_panel.py`.

### 13.9 A complete first-time demo sequence (checklist)

1. Connect **camera** and **SLM monitor**.
2. `python scripts/launch_demo.py`
3. **Fiber Authentication:** pick `FiberN`, wait for **Loaded: FiberN**.
4. **SLM:** Refresh displays → pick SLM screen → optional fullscreen → **Move SLM to Selected Screen**.
5. **Camera:** MindVision **or** Start Camera **or** Load Video File.
6. Confirm **live speckle** in the **center preview**.
7. Type a letter → **Send to SLM**; verify the **external** display shows the glyph.
8. Enable **Recognition active**; read predictions (prefer smoothed / panel text for demos).

For even more GUI narrative (mirrors an internal tutorial style), see **`docs/gui_tutorial.md`**—but verify paths like **`results/fiber_auth/fiber_models/`** against this README if the older text mentions `checkpoints/` alone.

---

## 14. Experiment framework (`analysis/`)

Unified runner:

```bash
python scripts/run_experiment.py <experiment_name> --config config/<name>.yaml
```

Convenience wrappers (examples):

```bash
python scripts/run_system_setup.py --config config/system_setup.yaml
python scripts/run_length_optimization.py --config config/length_optimization.yaml
python scripts/run_dual_channel_analysis.py --config config/dual_channel.yaml
python scripts/run_common_mode_eval.py --config config/common_mode.yaml
python scripts/run_authentication_eval.py --config config/authentication.yaml
python scripts/run_demo.py --config config/demo.yaml
```

Override YAML keys from CLI:

```bash
python scripts/run_experiment.py demo --config config/demo.yaml --set output.name=my_smoke --set seed=42
```

Each run creates **`results/<output.name>/`** containing `report.md`, `manifest.json`, `figures/`, `tables/`, `run.log`, etc.

**Deep reference:** `docs/experiments.md`.

---

## 15. Figures and where files are written

| Output | Produced by | Typical location |
|--------|-------------|-------------------|
| Per-run analysis plots | `analysis/` | `results/<run>/figures/` |
| “Paper pack” auth/training figures | `make_paper_figures.py` | `figures/` (flat `fig_*` names) |
| Journal-style re-render | `make_publication_figures.py` | `figures_publication/` |
| Extension dataset analytics | `analyze_new_datasets.py` | `figures/new_datasets_analysis/` |

**Archiving / Which PNG belongs to which experiment?** See **`docs/output_organization.md`**.

---

## 16. Automation: screenshots and dashboards

**Result browser (PySide6):**

```bash
python scripts/launch_dashboard.py
python scripts/launch_dashboard.py /path/to/results
```

**Automated PNG grabs of the live GUI** (needs a real display; for documentation maintainers):

```bash
python scripts/capture_manual_screenshots.py --auto --native
```

Default output directory: `figures/softcopyright/` (gitignored).

---

## 17. Sharing the demo on another computer

Minimum **code + assets**:

| Path | Why |
|------|-----|
| `gui/` | Application + MindVision bundle |
| `scripts/launch_demo.py` | Entry point |
| `models.py` | Architecture definition |
| `letter_images/` | SLM letters |
| `requirements.txt` | Dependencies |

Minimum **learned weights** (example layout):

```text
results/fiber_auth/fiber_models/Fiber1.pth … Fiber5.pth
```

**You do not need** the full `videocapture/` tree **only for inference**, but you **do** need it for training or `analysis/` experiments that read raw videos.

---

## 18. Troubleshooting

### Python import errors

Run from **repo root**, or use the provided `scripts/*.py` which adjust `sys.path`.

### `No module named torch` / `PySide6`

Re-run `pip install -r requirements.txt` inside your **activated** venv.

### GUI: no fiber models

Create `results/fiber_auth/fiber_models/` and place `Fiber*.pth` there, or run `fiber_auth_eval.py`.

### GUI: MindVision not detected (macOS)

- USB reconnect, wait a few seconds.
- **System Settings → Privacy & Security → Camera** → allow **Terminal** / **Python** / **Cursor** host app.
- If you rebuilt the repo, confirm `gui/libmvsdk.dylib` exists.

If macOS reports an **ad-hoc signature** problem:

```bash
codesign --force --sign - gui/libmvsdk.dylib
```

### GUI: camera works but SLM is black

- Wrong **SLM screen** selected—open **SLM screen** dropdown and re-pick the external display.
- Click **Move SLM to Selected Screen** then **Send to SLM** again.
- Use **Test SLM Output** and look at the **projector / SLM monitor**, not the center preview.

### Training: very low accuracy

- Verify videos actually contain speckle (open one in a media player).
- Ensure enough frames per file.
- Try `--lr 1e-4`, tune `--clip_len` / `--stride`.

### GPU not used

```bash
python -c "import torch; print('CUDA', torch.cuda.is_available()); print('MPS', torch.backends.mps.is_available())"
```

Install the correct PyTorch build for your hardware.

---

## 19. Frequently asked questions

**Do I need internet after `pip install`?**  
No for running local experiments; only for initial package download and optional pretrained weight fetch.

**Can I use fewer than 26 letters?**  
Labels come from filenames; you can adapt if you change `num_classes` and training data consistently (advanced).

**Why are `results/` and `figures/` empty after clone?**  
They are **generated outputs**, not source code. Train or run experiments locally.

**Where is the “login screen”?**  
There is **no** password login. The main window **is** the application entry point.

---

## 20. Related documents in `docs/`

| File | Contents |
|------|-----------|
| `docs/gui_tutorial.md` | Long GUI walkthrough (cross-check model paths with §13 here). |
| `docs/experiments.md` | Full `analysis/` reference. |
| `docs/output_organization.md` | How `figures/` and `results/` relate; archival hints. |
| `docs/legacy/usage.md` | Older CLI notes. |

---

## 21. License

MIT License
