# Speckle-PUF Experiment Framework

This document describes the `analysis/` package — a config-driven, reproducible
pipeline for running every experiment and figure associated with the
dual-channel optical PUF paper.

All outputs are written under `results/<run-name>/` and contain:

```
results/<run-name>/
├── config_snapshot.yaml      full copy of the YAML that was used
├── manifest.json             provenance (git SHA, host, captures, artefacts)
├── summary.json              compact, machine-readable metrics summary
├── report.md                 human-readable markdown report with figures/tables
├── run.log                   per-run logger output
├── figures/                  PNG + PDF + SVG for every chart
├── tables/                   CSV for every result table
└── cache/                    optional preprocessed-feature cache
```

---

## 1. Architecture overview

```
analysis/
├── utils/                   config loading, logging, seeding, typed dataclasses
├── io/                      video I/O, dataset discovery, manifests
├── caching/                 on-disk NumPy feature cache
├── preprocessing/           frame pipeline (grayscale, ROI, resize, normalise)
├── metrics/                 distances, auth metrics, profiles, stability, CIs
├── plotting/                matplotlib style + publication-grade chart helpers
├── reporting/               JSON/CSV/Markdown writers, ExperimentReport
└── experiments/             one class per paper experiment + unified registry
```

The flow for any experiment is the same:

1. **Load config** (`ExperimentConfig` wraps YAML with attribute access).
2. **Discover captures** (`DatasetLayout.from_config` + `discover_captures`).
3. **Extract features** (`extract_features` pipes captures through `Pipeline`
   and stores results in a deterministic cache).
4. **Run analyses** (metric modules).
5. **Emit artefacts** (`ExperimentContext.add_plot` / `add_report` +
   `save_figure` writes PNG/PDF/SVG + source CSV).
6. **Write manifest + summary** automatically.

---

## 2. Running experiments

Every experiment has a YAML config in `config/` and a convenience script in
`scripts/`. A unified runner is also provided.

### 2.1 System setup audit (section 3.1)

```
python scripts/run_experiment.py system_setup \
    --config config/system_setup.yaml
# or: python scripts/run_system_setup.py --config config/system_setup.yaml
```

Produces `captures_manifest.csv` + a markdown audit showing which challenges
are missing, the frame-count distribution, and any ingested power metadata.

### 2.2 Fiber length optimisation (section 3.2)

```
python scripts/run_experiment.py length_optimization \
    --config config/length_optimization.yaml
```

Evaluates transmission loss, intra/inter class separability, and pixel entropy
on each length group. Writes `optimal_length.json` containing the recommended
length plus the reasoning (ratio is maximised subject to
`recommendation.green_loss_threshold_db`).

### 2.3 Dual-channel characterisation (section 3.3)

```
python scripts/run_experiment.py dual_channel \
    --config config/dual_channel.yaml
```

Sub-analyses:

* `time_stability.csv` — consecutive + first-frame NCC per (fiber, channel).
* `perturbation_sensitivity.csv` — NCC drop between baseline and perturbed
  condition per (fiber, channel).
* `profile_summary.csv` — FWHM and Gaussian-fit σ of the radial profile of
  each channel's mean response image.

Figures: `time_stability`, `perturbation_sensitivity`, `profile_panel`.

### 2.4 Common-mode suppression (section 3.4)

```
python scripts/run_experiment.py common_mode \
    --config config/common_mode.yaml
```

Compares the raw green feature against the green/red ratio feature under two
shared-disturbance conditions. Emits `cv_comparison` and `reinstall_comparison`
figures plus the suppression summary.

### 2.5 Authentication performance (section 3.5)

```
python scripts/run_experiment.py authentication \
    --config config/authentication.yaml
```

Produces the confusion matrix, verification ROC, AUC, EER, top-k accuracy,
known- vs unknown-challenge identification, per-pair confusion counts and an
optional temporal-drift number.

### 2.6 Live demo / offline replay (section 3.6)

```
# scripted offline replay (CI-friendly)
python scripts/run_experiment.py demo --config config/demo.yaml

# launch the live PySide6 GUI
python scripts/run_experiment.py demo --config config/demo.yaml --set mode=gui
```

The live GUI delegates to `scripts/launch_demo.py` so the existing CCD /
camera integration keeps working unchanged.

### 2.7 Experiment result browser

```
python scripts/launch_dashboard.py
# or: python scripts/launch_dashboard.py path/to/results
```

Dark-themed PySide6 dashboard that discovers every `results/<run>/` folder,
shows the metric cards, figure gallery, table viewer, markdown report, and
run log in one polished window.

---

## 3. Config reference

Every YAML file supports the same top-level schema, with section-specific
extensions noted in the example files.

```yaml
seed: 0

output:
  root: results          # resolved against the repo root when configs live in config/
  name: my_experiment

dataset:
  root: ../videocapture
  layout: domain_fiber_letter     # also: length_fiber_repeat, session_fiber_channel, explicit
  fibers: [...]
  domains: [...]
  channels: [...]
  sessions: [...]
  length_groups: [...]
  fiber_lookup:                   # per-fiber metadata (length group, length mm, …)
    Fiber1: {length_group: 11cm, length_mm: 110}
  domain_map:                     # map physical folder names to channel/condition
    Green: {channel: green, condition: side_green}

preprocess:
  grayscale: true
  center_crop_size: 400
  resize: 112
  normalize: minmax               # none | minmax | zscore | unit
  frame_strategy: middle          # middle | uniform | random | all
  n_frames: 1
  roi: [x, y, w, h]               # optional explicit ROI
  subtract_background: false
  background_path: null

cache:
  enabled: true
```

CLI overrides: `--set dotted.key=value` may be passed multiple times to
tweak configs without editing the YAML (e.g. `--set output.name=smoke`).

---

## 4. Extending the framework

Adding a new experiment is three steps:

1. Subclass `BaseExperiment` in `analysis/experiments/<name>.py` and implement
   `execute(ctx)`.
2. Register it in `analysis/experiments/__init__.py::EXPERIMENT_REGISTRY`.
3. Drop a YAML config in `config/<name>.yaml` and (optionally) a
   `scripts/run_<name>.py` wrapper.

Use the existing modules as reference: `system_setup.py` is the most compact
example, `authentication.py` the most feature-rich.

---

## 5. Performance & determinism

* Frame I/O is lazy — videos are never materialised end-to-end.
* Feature extraction is cached keyed by `(path, preprocess config, sampling
  config)` with a version-aware `meta.json`; repeated runs with the same
  config reuse the cache automatically.
* All distance / ROC / NCC computations are vectorised NumPy pairwise ops.
* `seed_everything(seed)` is called automatically from every experiment,
  seeding `random`, `numpy`, and `torch` (if installed) to keep the split
  and any stochastic pass-throughs deterministic.
* Every output is committed as CSV/JSON/SVG/PDF/PNG so the figures can be
  regenerated or re-typeset from the saved source data without re-running
  feature extraction.
