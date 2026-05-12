# Output directories: roles and traceability

Generated data (videos, `results/`, most of `figures/`, weights) is typically **gitignored**. This document is the **primary map** for finding which folder answers which scientific or documentation question.

---

## 1. Directory roles

| Location | Role |
|----------|------|
| **`results/<run_name>/`** | **Most traceable** output of the `analysis/` framework: `report.md`, `manifest.json`, `figures/`, `tables/`, `run.log`, optional `cache/`. One run = one config + one provenance record. |
| **`figures/`** (repo root) | **Paper “main figure pack”**: mostly flat `fig_*` from **`scripts/make_paper_figures.py`**. Summarizes training + **fiber_auth** storyline; **not** tied to a single `results/<run>/` folder unless you document it. |
| **`figures_publication/`** | **Journal typography** re-export from **`scripts/make_publication_figures.py`**. Does not overwrite root `figures/` by design. |
| **`figures/softcopyright/`** | **GUI screenshots** for registration / legal packs (`scripts/capture_manual_screenshots.py`). **Not** core experimental science figures. |
| **`figures/new_datasets_analysis/`** | **Extension-dataset** analysis plots (`scripts/analyze_new_datasets.py`). |
| **`experiment_archive/`** | **Your** dated snapshots from **`scripts/archive_experiment_snapshot.py`** — local only, **never** commit. |
| **`archive/`** | **Legacy code / old scripts** shipped with the repo (see `archive/README.md`). Not the same as `experiment_archive/`. |
| **`checkpoints/`**, **`results/fiber*/`** | Training outputs from the **root training stack** (temporal-split trainers, etc.). |
| **`results/fiber_auth/`** | **PUF matrix** evaluation outputs + **`fiber_models/*.pth`** expected by the live demo. |
| **Historical campaign trees** (`LengthOptimize/`, `disturbance_sensitivity/`, `fiber_loss/`, `long_term_stability/`, `power_common_mode/`, `Green/`, …) | **Ad hoc or historical** data campaigns. **Do not move** without checking script paths; see `docs/repository_inventory.md` after running the inventory script. |

---

## 2. How to identify a figure’s source (priority order)

1. **`results/<run>/report.md`** — lists artefacts for that `analysis/` run (figures + captions + CSV sources).
2. **`results/<run>/manifest.json`** — machine-readable list of inputs, git SHA, artefacts.
3. **Flat `figures/fig_*`** — infer from **`scripts/make_paper_figures.py`** (grep the stem) and tie to **`results/fiber_auth/auth_matrix.json`**, per-fiber `results/fiber*/training_log.csv`, etc.
4. **`docs/generated_figures_manifest.csv`** — run `python scripts/inventory_repository.py` to refresh a **filename → likely script** table for images under `figures/`.
5. **`figures/README.md`** — explains the purpose of this directory vs single runs.
6. **Git + command history** — `git rev-parse HEAD`, shell history, or metadata in **`experiment_archive/.../GIT_COMMIT.txt`** after you snapshot.

---

## 3. Recommended paper workflow

1. **Run experiments** — `python scripts/run_experiment.py …` so each claim has a `results/<run>/` folder with `report.md` + `manifest.json`.
2. **Generate consolidated paper figures** — `python scripts/make_paper_figures.py` (writes root `figures/`).
3. **Optional journal pass** — `python scripts/make_publication_figures.py` → `figures_publication/`.
4. **Archive a snapshot** — `python scripts/archive_experiment_snapshot.py --tag <label> --apply` (add `--include-models` if you need `.pth` inside the snapshot).
5. **Write the paper from the archived copy** + cite the snapshot README / manifest.
6. **Do not hand-edit archived outputs** — if something is wrong, regenerate from code and create a **new** archive folder with a new tag.

---

## 4. What not to commit

| Item | Reason |
|------|--------|
| **`experiment_archive/`** | Large, personal snapshots; listed in `.gitignore`. |
| **Raw videos** | `videocapture/`, `video_capture/`, and `*.mp4` / `*.avi` / `*.mov` patterns (see `.gitignore`). |
| **Caches** | `.cache/`, `.analysis_cache/`, `analysis_cache/`, `__pycache__/`, per-run `cache/` under `results/` when huge. |
| **Model weights** | `*.pth`, `*.pt`, `*.ckpt` unless you intentionally ship small fixtures (default: ignore). |
| **Full root `figures/` PNG trees** | Ignored via `figures/.gitignore`; only **`figures/README.md`** and **`figures/.gitignore`** stay in Git. |

---

## 5. Automation in this repo

| Script | Purpose |
|--------|---------|
| **`scripts/inventory_repository.py`** | Read-only scan → `docs/repository_inventory.md` + `.csv` + `docs/generated_figures_manifest.csv`. |
| **`scripts/archive_experiment_snapshot.py`** | **Copy-only** snapshot under `experiment_archive/`; **dry-run by default**, `--apply` to copy. |

---

## Short answers

| Question | Answer |
|----------|--------|
| Single-run truth? | **`results/<run_name>/`** |
| “Official” bundled paper plots from training/PUF? | **`figures/`** after **`make_paper_figures.py`** |
| Journal-style figures? | **`figures_publication/`** |
| Softcopyright UI shots? **`figures/softcopyright/`** |
| Local dated bundle for sharing? | **`experiment_archive/<timestamp>_<tag>/`** after **`archive_experiment_snapshot.py --apply`** |
