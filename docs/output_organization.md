# Where outputs live (figures, tables, raw bundles)

Everything under `results/`, `figures/`, `figures_publication/`, and most large data trees is **gitignored**. The repository only holds the **code to regenerate** outputs. Use this note when you archive a “frozen” result set on disk.

## 1. Canonical per-experiment runs (`analysis/` framework)

**Path pattern:** `results/<run_name>/` (from `config/*.yaml` → `output.root` + `output.name`)

**Contents (typical):**

| Item | Role |
|------|------|
| `report.md` | Human-readable narrative + list of artefacts |
| `manifest.json` | Machine-readable run metadata |
| `summary.json` | Quick numeric summary when present |
| `figures/` | PNG/PDF/SVG produced **for this run only** |
| `tables/` | CSV / small structured exports |
| `run.log` | Text log |
| `cache/` or nested cache | Derived numpy cache (optional) |

**How to know which figure belongs to which experiment:** open `results/<run_name>/report.md` or `manifest.json`; every plot registered via `ExperimentContext.add_plot` is listed there with filenames under `figures/`.

**CLI entry:** `python scripts/run_experiment.py <experiment> --config config/<name>.yaml`

This is the **primary** place for “formal” analysis experiments (paper sections 3.1–3.6 style).

## 2. Consolidated training / PUF paper figures (flat `figures/`)

**Generator:** `scripts/make_paper_figures.py`  
**Reads from:** mainly `results/fiber_auth/`, per-fiber `results/fiber*/`, and `videocapture/`  
**Writes to:** repo root `figures/` (flat), e.g. `fig_auth_matrix`, `fig_training_curves`, …

These are **not** tied to a single `results/<run>/` folder; they are a **curated bundle** for the main authentication / training storyline. If you archive them, record **which** `auth_matrix.json`, checkpoints, and git commit you used.

**Subfolders you may also have:**

| Subpath | Typical source |
|---------|----------------|
| `figures/new_datasets_analysis/` | `scripts/analyze_new_datasets.py` (extension datasets) |
| `figures/softcopyright/` | `scripts/capture_manual_screenshots.py` (UI grabs for legal docs) |
| `figures/patent/` | Ad hoc / manual (if you use it) |

## 3. Journal-style re-export (`figures_publication/`)

**Generator:** `scripts/make_publication_figures.py`  
**Writes to:** `figures_publication/` (by design **does not** overwrite root `figures/`)

Use this as a **second layer** (fonts, layout) for submission. Archive it together with the same `results/` snapshot you used for inputs.

## 4. Large data trees next to the repo (not under `results/`)

Examples (often gitignored): `LengthOptimize/`, `disturbance_sensitivity/`, `fiber_loss/`, `long_term_stability/`, `power_common_mode/`, `figures_publication/`.

These are usually **raw or intermediate campaign data**. Prefer recording:

- absolute or relative path root,
- date,
- git commit,
- which script produced which plot (see script headers).

## 5. Suggested local archive layout (one glance = one study)

Keep this **outside** git or in a dated folder you control, for example:

```text
experiment_archive/
  2026-05-11_main_paper_bundle/
    GIT_COMMIT.txt              # output of: git rev-parse HEAD
    README.txt                    # one paragraph: what this freeze is for
    results/                      # copy of relevant runs, e.g. fiber_auth, length_optimization runs
    figures/                      # optional copy after make_paper_figures.py
    figures_publication/          # optional copy after make_publication_figures.py
    config_snapshots/             # copy of config/*.yaml used
```

Inside `README.txt`, list **each** `results/<run>/` you copied and one line: “figures in this run → see that folder’s report.md”.

## Short answer

| Question | Answer |
|----------|--------|
| Which folder is “the official experiment output” for `analysis/`? | **`results/<run_name>/`** (with `figures/` + `tables/` inside it). |
| Which folder is “the official paper figure pack” from training/PUF eval? | **Root `figures/`** after **`make_paper_figures.py`** (flat `fig_*` files). |
| Which is the “nicer journal” pack? | **`figures_publication/`** after **`make_publication_figures.py`**. |
| How do I map image → experiment? | Start from **`results/<run>/report.md`** or **`manifest.json`**; for flat `fig_*`, cross-check **`make_paper_figures.py`** section names and input paths. |
