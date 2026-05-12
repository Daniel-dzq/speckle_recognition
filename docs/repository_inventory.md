# Repository inventory

Generated: **2026-05-12T15:17:07** (local time)

This file is **machine-regenerated**. Re-run:

```bash
python scripts/inventory_repository.py
```

---

## Safety note (historical campaign directories)

The following top-level names are classified as **historical_campaigns**. **Keep them in place** until you confirm no active script depends on their paths; do not move into `archive/legacy_campaigns/` without a code audit:

- `Green/`
- `LengthOptimize/`
- `disturbance_sensitivity/`
- `fiber_loss/`
- `long_term_stability/`
- `power_common_mode/`

---

## Category legend

| Tag | Meaning |
|-----|---------|
| `code_core` | Application / library / config / docs |
| `experiment_results` | `results/*`, checkpoints, training outputs |
| `paper_figures` | Root `figures/` (paper bundle) |
| `publication_figures` | `figures_publication/` |
| `software_copyright_materials` | Typically `figures/softcopyright/` (inside `figures/`) |
| `raw_or_input_data` | Video roots, letter images parent if raw, etc. |
| `historical_campaigns` | Ad hoc campaign trees — **do not relocate blindly** |
| `cache_or_temp` | Caches and temp |
| `experiment_archive` | Local snapshot root (see `scripts/archive_experiment_snapshot.py`) |
| `unknown` | Review manually |

---

## Top-level scan

| Path | Category | Files | Size (MB) | Latest mtime | report.md | manifest.json | weights | video | cache-like | Top extensions |
|------|----------|-------|-----------|--------------|-----------|---------------|---------|-------|------------|----------------|
| `.analysis_cache/` | cache_or_temp | 1740 | 11.505 | 2026-05-04T23:12:35 | 0 | 0 | 0 | 0 | 1 |  |
| `.DS_Store` | cache_or_temp | 1 | 0.014 | 2026-05-11T01:39:50 | 0 | 0 | 0 | 0 | 0 | (no_ext) |
| `.gitignore` | code_core | 1 | 0.004 | 2026-05-12T15:16:26 | 0 | 0 | 0 | 0 | 0 | (no_ext) |
| `__pycache__/` | cache_or_temp | 5 | 0.084 | 2026-04-24T20:28:39 | 0 | 0 | 0 | 0 | 0 |  |
| `analysis/` | code_core | 34 | 0.208 | 2026-05-11T01:21:52 | 0 | 0 | 0 | 0 | 0 |  |
| `archive/` | code_core | 13 | 24.353 | 2026-04-21T22:16:45 | 0 | 0 | 0 | 0 | 0 |  |
| `checkpoints/` | experiment_results | 7 | 299.334 | 2026-04-15T13:49:10 | 0 | 0 | 1 | 0 | 0 |  |
| `config/` | code_core | 7 | 0.01 | 2026-05-11T01:21:52 | 0 | 0 | 0 | 0 | 0 |  |
| `dataset.py` | code_core | 1 | 0.008 | 2026-04-14T00:07:28 | 0 | 0 | 0 | 0 | 0 | .py |
| `disturbance_sensitivity/` | historical_campaigns | 75 | 29.736 | 2026-04-27T20:45:55 | 0 | 0 | 0 | 0 | 0 |  |
| `docs/` | code_core | 8 | 0.062 | 2026-05-12T15:16:39 | 0 | 0 | 0 | 0 | 0 |  |
| `fiber_loss/` | historical_campaigns | 5 | 0.041 | 2026-04-27T15:58:51 | 0 | 0 | 0 | 0 | 0 |  |
| `figures/` | paper_figures | 64 | 48.23 | 2026-05-12T15:16:15 | 0 | 0 | 0 | 0 | 0 |  |
| `figures_publication/` | publication_figures | 21 | 39.018 | 2026-05-05T16:47:56 | 0 | 0 | 0 | 0 | 0 |  |
| `gui/` | code_core | 32 | 56.079 | 2026-05-12T14:56:22 | 0 | 0 | 0 | 0 | 0 |  |
| `LengthOptimize/` | historical_campaigns | 250 | 129.337 | 2026-04-25T14:39:18 | 0 | 0 | 0 | 0 | 0 |  |
| `letter_images/` | code_core | 26 | 0.527 | 2026-04-14T22:31:58 | 0 | 0 | 0 | 0 | 0 |  |
| `long_term_stability/` | historical_campaigns | 195 | 74.825 | 2026-04-27T20:49:43 | 0 | 0 | 0 | 0 | 0 |  |
| `models.py` | code_core | 1 | 0.003 | 2026-04-14T00:07:16 | 0 | 0 | 0 | 0 | 0 | .py |
| `paper.docx` | unknown | 1 | 0.475 | 2026-05-06T17:41:41 | 0 | 0 | 0 | 0 | 0 | .docx |
| `power_common_mode/` | historical_campaigns | 600 | 144.629 | 2026-04-28T12:54:13 | 0 | 0 | 0 | 0 | 0 |  |
| `README.md` | code_core | 1 | 0.021 | 2026-05-12T15:16:58 | 0 | 0 | 0 | 0 | 0 | .md |
| `requirements.txt` | code_core | 1 | 0.001 | 2026-05-11T01:21:52 | 0 | 0 | 0 | 0 | 0 | .txt |
| `results/` | experiment_results | 1137 | 587.456 | 2026-05-11T01:21:52 | 1 | 1 | 1 | 0 | 1 |  |
| `scripts/` | code_core | 33 | 0.394 | 2026-05-12T15:17:05 | 0 | 0 | 0 | 0 | 0 |  |
| `test_mac_slm_output.py` | code_core | 1 | 0.004 | 2026-05-11T01:21:52 | 0 | 0 | 0 | 0 | 0 | .py |
| `train_eval.py` | code_core | 1 | 0.011 | 2026-04-14T00:22:32 | 0 | 0 | 0 | 0 | 0 | .py |
| `unified_dataset.py` | code_core | 1 | 0.039 | 2026-04-15T12:52:56 | 0 | 0 | 0 | 0 | 0 | .py |
| `video_capture/` | raw_or_input_data | 0 | 0.0 |  | 0 | 0 | 0 | 0 | 0 |  |
| `videocapture/` | raw_or_input_data | 390 | 143431.606 | 2026-04-15T17:33:24 | 0 | 0 | 0 | 1 | 0 |  |
| `技术交底书_光纤PUF认证.docx` | unknown | 1 | 0.338 | 2026-04-27T16:07:55 | 0 | 0 | 0 | 0 | 0 | .docx |
| `电学领域技术交底书模板-新 .doc` | unknown | 1 | 0.025 | 2026-04-27T15:24:15 | 0 | 0 | 0 | 0 | 0 | .doc |
| `计算机软件著作权登记申请表 .docx` | unknown | 1 | 0.017 | 2026-04-25T10:35:21 | 0 | 0 | 0 | 0 | 0 | .docx |
| `软著_源程序代码.docx` | unknown | 1 | 0.102 | 2026-04-25T10:55:43 | 0 | 0 | 0 | 0 | 0 | .docx |
| `软著_申请表填写汇总.docx` | unknown | 1 | 0.038 | 2026-04-25T11:00:27 | 0 | 0 | 0 | 0 | 0 | .docx |
| `软著_软件说明书_预审修订版.docx` | unknown | 1 | 5.833 | 2026-05-06T17:28:07 | 0 | 0 | 0 | 0 | 0 | .docx |
| `软著_软件说明书_预审最终修改版.docx` | unknown | 1 | 7.109 | 2026-05-11T02:10:38 | 0 | 0 | 0 | 0 | 0 | .docx |

---

## `results/` sub-runs (if present)

| Run folder | Files | Size (MB) | Latest mtime | report | manifest | figures dir | tables dir | weights | video | Top extensions |
|------------|-------|-----------|--------------|--------|----------|-------------|------------|---------|-------|----------------|
| `results/cross_fiber` | 4 | 0.083 | 2026-04-14T00:35:32 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| `results/fiber1` | 8 | 42.882 | 2026-04-13T23:06:42 | 0 | 0 | 0 | 0 | 1 | 0 |  |
| `results/fiber2` | 8 | 42.889 | 2026-04-13T23:14:52 | 0 | 0 | 0 | 0 | 1 | 0 |  |
| `results/fiber3` | 8 | 42.886 | 2026-04-13T23:22:34 | 0 | 0 | 0 | 0 | 1 | 0 |  |
| `results/fiber4` | 8 | 42.888 | 2026-04-13T23:31:50 | 0 | 0 | 0 | 0 | 1 | 0 |  |
| `results/fiber5` | 8 | 42.889 | 2026-04-13T23:41:02 | 0 | 0 | 0 | 0 | 1 | 0 |  |
| `results/fiber_auth` | 8 | 213.79 | 2026-04-15T14:49:24 | 0 | 0 | 0 | 0 | 1 | 0 |  |
| `results/green_partial_32` | 19 | 31.987 | 2026-04-24T10:58:47 | 1 | 0 | 1 | 1 | 0 | 0 |  |
| `results/length_optimization_green` | 1024 | 65.776 | 2026-04-25T14:44:18 | 1 | 1 | 1 | 1 | 0 | 0 |  |
| `results/length_optimize_current` | 19 | 60.352 | 2026-05-05T15:56:08 | 1 | 0 | 1 | 1 | 0 | 0 |  |
| `results/loss_analysis_32` | 8 | 0.232 | 2026-05-11T01:21:52 | 1 | 0 | 1 | 0 | 0 | 0 |  |
| `results/unified` | 13 | 0.658 | 2026-04-15T13:47:04 | 0 | 0 | 0 | 0 | 0 | 0 |  |

---

## Figure file manifest

Saved **`docs/generated_figures_manifest.csv`** (52 image files under `figures/`).

---

## See also

- [`output_organization.md`](output_organization.md) — roles of outputs and paper workflow.
