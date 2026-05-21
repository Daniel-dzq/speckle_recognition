# release_minimal — Final validation report

**Date:** 2026-05-21  
**Path:** `release_minimal/` (~50 GB)  
**Policy:** Copy-only from dev repo; not committed/pushed.

---

## 1. Folder integrity

| Path | Status |
|------|--------|
| `README.md` | OK |
| `input.pptx` | OK |
| `challenge_inputs/manifest.json` | OK (relative `source_pptx`: `input.pptx`) |
| `data/recognition_dataset/` | OK |
| `models/final_15fibers/Fiber1.pth` … `Fiber15.pth` | OK (15/15) |
| `outputs/final_15fiber_training/summary_15fibers.csv` | OK |
| `outputs/final_15fiber_training/auth_matrix_15x15.png` | OK |
| `figures/Fig2_length_optimization.png` | **MISSING** (see §8) |
| `experiments/length_optimization/` | OK |
| `experiments/fiber_loss/` | OK |
| `experiments/long_term_stability/` | OK |
| `experiments/disturbance_sensitivity/` | OK |

**Also present:** `figures/Fig2_length_optimization_regen.{png,pdf,svg}`, `Fig2_length_optimization_word.*`, `Fig2_length_optimization_data_summary.csv`, `Fig2_length_optimization_report.md`, `figures/auth_matrix_15x15.png`, `outputs/.../auth_matrix_15x15.csv`, GUI SDK, `analysis/`, experiment READMEs.

---

## 2. Dataset integrity

| Check | Result |
|-------|--------|
| Total videos | **240** (`.avi`) |
| Domains | **2** — `GreenAndRed`, `RedChange` |
| Fibers per domain | **15** each — `Fiber1` … `Fiber15` |
| Videos per fiber per domain | **8** each |
| Labels (all fibers, both domains) | `1`, `2`, `3`, `a`, `b`, `c`, `boy`, `girl` |
| `gril` typo file | **Not present** |

**Expected layout:** 2 × 15 × 8 = 240 videos.

---

## 3. Model integrity

| Check | Result |
|-------|--------|
| `.pth` files in `models/final_15fibers/` | **15** (`Fiber1.pth` … `Fiber15.pth`) |
| `label_map.json` classes | `1`, `2`, `3`, `a`, `b`, `c`, `boy`, `girl` |
| `num_classes` | **8** |

---

## 4. Experiment folders

| Experiment | data | outputs | scripts | README | Relative paths |
|------------|------|---------|---------|--------|----------------|
| `length_optimization` | OK (~130M) | OK (1029 files) | OK (13 `.py`) | OK | **Pass** |
| `fiber_loss` | OK (5 ×lsx) | Empty dir | Empty dir* | OK | **Pass** |
| `long_term_stability` | OK (~75M) | OK (3 files) | OK (1 script) | OK | **Pass** |
| `disturbance_sensitivity` | OK (~30M) | OK (3 files) | OK (1 script) | OK | **Pass** |

\* `fiber_loss/scripts/` and `outputs/` exist but are empty by design; loss reproduction is via `figures/generate_fig2_length_optimization.py` and length-optimization pipeline (documented in README).

**Absolute path grep** (text types; exclude `data/`, `models/`, `win_sdk/`):

```bash
grep -R "/Users/ziqidai" release_minimal \
  --include="*.py" --include="*.md" --include="*.txt" \
  --include="*.json" --include="*.yaml" --include="*.yml" --include="*.csv" \
  --exclude-dir=data --exclude-dir=models --exclude-dir=win_sdk
```

**Result: 0 matches** (metadata artifacts patched to release-relative paths).

**Fig2 regeneration:** `python3 figures/generate_fig2_length_optimization.py` writes only under `release_minimal/figures/` (`Fig2_length_optimization_regen.*`); not in parent repo.

---

## 5. Smoke tests

Environment: **`conda` env `recognition`** (has `torch`, `PySide6`). System `python3` alone lacks `torch`.

| Test | `recognition` env | Bare `python3` |
|------|-------------------|----------------|
| `python scripts/train_final_15fibers.py --help` | **OK** | FAIL (no torch) |
| `python scripts/export_ppt_challenges.py --help` | **OK** | **OK** |
| `labels_match("A", "a")` → `True` | **OK** (`gui.challenge_widgets`) | N/A |
| `labels_match("girl", "gril")` → `False` | **OK** | N/A |
| Core imports (`final_fiber_dataset`, `unified_dataset`, `train_eval`, `models`, `gui.challenge_manifest`) | **OK** | FAIL (no torch) |

**Note:** `labels_match` lives in `gui/challenge_widgets.py`, not `final_fiber_dataset.py`.

**Challenge inputs:** 8 PNGs + `manifest.json` (8 challenges).

---

## 6. Junk check

| Item | Count | Location |
|------|-------|----------|
| `__pycache__/` dirs | **1** | `experiments/length_optimization/scripts/paper_figures/__pycache__/` |
| `*.pyc` | **2** | Same directory |
| `.DS_Store` | **0** | — |

**Recommendation before zip:** `find release_minimal -name '__pycache__' -exec rm -rf {} +` and delete `*.pyc`.

---

## 7. Size

| Path | Size |
|------|------|
| `release_minimal/` (total) | **~50 GB** |
| `release_minimal/data/` | **~49 GB** |
| `release_minimal/models/` | **~641 MB** |
| `release_minimal/experiments/` | **~305 MB** |

---

## 8. Final verdict

### Ready to package: **YES** (with minor fixes recommended)

The release is structurally complete for recognition training, GUI demo, auth-matrix results, and physical-characterization experiments. Packaging can proceed after optional cleanup below.

### Unresolved / minor issues

1. **Bundled Fig2 filenames missing:** `figures/Fig2_length_optimization.png`, `.pdf`, `.svg` are not present. Equivalents exist as `Fig2_length_optimization_regen.*` and `Fig2_length_optimization_word.*`. **Fix:** copy or symlink `*_regen.*` → `Fig2_length_optimization.*` before distribution, or update README to reference `_regen` names only.

2. **Junk:** one `__pycache__` under length-optimization `paper_figures/` — remove before zip.

3. **`fiber_loss`:** empty `scripts/` and `outputs/` directories (intentional; documented).

4. **`power_loss.csv`:** not in release; Excel loss path works for Fig2.

5. **Smoke tests:** document `pip install -r requirements.txt` or use `recognition` conda env; bare system Python is insufficient for training imports.

6. **Size:** ~50 GB — use external storage / ZIP split / Git LFS; do not commit whole tree to normal Git.

### Passed (confirmed earlier + this run)

- Absolute path check on code/docs/metadata  
- Fig2 regen under `release_minimal/figures/` only  
- 240 videos, 2 domains, 15 fibers, 8 labels  
- 15 models, 8 classes  
- All four experiment trees with data + READMEs  
- Training summary and 15×15 auth matrix outputs  

---

*Generated by validation pass on `release_minimal/`.*
