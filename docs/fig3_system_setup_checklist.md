# Fig. 3 system setup — checklist only (no artwork changes in this task)

Candidate files (also mirrored under `paper_assets/png/` per `paper_assets/INDEX.csv`).

| File | Verdict |
|------|---------|
| `figures/patent/optical_path_render.png` | **use_as_final** for a PR-style English figure (labels present; dual path geometry correct). |
| `figures/patent/optical_path_clean.png` | **use_after_minor_label_fix** — same 3D geometry as render, but **no component labels**; suitable if the journal supplies captions only. |
| `figures/patent/fig1_system.png` | **use_after_minor_label_fix** — block diagram is **topologically correct** (green side illumination, red end-face, camera at far end, SLM on green arm only) but text is **Chinese** and green wavelength is **520 nm** (manuscript may cite **532 nm**). |
| `figures/patent/fig3_cmr.png` | **reject_and_redraw** **for §3.1 system Fig. 3** — this is a **common-mode / η concept sketch** (two time-series panels), not an optical bench layout; also Chinese-only. Do **not** substitute for `fig3_system_setup`. |
| `figures/patent/fig2_modes.png` / `fig4_flowchart.png` | Not full system schematics — **do not use** as Fig. 3. |

---

## Checklist (physics)

| Item | `optical_path_render` | `optical_path_clean` | `fig1_system` |
|------|------------------------|----------------------|---------------|
| Green **side-polished** (not end-face) launch | Yes | Yes | Yes |
| Red **end-face** launch | Yes | Yes | Yes |
| **SLM** only in green path | Yes | Yes | Yes |
| Output / speckle coupling to camera at **opposite** end from red input axis | Yes | Yes | Yes |
| Camera at **output** end (CMOS ≈ CCD role) | Yes | Yes | Yes |
| POF / fiber / side window / camera called out | Yes (wording varies) | No text | Yes (Chinese) |
| Fiber holder / computer in panel | Optional / not shown | Not shown | Computer block shown in `fig1_system`; holder not explicit |

---

## Mis-label risk

- Patent art uses **520 nm** green label; your narrative may require **532 nm** — harmonize in caption or replace label in a **future** vector edit (outside this task).

---

## Recommended Word insertion path (Fig. 3)

- **Primary:** `figures/patent/optical_path_render.png` (mirror: `paper_assets/png/figures_patent_optical_path_render.png`).
