# Top-journal figure style guide (Photonics Research / Light / Nature Photonics family)

Target: **clean physical-science figure aesthetics** — legibility at single-column width, minimal ink, color used for **meaning** not decoration.

---

## 1. Global rules

- **Background:** pure white (`facecolor` / export `white`); no grey canvas.
- **Borders:** remove **top** and **right** spines; avoid second “frame” boxes around panels.
- **Typography:** one family (Arial or Helvetica common); **axis labels 8–10 pt**, **tick labels 7–9 pt**; consistent weight.
- **Panel labels:** **(a) (b) (c)** **bold**, upper-left inside panel; same offset for every panel.
- **Export:** **300 dpi PNG** for Word; keep **PDF or SVG** as master for journal production.
- **Palette discipline:** **≤ 3–5** distinguishable colors total per multi-panel figure; **no** default saturated matplotlib rainbow.
- **Units:** every axis states **physical units**; dimensionless quantities explicitly marked (`Norm.`, `a.u.` only if unavoidable).
- **Caption placement:** **do not** paste long figure captions into the graphic — short on-figure annotations only (e.g. `×8.9`).

---

## 2. Color semantics (lock for this project)

| Quantity / channel | Suggested color | Avoid |
|--------------------|-----------------|-------|
| **Red (650 nm) channel** | deep red `#c0392b` or `#b2182b` | pink/orange masquerading as red |
| **Green (532 nm) channel** | muted green `#1b783f` or `#238b45` | neon lime |
| **η = G/R** (ratio feature) | **blue** `#2166ac` or **purple** `#6a3d9a` | reusing green |
| **Raw green intensity** (baseline) | desaturated green `#8fbc8f` or neutral grey-green | identical to η |
| **Inter-class / impostor** | warm terracotta `#d95f02` | |
| **Intra-class / genuine** | cool blue `#4575b4` | swapping warm/cool vs text |

---

## 3. Fig. 4 — length optimization (triple panel only)

- **Layout:** **1×3** horizontal **or fixed 2×2 with only three active axes** — **no** speckle montage in this figure; **no** standalone fourth “ratio-only” panel.
- **Panel (b):** grouped bars for intra vs inter **+** **right-axis line** for **inter/intra distance ratio**; verify ratio tick span ~**1.1–1.7** so **1.5653** cannot be read as **156.53**.
- **Optimum mark:** subtle **vertical grey line** or small marker at **9 cm** — avoid full-height saturated fill.
- **X-axis label:** **exactly** `Total fiber length (cm)`.
- **Forbidden inside Fig. 4:** authentication matrix, cross-fiber accuracy, confidence histograms, wording **“green propagation 9 cm is optimal”** (must stress **total** fiber length).
- **Footnote:** one line max on-figure; full paragraph → main caption.

---

## 4. Fig. 7 — dual-channel characterization

- **Panels (a)(b):** **shared logical y-range** when both report correlation-like metrics (0–1) for immediate comparison.
- **Panel (c) speckles (if shown):** identical crop size, **shared intensity normalization**, **scale bar** or caption note `Normalized intensity`.
- **Radial profiles:** **red curve vs green curve**; green should appear **wider / smoother** only if data support it — label data source (experiment vs proxy frame).
- **Integrity:** **no synthetic speckle** labeled as measurement; if using a single video frame, state “representative frame”.
- **Forbidden:** confusion matrix, cross-fiber bars, training curves.

---

## 5. Fig. 8 — common-mode suppression

- **Panel (a):** two bars — **38.2%** vs **4.3%** — large enough tick spacing; annotate **≈ 8.9× reduction** (or **≈ 1/9**) **between** bars or bracket.
- **Panel (b):** paired bars for reinstall narrative; **+28%** gain with **arrow** or parenthetical call-out vs raw baseline.
- **Notation:** Unicode **η** or plain text `eta = G/R` — **no** stray LaTeX fragments like `$eta$` in exported SVG text.
- **Forbidden:** authentication matrix, cross-fiber accuracy panels.

---

## 6. Figure package for each final release

For every production figure:

1. **PNG** — Word / internal review.  
2. **SVG or PDF** — publisher-ready vector.  
3. **CSV (or JSON)** — underlying numeric series archived beside figure (e.g. `results/length_optimization_green/tables/per_length_summary.csv`).  
4. **Manifest row** — e.g. extend `figures_competition/manifest.csv` or journal `INDEX.csv` with `source_script`, `source_data`, `mtime`.

---

## 7. Peer-review defensive checks (numbers)

- Fig. 4 optimal row must match **`optimal_length.json`**: **28.39 dB**, **1.5653**, **6.183 bit**, **9.0 cm total length**.  
- Scan supplementary exports for legacy tokens: **1.5779**, **6.8369**, **156.53**, **intra/inter** wording — treat as **errors** if appearing in Fig. 4 context.
