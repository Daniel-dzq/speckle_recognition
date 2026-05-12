# Final Word figure insertion list (策划书 alignment)

Use paths relative to repository root unless Word requires absolute paths. **Do not** insert `publication_fig03/04_length_optimization` as Fig. 4 for §3.2(1).

---

**Figure 3:**  
**Use file:** `figures/patent/optical_path_clean.png` (confirm with PI; alternates `figures/patent/optical_path_render.png`, `figures/patent/fig1_system.png`)  
**Section:** 3.1  
**Replace placeholder:** 图3 POF-PUF系统实验装置及原理示意图  
**Caption:** Schematic of the side-polished POF PUF readout: 532 nm DPSSL through beam shaping / SLM to the **side window**, 650 nm laser to the **fiber end-face**, and speckle imaging with a lens/objective onto the CCD. (Final caption to match journal tense and abbreviation policy.)

---

**Figure 4:**  
**Use file:** `figures_competition/fig4_length_optimization.png`  
**Vector for press:** `figures_competition/fig4_length_optimization.pdf` or `.svg`  
**Section:** 3.2(1)  
**Replace placeholder:** 图4 不同长度光纤…（损耗、类间/类内分离度、熵）  
**Caption:** Performance vs **total fiber length**: (a) red/green transmission loss; (b) intra- and inter-class ROI L₂ distances with **inter/intra distance ratio** on the right axis; (c) output speckle entropy. Vertical line: selected **9 cm** total length (green loss ≈ 28.39 dB, inter/intra ratio ≈ 1.5653, entropy ≈ 6.183 bit).

---

**Figure 7:**  
**Use file:** `figures_competition/fig7_dual_channel_characterization.png`  
**Section:** 3.2(3)  
**Replace placeholder:** 图7 双通道激励特性验证  
**Caption:** Dual-channel characterization: (a) intra-class correlation for red vs green temporal stability; (b) relative correlation before/after micro-bending (values per manuscript where raw traces omitted); (c) normalized radial mean intensity comparing representative side-launched green vs end-face red frames. State in caption which panels use measured vs summary statistics.

---

**Figure 8:**  
**Use file:** `figures_competition/fig8_common_mode_suppression.png`  
**Section:** immediately after the common-mode / η = G/R paragraph in §3.2  
**Replace placeholder:** 图8(a) 功率波动…；图8(b) 机械重装…  
**Caption:** (a) Coefficient of variation of raw green intensity versus **η = G/R**, showing ≈8.9× suppression of common-mode fluctuation; (b) schematic reinstall robustness: intra-class correlation improves with η relative to raw green intensity features (≈+28% per proposal constants — update if recomputed from `power_common_mode/`).



---

**Figure 12 (or equivalent §3.3 figure):**  
**Use file (pick one layout):** `figures_publication/publication_fig04_cross_fiber_auth.png` **or** `figures/fig_auth_matrix.png`  
**Section:** 3.3  
**Replace placeholder:** （识别性能 / 认证矩阵 / 跨纤拒绝）  
**Caption:** Recognition performance: same-fiber authentication accuracy and cross-fiber rejection (confusion matrix / summary panels from `results/fiber_auth/auth_matrix.json`).

---

**Optional supplementary / methods explorer (not Fig. 7):**  
`figures_publication/publication_fig05_dual_channel_robustness.png` — metrics grid from `figures/new_datasets_analysis/metrics_summary.json`.
