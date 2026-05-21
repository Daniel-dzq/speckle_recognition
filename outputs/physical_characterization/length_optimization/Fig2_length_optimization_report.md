# Figure 2 — fiber length comparison (generation report)

**Figure 2.** Performance comparison of fibers with different lengths. (a) Red and green transmission loss versus total fiber length. (b) Intra-class distance, inter-class distance, and inter/intra ratio versus total fiber length (ratio on right axis). (c) Shannon entropy of output speckle versus total fiber length.

## Exact plotted values

| total_length_cm | green_loss_mean | green_loss_std | red_loss_mean | red_loss_std | intra_mean | intra_std | inter_mean | inter_std | inter_intra_ratio | entropy_mean | entropy_std |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8.0 | 24.654176019575356 | 0.054943364312514495 | 4.303473805557611 | 0.044306633798639646 | 20.6058 | 3.93348 | 23.985 | 1.37843 | 1.164 | 5.77315 | 0.35787 |
| 9.0 | 28.386668952041003 | 0.23693820179502983 | 4.68623581530537 | 0.14930017626551803 | 13.2083 | 2.53378 | 20.6754 | 1.66372 | 1.56533 | 6.18342 | 0.62397 |
| 11.0 | 31.652258168649276 | 0.27580036337097047 | 5.1021497914965135 | 0.11311075947629605 | 22.5466 | 4.07308 | 27.7392 | 1.67322 | 1.23031 | 5.40523 | 0.495342 |
| 13.0 | 32.62630753698555 | 0.3455514700131221 | 5.27290787043684 | 0.06958407404814829 | 20.9797 | 5.08513 | 23.0601 | 2.06252 | 1.09916 | 5.20146 | 0.362018 |
| 16.0 | 34.689286048405044 | 0.5585402136156128 | 5.7443522940383485 | 0.304176486707768 | 12.4117 | 0.916246 | 15.6427 | 0.532551 | 1.26032 | 5.61424 | 0.345295 |

### 8 cm transmission loss

- **Loaded successfully:** **yes**.

## Data sources

- Metrics table: `experiments/length_optimization/outputs/fig3/Fig3_length_optimization_data.csv`
- Fiber loss loader: `fiber_loss/*.xlsx (Input/Output columns → transmission_loss_db)`

### Fiber loss inputs read

- `experiments/fiber_loss/data/Fiber13cm.xlsx`
- `experiments/fiber_loss/data/fiber11cm.xlsx`
- `experiments/fiber_loss/data/fiber16cm.xlsx`
- `experiments/fiber_loss/data/fiber8cm.xlsx`
- `experiments/fiber_loss/data/fiber9cm.xlsx`

## Figure export settings

- Layout: **1 row × 3 columns**, **figsize (15, 4)**, **dpi 600**, white background.
- Word-oriented export: **`Fig2_length_optimization_word.{png,pdf,svg}`**, **3 rows × 1 column**, **figsize ≈ (6.8, 9.5)**, **dpi 600**, shared x-axis with tick labels on each row; **Total fiber length (cm)** label on bottom panel only.
- Panel titles (**Transmission loss**, **Distance metrics**, **Shannon entropy**) use **`ax.set_title`** above each subplot (**outside** the axes; **pad ≈ 10**, centered); letters **a–c** remain inside the upper-left.
- **No global title**, **no geometry panel**, **no propagation-length annotation on axes**.
- Optimal length marker: vertical dashed line at **9 cm**.

## Pipeline references

- Optimal JSON: `experiments/length_optimization/outputs/length_optimization_green/optimal_length.json`
- Config: `experiments/length_optimization/scripts/length_optimization_green.yaml`
- **`optimal_length.json`** recommends **`Fiber9cm`** with **`recommended_total_length_cm = 9.0`** (pipeline optimum consistent with **9 cm total fiber length**).

## Raw JPG cohort

- **`LengthOptimize/Green/`** raw captures were **not** re-ingested here; metrics come from the exported CSV.
