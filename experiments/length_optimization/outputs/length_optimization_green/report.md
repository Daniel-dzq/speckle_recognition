# Experiment 3.2 — Fiber Length Optimisation

Comprehensive analysis of candidate **total fiber lengths** (paper Section 3.2 / manuscript Figure 4): transmission loss, intra/inter-class separability, and pixel entropy. **Horizontal axis:** total fiber length in cm (8, 9, 11, 13, 16 for Fiber8cm … Fiber16cm). Optional ``green_prop_mm`` in YAML is **auxiliary side-polish geometry only**, not the length-comparison axis and not “the optimal green propagation length”. **Naming:** *Fiber9cm* means the **9 cm total length** sample batch. The side-polished coupling region sits at a fixed position in the fixture; do not confuse that layout with the total-length label. Error bars: std across 5 fibers per length group.

## Per-length summary (aggregated across 5 fibers)

| Length group | Total length (cm) | Aux. green path (cm) | # Fibers | Entropy (bits) | ±std | Intra dist | ±std | Inter dist | ±std | Inter/intra dist. ratio | Green loss (dB) | ±std | Red loss (dB) | ±std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fiber8cm | 8.0 | 1.0 | 5 | 5.773 | 0.358 | 20.606 | 3.933 | 23.985 | 1.378 | 1.1640 | — | 0.00 | — | 0.00 |
| Fiber9cm | 9.0 | 2.0 | 5 | 6.183 | 0.624 | 13.208 | 2.534 | 20.675 | 1.664 | 1.5653 | 28.39 | 0.24 | 4.69 | 0.68 |
| Fiber11cm | 11.0 | 4.0 | 5 | 5.405 | 0.495 | 22.547 | 4.073 | 27.739 | 1.673 | 1.2303 | 31.65 | 0.28 | 4.97 | 0.31 |
| Fiber13cm | 13.0 | 6.0 | 5 | 5.201 | 0.362 | 20.980 | 5.085 | 23.060 | 2.063 | 1.0992 | 32.63 | 0.35 | 4.69 | 0.68 |
| Fiber16cm | 16.0 | 9.0 | 5 | 5.614 | 0.345 | 12.412 | 0.916 | 15.643 | 0.533 | 1.2603 | 34.69 | 0.56 | 5.56 | 0.61 |

## Per-fiber detail

| Length group | Fiber | Total length (cm) | Aux. green path (cm) | Entropy (bits) | Intra dist | Green loss (dB) | Red loss (dB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fiber11cm | Fiber1 | 11.0 | 4.0 | 5.212 | 25.788 | 31.64 | 5.26 |
| Fiber11cm | Fiber2 | 11.0 | 4.0 | 6.326 | 17.252 | 31.64 | 5.18 |
| Fiber11cm | Fiber3 | 11.0 | 4.0 | 5.347 | 26.657 | 32.10 | 4.40 |
| Fiber11cm | Fiber4 | 11.0 | 4.0 | 5.309 | 25.082 | 31.23 | 4.93 |
| Fiber11cm | Fiber5 | 11.0 | 4.0 | 4.832 | 17.954 | 31.64 | 5.09 |
| Fiber13cm | Fiber1 | 13.0 | 6.0 | 5.638 | 27.407 | 32.61 | 5.15 |
| Fiber13cm | Fiber2 | 13.0 | 6.0 | 5.622 | 25.375 | 32.61 | 4.59 |
| Fiber13cm | Fiber3 | 13.0 | 6.0 | 4.972 | 20.597 | 33.19 | 4.30 |
| Fiber13cm | Fiber4 | 13.0 | 6.0 | 4.749 | 13.111 | 32.10 | 3.71 |
| Fiber13cm | Fiber5 | 13.0 | 6.0 | 5.026 | 18.408 | 32.61 | 5.70 |
| Fiber16cm | Fiber1 | 16.0 | 9.0 | 5.456 | 11.862 | 34.65 | 6.23 |
| Fiber16cm | Fiber2 | 16.0 | 9.0 | 5.696 | 12.390 | 34.65 | 4.47 |
| Fiber16cm | Fiber3 | 16.0 | 9.0 | 5.318 | 12.551 | 34.65 | 5.63 |
| Fiber16cm | Fiber4 | 16.0 | 9.0 | 5.350 | 11.253 | 35.62 | 6.02 |
| Fiber16cm | Fiber5 | 16.0 | 9.0 | 6.252 | 14.003 | 33.86 | 5.44 |
| Fiber8cm | Fiber1 | 8.0 | 1.0 | 5.372 | 22.447 | — | — |
| Fiber8cm | Fiber2 | 8.0 | 1.0 | 5.366 | 25.931 | — | — |
| Fiber8cm | Fiber3 | 8.0 | 1.0 | 6.259 | 18.297 | — | — |
| Fiber8cm | Fiber4 | 8.0 | 1.0 | 5.824 | 14.403 | — | — |
| Fiber8cm | Fiber5 | 8.0 | 1.0 | 6.045 | 21.951 | — | — |
| Fiber9cm | Fiber1 | 9.0 | 2.0 | 6.714 | 12.389 | 28.03 | 5.15 |
| Fiber9cm | Fiber2 | 9.0 | 2.0 | 5.764 | 18.080 | 28.63 | 4.59 |
| Fiber9cm | Fiber3 | 9.0 | 2.0 | 6.739 | 12.881 | 28.63 | 4.30 |
| Fiber9cm | Fiber4 | 9.0 | 2.0 | 6.542 | 10.776 | 28.42 | 3.71 |
| Fiber9cm | Fiber5 | 9.0 | 2.0 | 5.158 | 11.915 | 28.22 | 5.70 |

## Three-criterion recommendation (Section 3.2)

The paper selects the batch whose **total fiber length** simultaneously satisfies: (1) green-channel loss below the acceptable threshold [**loss gate**]; (2) **highest inter/intra distance ratio** — best balance of uniqueness vs. stability; (3) pixel entropy near saturation — sufficient output randomness.

- **Recommended batch (length group)**: Fiber9cm
- **Total fiber length (cm, nominal)**: 9.0
- **Auxiliary geometry (green path, mm)**: 20
- **Inter/intra distance ratio**: 1.5653
- **Pixel entropy (bits)**: 6.183
- **Green loss (dB)**: 28.39
- **Red loss (dB)**: 4.69
- **Loss gate threshold (dB)**: 40.000
- **Reason**: Section 3.2 three-criterion selection: (1) green loss ≤ 40.0 dB [gate]; (2) highest inter/intra distance ratio = 1.5653 [primary]; (3) pixel entropy = 6.183 bits [context]. Recommended batch: Fiber9cm (total fiber length 9.0 cm — not green-path length).

### Criterion-by-criterion scores

| Length group | Inter/intra dist. ratio | Ratio score (norm) | Entropy (bits) | Entropy score (norm) | Green loss (dB) | Passes loss gate |
| --- | --- | --- | --- | --- | --- | --- |
| Fiber8cm | 1.1640 | 0.7436 | 5.773 | 0.9336 | — | ✓ |
| Fiber9cm | 1.5653 | 1.0000 | 6.183 | 1.0000 | 28.39 | ✓ |
| Fiber11cm | 1.2303 | 0.7860 | 5.405 | 0.8741 | 31.65 | ✓ |
| Fiber13cm | 1.0992 | 0.7022 | 5.202 | 0.8412 | 32.63 | ✓ |
| Fiber16cm | 1.2603 | 0.8051 | 5.614 | 0.9080 | 34.69 | ✓ |

## Methodology notes

- **Pixel entropy**: Shannon entropy of raw 8-bit grayscale pixel values (256-bin histogram, no normalisation). Computed on the per-fiber mean image (average of all 10 repeats) within the configured 400×400 ROI. Matches paper Section 3.2 definition exactly.
- **Intra-class distance**: mean pairwise Euclidean L2 distance among the 10 repeat captures of the same fiber (measures capture-to-capture stability). Error bars = std across 5 fibers.
- **Inter-class distance**: mean pairwise Euclidean L2 distance between captures of different fibers in the same length group (measures uniqueness). Error bars = std of per-fiber mean inter distances across 5 fibers.
- **Inter/intra distance ratio**: inter-class distance divided by intra-class distance (higher ⇒ better separability under this definition).
- **Distance features**: centre-cropped to 400×400 then resized to 112×112, per-image min-max normalised before L2 comparison.
- **Total fiber length**: nominal specimen length in centimetres (**Fiber9cm ⇒ 9 cm total**). The naming **Fiber9cm** refers to the total fiber length; the side-polished region is located at a fixed geometry in the setup and should not be confused with the total-length label.
- **Auxiliary `green_prop_mm`**: recorded side-polish geometry for each batch only; **not** used as the principal horizontal axis for Figure 4.
- **Loss data**: input power unified at 1460 µW for green channel across all length groups, enabling direct cross-group comparison. High absolute loss (~28–35 dB) is inherent to side-coupling geometry and is consistent across groups, so the inter/intra distance ratio remains the primary length-selection criterion.

## Generated figures

- `figures/loss_vs_length.*` — (a) Transmission loss vs **total fiber length (cm)**
- `figures/separability_vs_length.*` — (b) Intra/inter distance + **inter/intra distance ratio**
- `figures/entropy_vs_length.*` — (c) Pixel entropy vs **total fiber length (cm)**
- `figures/speckle_montage.*` — (d) Representative speckle images per group
- `figures/figure2_combined.*` — Paper Figure 2: all three panels combined
