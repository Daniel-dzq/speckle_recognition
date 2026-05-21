# Fig4_challenge_speckle report

## Data sources
- `challenge_inputs/manifest.json`
- `data/recognition_dataset/GreenAndRed/Fiber1/*.avi` (middle frame, Fiber1)

## Display
- Panel (a): challenge PNGs (original RGB).
- Panel (b): middle video frames, central ROI (55%), **rgb_video** display with shared 1–99.5% scaling and gamma=0.88 applied identically to every frame.
- Fiber1, GreenAndRed, middle frame — details for caption only (not overlaid on images).

## NCC
Pairwise NCC computed from grayscale ROI (report/CSV only). **Not shown** in the main figure.

## Figure role
Main-text challenge–response gallery (2 panels).

## Outputs
- `figures/paper/Fig4_challenge_speckle/Fig4_challenge_speckle.png`
- `figures/paper/Fig4_challenge_speckle/Fig4_challenge_speckle.pdf`
- `figures/paper/Fig4_challenge_speckle/Fig4_challenge_speckle.svg`
- `figures/paper/Fig4_challenge_speckle/Fig4_challenge_speckle_data_summary.csv`

## Caption draft
Eight SLM challenge patterns (a) and aligned Fiber1 speckle responses (b) illustrate class-dependent dual-channel optical signatures from real recordings.
