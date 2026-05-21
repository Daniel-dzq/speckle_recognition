# Experiment logic

## Step-by-step flow

1. **Challenge input** — A pattern (letter, digit, or avatar) is shown on the SLM from `challenge_inputs/` or typed manually in the GUI.
2. **Optical interaction** — Green and red channels illuminate the multimode PMMA fiber; the challenge pattern couples into the fiber input.
3. **Speckle response** — The camera records the output speckle field at the fiber output facet.
4. **Per-fiber decoding** — The authorized fiber model (Fiber1–Fiber15) classifies the clip into one of eight labels.
5. **Authentication** — If the normalized predicted label matches the normalized challenge label, the GUI reports **Access Granted**; otherwise **Access Denied** or **Unknown** (low confidence).
6. **Cross-fiber behavior** — A model tested on another fiber’s data should perform near chance (~12.5% for eight classes), supporting physical unclonability.

## Training vs live demo

- **Training** uses pre-recorded videos under `data/recognition_dataset/`.
- **Live demo** uses the same models on real-time camera frames (or a video file).

## Split strategy

`uniform_temporal`: clips from each video are shuffled with a fixed seed and split 70% / 15% / 15% (train / val / test). This samples the full timeline rather than using only the last segment for test.
