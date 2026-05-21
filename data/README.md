# Dataset (local copy, not tracked in Git)

This folder holds the final recognition dataset used for 15-fiber training.

## Layout

```
recognition_dataset/
  GreenAndRed/
    Fiber1/ … Fiber15/   (8 challenge videos per fiber)
  RedChange/
    Fiber1/ … Fiber15/   (8 challenge videos per fiber)
```

## Size

About **49 GB** (240 `.avi` files). Too large for normal Git hosting.

## Obtain the data

1. Copy from the full project `recognition_dataset/` tree, or
2. Use the same folder layout from your acquisition archive.

Training command (from release root):

```bash
python scripts/train_final_15fibers.py --data_root data/recognition_dataset ...
```

## Labels per video

Each fiber folder should contain eight classes:

`1`, `2`, `3`, `a`, `b`, `c`, `boy`, `girl`

If a typo file `gril.avi` appears, training maps it to `girl` (canonical 8-class map).
