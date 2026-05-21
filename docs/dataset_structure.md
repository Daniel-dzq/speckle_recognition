# Dataset structure

## Domains

| Folder | Role |
|--------|------|
| `GreenAndRed/` | Fixed red/green illumination condition |
| `RedChange/` | Dynamic red-channel condition |

## Fibers

`Fiber1` through `Fiber15` — one independent physical fiber per folder.

## Videos per fiber

Each fiber folder contains **eight** `.avi` files (one per class):

- Digits: `1.avi`, `2.avi`, `3.avi`
- Letters: `a.avi`, `b.avi`, `c.avi`
- Avatars: `boy.avi`, `girl.avi`

Total: **240 videos** (15 fibers × 2 domains × 8 classes).

## Label map

Canonical training labels (8 classes):

`1`, `2`, `3`, `a`, `b`, `c`, `boy`, `girl`

See `models/final_15fibers/label_map.json`.

## Typo alias

If `gril.avi` exists, it is mapped to `girl`. The current release dataset copy has no `gril.avi` file.
