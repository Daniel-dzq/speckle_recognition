#!/usr/bin/env python3
"""
Figure 4 — Challenge inputs and representative speckle frames (Fiber1, real data).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _figure_io import save_figure_triplet, set_archive_old, write_csv_rows, write_report  # noqa: E402
from _paths import CHALLENGE_DIR, CHALLENGE_MANIFEST, FIGURES_PAPER, ROOT, SPECKLE_VIDEO_DIR  # noqa: E402
from _style import FONT_AXIS, FONT_TICK, apply_paper_style, panel_label  # noqa: E402

OUT_DIR = FIGURES_PAPER / "Fig4_challenge_speckle"
BASE = "Fig4_challenge_speckle"
FIG_W, FIG_H = 14.0, 4.2

LABEL_TO_VIDEO = {
    "A": "a.avi",
    "B": "b.avi",
    "C": "c.avi",
    "1": "1.avi",
    "2": "2.avi",
    "3": "3.avi",
    "boy": "boy.avi",
    "girl": "girl.avi",
}

ROI_FRAC = 0.55
DISPLAY_SIZE = 180
DISPLAY_GAMMA = 0.88


def load_challenge_order() -> List[Dict[str, str]]:
    if not CHALLENGE_MANIFEST.is_file():
        raise FileNotFoundError(CHALLENGE_MANIFEST)
    data = json.loads(CHALLENGE_MANIFEST.read_text(encoding="utf-8"))
    items = []
    for ch in data.get("challenges", []):
        label = ch["label"]
        rel = ch["image"]
        path = ROOT / rel if not Path(rel).is_absolute() else Path(rel)
        if not path.is_file():
            path = CHALLENGE_DIR / f"{label}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Missing challenge image for label {label}: {path}")
        items.append({"label": label, "image_path": str(path)})
    return items


def read_challenge_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def read_speckle_middle_frame(video_path: Path) -> Tuple[np.ndarray, str]:
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = max(0, n // 2) if n > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame {idx} from {video_path}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if frame.ndim == 2 or (frame.shape[2] == 1):
        mode = "grayscale_video"
    else:
        b, g, r = cv2.split(frame)
        if np.allclose(b, g) and np.allclose(g, r):
            mode = "grayscale_video"
        else:
            mode = "rgb_video"
    return rgb, mode


def center_roi_rgb(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    ch = max(8, int(h * ROI_FRAC))
    cw = max(8, int(w * ROI_FRAC))
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    crop = img[y0 : y0 + ch, x0 : x0 + cw]
    return cv2.resize(crop, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_AREA)


def normalize_rgb_display(crops: List[np.ndarray]) -> Tuple[List[np.ndarray], float, float]:
    """Shared percentile scaling and identical gamma for all speckle frames."""
    stack = np.stack([c.astype(np.float64) for c in crops], axis=0)
    vmin = float(np.percentile(stack, 1))
    vmax = float(np.percentile(stack, 99.5))
    if vmax <= vmin:
        vmax = vmin + 1.0
    out: List[np.ndarray] = []
    for c in crops:
        x = np.clip((c.astype(np.float64) - vmin) / (vmax - vmin), 0.0, 1.0)
        x = np.power(x, DISPLAY_GAMMA)
        out.append(x)
    return out, vmin, vmax


def center_roi_gray(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = g.shape
    ch = max(8, int(h * ROI_FRAC))
    cw = max(8, int(w * ROI_FRAC))
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    crop = g[y0 : y0 + ch, x0 : x0 + cw]
    return cv2.resize(crop, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_AREA)


def feature_vector(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float64).ravel()
    g = g - g.mean()
    norm = np.linalg.norm(g)
    if norm > 0:
        g = g / norm
    return g


def ncc_matrix(vectors: List[np.ndarray]) -> np.ndarray:
    n = len(vectors)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            mat[i, j] = float(np.dot(vectors[i], vectors[j]))
    return mat


def build_figure(
    challenges: List[Dict[str, str]],
    challenge_imgs: List[np.ndarray],
    speckle_display: List[np.ndarray],
    display_mode: str,
) -> plt.Figure:
    n = len(challenges)
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    gs = GridSpec(
        2, n, figure=fig,
        height_ratios=[1.0, 1.0],
        hspace=0.10,
        wspace=0.04,
        left=0.04,
        right=0.995,
        top=0.86,
        bottom=0.04,
    )

    for i, ch in enumerate(challenges):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(challenge_imgs[i])
        ax.set_title(ch["label"], fontsize=FONT_TICK + 1, pad=3, color="#111111")
        ax.axis("off")
        if i == 0:
            panel_label(ax, "a")
            ax.text(0.0, 1.22, "Challenge patterns", transform=ax.transAxes,
                    fontsize=FONT_AXIS, fontweight="bold", ha="left", va="bottom")

    for i in range(n):
        ax = fig.add_subplot(gs[1, i])
        img = speckle_display[i]
        if display_mode == "grayscale_video" and img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(img)
        ax.axis("off")
        if i == 0:
            panel_label(ax, "b")
            ax.text(0.0, 1.22, "Real speckle responses", transform=ax.transAxes,
                    fontsize=FONT_AXIS, fontweight="bold", ha="left", va="bottom")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 4 (challenge vs speckle examples).")
    parser.add_argument("--archive-old", action="store_true", help="Archive existing outputs before overwrite.")
    args = parser.parse_args()
    set_archive_old(args.archive_old)

    apply_paper_style()
    mpl.rcParams["figure.constrained_layout.use"] = False

    challenges = load_challenge_order()
    challenge_imgs: List[np.ndarray] = []
    speckle_rgb: List[np.ndarray] = []
    gray_crops: List[np.ndarray] = []
    video_modes: List[str] = []
    rows: List[Dict[str, object]] = []

    for ch in challenges:
        label = ch["label"]
        img_path = Path(ch["image_path"])
        vid_name = LABEL_TO_VIDEO.get(label)
        if vid_name is None:
            raise KeyError(f"No video mapping for challenge label: {label}")
        video_path = SPECKLE_VIDEO_DIR / vid_name
        challenge_imgs.append(read_challenge_rgb(img_path))
        frame, vmode = read_speckle_middle_frame(video_path)
        speckle_rgb.append(frame)
        video_modes.append(vmode)
        gray_crops.append(center_roi_gray(frame))
        rows.append({
            "record_type": "pairing",
            "challenge_label": label,
            "challenge_image": str(img_path.relative_to(ROOT)),
            "speckle_video": str(video_path.relative_to(ROOT)),
            "frame_index": "middle",
            "video_color_mode": vmode,
        })

    display_mode = "rgb_video" if any(m == "rgb_video" for m in video_modes) else "grayscale_video"
    roi_crops = [center_roi_rgb(f) for f in speckle_rgb]
    speckle_display, vmin, vmax = normalize_rgb_display(roi_crops)
    rows.append({
        "record_type": "display_normalization",
        "display_mode": display_mode,
        "vmin_percentile_1": vmin,
        "vmax_percentile_99_5": vmax,
        "gamma": DISPLAY_GAMMA,
        "roi_fraction": ROI_FRAC,
    })

    vectors = [feature_vector(g) for g in gray_crops]
    sim = ncc_matrix(vectors)
    labels = [c["label"] for c in challenges]
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            rows.append({
                "record_type": "speckle_ncc",
                "challenge_i": li,
                "challenge_j": lj,
                "ncc": float(sim[i, j]),
            })
    rows.append({
        "record_type": "ncc_panel_decision",
        "included_in_figure": False,
        "reason": "NCC heatmap omitted from main figure; values in CSV/report only.",
    })

    fig = build_figure(challenges, challenge_imgs, speckle_display, display_mode)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / BASE
    paths = save_figure_triplet(fig, stem)
    plt.close(fig)

    csv_path = OUT_DIR / f"{BASE}_data_summary.csv"
    write_csv_rows(
        csv_path,
        [
            "record_type", "challenge_label", "challenge_image", "speckle_video", "frame_index",
            "video_color_mode", "challenge_i", "challenge_j", "ncc", "included_in_figure", "reason",
            "display_mode", "vmin_percentile_1", "vmax_percentile_99_5", "gamma", "roi_fraction",
        ],
        rows,
    )

    report = f"""# {BASE} report

## Data sources
- `{CHALLENGE_MANIFEST.relative_to(ROOT)}`
- `{SPECKLE_VIDEO_DIR.relative_to(ROOT)}/*.avi` (middle frame, Fiber1)

## Display
- Panel (a): challenge PNGs (original RGB).
- Panel (b): middle video frames, central ROI ({ROI_FRAC:.0%}), **{display_mode}** display with shared 1–99.5% scaling and gamma={DISPLAY_GAMMA} applied identically to every frame.
- Fiber1, GreenAndRed, middle frame — details for caption only (not overlaid on images).

## NCC
Pairwise NCC computed from grayscale ROI (report/CSV only). **Not shown** in the main figure.

## Figure role
Main-text challenge–response gallery (2 panels).

## Outputs
""" + "\n".join(f"- `{p.relative_to(ROOT)}`" for p in paths + [csv_path]) + """

## Caption draft
Eight SLM challenge patterns (a) and aligned Fiber1 speckle responses (b) illustrate class-dependent dual-channel optical signatures from real recordings.
"""
    write_report(OUT_DIR / f"{BASE}_report.md", report)
    print(f"Wrote {stem}.png/pdf/svg ({len(challenges)} classes, display={display_mode}, NCC panel=no)")


if __name__ == "__main__":
    main()
