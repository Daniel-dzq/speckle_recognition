#!/usr/bin/env python
"""
Generate all publication-quality figures from existing experimental data.

Usage:
    python scripts/make_paper_figures.py

Outputs go to  figures/  in PNG (600 dpi), PDF (vector), and SVG.
"""

import os, sys, json, csv, warnings
import numpy as np

# Ensure the scripts directory is on the path so plot_style can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

from plot_style import (
    apply_style, save_figure, add_panel_label,
    PALETTE, FIBER_COLORS, DOMAIN_COLORS, DOMAIN_LABELS,
    DEEP_BLUE, TEAL, MUTED_ORANGE, MUTED_RED, SOFT_PURPLE, SLATE_GRAY, GOLD,
    SINGLE_COL_W, DOUBLE_COL_W, GOLDEN_RATIO,
    FONT_SIZE_SMALL, FONT_SIZE_NORMAL, FONT_SIZE_LARGE, FONT_SIZE_TITLE,
    HEATMAP_CMAP,
)

apply_style()

FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

RESULTS_DIR     = os.path.join(ROOT_DIR, "results")
FIBER_AUTH_DIR  = os.path.join(RESULTS_DIR, "fiber_auth")
VIDEO_DIR       = os.path.join(ROOT_DIR, "videocapture")
FIBERS          = ["Fiber1", "Fiber2", "Fiber3", "Fiber4", "Fiber5"]
DOMAINS         = ["green_only", "red_green_fixed", "red_green_dynamic"]
DOMAIN_DIRS     = {"green_only": "Green", "red_green_fixed": "GreenAndRed",
                   "red_green_dynamic": "RedChange"}

created  = []
skipped  = []


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_auth_matrix():
    path = os.path.join(FIBER_AUTH_DIR, "auth_matrix.json")
    with open(path) as f:
        return json.load(f)


def load_training_log(fiber_name):
    path = os.path.join(RESULTS_DIR, fiber_name.lower().replace("fiber", "fiber"),
                        "training_log.csv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def load_test_predictions(fiber_name):
    path = os.path.join(RESULTS_DIR, fiber_name.lower().replace("fiber", "fiber"),
                        "test_predictions.csv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def extract_frame(video_path, frame_idx=None):
    """Extract a single frame from a video. Returns BGR numpy array or None."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx is None:
        frame_idx = total // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, total - 1))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE A — 5×5 Authentication Heatmap
# ═══════════════════════════════════════════════════════════════════════════

def fig_a_auth_heatmap():
    print("[A] Generating 5×5 authentication heatmap ...")
    data = load_auth_matrix()
    matrix = data["matrix"]

    arr = np.array([[matrix[mf][df] for df in FIBERS] for mf in FIBERS])

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W + 0.6, SINGLE_COL_W + 0.3))

    cmap = LinearSegmentedColormap.from_list(
        "auth_cmap",
        ["#F7F7F7", "#FDD49E", "#FDBB84", "#FC8D59", "#E34A33", "#B30000"],
        N=256,
    )
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=100, aspect="equal")

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([f.replace("Fiber", "F") for f in FIBERS], fontsize=FONT_SIZE_NORMAL)
    ax.set_yticklabels([f.replace("Fiber", "F") for f in FIBERS], fontsize=FONT_SIZE_NORMAL)
    ax.set_xlabel("Test data (fiber)", fontsize=FONT_SIZE_NORMAL)
    ax.set_ylabel("Enrolled model (fiber)", fontsize=FONT_SIZE_NORMAL)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_linewidth(0.6)
    ax.spines["right"].set_linewidth(0.6)

    ax.tick_params(top=True, bottom=True, left=True, right=True,
                   labeltop=True, labelbottom=False)

    for i in range(5):
        for j in range(5):
            val = arr[i, j]
            color = "white" if val > 55 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.1f}",
                    ha="center", va="center", fontsize=FONT_SIZE_NORMAL,
                    color=color, fontweight=weight)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06, shrink=0.85)
    cbar.set_label("Recognition accuracy (%)", fontsize=FONT_SIZE_SMALL)
    cbar.ax.tick_params(labelsize=FONT_SIZE_SMALL)

    avg_auth   = data["authorized_avg"]
    avg_unauth = data["unauthorized_avg"]
    gap        = data["auth_gap_pp"]
    ax.set_title(
        f"Authorized avg {avg_auth:.1f}%  |  Unauthorized avg {avg_unauth:.1f}%  |  Gap {gap:.1f} pp",
        fontsize=FONT_SIZE_SMALL, fontweight="normal", pad=22, color=SLATE_GRAY,
    )

    stem = os.path.join(FIGURES_DIR, "fig_auth_matrix")
    saved = save_figure(fig, stem)
    plt.close(fig)
    created.extend(saved)
    print(f"    Saved: {', '.join(os.path.basename(s) for s in saved)}")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE B — Same-fiber per-domain accuracy (grouped bars)
# ═══════════════════════════════════════════════════════════════════════════

def fig_b_per_domain_bars():
    print("[B] Generating same-fiber per-domain bar chart ...")
    data = load_auth_matrix()
    per_domain = data["same_fiber_per_domain"]

    n_fibers  = len(FIBERS)
    n_domains = len(DOMAINS)
    x = np.arange(n_fibers)
    bar_w = 0.22

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W * 0.75, DOUBLE_COL_W * 0.75 / GOLDEN_RATIO))

    for k, dom in enumerate(DOMAINS):
        vals = [per_domain[f][dom] for f in FIBERS]
        offset = (k - 1) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=DOMAIN_LABELS[dom],
                       color=DOMAIN_COLORS[dom], edgecolor="white", linewidth=0.4,
                       zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{val:.0f}", ha="center", va="bottom",
                    fontsize=5.5, color=DOMAIN_COLORS[dom])

    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("Fiber", "Fiber ") for f in FIBERS])
    ax.set_ylabel("Same-fiber accuracy (%)")
    ax.set_ylim(75, 105)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.legend(loc="lower left", ncol=1, fontsize=FONT_SIZE_SMALL)

    ax.axhline(y=100, color="#CCCCCC", linewidth=0.4, zorder=1)

    stem = os.path.join(FIGURES_DIR, "fig_same_fiber_per_domain")
    saved = save_figure(fig, stem)
    plt.close(fig)
    created.extend(saved)
    print(f"    Saved: {', '.join(os.path.basename(s) for s in saved)}")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE C — Authorized vs Unauthorized Gap
# ═══════════════════════════════════════════════════════════════════════════

def fig_c_auth_gap():
    print("[C] Generating authorized vs unauthorized gap figure ...")
    data = load_auth_matrix()
    matrix = data["matrix"]

    diag_vals     = [matrix[f][f] for f in FIBERS]
    off_diag_vals = []
    for mf in FIBERS:
        for df in FIBERS:
            if mf != df:
                off_diag_vals.append(matrix[mf][df])

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, DOUBLE_COL_W / 2.3),
                             gridspec_kw={"width_ratios": [1, 1.6]})

    # --- Panel (a): bar comparison ---
    ax = axes[0]
    avg_auth   = np.mean(diag_vals)
    avg_unauth = np.mean(off_diag_vals)
    chance     = 100.0 / 26

    bars = ax.bar([0, 1], [avg_auth, avg_unauth],
                  color=[DEEP_BLUE, MUTED_RED], width=0.55, edgecolor="white",
                  linewidth=0.5, zorder=3)

    ax.axhline(y=chance, color=SLATE_GRAY, linewidth=0.7, linestyle="--", zorder=2)
    ax.text(1.45, chance + 1.0, f"Chance\n({chance:.1f}%)",
            fontsize=5.5, color=SLATE_GRAY, va="bottom")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Authorized\n(same fiber)", "Unauthorized\n(cross fiber)"],
                        fontsize=FONT_SIZE_SMALL)
    ax.set_ylabel("Recognition accuracy (%)")
    ax.set_ylim(0, 110)

    ax.text(0, avg_auth + 1.5, f"{avg_auth:.1f}%", ha="center", va="bottom",
            fontsize=FONT_SIZE_NORMAL, fontweight="bold", color=DEEP_BLUE)
    ax.text(1, avg_unauth + 1.5, f"{avg_unauth:.1f}%", ha="center", va="bottom",
            fontsize=FONT_SIZE_NORMAL, fontweight="bold", color=MUTED_RED)

    gap = avg_auth - avg_unauth
    mid_y = (avg_auth + avg_unauth) / 2
    ax.annotate("", xy=(0.5, avg_auth - 1), xytext=(0.5, avg_unauth + 1),
                arrowprops=dict(arrowstyle="<->", color=SLATE_GRAY, lw=0.8))
    ax.text(0.5, mid_y, f"Gap\n{gap:.1f} pp", ha="center", va="center",
            fontsize=FONT_SIZE_SMALL, color=SLATE_GRAY,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none"))

    add_panel_label(ax, "(a)")

    # --- Panel (b): strip / swarm plot of individual values ---
    ax2 = axes[1]

    for i, val in enumerate(diag_vals):
        ax2.scatter(0, val, color=FIBER_COLORS[FIBERS[i]], s=40, zorder=4,
                    edgecolors="white", linewidths=0.4)

    np.random.seed(42)
    for val in off_diag_vals:
        jitter = np.random.uniform(-0.18, 0.18)
        ax2.scatter(1 + jitter, val, color=MUTED_RED, s=12, alpha=0.55, zorder=3,
                    edgecolors="none")

    ax2.scatter(0, avg_auth, marker="_", color="black", s=200, linewidths=1.5, zorder=5)
    ax2.scatter(1, avg_unauth, marker="_", color="black", s=200, linewidths=1.5, zorder=5)

    ax2.axhline(y=chance, color=SLATE_GRAY, linewidth=0.7, linestyle="--", zorder=1)
    ax2.text(1.55, chance + 0.5, f"Chance ({chance:.1f}%)",
             fontsize=5.5, color=SLATE_GRAY, va="bottom")

    legend_patches = [mpatches.Patch(color=FIBER_COLORS[f],
                      label=f.replace("Fiber", "F")) for f in FIBERS]
    ax2.legend(handles=legend_patches, loc="center right", fontsize=5.5,
               title="Fibers", title_fontsize=6, handlelength=1.0)

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Authorized\n(5 diagonal)", "Unauthorized\n(20 off-diagonal)"],
                        fontsize=FONT_SIZE_SMALL)
    ax2.set_ylabel("Recognition accuracy (%)")
    ax2.set_ylim(-3, 108)

    add_panel_label(ax2, "(b)")

    stem = os.path.join(FIGURES_DIR, "fig_auth_gap")
    saved = save_figure(fig, stem)
    plt.close(fig)
    created.extend(saved)
    print(f"    Saved: {', '.join(os.path.basename(s) for s in saved)}")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE D — Confidence Score Distributions
# ═══════════════════════════════════════════════════════════════════════════

def fig_d_score_distributions():
    print("[D] Generating confidence score distributions ...")

    auth_confs   = []
    unauth_confs = []

    for fiber in FIBERS:
        preds = load_test_predictions(fiber.lower().replace("fiber", "fiber"))
        if preds is None:
            continue
        for row in preds:
            conf = float(row["confidence"])
            correct = (row["true_label"] == row["pred_label"])
            auth_confs.append(conf)

    if not auth_confs:
        skipped.append(("fig_auth_scores", "No prediction files found"))
        print("    SKIPPED: no prediction files with confidence scores.")
        return

    auth_confs_arr = np.array(auth_confs)

    auth_json = load_auth_matrix()
    matrix = auth_json["matrix"]
    off_diag_accs = []
    for mf in FIBERS:
        for df in FIBERS:
            if mf != df:
                off_diag_accs.append(matrix[mf][df])
    off_diag_arr = np.array(off_diag_accs) / 100.0

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, DOUBLE_COL_W / 2.5))

    # --- Panel (a): authorized confidence distribution ---
    ax = axes[0]
    bins = np.linspace(0, 1, 30)
    ax.hist(auth_confs_arr, bins=bins, color=DEEP_BLUE, alpha=0.8, edgecolor="white",
            linewidth=0.3, zorder=3)
    median_c = np.median(auth_confs_arr)
    ax.axvline(median_c, color="black", linewidth=0.8, linestyle="--", zorder=4)
    ax.text(median_c + 0.02, ax.get_ylim()[1] * 0.9,
            f"Median = {median_c:.2f}", fontsize=FONT_SIZE_SMALL, va="top")
    ax.set_xlabel("Prediction confidence")
    ax.set_ylabel("Count (authorized samples)")
    add_panel_label(ax, "(a)")

    # --- Panel (b): off-diagonal accuracy distribution ---
    ax2 = axes[1]
    ax2.hist(off_diag_arr * 100, bins=np.linspace(0, 15, 20),
             color=MUTED_RED, alpha=0.8, edgecolor="white", linewidth=0.3, zorder=3)
    chance = 100.0 / 26
    ax2.axvline(chance, color=SLATE_GRAY, linewidth=0.8, linestyle="--", zorder=4)
    ymax_b = ax2.get_ylim()[1]
    ax2.text(chance + 0.3, ymax_b * 0.85,
             f"Chance = {chance:.1f}%", fontsize=FONT_SIZE_SMALL, va="top", color=SLATE_GRAY)
    ax2.set_xlabel("Cross-fiber accuracy (%)")
    ax2.set_ylabel("Count (unauthorized evaluations)")
    add_panel_label(ax2, "(b)")

    stem = os.path.join(FIGURES_DIR, "fig_auth_scores")
    saved = save_figure(fig, stem)
    plt.close(fig)
    created.extend(saved)
    print(f"    Saved: {', '.join(os.path.basename(s) for s in saved)}")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE E — Speckle Example Panels
# ═══════════════════════════════════════════════════════════════════════════

def fig_e_speckle_examples():
    print("[E] Generating speckle example panels ...")
    import cv2

    letters_across_fibers = ["A", "H", "S"]
    domains_for_same_fiber = ["Green", "GreenAndRed", "RedChange"]
    example_fiber = "Fiber1"

    n_rows = 2
    n_cols = max(len(FIBERS), len(domains_for_same_fiber))

    fig_w = DOUBLE_COL_W
    cell_size = fig_w / n_cols
    fig_h = cell_size * n_rows + 0.8

    # --- Row 1: same letter "A", across 5 fibers, Green domain ---
    # --- Row 2: same fiber (Fiber1), letter "A", across 3 domains ---

    row1_images = []
    row1_labels = []
    letter_row1 = "A"
    for fiber in FIBERS:
        vpath = os.path.join(VIDEO_DIR, "Green", fiber, f"{letter_row1}.avi")
        frame = extract_frame(vpath)
        if frame is not None:
            row1_images.append(frame)
            row1_labels.append(fiber.replace("Fiber", "F"))

    row2_images = []
    row2_labels = []
    domain_label_map = {"Green": "Green only", "GreenAndRed": "Green + Red", "RedChange": "Red sweep"}
    for dom_dir in domains_for_same_fiber:
        vpath = os.path.join(VIDEO_DIR, dom_dir, example_fiber, f"{letter_row1}.avi")
        frame = extract_frame(vpath)
        if frame is not None:
            row2_images.append(frame)
            row2_labels.append(domain_label_map.get(dom_dir, dom_dir))

    if not row1_images and not row2_images:
        skipped.append(("fig_speckle_examples", "Could not read any video frames"))
        print("    SKIPPED: could not read video frames.")
        return

    total_cols = max(len(row1_images), len(row2_images))
    fig, axes = plt.subplots(2, total_cols,
                             figsize=(fig_w, fig_w / total_cols * 2 + 0.7))
    if total_cols == 1:
        axes = axes.reshape(2, 1)

    crop = 200

    for col_idx in range(total_cols):
        # Row 1
        ax = axes[0, col_idx]
        if col_idx < len(row1_images):
            img = row1_images[col_idx]
            h, w = img.shape[:2]
            img_cropped = img[crop:h - crop, crop:w - crop]
            ax.imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
            ax.set_title(row1_labels[col_idx], fontsize=FONT_SIZE_SMALL, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.4)
            spine.set_color("#CCCCCC")

        # Row 2
        ax2 = axes[1, col_idx]
        if col_idx < len(row2_images):
            img = row2_images[col_idx]
            h, w = img.shape[:2]
            img_cropped = img[crop:h - crop, crop:w - crop]
            ax2.imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
            ax2.set_title(row2_labels[col_idx], fontsize=FONT_SIZE_SMALL, pad=3)
        else:
            ax2.set_visible(False)
        ax2.set_xticks([])
        ax2.set_yticks([])
        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.4)
            spine.set_color("#CCCCCC")

    axes[0, 0].set_ylabel(f'Letter "{letter_row1}"\nacross fibers',
                           fontsize=FONT_SIZE_SMALL, labelpad=8)
    axes[1, 0].set_ylabel(f'{example_fiber}\nacross domains',
                           fontsize=FONT_SIZE_SMALL, labelpad=8)

    add_panel_label(axes[0, 0], "(a)", x=-0.08, y=1.15)
    add_panel_label(axes[1, 0], "(b)", x=-0.08, y=1.15)

    stem = os.path.join(FIGURES_DIR, "fig_speckle_examples")
    saved = save_figure(fig, stem)
    plt.close(fig)
    created.extend(saved)
    print(f"    Saved: {', '.join(os.path.basename(s) for s in saved)}")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE F — NCC / HD Distribution
# ═══════════════════════════════════════════════════════════════════════════

def fig_f_ncc_hd():
    """
    NCC / HD distribution for PUF uniqueness evaluation.

    Intra-fiber (genuine): same fiber, same challenge letter, different
        temporal frames within the same video → measures repeatability.
    Inter-fiber (impostor): different fibers, same challenge letter, same
        temporal position → measures device uniqueness.
    """
    print("[F] Computing NCC / HD from representative speckle frames ...")
    import cv2

    target_size = (256, 256)
    letters_to_sample = list("ABCDEFGHIJKLMNOP")

    # Pre-extract multiple temporal frames per (fiber, letter)
    TEMPORAL_OFFSETS = [20, 50, 80, 110, 140]

    fiber_letter_frames = {}  # (fiber, letter) -> list of grayscale arrays
    for fiber in FIBERS:
        for letter in letters_to_sample:
            vpath = os.path.join(VIDEO_DIR, "Green", fiber, f"{letter}.avi")
            frames = []
            for t in TEMPORAL_OFFSETS:
                raw = extract_frame(vpath, frame_idx=t)
                if raw is not None:
                    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, target_size).astype(np.float64)
                    frames.append(gray)
            if frames:
                fiber_letter_frames[(fiber, letter)] = frames

    if not fiber_letter_frames:
        skipped.append(("fig_ncc_hd", "No frames decoded"))
        print("    SKIPPED: could not decode frames.")
        return

    def ncc(a, b):
        a_norm = a - a.mean()
        b_norm = b - b.mean()
        denom = np.sqrt(np.sum(a_norm**2) * np.sum(b_norm**2))
        if denom < 1e-12:
            return 0.0
        return float(np.sum(a_norm * b_norm) / denom)

    def hamming_distance(a, b):
        a_bin = (a > np.median(a)).astype(np.uint8).ravel()
        b_bin = (b > np.median(b)).astype(np.uint8).ravel()
        return float(np.mean(a_bin != b_bin))

    intra_ncc, inter_ncc = [], []
    intra_hd,  inter_hd  = [], []

    for letter in letters_to_sample:
        # --- Intra-fiber (genuine): same fiber, same letter, different times
        for fiber in FIBERS:
            frames = fiber_letter_frames.get((fiber, letter), [])
            for i in range(len(frames)):
                for j in range(i + 1, len(frames)):
                    intra_ncc.append(ncc(frames[i], frames[j]))
                    intra_hd.append(hamming_distance(frames[i], frames[j]))

        # --- Inter-fiber (impostor): different fibers, same letter, same time
        for i, f1 in enumerate(FIBERS):
            for j, f2 in enumerate(FIBERS):
                if j <= i:
                    continue
                frames1 = fiber_letter_frames.get((f1, letter), [])
                frames2 = fiber_letter_frames.get((f2, letter), [])
                n = min(len(frames1), len(frames2))
                for k in range(n):
                    inter_ncc.append(ncc(frames1[k], frames2[k]))
                    inter_hd.append(hamming_distance(frames1[k], frames2[k]))

    intra_ncc = np.array(intra_ncc)
    inter_ncc = np.array(inter_ncc)
    intra_hd  = np.array(intra_hd)
    inter_hd  = np.array(inter_hd)

    if len(intra_ncc) == 0 or len(inter_ncc) == 0:
        skipped.append(("fig_ncc_hd", "Insufficient pairs for comparison"))
        print("    SKIPPED: insufficient data for intra/inter comparison.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, DOUBLE_COL_W / 2.5))

    # Panel (a): NCC distributions
    ax = axes[0]
    all_ncc = np.concatenate([intra_ncc, inter_ncc])
    lo, hi = max(all_ncc.min() - 0.05, -0.1), min(all_ncc.max() + 0.05, 1.05)
    bins_ncc = np.linspace(lo, hi, 50)
    ax.hist(inter_ncc, bins=bins_ncc, alpha=0.7, color=MUTED_RED,
            edgecolor="white", linewidth=0.3, label="Inter-fiber (impostor)", density=True, zorder=3)
    ax.hist(intra_ncc, bins=bins_ncc, alpha=0.7, color=DEEP_BLUE,
            edgecolor="white", linewidth=0.3, label="Intra-fiber (genuine)", density=True, zorder=4)
    ax.set_xlabel("Normalized cross-correlation (NCC)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=FONT_SIZE_SMALL, loc="upper left")

    mu_i, s_i = inter_ncc.mean(), inter_ncc.std()
    mu_g, s_g = intra_ncc.mean(), intra_ncc.std()
    ax.text(0.97, 0.97,
            f"Inter: {mu_i:.3f} \u00b1 {s_i:.3f}\n"
            f"Intra: {mu_g:.3f} \u00b1 {s_g:.3f}",
            transform=ax.transAxes, fontsize=5.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", alpha=0.9))
    add_panel_label(ax, "(a)")

    # Panel (b): HD distributions
    ax2 = axes[1]
    all_hd = np.concatenate([intra_hd, inter_hd])
    lo_h, hi_h = max(all_hd.min() - 0.02, 0.0), min(all_hd.max() + 0.02, 0.65)
    bins_hd = np.linspace(lo_h, hi_h, 50)
    ax2.hist(inter_hd, bins=bins_hd, alpha=0.7, color=MUTED_RED,
             edgecolor="white", linewidth=0.3, label="Inter-fiber (impostor)", density=True, zorder=3)
    ax2.hist(intra_hd, bins=bins_hd, alpha=0.7, color=DEEP_BLUE,
             edgecolor="white", linewidth=0.3, label="Intra-fiber (genuine)", density=True, zorder=4)
    ax2.axvline(0.5, color=SLATE_GRAY, linewidth=0.7, linestyle="--", zorder=2)
    ax2.text(0.502, 0.92, "Ideal = 0.50",
             transform=ax2.get_xaxis_transform(),
             fontsize=5.5, color=SLATE_GRAY, va="top", rotation=90)
    ax2.set_xlabel("Hamming distance (HD)")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=FONT_SIZE_SMALL, loc="upper right")

    mu_ih, s_ih = inter_hd.mean(), inter_hd.std()
    mu_gh, s_gh = intra_hd.mean(), intra_hd.std()
    ax2.text(0.03, 0.97,
             f"Inter: {mu_ih:.3f} \u00b1 {s_ih:.3f}\n"
             f"Intra: {mu_gh:.3f} \u00b1 {s_gh:.3f}",
             transform=ax2.transAxes, fontsize=5.5, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", alpha=0.9))
    add_panel_label(ax2, "(b)")

    stem = os.path.join(FIGURES_DIR, "fig_ncc_hd")
    saved = save_figure(fig, stem)
    plt.close(fig)
    created.extend(saved)
    print(f"    Saved: {', '.join(os.path.basename(s) for s in saved)}")
    print(f"    Intra-fiber (genuine)  — {len(intra_ncc)} pairs")
    print(f"    Inter-fiber (impostor) — {len(inter_ncc)} pairs")
    print(f"    Stats  NCC — inter: {mu_i:.4f}\u00b1{s_i:.4f}, intra: {mu_g:.4f}\u00b1{s_g:.4f}")
    print(f"    Stats  HD  — inter: {mu_ih:.4f}\u00b1{s_ih:.4f}, intra: {mu_gh:.4f}\u00b1{s_gh:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE G — Training Curves Summary
# ═══════════════════════════════════════════════════════════════════════════

def fig_g_training_curves():
    print("[G] Generating training curve summary ...")

    all_logs = {}
    for fiber in FIBERS:
        log = load_training_log(fiber.lower().replace("fiber", "fiber"))
        if log is not None:
            all_logs[fiber] = log

    if not all_logs:
        skipped.append(("fig_training_curves", "No training logs found"))
        print("    SKIPPED: no training logs found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, DOUBLE_COL_W / 2.5))

    # Panel (a): training loss
    ax = axes[0]
    for fiber, log in all_logs.items():
        epochs = [row["epoch"] for row in log]
        losses = [row["train_loss"] for row in log]
        ax.plot(epochs, losses, color=FIBER_COLORS[fiber],
                label=fiber.replace("Fiber", "F"), linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.legend(fontsize=5.5, ncol=2)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    add_panel_label(ax, "(a)")

    # Panel (b): validation accuracy
    ax2 = axes[1]
    for fiber, log in all_logs.items():
        epochs  = [row["epoch"] for row in log]
        val_acc = [row["val_acc"] for row in log]
        ax2.plot(epochs, val_acc, color=FIBER_COLORS[fiber],
                 label=fiber.replace("Fiber", "F"), linewidth=1.0)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation accuracy (%)")
    ax2.set_ylim(50, 102)
    ax2.legend(fontsize=5.5, ncol=2)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    add_panel_label(ax2, "(b)")

    stem = os.path.join(FIGURES_DIR, "fig_training_curves")
    saved = save_figure(fig, stem)
    plt.close(fig)
    created.extend(saved)
    print(f"    Saved: {', '.join(os.path.basename(s) for s in saved)}")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE H (bonus) — Per-fiber test accuracy summary bar
# ═══════════════════════════════════════════════════════════════════════════

def fig_h_test_accuracy_summary():
    print("[H] Generating per-fiber test accuracy summary ...")

    data = load_auth_matrix()
    same_acc = data["same_fiber_accuracy"]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W + 0.5, SINGLE_COL_W / GOLDEN_RATIO))

    vals = [same_acc[f] for f in FIBERS]
    colors = [FIBER_COLORS[f] for f in FIBERS]
    x = np.arange(len(FIBERS))

    bars = ax.bar(x, vals, color=colors, width=0.55, edgecolor="white",
                  linewidth=0.5, zorder=3)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=FONT_SIZE_SMALL, fontweight="bold")

    avg = np.mean(vals)
    ax.axhline(avg, color=SLATE_GRAY, linewidth=0.8, linestyle="--", zorder=2)
    ax.text(4.4, avg + 0.5, f"Avg {avg:.1f}%",
            fontsize=FONT_SIZE_SMALL, color=SLATE_GRAY, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("Fiber", "Fiber ") for f in FIBERS])
    ax.set_ylabel("Same-fiber accuracy (%)")
    ax.set_ylim(85, 103)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))

    stem = os.path.join(FIGURES_DIR, "fig_test_accuracy_summary")
    saved = save_figure(fig, stem)
    plt.close(fig)
    created.extend(saved)
    print(f"    Saved: {', '.join(os.path.basename(s) for s in saved)}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Paper Figure Generation")
    print("=" * 60)
    print(f"  Output directory: {FIGURES_DIR}")
    print()

    warnings.filterwarnings("ignore", category=UserWarning)

    fig_a_auth_heatmap()
    fig_b_per_domain_bars()
    fig_c_auth_gap()
    fig_d_score_distributions()
    fig_e_speckle_examples()
    fig_f_ncc_hd()
    fig_g_training_curves()
    fig_h_test_accuracy_summary()

    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    unique_stems = sorted(set(os.path.splitext(os.path.basename(f))[0] for f in created))
    print(f"  Created {len(unique_stems)} figures ({len(created)} files):")
    for stem in unique_stems:
        exts = [os.path.splitext(f)[1] for f in created if os.path.basename(f).startswith(stem)]
        print(f"    {stem}  [{', '.join(exts)}]")

    if skipped:
        print(f"\n  Skipped {len(skipped)}:")
        for name, reason in skipped:
            print(f"    {name}: {reason}")
    else:
        print("\n  No figures skipped.")

    print()


if __name__ == "__main__":
    main()
