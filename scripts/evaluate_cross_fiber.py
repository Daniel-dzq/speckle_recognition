#!/usr/bin/env python3
"""
Cross-fiber evaluation matrix.

For each trained fiber model, evaluate it on the test set of EVERY fiber.
This produces the key PUF uniqueness evidence:
  - Diagonal entries should be high (correct fiber model on its own data)
  - Off-diagonal entries should be low (wrong fiber model fails on foreign data)

Usage:
    python scripts/evaluate_cross_fiber.py
    python scripts/evaluate_cross_fiber.py --clip_len 16 --stride 8
    python scripts/evaluate_cross_fiber.py --checkpoints_dir checkpoints --output_dir results/cross_fiber

Output:
    results/cross_fiber/cross_fiber_accuracy.csv
    results/cross_fiber/cross_fiber_heatmap.png
    results/cross_fiber/summary.md
"""

import os
import sys
import csv
import json
import glob
import argparse
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import prepare_data, SpeckleClipDataset
from models import get_model
from train_eval import evaluate


# ============================================================================
#  Helpers
# ============================================================================

VIDEO_CAPTURE_DIR = os.path.join(ROOT, "video_capture")
CHECKPOINTS_DIR   = os.path.join(ROOT, "checkpoints")
CROSS_RESULTS_DIR = os.path.join(ROOT, "results", "cross_fiber")


def fiber_key(name: str) -> str:
    return name.lower().replace(" ", "_")


def discover_fibers(video_dir: str):
    """Return list of fiber folder names that have .avi files."""
    fibers = []
    if not os.path.isdir(video_dir):
        return fibers
    for d in sorted(os.listdir(video_dir)):
        dpath = os.path.join(video_dir, d)
        if not os.path.isdir(dpath):
            continue
        avis = glob.glob(os.path.join(dpath, "*.avi")) + glob.glob(os.path.join(dpath, "*.AVI"))
        if avis:
            fibers.append(d)
    return fibers


def discover_checkpoints(ckpt_dir: str):
    """Return dict {fiber_key: checkpoint_path} for all *_best.pth files."""
    result = {}
    if not os.path.isdir(ckpt_dir):
        return result
    for f in sorted(os.listdir(ckpt_dir)):
        if f.endswith("_best.pth"):
            key = f[:-len("_best.pth")]
            result[key] = os.path.join(ckpt_dir, f)
    return result


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    """Load model, return (model, clip_len, img_size, class_names, fiber_name)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_type  = ckpt.get("model_type",  "cnn_pool")
    num_classes = ckpt.get("num_classes", 26)
    class_names = ckpt.get("class_names", [chr(65 + i) for i in range(26)])
    clip_len    = ckpt.get("clip_len",    16)
    img_size    = ckpt.get("img_size",    224)
    fiber_name  = ckpt.get("fiber_name",  os.path.basename(ckpt_path))

    model = get_model(model_type, num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    return model, clip_len, img_size, class_names, fiber_name


def build_test_loader(fiber_dir: str, clip_len: int, img_size: int, batch_size: int):
    """Load a fiber's test set. Returns (test_loader, test_clips, class_names)."""
    all_frames, _, _, test_clips, class_names = prepare_data(
        fiber_dir, clip_len=clip_len, stride=clip_len, img_size=img_size
    )
    if len(test_clips) == 0:
        return None, [], class_names

    test_ds = SpeckleClipDataset(test_clips, all_frames, clip_len, augment=False)
    loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=0, pin_memory=False)
    return loader, test_clips, class_names


def eval_model_on_loader(model, loader, device):
    """Run inference, return accuracy (%)."""
    if loader is None:
        return float("nan")
    criterion = nn.CrossEntropyLoss()
    _, acc, _, _, _ = evaluate(model, loader, criterion, device, "cross-eval")
    return round(acc, 2)


# ============================================================================
#  Heatmap
# ============================================================================

def save_heatmap(matrix: dict, model_fibers: list, test_fibers: list, output_dir: str):
    """Save cross-fiber accuracy heatmap as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_rows = len(model_fibers)
        n_cols = len(test_fibers)
        data   = np.zeros((n_rows, n_cols))

        for i, mf in enumerate(model_fibers):
            for j, tf in enumerate(test_fibers):
                v = matrix.get(mf, {}).get(tf, float("nan"))
                data[i, j] = v if not np.isnan(v) else -1.0

        fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.2), max(5, n_rows * 1.0)))

        masked = np.ma.masked_where(data < 0, data)
        im     = ax.imshow(masked, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy (%)")

        row_labels = [f"model_{mf}" for mf in model_fibers]
        col_labels = [f"data_{tf}"  for tf in test_fibers]

        ax.set_xticks(range(n_cols))
        ax.set_yticks(range(n_rows))
        ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_xlabel("Test Data (from fiber)")
        ax.set_ylabel("Model (trained on fiber)")
        ax.set_title("Cross-Fiber Accuracy Matrix (%)\n(Diagonal = correct fiber, Off-diagonal = wrong fiber)")

        for i in range(n_rows):
            for j in range(n_cols):
                v = data[i, j]
                if v >= 0:
                    color = "white" if v > 60 or v < 20 else "black"
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=8, color=color, fontweight="bold" if i == j else "normal")

        # Highlight diagonal
        for k in range(min(n_rows, n_cols)):
            if model_fibers[k] == test_fibers[k]:
                rect = plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                     fill=False, edgecolor="blue", linewidth=2.5)
                ax.add_patch(rect)

        fig.tight_layout()
        save_path = os.path.join(output_dir, "cross_fiber_heatmap.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Heatmap saved: {save_path}")
    except Exception as e:
        print(f"  [WARNING] Could not save heatmap: {e}")


# ============================================================================
#  Summary markdown
# ============================================================================

def save_summary_md(
    matrix: dict,
    model_fibers: list,
    test_fibers: list,
    output_dir: str,
):
    chance = 100.0 / 26.0

    diagonal_vals     = []
    off_diagonal_vals = []

    for mf in model_fibers:
        for tf in test_fibers:
            v = matrix.get(mf, {}).get(tf, None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if fiber_key(mf) == fiber_key(tf):
                diagonal_vals.append((mf, tf, v))
            else:
                off_diagonal_vals.append((mf, tf, v))

    diag_accs = [x[2] for x in diagonal_vals]
    off_accs  = [x[2] for x in off_diagonal_vals]

    mean_diag = np.mean(diag_accs) if diag_accs else float("nan")
    mean_off  = np.mean(off_accs)  if off_accs  else float("nan")
    max_off   = max(off_accs)      if off_accs  else float("nan")

    # Find anomalous cross-fiber pairs
    threshold = min(mean_diag * 0.5, 50.0) if not np.isnan(mean_diag) else 50.0
    anomalies = [(mf, tf, v) for mf, tf, v in off_diagonal_vals if v > threshold]

    lines = [
        "# Cross-Fiber Evaluation Summary",
        "",
        "## Overview",
        "",
        f"- Number of fiber models evaluated: {len(model_fibers)}",
        f"- Number of test datasets: {len(test_fibers)}",
        f"- Chance level (1/26 random): **{chance:.2f}%**",
        "",
        "## Key Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean diagonal accuracy (correct fiber) | **{mean_diag:.2f}%** |",
        f"| Mean off-diagonal accuracy (wrong fiber) | **{mean_off:.2f}%** |",
        f"| Max off-diagonal accuracy | {max_off:.2f}% |",
        f"| Diagonal advantage | {mean_diag - mean_off:.2f}% |",
        "",
        "## Per-Fiber Diagonal Accuracy",
        "",
        "| Fiber | Own-Fiber Accuracy |",
        "|-------|--------------------|",
    ]
    for mf, tf, v in sorted(diagonal_vals, key=lambda x: -x[2]):
        lines.append(f"| {mf} | {v:.2f}% |")

    lines += [
        "",
        "## Diagonal Dominance Analysis",
        "",
    ]

    if not np.isnan(mean_diag) and not np.isnan(mean_off):
        if mean_diag > mean_off * 2 and mean_diag > chance * 2:
            lines.append(
                "**STRONG diagonal dominance detected.** "
                "The correct fiber model significantly outperforms wrong-fiber models. "
                "This supports the PUF uniqueness claim: each fiber produces a unique "
                "speckle encoding that cannot be decoded by a model trained on a different fiber."
            )
        elif mean_diag > mean_off * 1.3 and mean_diag > chance * 1.5:
            lines.append(
                "**MODERATE diagonal dominance detected.** "
                "Correct fiber models generally outperform wrong-fiber models, "
                "but the separation is not as strong as ideal. "
                "Consider longer training or more epochs for cleaner results."
            )
        else:
            lines.append(
                "**WEAK or NO diagonal dominance detected.** "
                "The cross-fiber separation is not clear. "
                "Possible explanations: insufficient training epochs, "
                "similar speckle statistics across fibers, "
                "or model underfitting. Check individual fiber metrics."
            )

    if anomalies:
        lines += [
            "",
            "## Anomalous Cross-Fiber Pairs",
            "",
            f"The following model-data combinations show unusually high cross-fiber accuracy (>{threshold:.0f}%),",
            "which may indicate partial transferability between these fibers:",
            "",
            "| Model Fiber | Test Data Fiber | Accuracy |",
            "|-------------|-----------------|----------|",
        ]
        for mf, tf, v in sorted(anomalies, key=lambda x: -x[2]):
            lines.append(f"| {mf} | {tf} | {v:.2f}% |")
    else:
        lines += [
            "",
            "## Cross-Fiber Transferability",
            "",
            "No anomalous cross-fiber accuracy pairs found above the threshold. "
            "This supports the claim that fiber models are not transferable across different fibers.",
        ]

    lines += [
        "",
        "## Conclusion",
        "",
        "Based on the cross-fiber evaluation matrix:",
        "",
    ]

    if not np.isnan(mean_diag) and not np.isnan(mean_off):
        if mean_diag > mean_off * 2:
            lines += [
                f"- Correct-fiber models achieve {mean_diag:.1f}% average accuracy on their own data.",
                f"- Wrong-fiber models achieve only {mean_off:.1f}% on foreign data.",
                f"- This represents a {mean_diag/mean_off:.1f}x performance gap, strongly supporting PUF uniqueness.",
                "- **Conclusion: Each fiber acts as a unique physical key. Wrong keys fail to decode the data.**",
            ]
        else:
            lines += [
                f"- Correct-fiber accuracy: {mean_diag:.1f}%",
                f"- Wrong-fiber accuracy: {mean_off:.1f}%",
                "- The separation requires further investigation or improved training.",
            ]

    lines += ["", "---", f"*Generated automatically by evaluate_cross_fiber.py*"]

    path = os.path.join(output_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Summary markdown saved: {path}")


# ============================================================================
#  Main
# ============================================================================

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("  Speckle-PUF  |  Cross-Fiber Evaluation")
    print("=" * 80)
    print(f"  Device          : {device}")
    if device.type == "cuda":
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
    print(f"  Checkpoints dir : {args.checkpoints_dir}")
    print(f"  Video dir       : {args.video_dir}")
    print(f"  Output dir      : {args.output_dir}")
    print()

    # Discover models and data fibers
    ckpt_map = discover_checkpoints(args.checkpoints_dir)
    if not ckpt_map:
        sys.exit(f"[ERROR] No *_best.pth checkpoints found in {args.checkpoints_dir}\n"
                 "  Run: python scripts/train_all_fibers.py  first.")

    data_fibers = discover_fibers(args.video_dir)
    if not data_fibers:
        sys.exit(f"[ERROR] No fiber data found in {args.video_dir}")

    print(f"  Models found    : {list(ckpt_map.keys())}")
    print(f"  Data fibers     : {data_fibers}")
    print()

    # Match model keys to fiber names
    # ckpt_map keys are sanitized (e.g., "fiber1", "test_fiber")
    # data_fibers are original names (e.g., "Fiber1", "test fiber")
    def match_data_fiber(model_key):
        """Find the data fiber whose key matches the model key."""
        for df in data_fibers:
            if fiber_key(df) == model_key:
                return df
        return None

    # Matrix: matrix[model_fiber_key][test_fiber_name] = accuracy
    matrix     = {mk: {} for mk in ckpt_map}
    model_fiber_names = []  # display names for model axis

    total_evals = len(ckpt_map) * len(data_fibers)
    done = 0
    t0_total = time.time()

    # Outer loop: test data fiber (load data once, evaluate all models)
    for tf in data_fibers:
        tf_dir = os.path.join(args.video_dir, tf)
        print(f"\n--- Loading test data for: {tf} ---")

        # We need to find clip_len and img_size from some model checkpoint
        # Use the first available checkpoint as reference
        first_ckpt = next(iter(ckpt_map.values()))
        ref_ckpt   = torch.load(first_ckpt, map_location="cpu", weights_only=False)
        clip_len   = ref_ckpt.get("clip_len", args.clip_len)
        img_size   = ref_ckpt.get("img_size",  args.img_size)

        try:
            test_loader, test_clips, data_class_names = build_test_loader(
                tf_dir, clip_len, img_size, args.batch_size
            )
            if test_loader is None:
                print(f"  [WARNING] No test clips for {tf}, skipping.")
                for mk in ckpt_map:
                    matrix[mk][tf] = float("nan")
                done += len(ckpt_map)
                continue
            print(f"  Test clips: {len(test_clips)}")
        except Exception as e:
            print(f"  [ERROR] Failed to load data for {tf}: {e}")
            for mk in ckpt_map:
                matrix[mk][tf] = float("nan")
            done += len(ckpt_map)
            continue

        # Inner loop: each model
        for mk, ckpt_path in ckpt_map.items():
            done += 1
            print(f"  [{done}/{total_evals}] model={mk}  data={tf} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                model, model_clip_len, model_img_size, model_class_names, fiber_name = \
                    load_model_from_checkpoint(ckpt_path, device)

                # If model was trained with different clip_len/img_size, rebuild loader
                if model_clip_len != clip_len or model_img_size != img_size:
                    test_loader, test_clips, _ = build_test_loader(
                        tf_dir, model_clip_len, model_img_size, args.batch_size
                    )

                acc = eval_model_on_loader(model, test_loader, device)
                dt  = time.time() - t0
                print(f"acc={acc:.1f}%  ({dt:.1f}s)")
                matrix[mk][tf] = acc

                # Store display name for model axis
                display = fiber_name if fiber_name else mk
                if display not in model_fiber_names:
                    model_fiber_names.append(display)

            except Exception as e:
                print(f"ERROR: {e}")
                matrix[mk][tf] = float("nan")

        # Free memory
        del test_loader
        import gc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_elapsed = time.time() - t0_total

    # Build ordered lists for output
    # Model axis: use original fiber names in the same order as ckpt_map
    model_keys    = list(ckpt_map.keys())
    model_display = []
    for mk in model_keys:
        # Try to recover original fiber name from checkpoint
        try:
            ckpt = torch.load(ckpt_map[mk], map_location="cpu", weights_only=False)
            name = ckpt.get("fiber_name", mk)
        except Exception:
            name = mk
        model_display.append(name)

    # ── Save CSV ────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "cross_fiber_accuracy.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["model\\test_data"] + data_fibers
        writer.writerow(header)
        for mk, mdisp in zip(model_keys, model_display):
            row = [mdisp]
            for tf in data_fibers:
                v = matrix[mk].get(tf, float("nan"))
                row.append(f"{v:.2f}" if not (isinstance(v, float) and np.isnan(v)) else "NaN")
            writer.writerow(row)
    print(f"\n  CSV saved: {csv_path}")

    # Print table to console
    print(f"\n{'=' * 80}")
    print("  Cross-Fiber Accuracy Matrix (%)")
    print(f"{'=' * 80}")
    col_w = 12
    header_str = f"  {'Model / Data':<22}" + "".join(f"{tf:>{col_w}}" for tf in data_fibers)
    print(header_str)
    print("  " + "-" * (22 + col_w * len(data_fibers)))
    for mk, mdisp in zip(model_keys, model_display):
        row_str = f"  {mdisp:<22}"
        for tf in data_fibers:
            v = matrix[mk].get(tf, float("nan"))
            s = f"{v:.1f}%" if not (isinstance(v, float) and np.isnan(v)) else "N/A"
            marker = " *" if fiber_key(mdisp) == fiber_key(tf) else "  "
            row_str += f"{s + marker:>{col_w}}"
        print(row_str)
    print(f"\n  (* = same fiber, diagonal entries)")
    print(f"  Total evaluation time: {total_elapsed:.0f}s")

    # ── Save heatmap ─────────────────────────────────────────────────────
    save_heatmap(matrix, model_keys, data_fibers, args.output_dir)

    # ── Save summary markdown ────────────────────────────────────────────
    save_summary_md(matrix, model_display, data_fibers, args.output_dir)

    # ── Save JSON ────────────────────────────────────────────────────────
    json_data = {
        "model_fibers": model_display,
        "test_fibers":  data_fibers,
        "matrix":       {model_display[i]: matrix[mk] for i, mk in enumerate(model_keys)},
        "chance_level": round(100.0 / 26.0, 4),
    }
    json_path = os.path.join(args.output_dir, "cross_fiber_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"  JSON saved: {json_path}")

    print("=" * 80)


def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-fiber evaluation matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--checkpoints_dir", type=str, default=CHECKPOINTS_DIR)
    p.add_argument("--video_dir",       type=str, default=VIDEO_CAPTURE_DIR)
    p.add_argument("--output_dir",      type=str, default=CROSS_RESULTS_DIR)
    p.add_argument("--clip_len",        type=int, default=16)
    p.add_argument("--img_size",        type=int, default=224)
    p.add_argument("--batch_size",      type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    main()
