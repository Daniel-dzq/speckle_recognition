#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training and Evaluation Module
==============================

Contains:
  - Single-epoch training loop (tqdm progress bar)
  - Validation / test evaluation
  - Model saving (best model by val acc)
  - Early stopping
  - Output saving: training log CSV, confusion matrix PNG, per-class metrics CSV, per-clip predictions CSV
"""

import os
import sys
import csv
import copy
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
#  Training / Evaluation Core Functions
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """Single-epoch training loop. Returns (avg_loss, accuracy%)."""
    model.train()
    total_loss = correct = total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{total_epochs} [train]",
                dynamic_ncols=True, leave=False, file=sys.stderr)

    for clips, labels in pbar:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(clips)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == labels).sum().item()
        total += bs

        pbar.set_postfix(
            loss=f"{total_loss / total:.4f}",
            acc=f"{100 * correct / total:.1f}%",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return total_loss / total, 100 * correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "val",
) -> Tuple[float, float, list, list, list]:
    """
    Evaluate model. Returns (avg_loss, accuracy%, all_preds, all_labels, all_probs).
    all_probs is the full softmax probability vector for each sample.
    """
    model.eval()
    total_loss = correct = total = 0
    all_preds: list = []
    all_labels: list = []
    all_probs: list = []

    n_batches = len(loader)
    for batch_idx, (clips, labels) in enumerate(loader):
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(clips)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += bs

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().numpy())

        # Single-line overwriting progress (works in any terminal)
        pct = (batch_idx + 1) / n_batches * 100
        acc_now = 100 * correct / total
        print(f"\r  [{desc}] {batch_idx+1}/{n_batches}  "
              f"loss={total_loss/total:.4f}  acc={acc_now:.1f}%  ",
              end="", flush=True)

    print()  # newline after the overwriting line

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100 * correct / total if total > 0 else 0.0

    return avg_loss, accuracy, all_preds, all_labels, all_probs


# ============================================================================
#  Main Training Loop
# ============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    device: torch.device,
    output_dir: str,
) -> Tuple[nn.Module, float, list]:
    """
    Full training loop: AdamW + CosineAnnealing + Early Stopping.
    Returns (model_with_best_weights, best_val_acc, history).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_val_acc = 0.0
    best_state = None
    patience_cnt = 0
    history: List[dict] = []

    print(f"\n{'=' * 80}")
    print(f"  Training started: model={args.model_type}, epochs={args.epochs}, "
          f"lr={args.lr}, device={device}")
    print(f"{'=' * 80}\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )

        val_loss, val_acc, _, _, _ = evaluate(
            model, val_loader, criterion, device, "val"
        )

        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        marker = "*" if val_acc > best_val_acc else " "
        print(
            f" {marker} Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:6.2f}% | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:6.2f}% | lr={lr:.2e}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 4),
            "lr": lr,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_cnt = 0

            torch.save({
                "model_state_dict": best_state,
                "model_type": args.model_type,
                "num_classes": args.num_classes,
                "class_names": args.class_names,
                "clip_len": args.clip_len,
                "img_size": args.img_size,
                "best_val_acc": best_val_acc,
                "epoch": epoch,
            }, os.path.join(output_dir, "best_model.pth"))
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\n  Early stopping: val_acc not improved for {args.patience} epochs")
                break

    print(f"\n{'=' * 80}")
    print(f"  Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'=' * 80}\n")

    # Save training log CSV
    log_path = os.path.join(output_dir, "training_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
        )
        writer.writeheader()
        writer.writerows(history)
    print(f"  Training log saved: {log_path}")

    model.load_state_dict(best_state)
    return model, best_val_acc, history


# ============================================================================
#  Test and Output
# ============================================================================

def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    test_clips: List[dict],
    class_names: List[str],
    device: torch.device,
    output_dir: str,
) -> float:
    """
    Evaluate on the test set and save all required output files:
      1. confusion_matrix.png
      2. per_class_metrics.csv
      3. test_predictions.csv
    Returns test accuracy.
    """
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, all_preds, all_labels, all_probs = evaluate(
        model, test_loader, criterion, device, "test"
    )

    print(f"\n{'=' * 80}")
    print(f"  Test results:  loss = {test_loss:.4f},  accuracy = {test_acc:.2f}%")
    print(f"{'=' * 80}\n")

    # ── Classification report (terminal) ─────────────────────────────────
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(report)

    # ── 1. Per-class precision / recall / F1 (CSV) ───────────────────────
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    metrics_path = os.path.join(output_dir, "per_class_metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for i, name in enumerate(class_names):
            writer.writerow([
                name,
                f"{precision[i]:.4f}",
                f"{recall[i]:.4f}",
                f"{f1[i]:.4f}",
                int(support[i]),
            ])
    print(f"  Per-class metrics saved: {metrics_path}")

    # ── 2. Per-clip test predictions (CSV) ───────────────────────────────
    preds_path = os.path.join(output_dir, "test_predictions.csv")
    with open(preds_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "true_label", "pred_label", "confidence",
            "video_name", "start_frame", "end_frame",
        ])
        for i, clip_info in enumerate(test_clips):
            true_label = class_names[clip_info["label"]]
            pred_label = class_names[all_preds[i]]
            confidence = float(all_probs[i][all_preds[i]])
            writer.writerow([
                true_label,
                pred_label,
                f"{confidence:.4f}",
                clip_info["video_name"],
                clip_info["start_frame"],
                clip_info["end_frame"],
            ])
    print(f"  Test predictions saved: {preds_path}")

    # ── 3. Confusion matrix (PNG) ───────────────────────────────────────
    _save_confusion_matrix(all_labels, all_preds, class_names, output_dir)

    return test_acc


# ============================================================================
#  Confusion Matrix Visualization
# ============================================================================

def _save_confusion_matrix(
    labels: list,
    preds: list,
    class_names: List[str],
    output_dir: str,
) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    n = len(class_names)

    fig_size = max(8, n * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_fs = max(6, 11 - n // 6)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=tick_fs)
    ax.set_yticklabels(class_names, fontsize=tick_fs)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix (Test Set)", fontsize=14)

    thresh = cm.max() / 2.0
    cell_fs = max(5, 9 - n // 6)
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=cell_fs,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    save_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved: {save_path}")
