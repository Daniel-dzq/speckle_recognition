#!/usr/bin/env python3
"""
Speckle-PUF CNN Training Script
================================

Background:
  Input is speckle video frames captured after SLM-encoded letters pass through
  multimode/plastic optical fiber. The goal is to recognize the encoded letter
  (decode) or identify the fiber (fiber_id) from the otherwise visually
  indistinguishable speckle patterns.

Data structure:

  Flat (single fiber, current):
    DATA_DIR/
      A/  frame_00001.png, frame_00002.png, ...
      B/  ...

  Multi-fiber (future extension):
    DATA_DIR/
      fiber_01/
        A/  frame_*.png
        B/  ...
      fiber_02/
        A/  ...

Temporal split (no clip leakage):
  Each (fiber, letter) segment is split chronologically so that clips from
  the same video never appear in both train and test:
    First  70% of frames -> train
    Middle 15% of frames -> val
    Last   15% of frames -> test

Tasks:
  decode   -> recognize the encoded letter/symbol (class = subfolder name)
  fiber_id -> identify which fiber (class = fiber_xx directory name)

Usage:
  python train_model.py                                # default: ./screenshots, decode
  python train_model.py --data my_data                 # custom data directory
  python train_model.py --task fiber_id                # fiber identification
  python train_model.py --epochs 100 --lr 5e-4         # tune hyperparameters
  python train_model.py --data data --task decode --output model_decode.pth
"""

import os
import sys
import glob
import copy
import argparse
import random
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix

# tqdm is optional; degrades gracefully if not installed
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Tip: install tqdm for nicer progress bars  ->  pip install tqdm\n")

# ============================================================================
#                              Default Configuration
# ============================================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(BASE_DIR, "screenshots")
DEFAULT_OUT  = os.path.join(BASE_DIR, "model.pth")

IMAGE_SIZE  = 128
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-3
PATIENCE    = 10
TRAIN_RATIO = 0.70   # first 70% of frames -> train
VAL_RATIO   = 0.15   # middle 15% -> val
# TEST_RATIO = 0.15  # last 15% -> test (implicit)
NUM_WORKERS = 0      # must be 0 on Windows for DataLoader multiprocessing


# ============================================================================
#                        Data Structure Detection & Sample Collection
# ============================================================================

def _has_images(folder: str) -> bool:
    """Return True if the directory contains image files directly."""
    for ext in ("*.png", "*.jpg", "*.PNG", "*.JPG"):
        if glob.glob(os.path.join(folder, ext)):
            return True
    return False


def detect_structure(data_dir: str) -> str:
    """
    Detect the directory hierarchy of the data folder.

    Returns:
      'flat'  -> data_dir/classX/frame_*.png
      'fiber' -> data_dir/fiberX/classX/frame_*.png
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    subdirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    if not subdirs:
        raise ValueError(f"No subdirectories found in data directory: {data_dir}")

    first = os.path.join(data_dir, subdirs[0])
    if _has_images(first):
        return 'flat'

    # Check for fiberX/classX two-level structure
    sub_subdirs = [
        d for d in os.listdir(first)
        if os.path.isdir(os.path.join(first, d))
    ]
    if sub_subdirs and _has_images(os.path.join(first, sub_subdirs[0])):
        return 'fiber'

    return 'flat'   # fallback


def collect_samples(
    data_dir: str,
    task: str = 'decode',
) -> Tuple[List, List, List, List[str]]:
    """
    Collect samples and apply temporal train/val/test split.
    Returns train / val / test sample lists and class name list.

    Split principle:
      - Flat structure: sort frames by filename within each class folder, then split chronologically.
      - Multi-fiber structure: split each (fiber, letter) segment independently so clips from
        the same video never appear in both train and test.
    """
    structure = detect_structure(data_dir)
    print(f"\n[Data] Structure: {structure}  Task: {task}")

    # segments: list of (class_label, sorted_frame_paths)
    # Each segment corresponds to one "video unit" and is split internally.
    segments: List[Tuple[str, List[str]]] = []
    class_names_set = set()

    if structure == 'flat':
        # data_dir/classX/frame_*.png
        for cls_dir in sorted(os.listdir(data_dir)):
            cls_path = os.path.join(data_dir, cls_dir)
            if not os.path.isdir(cls_path):
                continue
            imgs = sorted(
                glob.glob(os.path.join(cls_path, "*.png")) +
                glob.glob(os.path.join(cls_path, "*.jpg"))
            )
            if not imgs:
                continue
            class_label = cls_dir if task == 'decode' else 'fiber_default'
            class_names_set.add(class_label)
            segments.append((class_label, imgs))

    else:  # fiber
        # data_dir/fiberX/classX/frame_*.png
        for fiber_dir in sorted(os.listdir(data_dir)):
            fiber_path = os.path.join(data_dir, fiber_dir)
            if not os.path.isdir(fiber_path):
                continue
            for letter_dir in sorted(os.listdir(fiber_path)):
                letter_path = os.path.join(fiber_path, letter_dir)
                if not os.path.isdir(letter_path):
                    continue
                imgs = sorted(
                    glob.glob(os.path.join(letter_path, "*.png")) +
                    glob.glob(os.path.join(letter_path, "*.jpg"))
                )
                if not imgs:
                    continue
                class_label = letter_dir if task == 'decode' else fiber_dir
                class_names_set.add(class_label)
                segments.append((class_label, imgs))

    if not segments:
        raise RuntimeError("No images found. Please check the data directory structure.")

    class_names  = sorted(class_names_set)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    train_s, val_s, test_s = [], [], []

    print(f"\n{'Class':>12}  {'Total':>6}  {'Train':>6}  {'Val':>6}  {'Test':>6}")
    print("-" * 50)

    # Aggregate per-class frame counts for display
    cls_stat: Dict[str, Dict[str, int]] = {
        c: {'total': 0, 'train': 0, 'val': 0, 'test': 0}
        for c in class_names
    }

    for class_label, imgs in segments:
        n         = len(imgs)
        train_end = int(n * TRAIN_RATIO)
        val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))
        idx       = class_to_idx[class_label]

        t_part = [(p, idx) for p in imgs[:train_end]]
        v_part = [(p, idx) for p in imgs[train_end:val_end]]
        s_part = [(p, idx) for p in imgs[val_end:]]

        train_s.extend(t_part)
        val_s.extend(v_part)
        test_s.extend(s_part)

        stat = cls_stat[class_label]
        stat['total'] += n
        stat['train'] += len(t_part)
        stat['val']   += len(v_part)
        stat['test']  += len(s_part)

    for cls in class_names:
        s = cls_stat[cls]
        print(f"  {cls:>10}  {s['total']:>6}  {s['train']:>6}  {s['val']:>6}  {s['test']:>6}")

    total = len(train_s) + len(val_s) + len(test_s)
    print("-" * 50)
    print(f"  {'Total':>10}  {total:>6}  {len(train_s):>6}  {len(val_s):>6}  {len(test_s):>6}\n")

    return train_s, val_s, test_s, class_names


# ============================================================================
#                              Dataset
# ============================================================================

class SpeckleDataset(Dataset):
    """Frame-level speckle image dataset."""

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================================================================
#                              CNN Model
# ============================================================================

class SimpleCNN(nn.Module):
    """
    4-layer grayscale CNN for speckle frame classification.
    Input: 1 x H x W. FC layer dimensions are computed automatically
    from image_size, supporting any multiple of 16.
    """

    def __init__(self, num_classes: int, image_size: int = 128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1,   32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32,  64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )

        # 4x MaxPool2d(2) -> spatial dimensions shrink by factor of 16
        feat_h    = image_size // 16
        feat_dim  = 256 * feat_h * feat_h

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================================================
#                         DataLoader Construction
# ============================================================================

def make_loaders(
    train_s, val_s, test_s,
    image_size: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build three DataLoaders; train split includes data augmentation."""

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    pin = torch.cuda.is_available()

    def _loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=NUM_WORKERS, pin_memory=pin,
            persistent_workers=False,
        )

    return (
        _loader(SpeckleDataset(train_s, train_tf), shuffle=True),
        _loader(SpeckleDataset(val_s,   eval_tf),  shuffle=False),
        _loader(SpeckleDataset(test_s,  eval_tf),  shuffle=False),
    )


# ============================================================================
#                        Evaluation Function
# ============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, list, list]:
    """Returns (mean_loss, acc_pct, all_preds, all_labels)."""
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out  = model(x)
        loss = criterion(out, y)

        total_loss += loss.item() * y.size(0)
        preds       = out.argmax(1)
        correct    += (preds == y).sum().item()
        total      += y.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    return total_loss / total, 100 * correct / total, all_preds, all_labels


# ============================================================================
#                        Main Training Loop
# ============================================================================

def train(args, train_loader, val_loader, test_loader,
          class_names: List[str], device: torch.device) -> dict:
    """Train, validate, and test; save the best model."""

    num_classes = len(class_names)
    model       = SimpleCNN(num_classes=num_classes, image_size=args.image_size).to(device)
    criterion   = nn.CrossEntropyLoss()
    optimizer   = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_state   = None
    patience_cnt = 0
    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"[Training] device={device}  classes={num_classes}  "
          f"epochs={args.epochs}  lr={args.lr}  image_size={args.image_size}")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        total_loss = correct = total = 0

        if HAS_TQDM:
            batches = tqdm(
                train_loader,
                desc=f"Epoch {epoch:03d}/{args.epochs}",
                ncols=85, leave=False,
            )
        else:
            batches = train_loader

        for x, y in batches:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct    += (out.argmax(1) == y).sum().item()
            total      += y.size(0)

        scheduler.step()
        train_loss = total_loss / total
        train_acc  = 100 * correct / total

        # ── Val ────────────────────────────────────────────────────────────
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]['lr']

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        star = "*" if val_acc > best_val_acc else " "
        print(f"{star} Epoch {epoch:03d}/{args.epochs} | "
              f"loss {train_loss:.4f} | train {train_acc:6.2f}% | "
              f"val {val_acc:6.2f}% | lr {current_lr:.2e}")

        # ── Early Stopping & Best Model ────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_cnt = 0
            best_state   = copy.deepcopy(model.state_dict())
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\n  Early stopping: val_acc did not improve for {args.patience} epochs.")
                break

    print("=" * 80)
    print(f"[Training complete] Best val accuracy: {best_val_acc:.2f}%")

    # Load best weights and evaluate on test set
    model.load_state_dict(best_state)
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)

    print(f"\n[Test set]  loss {test_loss:.4f}   acc {test_acc:.2f}%")
    print("\nClassification report:")
    print(classification_report(
        labels, preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    ))

    # Confusion matrix
    _save_confusion_matrix(
        labels, preds, class_names,
        save_path=os.path.join(os.path.dirname(args.output), "confusion_matrix.png"),
    )

    # Save checkpoint
    checkpoint = {
        "model_state_dict": best_state,
        "model":            best_state,       # legacy key compatibility
        "class_names":      class_names,
        "letters":          class_names,       # legacy key compatibility
        "num_classes":      num_classes,
        "image_size":       args.image_size,
        "task":             args.task,
        "best_val_acc":     best_val_acc,
        "test_acc":         test_acc,
        "history":          history,
    }
    torch.save(checkpoint, args.output)
    print(f"\n[Saved] model  -> {args.output}")

    history["best_val_acc"] = best_val_acc
    history["test_acc"]     = test_acc
    return history


# ============================================================================
#                        Confusion Matrix
# ============================================================================

def _save_confusion_matrix(
    labels: list, preds: list,
    class_names: List[str],
    save_path: str,
) -> None:
    """Save the confusion matrix as a PNG image."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        cm   = confusion_matrix(labels, preds)
        n    = len(class_names)
        size = max(5, n * 0.6)
        fig, ax = plt.subplots(figsize=(size, size))

        im = ax.imshow(cm, cmap='Blues')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=max(6, 10 - n // 5))
        ax.set_yticklabels(class_names, fontsize=max(6, 10 - n // 5))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        thresh = cm.max() / 2.0
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center', fontsize=max(6, 9 - n // 6),
                        color='white' if cm[i, j] > thresh else 'black')

        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"[Saved] confusion matrix -> {save_path}")

    except Exception as e:
        print(f"  Warning: could not save confusion matrix (training results unaffected): {e}")


# ============================================================================
#                         Argument Parsing & Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Speckle-PUF CNN Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--data',
                        default=DEFAULT_DATA,
                        help=f'Data directory (default: {DEFAULT_DATA})')
    parser.add_argument('--output',
                        default=DEFAULT_OUT,
                        help=f'Model save path (default: {DEFAULT_OUT})')
    parser.add_argument('--task',
                        default='decode',
                        choices=['decode', 'fiber_id'],
                        help='Task type: decode=recognize letters  fiber_id=identify fiber (default: decode)')
    parser.add_argument('--epochs',
                        type=int,   default=EPOCHS,
                        help=f'Max training epochs (default: {EPOCHS})')
    parser.add_argument('--lr',
                        type=float, default=LR,
                        help=f'Initial learning rate (default: {LR})')
    parser.add_argument('--batch',
                        type=int,   default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--image-size',
                        type=int,   default=IMAGE_SIZE,
                        dest='image_size',
                        help=f'Input image size, must be a multiple of 16 (default: {IMAGE_SIZE})')
    parser.add_argument('--patience',
                        type=int,   default=PATIENCE,
                        help=f'Early stopping patience (default: {PATIENCE})')
    parser.add_argument('--seed',
                        type=int,   default=42,
                        help='Random seed (default: 42)')
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("  Speckle-PUF CNN Training")
    print("=" * 80)
    print(f"  Data dir    : {args.data}")
    print(f"  Task        : {args.task}")
    print(f"  Output      : {args.output}")
    print(f"  Device      : {device}")
    if torch.cuda.is_available():
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print()

    # Collect samples (temporal split)
    train_s, val_s, test_s, class_names = collect_samples(args.data, args.task)

    if not train_s:
        sys.exit("Error: no training samples found. Check data directory structure.")

    if len(class_names) < 2:
        print(f"Warning: only {len(class_names)} class(es) found. At least 2 are recommended.")

    # Build DataLoaders
    train_loader, val_loader, test_loader = make_loaders(
        train_s, val_s, test_s,
        args.image_size, args.batch,
    )

    # Train
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    train(args, train_loader, val_loader, test_loader, class_names, device)


if __name__ == "__main__":
    main()
