#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speckle-PUF Inference Script
=============================
Classify speckle frames in a folder, outputting per-frame predictions and video-level majority vote.

Usage:
    python scripts/predict.py --model checkpoints/fiber1_best.pth --test-dir test/
    python scripts/predict.py --model checkpoints/fiber1_best.pth --test-dir screenshots/A --ground-truth A
    python scripts/predict.py --model checkpoints/fiber1_best.pth --test-dir screenshots/A --ground-truth A --top-k 3
"""

import os
import glob
import argparse
from collections import Counter

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# ============================================================================
#   SimpleCNN — must match archive/train_model.py / archive/deep_learning_gui.py exactly
#   Attribute names: self.features / self.classifier
# ============================================================================

class SimpleCNN(nn.Module):
    """
    4-layer convolutional CNN for 1×H×W grayscale speckle frames.
    FC layer dimension auto-computed from image_size; supports any multiple of 16.
    """

    def __init__(self, num_classes: int, image_size: int = 128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1,   32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32,  64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )

        feat_h   = image_size // 16
        feat_dim = 256 * feat_h * feat_h

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
#   Model loading (compatible with both checkpoint formats)
# ============================================================================

def load_model(model_path: str):
    """
    Load the model and return (model, class_names, image_size).
    Compatible with both old and new checkpoint formats.
    """
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Compatible with old/new state_dict key formats
    state_dict = (checkpoint.get('model_state_dict')
                  or checkpoint.get('model')
                  or checkpoint)

    # Class info
    class_names = (checkpoint.get('class_names')
                   or checkpoint.get('letters')
                   or [])
    if not class_names:
        raise ValueError("No class info found in checkpoint (class_names / letters)")

    num_classes = checkpoint.get('num_classes', len(class_names))
    image_size  = checkpoint.get('image_size', 128)

    model = SimpleCNN(num_classes=num_classes, image_size=image_size)
    model.load_state_dict(state_dict)
    model.eval()

    return model, class_names, image_size


# ============================================================================
#   Inference Functions
# ============================================================================

def predict_folder(
    model: nn.Module,
    folder_path: str,
    class_names: list,
    image_size: int,
    device: torch.device,
    ground_truth: str = None,
    top_k: int = 1,
    verbose: bool = True,
) -> list:
    """
    Predict all images in a folder. Returns [(filename, pred_class, confidence), ...].

    Args:
        ground_truth : If provided, also computes frame-level accuracy and video-level correctness.
        top_k        : Show Top-K predicted classes (default 1).
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    image_files = sorted(
        glob.glob(os.path.join(folder_path, "*.png")) +
        glob.glob(os.path.join(folder_path, "*.jpg"))
    )

    if not image_files:
        print(f"⚠  No images found in {folder_path}")
        return []

    if verbose:
        print(f"\nFound {len(image_files)} images")
        print("=" * 65)
        print(f"  {'Filename':<30} {'Prediction':<8} {'Confidence':>8}")
        print("-" * 65)

    results = []

    for img_path in image_files:
        try:
            img    = Image.open(img_path).convert('L')
            tensor = transform(img).unsqueeze(0).to(device)
        except Exception as e:
            print(f"  ⚠  Skipping {os.path.basename(img_path)}: {e}")
            continue

        with torch.no_grad():
            logits = model(tensor)
            prob   = torch.softmax(logits, dim=1)[0]

        if top_k == 1:
            conf, pred_idx = torch.max(prob, 0)
            pred_class = class_names[pred_idx.item()]
            conf_pct   = conf.item() * 100

            if verbose:
                fname = os.path.basename(img_path)
                print(f"  {fname:<30} {pred_class:<8} {conf_pct:>7.2f}%")

            results.append((os.path.basename(img_path), pred_class, conf_pct))

        else:
            topk_vals, topk_idxs = torch.topk(prob, min(top_k, len(class_names)))
            pred_class = class_names[topk_idxs[0].item()]
            conf_pct   = topk_vals[0].item() * 100

            if verbose:
                fname = os.path.basename(img_path)
                topk_str = "  ".join(
                    f"{class_names[i.item()]}({v.item()*100:.1f}%)"
                    for i, v in zip(topk_idxs, topk_vals)
                )
                print(f"  {fname:<30} {topk_str}")

            results.append((os.path.basename(img_path), pred_class, conf_pct))

    if not results:
        return results

    # ── Statistics ────────────────────────────────────────────────────────
    all_preds = [r[1] for r in results]
    counter   = Counter(all_preds)
    majority  = counter.most_common(1)[0][0]

    print("=" * 65)
    print(f"\nVideo-level majority vote: {majority}")
    print(f"Frame-level counts: {dict(counter)}")

    if ground_truth is not None:
        frame_acc = sum(p == ground_truth for p in all_preds) / len(all_preds) * 100
        video_ok  = (majority == ground_truth)
        print(f"\nGround truth : {ground_truth}")
        print(f"Frame accuracy: {frame_acc:.1f}%")
        print(f"Video result : {'CORRECT' if video_ok else 'WRONG'}  "
              f"(vote={majority}, truth={ground_truth})")

    return results


def predict_single_image(
    model: nn.Module,
    image_path: str,
    class_names: list,
    image_size: int,
    device: torch.device,
) -> tuple:
    """Predict a single image. Returns (pred_class, confidence_pct)."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    img    = Image.open(image_path).convert('L')
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.softmax(model(tensor), dim=1)
        conf, pred_idx = torch.max(prob, 1)

    return class_names[pred_idx.item()], conf.item() * 100


# ============================================================================
#   CLI entry point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Speckle-PUF inference script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--model',
                        required=True,
                        help='Path to model file (.pth)')
    parser.add_argument('--test-dir',
                        required=True, dest='test_dir',
                        help='Test image directory (frame_*.png / *.jpg)')
    parser.add_argument('--ground-truth',
                        default=None, dest='ground_truth',
                        help='Optional ground-truth class label for accuracy metrics')
    parser.add_argument('--top-k',
                        type=int, default=1, dest='top_k',
                        help='Show Top-K predictions (default 1)')
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("  Speckle-PUF Inference")
    print("=" * 65)
    print(f"  Device       : {device}")
    print(f"  Model        : {args.model}")
    print(f"  Test dir     : {args.test_dir}")
    if args.ground_truth:
        print(f"  Ground truth : {args.ground_truth}")
    print()

    # Loading model
    model, class_names, image_size = load_model(args.model)
    model = model.to(device)

    print(f"  Classes      : {class_names}")
    print(f"  Image size   : {image_size}")

    if not os.path.isdir(args.test_dir):
        # Single-image mode
        pred_class, confidence = predict_single_image(
            model, args.test_dir, class_names, image_size, device
        )
        print(f"\nPrediction: {pred_class}  Confidence: {confidence:.2f}%")
    else:
        # Batch folder mode
        predict_folder(
            model, args.test_dir, class_names, image_size, device,
            ground_truth=args.ground_truth,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
