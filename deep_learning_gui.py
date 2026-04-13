#!/usr/bin/env python3
"""
Deep Learning Training Suite - Unified GUI Application
Integrates video frame extraction, CNN training, Random Forest training,
model prediction, and feature visualization.
"""

import os
import sys
import glob
import copy
import string
import random
import threading
import queue
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional, Any

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# Visualization
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


# ============================================================================
#                              Utilities
# ============================================================================

class BackgroundTask:
    """Background task thread manager."""

    def __init__(self, root, task_func: Callable,
                 on_progress: Callable = None,
                 on_complete: Callable = None,
                 on_error: Callable = None):
        self.root = root
        self.task_func = task_func
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.on_error = on_error
        self.message_queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.thread = None

    def start(self, *args, **kwargs):
        """Start the background task."""
        self.stop_flag.clear()
        self.thread = threading.Thread(
            target=self._run_task,
            args=args,
            kwargs=kwargs,
            daemon=True
        )
        self.thread.start()
        self._poll_messages()

    def stop(self):
        """Request task stop."""
        self.stop_flag.set()

    def _run_task(self, *args, **kwargs):
        try:
            result = self.task_func(
                *args,
                progress_callback=self._send_progress,
                stop_flag=self.stop_flag,
                **kwargs
            )
            self.message_queue.put(("complete", result))
        except Exception as e:
            self.message_queue.put(("error", str(e)))

    def _send_progress(self, progress: float, message: str = ""):
        self.message_queue.put(("progress", (progress, message)))

    def _poll_messages(self):
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()
                if msg_type == "progress" and self.on_progress:
                    self.on_progress(data[0], data[1])
                elif msg_type == "complete":
                    if self.on_complete:
                        self.on_complete(data)
                    return
                elif msg_type == "error":
                    if self.on_error:
                        self.on_error(data)
                    return
        except queue.Empty:
            pass

        self.root.after(100, self._poll_messages)


# ============================================================================
#                           Video Processing
# ============================================================================

@dataclass
class VideoInfo:
    path: str
    name: str
    extension: str
    duration: float
    size_mb: float


class VideoProcessor:
    """Video frame extraction processor."""

    SUPPORTED_FORMATS = {'.avi', '.mp4', '.mkv', '.mov'}

    def __init__(self):
        self.ffmpeg_path = self._get_ffmpeg_path()

    def _get_ffmpeg_path(self) -> str:
        """Locate the FFmpeg executable."""
        # 1. Try imageio_ffmpeg
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass

        # 2. Try system PATH
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            if result.returncode == 0:
                return "ffmpeg"
        except Exception:
            pass

        # 3. Try common Windows installation paths
        common_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
            os.path.expanduser(r"~\ffmpeg\bin\ffmpeg.exe"),
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path

        return "ffmpeg"  # last resort default

    def scan_folder(self, folder: str) -> List[VideoInfo]:
        """Scan a folder for video files."""
        videos = []

        for ext in self.SUPPORTED_FORMATS:
            for case_ext in [ext, ext.upper()]:
                pattern = os.path.join(folder, f"*{case_ext}")
                for path in glob.glob(pattern):
                    name = os.path.splitext(os.path.basename(path))[0]
                    size_mb = os.path.getsize(path) / (1024 * 1024)
                    duration = self.get_video_duration(path)

                    videos.append(VideoInfo(
                        path=path,
                        name=name,
                        extension=ext.lower(),
                        duration=duration or 0.0,
                        size_mb=size_mb
                    ))

        # Deduplicate and sort
        seen = set()
        unique = []
        for v in sorted(videos, key=lambda x: x.name):
            if v.path not in seen:
                seen.add(v.path)
                unique.append(v)

        return unique

    def get_video_duration(self, video_path: str) -> Optional[float]:
        """Get video duration in seconds."""
        cmd = [self.ffmpeg_path, "-i", video_path]
        try:
            result = subprocess.run(
                cmd,
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="ignore"
            )

            for line in result.stderr.splitlines():
                if "Duration" in line:
                    time_str = line.split("Duration:")[1].split(",")[0].strip()
                    h, m, s = time_str.split(":")
                    return float(h) * 3600 + float(m) * 60 + float(s)
        except Exception:
            pass

        return None

    def extract_frames(self, video_path: str, output_dir: str,
                       target_frames: int = 100, image_format: str = "png",
                       progress_callback: Callable = None,
                       stop_flag: threading.Event = None) -> int:
        """Extract frames from a video file."""
        os.makedirs(output_dir, exist_ok=True)

        duration = self.get_video_duration(video_path)
        if not duration or duration <= 0:
            # Fall back to direct extraction without knowing duration
            duration = 60.0  # assume 60 s and let ffmpeg handle the rest

        fps = target_frames / duration
        fps = max(fps, 0.1)

        output_pattern = os.path.join(output_dir, f"frame_%05d.{image_format}")

        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-vf", f"fps={fps},format=gray",
            "-vsync", "vfr",
            "-y",
            output_pattern
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="ignore",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
        except Exception as e:
            print(f"FFmpeg error: {e}")
            return 0

        # Check output files even if returncode != 0 (ffmpeg sometimes returns 1 but still succeeds)

        files = glob.glob(os.path.join(output_dir, f"*.{image_format}"))
        return len(files)

    def extract_all(self, videos: List[VideoInfo], output_dir: str,
                    target_frames: int = 100, image_format: str = "png",
                    progress_callback: Callable = None,
                    stop_flag: threading.Event = None) -> int:
        """Extract frames from all videos in batch."""
        total = 0

        for i, video in enumerate(videos):
            if stop_flag and stop_flag.is_set():
                break

            video_output = os.path.join(output_dir, video.name)
            saved = self.extract_frames(
                video.path, video_output,
                target_frames, image_format
            )
            total += saved

            if progress_callback:
                progress = (i + 1) / len(videos) * 100
                progress_callback(progress, f"Processing: {video.name} ({saved} frames)")

        return total


# ============================================================================
#                             CNN Model
# ============================================================================

class SimpleCNN(nn.Module):
    """
    4-layer grayscale CNN for speckle frame classification.
    Input: 1 x H x W. FC layer dimensions auto-computed from image_size;
    supports any multiple of 16.
    """

    def __init__(self, num_classes: int = 26, image_size: int = 128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1,   32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32,  64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )

        # 4x MaxPool2d(2) -> spatial dims shrink by factor 16
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


class LightSignalDataset(Dataset):
    """Image dataset for speckle frames."""

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
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
#                           CNN Trainer
# ============================================================================

class CNNTrainer:
    """CNN model trainer."""

    def __init__(self, data_dir: str, device: str = "auto"):
        self.data_dir = data_dir
        self.device = self._parse_device(device)
        self.stop_requested = False
        self.model = None
        self.letters = None  # set dynamically from data

    def _parse_device(self, device: str) -> torch.device:
        """Parse device string to torch.device."""
        if device == "auto" or device.startswith("auto"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device.startswith("cuda:"):
            cuda_id = device.split()[0]  # strip GPU name in parentheses
            return torch.device(cuda_id)
        elif device == "cpu":
            return torch.device("cpu")
        else:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_data(self, image_size: int = 128, batch_size: int = 32,
                     train_split: float = 0.8):
        """
        Prepare datasets with temporal train/val split.

        Frames within each class folder are sorted by filename (chronological order)
        and split at train_split ratio:
          First train_split fraction  -> training set
          Last  (1-train_split) fraction -> validation set
        This prevents clip leakage across splits.

        For multi-fiber data (fiber_xx/letter/ hierarchy), use train_model.py
        (CLI) with collect_samples() for a more complete temporal split.
        """
        all_folders = sorted([
            f for f in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, f))
        ])

        if not all_folders:
            raise ValueError("No subdirectories found in data folder.")

        # Use folder names directly as class labels
        self.letters    = all_folders
        letter_to_idx   = {c: i for i, c in enumerate(self.letters)}

        train_samples, val_samples = [], []

        for folder in all_folders:
            label       = letter_to_idx[folder]
            folder_path = os.path.join(self.data_dir, folder)

            # Sort by filename to ensure chronological order
            imgs = sorted(
                glob.glob(os.path.join(folder_path, "*.png")) +
                glob.glob(os.path.join(folder_path, "*.jpg"))
            )

            if not imgs:
                continue

            # Temporal split: first train_split -> train, rest -> val
            split = int(len(imgs) * train_split)
            train_samples.extend([(p, label) for p in imgs[:split]])
            val_samples.extend(  [(p, label) for p in imgs[split:]])

        if not train_samples:
            raise ValueError("No training samples found. Check that the data directory contains images.")

        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        val_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        train_ds = LightSignalDataset(train_samples, train_tf)
        val_ds   = LightSignalDataset(val_samples,   val_tf)

        pin = self.device.type == "cuda"

        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=pin,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=pin,
        )

        return len(train_samples), len(val_samples)

    def train(self, epochs: int = 50, lr: float = 1e-3, patience: int = 8,
              image_size: int = 128, progress_callback: Callable = None,
              stop_flag: threading.Event = None) -> dict:
        """
        Train the CNN model.

        Features:
          - image_size passed to SimpleCNN to auto-compute FC dimensions
          - CosineAnnealingLR scheduler for stable convergence
          - Tracks both train_loss and val_loss for overfitting detection
          - copy.deepcopy saves best weights without shallow-copy issues
        """
        self.model = SimpleCNN(
            num_classes=len(self.letters), image_size=image_size
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc     = 0.0
        patience_cnt = 0
        history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            if stop_flag and stop_flag.is_set():
                break

            # ── Train ────────────────────────────────────────────────────
            self.model.train()
            correct = total = 0
            total_loss = 0.0

            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                out  = self.model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * y.size(0)
                correct    += (out.argmax(1) == y).sum().item()
                total      += y.size(0)

            train_acc  = 100 * correct / total
            train_loss = total_loss / total
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # ── Validation ───────────────────────────────────────────────
            self.model.eval()
            val_correct = val_total = 0
            val_loss_sum = 0.0

            with torch.no_grad():
                for x, y in self.val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out  = self.model(x)
                    val_loss_sum += criterion(out, y).item() * y.size(0)
                    val_correct  += (out.argmax(1) == y).sum().item()
                    val_total    += y.size(0)

            val_acc  = 100 * val_correct / val_total
            val_loss = val_loss_sum / val_total

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if progress_callback:
                progress = (epoch + 1) / epochs * 100
                msg = (f"Epoch {epoch+1}/{epochs} | "
                       f"Loss {train_loss:.4f} | "
                       f"Train {train_acc:.2f}% | "
                       f"Val {val_acc:.2f}% | "
                       f"LR {current_lr:.2e}")
                progress_callback(progress, msg)

            # ── Early Stopping & Best Model ───────────────────────────────
            if val_acc > best_acc:
                best_acc     = val_acc
                patience_cnt = 0
                # deepcopy ensures best weights are not overwritten by later epochs
                self.best_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    break

        history["best_acc"]   = best_acc
        history["image_size"] = image_size
        return history

    def save_model(self, path: str, image_size: int = 128):
        """Save model checkpoint."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        checkpoint = {
            "model": self.best_state if hasattr(self, 'best_state') else self.model.state_dict(),
            "letters": self.letters,
            "image_size": image_size,
            "num_classes": len(self.letters),
            "class_names": self.letters,
            "model_state_dict": self.best_state if hasattr(self, 'best_state') else self.model.state_dict()
        }

        torch.save(checkpoint, path)

    def request_stop(self):
        self.stop_requested = True


# ============================================================================
#                        Random Forest Trainer
# ============================================================================

class RFTrainer:
    """Random Forest model trainer."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.model = None
        self.letters = None  # set dynamically

    def extract_video_features(self, folder: str) -> List[float]:
        """Extract statistical features from a single video folder."""
        frames = sorted(
            glob.glob(os.path.join(folder, "*.png")) +
            glob.glob(os.path.join(folder, "*.jpg"))
        )

        if len(frames) == 0:
            raise RuntimeError(f"No images found in folder: {folder}")

        gray_means = []
        gray_stds = []
        edge_strength = []

        for f in frames:
            img = Image.open(f).convert("L")
            arr = np.array(img, dtype=np.float32)

            gray_means.append(arr.mean())
            gray_stds.append(arr.std())

            gx, gy = np.gradient(arr)
            edge_strength.append(np.mean(np.sqrt(gx**2 + gy**2)))

        gray_means = np.array(gray_means)
        gray_stds = np.array(gray_stds)
        edge_strength = np.array(edge_strength)

        diff = np.diff(gray_means)

        if len(gray_means) > 1:
            fft = np.fft.rfft(gray_means - gray_means.mean())
            fft_energy = np.mean(np.abs(fft))
        else:
            fft_energy = 0.0

        return [
            gray_means.mean(),
            gray_stds.mean(),
            edge_strength.mean(),
            diff.mean() if len(diff) else 0.0,
            diff.std() if len(diff) else 0.0,
            fft_energy
        ]

    def extract_all_features(self, progress_callback: Callable = None,
                             stop_flag: threading.Event = None) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from all video folders."""
        folders = sorted([
            f for f in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, f))
        ])

        if len(folders) == 0:
            raise ValueError("No subdirectories found in data folder.")

        num_classes = len(folders)
        self.letters = [chr(ord('A') + i) for i in range(num_classes)]

        X, y = [], []

        for i, (folder, letter) in enumerate(zip(folders, self.letters)):
            if stop_flag and stop_flag.is_set():
                break

            path = os.path.join(self.data_dir, folder)
            feat = self.extract_video_features(path)
            X.append(feat)
            y.append(letter)

            if progress_callback:
                progress = (i + 1) / len(folders) * 100
                progress_callback(progress, f"Extracting features: {folder} -> {letter}")

        return np.array(X), np.array(y)

    def train(self, n_estimators: int = 400, test_size: float = 0.2,
              progress_callback: Callable = None,
              stop_flag: threading.Event = None) -> dict:
        """
        Train a Random Forest model.

        Note: RF mode extracts only 1 feature vector per video (global statistics).
        With one video per class, effective sample count equals number of classes,
        so evaluation metrics are for reference only.
        CNN mode is recommended for more reliable frame-level evaluation.
        """
        X, y = self.extract_all_features(progress_callback, stop_flag)

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=n_estimators, random_state=42))
        ])

        if test_size > 0 and len(X) >= 4:
            # Attempt stratified split; fall back to plain split if insufficient samples
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            except ValueError:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            self.model.fit(X_train, y_train)
            train_acc = (self.model.predict(X_train) == y_train).mean() * 100
            val_acc   = (self.model.predict(X_val)   == y_val  ).mean() * 100
        else:
            # Too few samples to split; train on all data and report train accuracy (reference only)
            self.model.fit(X, y)
            train_acc = (self.model.predict(X) == y).mean() * 100
            val_acc   = train_acc
            if progress_callback:
                progress_callback(90, "Warning: too few samples; reporting train accuracy only (reference)")

        if progress_callback:
            progress_callback(100, f"Training complete | Train {train_acc:.2f}% | Val {val_acc:.2f}%")

        return {"train_acc": train_acc, "val_acc": val_acc}

    def save_model(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        joblib.dump({
            "model": self.model,
            "letters": self.letters,
            "model_type": "rf"
        }, path)


# ============================================================================
#                             Predictor
# ============================================================================

class Predictor:
    """Model predictor (supports CNN and Random Forest checkpoints)."""

    def __init__(self):
        self.model = None
        self.model_type = None
        self.class_names = None
        self.image_size = 128
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, path: str) -> dict:
        """Load a model checkpoint (auto-detects type)."""
        if path.endswith('.pth'):
            return self._load_cnn(path)
        elif path.endswith('.joblib'):
            return self._load_rf(path)
        else:
            # Try PyTorch format first
            try:
                return self._load_cnn(path)
            except Exception:
                return self._load_rf(path)

    def _load_cnn(self, path: str) -> dict:
        """Load a CNN checkpoint."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        # Support multiple checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        num_classes = checkpoint.get('num_classes', 26)
        self.class_names = checkpoint.get('class_names',
                          checkpoint.get('letters', list(string.ascii_uppercase)))
        self.image_size = checkpoint.get('image_size', 128)

        self.model = SimpleCNN(num_classes=num_classes, image_size=self.image_size)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.model_type = "cnn"

        return {
            "type": "CNN",
            "classes": len(self.class_names),
            "image_size": self.image_size
        }

    def _load_rf(self, path: str) -> dict:
        """Load a Random Forest checkpoint."""
        data = joblib.load(path)
        self.model = data['model']
        self.class_names = data.get('letters', list(string.ascii_uppercase))
        self.model_type = "rf"

        return {
            "type": "Random Forest",
            "classes": len(self.class_names)
        }

    def predict_single(self, image_path: str) -> Tuple[str, float]:
        """Predict a single image."""
        if self.model_type == "cnn":
            return self._predict_cnn(image_path)
        else:
            return self._predict_rf(image_path)

    def _predict_cnn(self, image_path: str) -> Tuple[str, float]:
        """CNN inference on a single image."""
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        image = Image.open(image_path).convert('L')
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = self.class_names[predicted.item()]
        confidence_score = confidence.item() * 100

        return predicted_class, confidence_score

    def _predict_rf(self, image_path: str) -> Tuple[str, float]:
        """RF inference on a single image (simplified feature extraction)."""
        img = Image.open(image_path).convert("L")
        arr = np.array(img, dtype=np.float32)

        gx, gy = np.gradient(arr)
        edge = np.mean(np.sqrt(gx**2 + gy**2))

        features = [arr.mean(), arr.std(), edge, 0.0, 0.0, 0.0]

        pred = self.model.predict([features])[0]
        proba = self.model.predict_proba([features])
        confidence = proba.max() * 100

        return pred, confidence

    def predict_batch(self, folder: str, progress_callback: Callable = None,
                      stop_flag: threading.Event = None) -> List[Tuple[str, str, float]]:
        """Run batch inference on all images in a folder."""
        image_files = glob.glob(os.path.join(folder, "*.png"))
        image_files.extend(glob.glob(os.path.join(folder, "*.jpg")))

        results = []

        for i, img_path in enumerate(sorted(image_files)):
            if stop_flag and stop_flag.is_set():
                break

            pred_class, confidence = self.predict_single(img_path)
            filename = os.path.basename(img_path)
            results.append((filename, pred_class, confidence))

            if progress_callback:
                progress = (i + 1) / len(image_files) * 100
                progress_callback(progress, f"{filename}: {pred_class} ({confidence:.1f}%)")

        return results


# ============================================================================
#                             Feature Visualizer
# ============================================================================

class Visualizer:
    """Feature visualizer for speckle class separability analysis."""

    def __init__(self, data_dir: str, group_size: int = 5):
        self.data_dir = data_dir
        self.group_size = group_size
        self.X = None
        self.y = None

    def extract_features(self, progress_callback: Callable = None,
                         stop_flag: threading.Event = None):
        """Extract features for visualization."""
        folders = sorted([
            f for f in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, f))
        ])

        X, y = [], []

        for idx, folder in enumerate(folders):
            if stop_flag and stop_flag.is_set():
                break

            label = chr(ord('A') + idx)
            folder_path = os.path.join(self.data_dir, folder)

            imgs = []
            for f in sorted(os.listdir(folder_path)):
                if f.lower().endswith((".png", ".jpg")):
                    try:
                        img = np.array(Image.open(os.path.join(folder_path, f)).convert("L"))
                        imgs.append(img)
                    except Exception:
                        pass

            for i in range(0, len(imgs) - self.group_size + 1, self.group_size):
                group = imgs[i:i + self.group_size]
                if len(group) == self.group_size:
                    feat = self._extract_group_features(group)
                    X.append(feat)
                    y.append(label)

            if progress_callback:
                progress = (idx + 1) / len(folders) * 100
                progress_callback(progress, f"Processing: {folder}")

        self.X = np.array(X)
        self.y = np.array(y)

        return len(X)

    def _extract_group_features(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract group-level statistical features."""
        stack = np.stack(images)
        mean = stack.mean()
        std = stack.std()

        edges = []
        for img in images:
            gx, gy = np.gradient(img.astype(np.float32))
            edge = np.mean(np.sqrt(gx**2 + gy**2))
            edges.append(edge)
        edge_density = np.mean(edges)

        return np.array([mean, std, edge_density])

    def create_3d_scatter(self) -> Figure:
        """Create 3D scatter plot of features."""
        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        colors = plt.cm.tab20(np.linspace(0, 1, len(set(self.y))))
        label_color = dict(zip(sorted(set(self.y)), colors))

        for label in sorted(set(self.y)):
            idxs = self.y == label
            ax.scatter(
                self.X[idxs, 0], self.X[idxs, 1], self.X[idxs, 2],
                label=label, color=label_color[label], s=30
            )

        ax.set_xlabel("Mean Gray")
        ax.set_ylabel("Std Gray")
        ax.set_zlabel("Edge Density")
        ax.set_title("3D Feature Distribution (A-Z)")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()

        return fig

    def create_2d_projections(self) -> Figure:
        """Create 2D projection plots."""
        fig = Figure(figsize=(15, 4))

        pairs = [(0, 1), (0, 2), (1, 2)]
        names = ["Mean", "Std", "Edge"]

        for i, (a, b) in enumerate(pairs):
            ax = fig.add_subplot(1, 3, i + 1)
            for label in sorted(set(self.y)):
                idxs = self.y == label
                ax.scatter(self.X[idxs, a], self.X[idxs, b], label=label, s=20)
            ax.set_xlabel(names[a])
            ax.set_ylabel(names[b])
            ax.set_title(f"{names[a]} vs {names[b]}")

        fig.tight_layout()
        return fig

    def create_distance_heatmap(self) -> Figure:
        """Create inter-class feature distance heatmap."""
        centers = {}
        for label in sorted(set(self.y)):
            centers[label] = self.X[self.y == label].mean(axis=0)

        labels = sorted(centers.keys())
        dist_matrix = np.zeros((len(labels), len(labels)))

        for i, la in enumerate(labels):
            for j, lb in enumerate(labels):
                dist_matrix[i, j] = np.linalg.norm(centers[la] - centers[lb])

        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        im = ax.imshow(dist_matrix, cmap="viridis")
        fig.colorbar(im, ax=ax, label="Euclidean Distance")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_title("Inter-class Feature Distance")
        fig.tight_layout()

        return fig


# ============================================================================
#                              GUI Widgets
# ============================================================================

class FolderSelector(ttk.Frame):
    """Folder selection widget."""

    def __init__(self, parent, label: str, on_select: Callable = None):
        super().__init__(parent)
        self.on_select = on_select

        ttk.Label(self, text=label, width=12).pack(side=tk.LEFT, padx=(0, 5))

        self.path_var = tk.StringVar()
        self.entry = ttk.Entry(self, textvariable=self.path_var, width=50)
        self.entry.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)

        ttk.Button(self, text="Browse...", command=self._browse).pack(side=tk.LEFT)

        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(self, textvariable=self.status_var, foreground="gray")
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))

    def _browse(self):
        path = filedialog.askdirectory()
        if path:
            self.path_var.set(path)
            if self.on_select:
                self.on_select(path)

    def get_path(self) -> str:
        return self.path_var.get()

    def set_status(self, text: str, color: str = "gray"):
        self.status_var.set(text)
        self.status_label.config(foreground=color)


class FileSelector(ttk.Frame):
    """File selection widget."""

    def __init__(self, parent, label: str, filetypes: List[Tuple[str, str]] = None,
                 on_select: Callable = None):
        super().__init__(parent)
        self.on_select = on_select
        self.filetypes = filetypes or [("All files", "*.*")]

        ttk.Label(self, text=label, width=12).pack(side=tk.LEFT, padx=(0, 5))

        self.path_var = tk.StringVar()
        self.entry = ttk.Entry(self, textvariable=self.path_var, width=50)
        self.entry.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)

        ttk.Button(self, text="Browse...", command=self._browse).pack(side=tk.LEFT)

        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(self, textvariable=self.status_var, foreground="gray")
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))

    def _browse(self):
        path = filedialog.askopenfilename(filetypes=self.filetypes)
        if path:
            self.path_var.set(path)
            if self.on_select:
                self.on_select(path)

    def get_path(self) -> str:
        return self.path_var.get()

    def set_status(self, text: str, color: str = "gray"):
        self.status_var.set(text)
        self.status_label.config(foreground=color)


class LogWidget(ttk.Frame):
    """Log output widget."""

    def __init__(self, parent, height: int = 10):
        super().__init__(parent)

        self.text = tk.Text(self, height=height, wrap=tk.WORD, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)

        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def log(self, message: str):
        self.text.configure(state=tk.NORMAL)
        self.text.insert(tk.END, message + "\n")
        self.text.see(tk.END)
        self.text.configure(state=tk.DISABLED)

    def clear(self):
        self.text.configure(state=tk.NORMAL)
        self.text.delete(1.0, tk.END)
        self.text.configure(state=tk.DISABLED)


# ============================================================================
#                              Tabs
# ============================================================================

class VideoTab(ttk.Frame):
    """Video frame extraction tab."""

    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self.processor = VideoProcessor()
        self.videos = []
        self.task = None
        self._create_widgets()

    def _create_widgets(self):
        self.video_selector = FolderSelector(self, "Video folder:", self._on_video_folder_select)
        self.video_selector.pack(fill=tk.X, pady=5)

        self.output_selector = FolderSelector(self, "Output folder:")
        self.output_selector.pack(fill=tk.X, pady=5)

        params_frame = ttk.LabelFrame(self, text="Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=10)

        ttk.Label(params_frame, text="Frames per video:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.frames_var = tk.StringVar(value="100")
        ttk.Entry(params_frame, textvariable=self.frames_var, width=10).grid(row=0, column=1, padx=5)

        ttk.Label(params_frame, text="Image format:").grid(row=0, column=2, padx=(20, 5))
        self.format_var = tk.StringVar(value="png")
        ttk.Radiobutton(params_frame, text="PNG", variable=self.format_var, value="png").grid(row=0, column=3)
        ttk.Radiobutton(params_frame, text="JPG", variable=self.format_var, value="jpg").grid(row=0, column=4)

        list_frame = ttk.LabelFrame(self, text="Video list", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        columns = ("name", "duration", "size")
        self.video_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=8)
        self.video_tree.heading("name", text="Video name")
        self.video_tree.heading("duration", text="Duration")
        self.video_tree.heading("size", text="Size (MB)")
        self.video_tree.column("name", width=300)
        self.video_tree.column("duration", width=100)
        self.video_tree.column("size", width=100)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.video_tree.yview)
        self.video_tree.configure(yscrollcommand=scrollbar.set)

        self.video_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, pady=5)

        self.start_btn = ttk.Button(btn_frame, text="Start Extraction", command=self._start_extraction)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self._stop_extraction, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, pady=5)

        self.log = LogWidget(self, height=6)
        self.log.pack(fill=tk.BOTH, expand=True, pady=5)

    def _on_video_folder_select(self, path: str):
        self.videos = self.processor.scan_folder(path)

        for item in self.video_tree.get_children():
            self.video_tree.delete(item)

        for video in self.videos:
            duration = f"{int(video.duration//60):02d}:{int(video.duration%60):02d}" if video.duration else "Unknown"
            self.video_tree.insert("", tk.END, values=(video.name, duration, f"{video.size_mb:.1f}"))

        self.video_selector.set_status(f"Found {len(self.videos)} video(s)", "green")

    def _start_extraction(self):
        video_dir = self.video_selector.get_path()
        output_dir = self.output_selector.get_path()

        if not video_dir or not output_dir:
            messagebox.showerror("Error", "Please select a video folder and an output folder.")
            return

        if not self.videos:
            messagebox.showerror("Error", "No video files found.")
            return

        try:
            target_frames = int(self.frames_var.get())
        except ValueError:
            messagebox.showerror("Error", "Frame count must be an integer.")
            return

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.log.clear()
        self.progress_var.set(0)

        def run_extraction(progress_callback, stop_flag):
            return self.processor.extract_all(
                self.videos, output_dir, target_frames,
                self.format_var.get(), progress_callback, stop_flag
            )

        self.task = BackgroundTask(
            self.winfo_toplevel(),
            run_extraction,
            on_progress=self._on_progress,
            on_complete=self._on_complete,
            on_error=self._on_error
        )
        self.task.start()

    def _stop_extraction(self):
        if self.task:
            self.task.stop()

    def _on_progress(self, progress: float, message: str):
        self.progress_var.set(progress)
        self.log.log(message)

    def _on_complete(self, total: int):
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress_var.set(100)
        self.log.log(f"\nDone! Extracted {total} frames total.")
        messagebox.showinfo("Done", f"Frame extraction complete!\nExtracted {total} frames.")

    def _on_error(self, error: str):
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.log.log(f"Error: {error}")
        messagebox.showerror("Error", error)


class TrainingTab(ttk.Frame):
    """Model training tab."""

    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self.trainer = None
        self.task = None
        self._create_widgets()

    def _create_widgets(self):
        self.data_selector = FolderSelector(self, "Training data:", self._on_data_folder_select)
        self.data_selector.pack(fill=tk.X, pady=5)

        type_frame = ttk.LabelFrame(self, text="Model type", padding=10)
        type_frame.pack(fill=tk.X, pady=10)

        self.model_type_var = tk.StringVar(value="cnn")
        ttk.Radiobutton(type_frame, text="CNN (Deep Learning)", variable=self.model_type_var,
                        value="cnn", command=self._toggle_params).pack(side=tk.LEFT, padx=20)
        ttk.Radiobutton(type_frame, text="Random Forest (Traditional ML)", variable=self.model_type_var,
                        value="rf", command=self._toggle_params).pack(side=tk.LEFT, padx=20)

        self.params_frame = ttk.Frame(self)
        self.params_frame.pack(fill=tk.X, pady=10)

        self.cnn_frame = ttk.LabelFrame(self.params_frame, text="CNN Parameters", padding=10)

        params = [
            ("Image size:", "image_size", "128"),
            ("Batch size:", "batch_size", "32"),
            ("Epochs:", "epochs", "50"),
            ("Learning rate:", "lr", "0.001"),
            ("Train ratio:", "train_split", "0.8"),
            ("ES patience:", "patience", "8"),
        ]

        self.cnn_vars = {}
        for i, (label, key, default) in enumerate(params):
            ttk.Label(self.cnn_frame, text=label).grid(row=i//3, column=(i%3)*2, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar(value=default)
            self.cnn_vars[key] = var
            ttk.Entry(self.cnn_frame, textvariable=var, width=10).grid(row=i//3, column=(i%3)*2+1, padx=5, pady=2)

        ttk.Label(self.cnn_frame, text="Device:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.device_var = tk.StringVar(value="auto")
        device_combo = ttk.Combobox(self.cnn_frame, textvariable=self.device_var, width=15, state="readonly")

        device_options = ["auto (auto-detect)", "cpu"]
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                device_options.append(f"cuda:{i} ({gpu_name[:20]})")

        device_combo['values'] = device_options
        device_combo.current(0)
        device_combo.grid(row=2, column=1, columnspan=3, sticky=tk.W, padx=5, pady=2)

        self.cnn_frame.pack(fill=tk.X)

        self.rf_frame = ttk.LabelFrame(self.params_frame, text="Random Forest Parameters", padding=10)

        ttk.Label(self.rf_frame, text="Num trees:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.rf_trees_var = tk.StringVar(value="400")
        ttk.Entry(self.rf_frame, textvariable=self.rf_trees_var, width=10).grid(row=0, column=1, padx=5)

        save_frame = ttk.LabelFrame(self, text="Model save", padding=10)
        save_frame.pack(fill=tk.X, pady=10)

        ttk.Label(save_frame, text="Save path:").pack(side=tk.LEFT, padx=5)
        self.save_path_var = tk.StringVar()
        ttk.Entry(save_frame, textvariable=self.save_path_var, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(save_frame, text="Browse...", command=self._browse_save_path).pack(side=tk.LEFT, padx=5)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, pady=5)

        self.start_btn = ttk.Button(btn_frame, text="Start Training", command=self._start_training)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self._stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        gpu_status = "Available" if torch.cuda.is_available() else "Unavailable"
        gpu_color = "green" if torch.cuda.is_available() else "red"
        self.gpu_label = ttk.Label(btn_frame, text=f"GPU: {gpu_status}", foreground=gpu_color)
        self.gpu_label.pack(side=tk.RIGHT, padx=10)

        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, pady=5)

        self.log = LogWidget(self, height=10)
        self.log.pack(fill=tk.BOTH, expand=True, pady=5)

    def _toggle_params(self):
        if self.model_type_var.get() == "cnn":
            self.rf_frame.pack_forget()
            self.cnn_frame.pack(fill=tk.X)
        else:
            self.cnn_frame.pack_forget()
            self.rf_frame.pack(fill=tk.X)

    def _on_data_folder_select(self, path: str):
        try:
            folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

            total_images = 0
            for folder in folders:
                folder_path = os.path.join(path, folder)
                imgs = glob.glob(os.path.join(folder_path, "*.png")) + \
                       glob.glob(os.path.join(folder_path, "*.jpg"))
                total_images += len(imgs)

            self.data_selector.set_status(f"{len(folders)} classes, {total_images} images", "green")
        except Exception as e:
            self.data_selector.set_status(f"Error: {e}", "red")

    def _browse_save_path(self):
        if self.model_type_var.get() == "cnn":
            path = filedialog.asksaveasfilename(
                defaultextension=".pth",
                filetypes=[("PyTorch model", "*.pth")]
            )
        else:
            path = filedialog.asksaveasfilename(
                defaultextension=".joblib",
                filetypes=[("Joblib model", "*.joblib")]
            )
        if path:
            self.save_path_var.set(path)

    def _start_training(self):
        data_dir = self.data_selector.get_path()
        save_path = self.save_path_var.get()

        if not data_dir:
            messagebox.showerror("Error", "Please select a training data folder.")
            return

        if not save_path:
            messagebox.showerror("Error", "Please select a model save path.")
            return

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.log.clear()
        self.progress_var.set(0)

        model_type = self.model_type_var.get()

        if model_type == "cnn":
            self._start_cnn_training(data_dir, save_path)
        else:
            self._start_rf_training(data_dir, save_path)

    def _start_cnn_training(self, data_dir: str, save_path: str):
        try:
            image_size = int(self.cnn_vars["image_size"].get())
            batch_size = int(self.cnn_vars["batch_size"].get())
            epochs = int(self.cnn_vars["epochs"].get())
            lr = float(self.cnn_vars["lr"].get())
            train_split = float(self.cnn_vars["train_split"].get())
            patience = int(self.cnn_vars["patience"].get())
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter: {e}")
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            return

        device_selection = self.device_var.get()
        self.trainer = CNNTrainer(data_dir, device=device_selection)
        self.log.log(f"Using device: {self.trainer.device}")

        def run_training(progress_callback, stop_flag):
            train_count, val_count = self.trainer.prepare_data(image_size, batch_size, train_split)
            progress_callback(5, f"Data ready: train={train_count}, val={val_count}")

            history = self.trainer.train(epochs, lr, patience, image_size, progress_callback, stop_flag)

            self.trainer.save_model(save_path, image_size)
            return history

        self.task = BackgroundTask(
            self.winfo_toplevel(),
            run_training,
            on_progress=self._on_progress,
            on_complete=self._on_cnn_complete,
            on_error=self._on_error
        )
        self.task.start()

    def _start_rf_training(self, data_dir: str, save_path: str):
        try:
            n_estimators = int(self.rf_trees_var.get())
        except ValueError:
            messagebox.showerror("Error", "Number of trees must be an integer.")
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            return

        self.trainer = RFTrainer(data_dir)

        def run_training(progress_callback, stop_flag):
            history = self.trainer.train(n_estimators, 0.2, progress_callback, stop_flag)
            self.trainer.save_model(save_path)
            return history

        self.task = BackgroundTask(
            self.winfo_toplevel(),
            run_training,
            on_progress=self._on_progress,
            on_complete=self._on_rf_complete,
            on_error=self._on_error
        )
        self.task.start()

    def _stop_training(self):
        if self.task:
            self.task.stop()
        if self.trainer:
            self.trainer.request_stop()

    def _on_progress(self, progress: float, message: str):
        self.progress_var.set(progress)
        self.log.log(message)

    def _on_cnn_complete(self, history: dict):
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress_var.set(100)

        best_acc = history.get("best_acc", 0)
        self.log.log(f"\nTraining complete! Best val accuracy: {best_acc:.2f}%")
        self.log.log(f"Model saved to: {self.save_path_var.get()}")
        messagebox.showinfo("Done", f"CNN training complete!\nBest val accuracy: {best_acc:.2f}%")

    def _on_rf_complete(self, history: dict):
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress_var.set(100)

        train_acc = history.get("train_acc", 0)
        self.log.log(f"\nTraining complete! Accuracy: {train_acc:.2f}%")
        self.log.log(f"Model saved to: {self.save_path_var.get()}")
        messagebox.showinfo("Done", f"Random Forest training complete!\nAccuracy: {train_acc:.2f}%")

    def _on_error(self, error: str):
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.log.log(f"Error: {error}")
        messagebox.showerror("Error", error)


class PredictTab(ttk.Frame):
    """Model prediction tab."""

    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self.predictor = Predictor()
        self.task = None
        self._create_widgets()

    def _create_widgets(self):
        self.model_selector = FileSelector(
            self, "Model file:",
            filetypes=[("Model files", "*.pth *.joblib"), ("PyTorch", "*.pth"), ("Joblib", "*.joblib")],
            on_select=self._on_model_select
        )
        self.model_selector.pack(fill=tk.X, pady=5)

        input_frame = ttk.LabelFrame(self, text="Input mode", padding=10)
        input_frame.pack(fill=tk.X, pady=10)

        self.input_mode_var = tk.StringVar(value="folder")
        ttk.Radiobutton(input_frame, text="Batch (folder)", variable=self.input_mode_var,
                        value="folder", command=self._toggle_input).pack(side=tk.LEFT, padx=20)
        ttk.Radiobutton(input_frame, text="Single image", variable=self.input_mode_var,
                        value="single", command=self._toggle_input).pack(side=tk.LEFT, padx=20)

        self.input_frame = ttk.Frame(self)
        self.input_frame.pack(fill=tk.X, pady=5)

        self.folder_selector = FolderSelector(self.input_frame, "Image folder:")
        self.folder_selector.pack(fill=tk.X)

        self.image_selector = FileSelector(
            self.input_frame, "Image file:",
            filetypes=[("Images", "*.png *.jpg *.jpeg")]
        )

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, pady=10)

        self.start_btn = ttk.Button(btn_frame, text="Start Prediction", command=self._start_prediction)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self._stop_prediction, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, pady=5)

        result_frame = ttk.LabelFrame(self, text="Prediction results", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        columns = ("filename", "prediction", "confidence")
        self.result_tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=12)
        self.result_tree.heading("filename", text="Filename")
        self.result_tree.heading("prediction", text="Prediction")
        self.result_tree.heading("confidence", text="Confidence")
        self.result_tree.column("filename", width=250)
        self.result_tree.column("prediction", width=100)
        self.result_tree.column("confidence", width=100)

        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=scrollbar.set)

        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.stats_label = ttk.Label(self, text="")
        self.stats_label.pack(fill=tk.X, pady=5)

    def _toggle_input(self):
        if self.input_mode_var.get() == "folder":
            self.image_selector.pack_forget()
            self.folder_selector.pack(fill=tk.X)
        else:
            self.folder_selector.pack_forget()
            self.image_selector.pack(fill=tk.X)

    def _on_model_select(self, path: str):
        try:
            info = self.predictor.load_model(path)
            self.model_selector.set_status(f"{info['type']}, {info['classes']} classes", "green")
        except Exception as e:
            self.model_selector.set_status(f"Load failed: {e}", "red")

    def _start_prediction(self):
        model_path = self.model_selector.get_path()

        if not model_path:
            messagebox.showerror("Error", "Please select a model file.")
            return

        if self.predictor.model is None:
            try:
                self.predictor.load_model(model_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
                return

        # Clear previous results
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)

        if self.input_mode_var.get() == "single":
            self._predict_single()
        else:
            self._predict_batch()

    def _predict_single(self):
        image_path = self.image_selector.get_path()

        if not image_path:
            messagebox.showerror("Error", "Please select an image file.")
            return

        try:
            pred_class, confidence = self.predictor.predict_single(image_path)
            filename = os.path.basename(image_path)

            self.result_tree.insert("", tk.END, values=(filename, pred_class, f"{confidence:.2f}%"))
            self.stats_label.config(text=f"Prediction: {pred_class}, Confidence: {confidence:.2f}%")

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")

    def _predict_batch(self):
        folder_path = self.folder_selector.get_path()

        if not folder_path:
            messagebox.showerror("Error", "Please select an image folder.")
            return

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress_var.set(0)

        def run_prediction(progress_callback, stop_flag):
            return self.predictor.predict_batch(folder_path, progress_callback, stop_flag)

        self.task = BackgroundTask(
            self.winfo_toplevel(),
            run_prediction,
            on_progress=self._on_progress,
            on_complete=self._on_complete,
            on_error=self._on_error
        )
        self.task.start()

    def _stop_prediction(self):
        if self.task:
            self.task.stop()

    def _on_progress(self, progress: float, message: str):
        self.progress_var.set(progress)

    def _on_complete(self, results: List[Tuple[str, str, float]]):
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress_var.set(100)

        stats = {}
        for filename, pred_class, confidence in results:
            self.result_tree.insert("", tk.END, values=(filename, pred_class, f"{confidence:.2f}%"))
            stats[pred_class] = stats.get(pred_class, 0) + 1

        stats_text = ", ".join([f"{k}: {v}" for k, v in sorted(stats.items())])
        self.stats_label.config(text=f"Total {len(results)} | {stats_text}")

    def _on_error(self, error: str):
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        messagebox.showerror("Error", error)


class VisualizeTab(ttk.Frame):
    """Feature visualization tab."""

    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self.visualizer = None
        self.figures = []
        self.current_fig_idx = 0
        self._create_widgets()

    def _create_widgets(self):
        self.data_selector = FolderSelector(self, "Data folder:", self._on_data_select)
        self.data_selector.pack(fill=tk.X, pady=5)

        params_frame = ttk.LabelFrame(self, text="Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=10)

        ttk.Label(params_frame, text="Group size:").pack(side=tk.LEFT, padx=5)
        self.group_size_var = tk.StringVar(value="5")
        ttk.Entry(params_frame, textvariable=self.group_size_var, width=10).pack(side=tk.LEFT, padx=5)

        options_frame = ttk.LabelFrame(self, text="Visualization types", padding=10)
        options_frame.pack(fill=tk.X, pady=10)

        self.show_3d_var = tk.BooleanVar(value=True)
        self.show_2d_var = tk.BooleanVar(value=True)
        self.show_heatmap_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(options_frame, text="3D Scatter", variable=self.show_3d_var).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(options_frame, text="2D Projections", variable=self.show_2d_var).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(options_frame, text="Distance Heatmap", variable=self.show_heatmap_var).pack(side=tk.LEFT, padx=10)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, pady=5)

        self.generate_btn = ttk.Button(btn_frame, text="Generate Visualization", command=self._generate)
        self.generate_btn.pack(side=tk.LEFT, padx=5)

        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, pady=5)

        self.canvas_frame = ttk.LabelFrame(self, text="Visualization results", padding=10)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        nav_frame = ttk.Frame(self)
        nav_frame.pack(fill=tk.X, pady=5)

        self.prev_btn = ttk.Button(nav_frame, text="< Prev", command=self._prev_fig, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=5)

        self.fig_label = ttk.Label(nav_frame, text="No visualization")
        self.fig_label.pack(side=tk.LEFT, padx=20)

        self.next_btn = ttk.Button(nav_frame, text="Next >", command=self._next_fig, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(nav_frame, text="Save current figure", command=self._save_current).pack(side=tk.RIGHT, padx=5)

    def _on_data_select(self, path: str):
        try:
            folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
            self.data_selector.set_status(f"Found {len(folders)} class(es)", "green")
        except Exception as e:
            self.data_selector.set_status(f"Error: {e}", "red")

    def _generate(self):
        data_dir = self.data_selector.get_path()

        if not data_dir:
            messagebox.showerror("Error", "Please select a data folder.")
            return

        try:
            group_size = int(self.group_size_var.get())
        except ValueError:
            messagebox.showerror("Error", "Group size must be an integer.")
            return

        self.generate_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)

        self.visualizer = Visualizer(data_dir, group_size)

        def run_visualization(progress_callback, stop_flag):
            count = self.visualizer.extract_features(progress_callback, stop_flag)

            figs = []
            if self.show_3d_var.get():
                figs.append(("3D Scatter", self.visualizer.create_3d_scatter()))
            if self.show_2d_var.get():
                figs.append(("2D Projections", self.visualizer.create_2d_projections()))
            if self.show_heatmap_var.get():
                figs.append(("Distance Heatmap", self.visualizer.create_distance_heatmap()))

            return figs

        task = BackgroundTask(
            self.winfo_toplevel(),
            run_visualization,
            on_progress=self._on_progress,
            on_complete=self._on_complete,
            on_error=self._on_error
        )
        task.start()

    def _on_progress(self, progress: float, message: str):
        self.progress_var.set(progress)

    def _on_complete(self, figures: List[Tuple[str, Figure]]):
        self.generate_btn.config(state=tk.NORMAL)
        self.progress_var.set(100)

        self.figures = figures
        self.current_fig_idx = 0

        if self.figures:
            self._show_current_fig()
            self.prev_btn.config(state=tk.NORMAL)
            self.next_btn.config(state=tk.NORMAL)

    def _on_error(self, error: str):
        self.generate_btn.config(state=tk.NORMAL)
        messagebox.showerror("Error", error)

    def _show_current_fig(self):
        if not self.figures:
            return

        name, fig = self.figures[self.current_fig_idx]

        # Update canvas
        self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig_label.config(text=f"{name} ({self.current_fig_idx + 1}/{len(self.figures)})")

    def _prev_fig(self):
        if self.figures:
            self.current_fig_idx = (self.current_fig_idx - 1) % len(self.figures)
            self._show_current_fig()

    def _next_fig(self):
        if self.figures:
            self.current_fig_idx = (self.current_fig_idx + 1) % len(self.figures)
            self._show_current_fig()

    def _save_current(self):
        if not self.figures:
            messagebox.showwarning("Warning", "No figure to save.")
            return

        name, fig = self.figures[self.current_fig_idx]
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF file", "*.pdf")],
            initialfile=f"{name}.png"
        )

        if path:
            fig.savefig(path, dpi=150, bbox_inches='tight')
            messagebox.showinfo("Saved", f"Figure saved to: {path}")


# ============================================================================
#                              Main Application
# ============================================================================

class DeepLearningApp:
    """Speckle-PUF Deep Learning Suite main application."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self._setup_window()
        self._create_menu()
        self._create_tabs()
        self._create_status_bar()

    def _setup_window(self):
        self.root.title("Speckle-PUF Deep Learning Suite")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)

        try:
            self.root.iconbitmap("icon.ico")
        except Exception:
            pass

    def _create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Exit", command=self.root.quit)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _create_tabs(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.video_tab = VideoTab(self.notebook)
        self.training_tab = TrainingTab(self.notebook)
        self.predict_tab = PredictTab(self.notebook)
        self.visualize_tab = VisualizeTab(self.notebook)

        self.notebook.add(self.video_tab, text="Video Extraction")
        self.notebook.add(self.training_tab, text="Model Training")
        self.notebook.add(self.predict_tab, text="Model Prediction")
        self.notebook.add(self.visualize_tab, text="Feature Visualization")

    def _create_status_bar(self):
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)

        ttk.Label(status_frame, text="Ready").pack(side=tk.LEFT)

        gpu_status = "GPU: Available" if torch.cuda.is_available() else "GPU: Unavailable"
        gpu_color = "green" if torch.cuda.is_available() else "gray"
        ttk.Label(status_frame, text=gpu_status, foreground=gpu_color).pack(side=tk.RIGHT)

    def _show_about(self):
        messagebox.showinfo(
            "About",
            "Speckle-PUF Deep Learning Suite v1.0\n\n"
            "Features:\n"
            "- Video frame extraction\n"
            "- CNN / Random Forest model training\n"
            "- Model prediction\n"
            "- Feature visualization\n\n"
            f"PyTorch: {torch.__version__}\n"
            f"GPU: {'Available' if torch.cuda.is_available() else 'Unavailable'}"
        )


def main():
    root = tk.Tk()
    app = DeepLearningApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
