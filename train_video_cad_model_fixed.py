import os
import glob
import string
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ================= Paths =================
# Use relative paths (relative to this script's directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "screenshots")
# ======================================


def extract_video_features(folder):
    frames = sorted(
        glob.glob(os.path.join(folder, "*.png")) +
        glob.glob(os.path.join(folder, "*.jpg"))
    )

    if len(frames) == 0:
        raise RuntimeError(f"No images in folder: {folder}")

    gray_means = []
    gray_stds = []
    edge_strength = []

    for f in frames:
        img = Image.open(f).convert("L")
        arr = np.array(img, dtype=np.float32)

        gray_means.append(arr.mean())
        gray_stds.append(arr.std())

        # Approximate edge strength via gradients (instead of cv2.Canny)
        gx, gy = np.gradient(arr)
        edge_strength.append(np.mean(np.sqrt(gx**2 + gy**2)))

    gray_means = np.array(gray_means)
    gray_stds = np.array(gray_stds)
    edge_strength = np.array(edge_strength)

    # Temporal change features
    diff = np.diff(gray_means)

    # FFT (safe when few frames)
    if len(gray_means) > 1:
        fft = np.fft.rfft(gray_means - gray_means.mean())
        fft_energy = np.mean(np.abs(fft))
    else:
        fft_energy = 0.0

    return [
        gray_means.mean(),        # mean brightness
        gray_stds.mean(),         # contrast
        edge_strength.mean(),     # structure strength (CAD edges)
        diff.mean() if len(diff) else 0.0,
        diff.std() if len(diff) else 0.0,
        fft_energy               # temporal pattern
    ]


def main():
    folders = sorted([
        f for f in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, f))
    ])

    assert len(folders) == 26, f"Expected 26 videos, found {len(folders)}"

    letters = list(string.ascii_uppercase)

    X, y = [], []

    print("Extracting CAD image sequence features...")
    for folder, letter in zip(folders, letters):
        path = os.path.join(DATA_DIR, folder)
        feat = extract_video_features(path)
        X.append(feat)
        y.append(letter)

        print(f"{folder} -> {letter}, features = {np.round(feat, 3)}")

    X = np.array(X)
    y = np.array(y)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            random_state=42
        ))
    ])

    model.fit(X, y)
    preds = model.predict(X)

    acc = (preds == y).mean() * 100
    print("\nVideo-level CAD recognition complete")
    print(f"Training set accuracy: {acc:.2f}%")

    print("\nPredictions:")
    for gt, pr in zip(y, preds):
        print(f"{gt} -> {pr}")


if __name__ == "__main__":
    main()
