import os
import glob
import string
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Use relative paths (relative to this script's directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "screenshots")


def extract_video_features(folder):
    frames = sorted(glob.glob(os.path.join(folder, "*.png")))

    gray_means = []
    gray_stds = []
    edge_density = []

    for f in frames:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        gray_means.append(img.mean())
        gray_stds.append(img.std())

        edges = cv2.Canny(img, 50, 150)
        edge_density.append(edges.mean() / 255.0)

    gray_means = np.array(gray_means)
    gray_stds = np.array(gray_stds)
    edge_density = np.array(edge_density)

    diff = np.diff(gray_means)
    fft = np.fft.rfft(gray_means - gray_means.mean())

    return [
        gray_means.mean(),
        gray_stds.mean(),
        edge_density.mean(),
        diff.mean(),
        diff.std(),
        np.mean(np.abs(fft))
    ]


def main():
    folders = sorted([
        f for f in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, f))
    ])

    assert len(folders) == 26, "Expected 26 videos (A-Z)"

    letters = list(string.ascii_uppercase)

    X, y = [], []

    print("Extracting CAD image sequence features...")
    for folder, letter in zip(folders, letters):
        path = os.path.join(DATA_DIR, folder)
        feat = extract_video_features(path)
        X.append(feat)
        y.append(letter)

        print(f"{folder} -> {letter}, features = {np.round(feat, 3)}")

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

    for gt, pr in zip(y, preds):
        print(f"{gt} -> {pr}")


if __name__ == "__main__":
    main()
