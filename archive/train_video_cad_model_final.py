import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import cv2  # edge detection only

# ================= Parameters =================
# Use relative paths (relative to this script's directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "screenshots")
GROUP_SIZE = 5
# ======================================

def load_gray_image(path):
    try:
        img = Image.open(path).convert("L")
        return np.array(img)
    except Exception:
        return None

def extract_features(images):
    stack = np.stack(images)

    mean = stack.mean()
    std = stack.std()

    edges = [cv2.Canny(img, 50, 150) for img in images]
    edge_density = np.mean([e.mean() for e in edges])

    return np.array([mean, std, edge_density])

X, y = [], []

print("Extracting CAD image group features...\n")

folders = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if os.path.isdir(os.path.join(IMAGE_DIR, f))
])

for idx, folder in enumerate(folders):
    folder_path = os.path.join(IMAGE_DIR, folder)

    label = chr(ord('A') + idx)

    image_paths = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith('.png')
    ])

    images = []
    for p in image_paths:
        img = load_gray_image(p)
        if img is not None:
            images.append(img)

    if len(images) < GROUP_SIZE:
        print(f"[WARN] {folder}: not enough images, skipping")
        continue

    for i in range(0, len(images) - GROUP_SIZE + 1, GROUP_SIZE):
        group = images[i:i + GROUP_SIZE]
        if len(group) < GROUP_SIZE:
            continue
        feat = extract_features(group)
        X.append(feat)
        y.append(label)

    print(f"{folder} -> {label}, samples {len(images)//GROUP_SIZE}")

print(f"\nTotal samples: {len(X)}")

if len(X) == 0:
    raise RuntimeError("No samples generated (ensure screenshots/ contains PNG files)")

X = np.array(X)
y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

clf = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

clf.fit(X_train, y_train)

print("\nTraining finished")
print(f"Train Acc: {accuracy_score(y_train, clf.predict(X_train)):.2%}")
print(f"Val   Acc: {accuracy_score(y_val, clf.predict(X_val)):.2%}")
print("\nClassification report:")
print(classification_report(y_val, clf.predict(X_val)))
