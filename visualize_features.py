import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ============ Parameters ============
# Use relative paths (relative to this script's directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "screenshots")
GROUP_SIZE = 5
# ==============================

def load_gray_image(path):
    try:
        return np.array(Image.open(path).convert("L"))
    except:
        return None

def extract_features(images):
    stack = np.stack(images)
    mean = stack.mean()
    std = stack.std()
    edges = [cv2.Canny(img, 50, 150) for img in images]
    edge_density = np.mean([e.mean() for e in edges])
    return np.array([mean, std, edge_density])

X, y = [], []

folders = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if os.path.isdir(os.path.join(IMAGE_DIR, f))
])

for idx, folder in enumerate(folders):
    label = chr(ord('A') + idx)
    folder_path = os.path.join(IMAGE_DIR, folder)

    imgs = []
    for f in sorted(os.listdir(folder_path)):
        if f.lower().endswith(".png"):
            img = load_gray_image(os.path.join(folder_path, f))
            if img is not None:
                imgs.append(img)

    for i in range(0, len(imgs) - GROUP_SIZE + 1, GROUP_SIZE):
        group = imgs[i:i + GROUP_SIZE]
        if len(group) == GROUP_SIZE:
            X.append(extract_features(group))
            y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Samples: {len(X)}")

# ================= 1. 3D scatter =================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.tab20(np.linspace(0, 1, len(set(y))))
label_color = dict(zip(sorted(set(y)), colors))

for label in sorted(set(y)):
    idxs = y == label
    ax.scatter(
        X[idxs, 0], X[idxs, 1], X[idxs, 2],
        label=label,
        color=label_color[label],
        s=30
    )

ax.set_xlabel("Mean Gray")
ax.set_ylabel("Std Gray")
ax.set_zlabel("Edge Density")
ax.set_title("3D Feature Distribution (A-Z)")
ax.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()

# ================= 2. 2D projections =================
pairs = [(0, 1), (0, 2), (1, 2)]
names = ["Mean", "Std", "Edge"]

plt.figure(figsize=(15, 4))
for i, (a, b) in enumerate(pairs):
    plt.subplot(1, 3, i + 1)
    for label in sorted(set(y)):
        idxs = y == label
        plt.scatter(
            X[idxs, a], X[idxs, b],
            label=label,
            s=20
        )
    plt.xlabel(names[a])
    plt.ylabel(names[b])
    plt.title(f"{names[a]} vs {names[b]}")
plt.tight_layout()
plt.show()

# ================= 3. Inter-class center distances =================
centers = {}
for label in sorted(set(y)):
    centers[label] = X[y == label].mean(axis=0)

labels = sorted(centers.keys())
dist_matrix = np.zeros((len(labels), len(labels)))

for i, la in enumerate(labels):
    for j, lb in enumerate(labels):
        dist_matrix[i, j] = np.linalg.norm(centers[la] - centers[lb])

plt.figure(figsize=(8, 6))
plt.imshow(dist_matrix, cmap="viridis")
plt.colorbar(label="Euclidean Distance")
plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)
plt.title("Inter-class Feature Distance")
plt.tight_layout()
plt.show()
