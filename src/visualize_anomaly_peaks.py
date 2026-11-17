
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from transformers import Dinov2Model


# -----------------------------------
# 1. Load DINOv2 and FAISS
# -----------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Dinov2Model.from_pretrained(
    "facebook/dinov2-small"
).to(device)

model.eval()

index = faiss.read_index(
    "outputs/features/screw_faiss.index"
)


# -----------------------------------
# 2. Load image
# -----------------------------------

image_path = Path(
    "data/screw/test/scratch_head/000.png"
)

image = Image.open(image_path).convert("RGB")
image = image.resize((518, 518))
image_array = np.array(image)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------------
# 3. Calculate anomaly heatmap
# -----------------------------------

image_tensor = preprocess(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(pixel_values=image_tensor)
    patches = outputs.last_hidden_state[:, 1:, :].squeeze(0)

query_vectors = np.ascontiguousarray(
    patches.cpu().numpy(),
    dtype=np.float32
)

distances, _ = index.search(query_vectors, 1)

scores = torch.from_numpy(
    np.sqrt(np.maximum(distances[:, 0], 0))
)

anomaly_map = F.interpolate(
    scores.reshape(1, 1, 37, 37),
    size=(518, 518),
    mode="bilinear",
    align_corners=False
).squeeze().numpy()


# -----------------------------------
# 4. Find top 5 separated anomaly peaks
# -----------------------------------

working_map = anomaly_map.copy()

peaks = []
suppression_radius = 30

for i in range(5):

    y, x = np.unravel_index(
        np.argmax(working_map),
        working_map.shape
    )

    peaks.append((int(x), int(y)))

    # Suppress nearby pixels so the next
    # point comes from a different region
    y1 = max(0, y - suppression_radius)
    y2 = min(518, y + suppression_radius + 1)
    x1 = max(0, x - suppression_radius)
    x2 = min(518, x + suppression_radius + 1)

    working_map[y1:y2, x1:x2] = -np.inf

print("Top anomaly peaks:", peaks)


# -----------------------------------
# 5. Visualize anomaly peaks
# -----------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(image_array)
axes[0].set_title("Original Screw")

axes[1].imshow(image_array)
axes[1].imshow(anomaly_map, cmap="jet", alpha=0.5)

for i, (x, y) in enumerate(peaks):

    axes[1].scatter(
        x, y,
        color="white",
        edgecolors="black",
        s=100
    )

    axes[1].text(
        x + 8,
        y - 8,
        str(i + 1),
        color="black",
        fontsize=12,
        fontweight="bold"
    )

axes[1].set_title("Top 5 Anomaly Peaks")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
