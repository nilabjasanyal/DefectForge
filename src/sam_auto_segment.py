
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from transformers import Dinov2Model
from segment_anything import sam_model_registry, SamPredictor


# -----------------------------------
# 1. Load models
# -----------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

dinov2 = Dinov2Model.from_pretrained(
    "facebook/dinov2-small"
).to(device)

dinov2.eval()

index = faiss.read_index(
    "outputs/features/screw_faiss.index"
)

sam = sam_model_registry["vit_b"](
    checkpoint="models/sam_vit_b_01ec64.pth"
).to(device)

sam.eval()
predictor = SamPredictor(sam)


# -----------------------------------
# 2. Load defective screw image
# -----------------------------------

image_path = Path(
    "data/screw/test/scratch_head/000.png"
)

image = Image.open(image_path).convert("RGB")
image = image.resize((518, 518))
image_array = np.array(image)


# -----------------------------------
# 3. Extract DINOv2 features
# -----------------------------------

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image_tensor = preprocess(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = dinov2(pixel_values=image_tensor)
    patches = outputs.last_hidden_state[:, 1:, :].squeeze(0)


# -----------------------------------
# 4. FAISS anomaly detection
# -----------------------------------

query_vectors = np.ascontiguousarray(
    patches.cpu().numpy(),
    dtype=np.float32
)

distances, indices = index.search(query_vectors, 1)

anomaly_scores = torch.from_numpy(
    np.sqrt(np.maximum(distances[:, 0], 0))
)

anomaly_map = anomaly_scores.reshape(1, 1, 37, 37)

anomaly_map = F.interpolate(
    anomaly_map,
    size=(518, 518),
    mode="bilinear",
    align_corners=False
)

anomaly_map = anomaly_map.squeeze().numpy()



# -----------------------------------
# 5. Find a point inside the actual defect
# Diagnostic experiment using ground truth
# -----------------------------------

mask_path = (
    Path("data/screw/ground_truth/scratch_head")
    / f"{image_path.stem}_mask.png"
)

ground_truth_image = Image.open(mask_path).convert("L")
ground_truth_image = ground_truth_image.resize(
    (518, 518),
    Image.Resampling.NEAREST
)

ground_truth = np.array(ground_truth_image) > 0

# Find all pixels belonging to the actual defect
ys, xs = np.where(ground_truth)

# Choose the approximate center of the defect
x = int(np.median(xs))
y = int(np.median(ys))

print("Ground-truth defect point:", (x, y))

input_point = np.array([[x, y]])
input_label = np.array([1])


# -----------------------------------
# 6. Generate SAM masks
# -----------------------------------


# Create a 50x50 bounding box around
# the highest anomaly point
box_size = 50
half_size = box_size // 2

x1 = max(0, x - half_size)
y1 = max(0, y - half_size)
x2 = min(517, x + half_size)
y2 = min(517, y + half_size)

input_box = np.array([x1, y1, x2, y2])

print("Bounding box:", input_box)

predictor.set_image(image_array)

with torch.no_grad():
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=True
    )

print("SAM confidence scores:", scores)



# -----------------------------------
# 7. Visualize results
# -----------------------------------

from matplotlib.patches import Rectangle

# First create the figure and axes
fig, axes = plt.subplots(1, 5, figsize=(20, 5))

# Original image
axes[0].imshow(image_array)
axes[0].set_title("Original Screw")

# FAISS anomaly heatmap
axes[1].imshow(anomaly_map, cmap="jet")

# Show highest-anomaly point
axes[1].scatter(
    x, y,
    color="white",
    marker="*",
    s=180
)

# Draw bounding box around the anomaly
axes[1].add_patch(
    Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        edgecolor="red",
        facecolor="none",
        linewidth=2
    )
)

axes[1].set_title("FAISS Heatmap + Auto Prompt")

# Display the three SAM segmentation masks
for i in range(3):
    axes[i + 2].imshow(image_array)

    axes[i + 2].imshow(
        np.ma.masked_where(~masks[i], masks[i]),
        cmap="autumn",
        alpha=0.5
    )

    axes[i + 2].scatter(
        x, y,
        color="cyan",
        marker="*",
        s=100
    )

    axes[i + 2].set_title(
        f"SAM Mask {i + 1} ({scores[i]:.3f})"
    )

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()


# -----------------------------------
# 8. Evaluate SAM segmentation with IoU
# -----------------------------------

# Load ground-truth defect mask
mask_path = (
    Path("data/screw/ground_truth/scratch_head")
    / f"{image_path.stem}_mask.png"
)

ground_truth_image = Image.open(mask_path).convert("L")
ground_truth_image = ground_truth_image.resize(
    (518, 518),
    Image.Resampling.NEAREST
)

ground_truth = np.array(ground_truth_image) > 0

print("\n--- SAM Segmentation Evaluation ---")

# Calculate IoU for each SAM mask
for i in range(3):

    predicted_mask = masks[i].astype(bool)

    intersection = np.logical_and(
        predicted_mask, ground_truth
    ).sum()

    union = np.logical_or(
        predicted_mask, ground_truth
    ).sum()

    iou = intersection / union if union > 0 else 0.0

    print(f"SAM Mask {i + 1} IoU: {iou:.4f}")


# -----------------------------------
# 9. Visualize ground truth vs SAM
# -----------------------------------

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(image_array)
axes[0].set_title("Original Screw")

axes[1].imshow(ground_truth, cmap="gray")
axes[1].set_title("Ground Truth Defect")

for i in range(2):
    axes[i + 2].imshow(masks[i], cmap="gray")
    axes[i + 2].set_title(f"SAM Mask {i + 1}")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
