
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import Dinov2Model
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
import numpy as np

import faiss
import numpy as np

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
# =================================================================
# Load our normal memory bank
# This was for pytorch.cdist() nearest-neighbor search
# memory_bank = torch.load( 
#     "outputs/features/coreset_5000.pt",
#     map_location=device,
#     weights_only=True
# )

# This is for FAISS nearest-neighbor search
# Load our saved FAISS index
index = faiss.read_index(
    "outputs/features/screw_faiss.index"
)

print("FAISS index size:", index.ntotal)
# =================================================================



# Load pretrained DINOv2
model = Dinov2Model.from_pretrained(
    "facebook/dinov2-small"
).to(device)

model.eval()

# Same preprocessing as our training images
preprocess = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Select a defective screw image
image_dir = Path("data/screw/test/scratch_head")
image_path = sorted(image_dir.glob("*.png"))[0]

print("Testing image:", image_path)

image = Image.open(image_path).convert("RGB")
image_tensor = preprocess(image).unsqueeze(0).to(device)

# Extract patch embeddings
with torch.no_grad():
    outputs = model(pixel_values=image_tensor)

patch_embeddings = outputs.last_hidden_state[:, 1:, :]
patch_embeddings = patch_embeddings.squeeze(0)

print("Patch embeddings:", patch_embeddings.shape)

# ========================================================================
# Compare every test patch with our memory bank (Pytorch nearest-neighbor search)
# with torch.no_grad():
#     distances = torch.cdist(
#         patch_embeddings.unsqueeze(0),
#         memory_bank.unsqueeze(0)
#     ).squeeze(0)
# Find the nearest normal embedding for every test patch
# anomaly_scores = distances.min(dim=1).values

# This is the FAISS nearest-neighbor search
# Move DINOv2 embeddings to CPU for FAISS
query_vectors = np.ascontiguousarray(
    patch_embeddings.detach().cpu().numpy(),
    dtype=np.float32
)

# Search for the nearest normal embedding for every patch
faiss_distances, faiss_indices = index.search(
    query_vectors, 1
)

# FAISS returns squared Euclidean distances
# Convert to ordinary Euclidean distances
anomaly_scores = torch.from_numpy(
    np.sqrt(np.maximum(faiss_distances[:, 0], 0))
).to(device)

print("FAISS anomaly scores shape:", anomaly_scores.shape)

# ========================================================================




print("Anomaly scores shape:", anomaly_scores.shape)
print("Minimum anomaly score:", anomaly_scores.min().item())
print("Maximum anomaly score:", anomaly_scores.max().item())
print("Mean anomaly score:", anomaly_scores.mean().item())

# ==============================================



# Convert 1,369 scores into a 37x37 grid
anomaly_map = anomaly_scores.reshape(37, 37)

print("Anomaly map shape:", anomaly_map.shape)

# Resize the anomaly map to match the original image
anomaly_map = anomaly_map.unsqueeze(0).unsqueeze(0)

anomaly_map = F.interpolate(
    anomaly_map,
    size=(518, 518),
    mode="bilinear",
    align_corners=False
)

# Convert to NumPy for visualization
anomaly_map = anomaly_map.squeeze().cpu().numpy()

# Show original image, heatmap, and overlay
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image.resize((518, 518)))
axes[0].set_title("Defective Screw")

axes[1].imshow(anomaly_map, cmap="jet")
axes[1].set_title("Anomaly Heatmap")

axes[2].imshow(image.resize((518, 518)))
axes[2].imshow(anomaly_map, cmap="jet", alpha=0.5)
axes[2].set_title("Heatmap Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()


# Load the ground-truth mask for this defective image
mask_path = (
    Path("data/screw/ground_truth/scratch_head")
    / f"{image_path.stem}_mask.png"
)

mask = Image.open(mask_path).convert("L")
mask = mask.resize((518, 518), Image.Resampling.NEAREST)

# Compare ground truth with predicted anomaly heatmap
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image.resize((518, 518)))
axes[0].set_title("Defective Screw")

axes[1].imshow(mask, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Ground Truth (Actual Defect)")

axes[2].imshow(anomaly_map, cmap="jet")
axes[2].set_title("Predicted Anomaly Heatmap")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

# ===========================================================================


# STEP 11: Calculate pixel-level AUROC

# Convert the ground-truth mask to a binary array
# 0 = normal pixel, 1 = defective pixel
ground_truth = np.array(mask) > 0

# Flatten both arrays into 1D lists
ground_truth_flat = ground_truth.flatten().astype(int)
anomaly_scores_flat = anomaly_map.flatten()

# Calculate AUROC
pixel_auroc = roc_auc_score(
    ground_truth_flat,
    anomaly_scores_flat
)

print("\n--- Pixel-Level Evaluation ---")
print("Ground truth pixels:", ground_truth_flat.shape)
print("Predicted scores:", anomaly_scores_flat.shape)
print(f"Pixel-level AUROC: {pixel_auroc:.4f}")
