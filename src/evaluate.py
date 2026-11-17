
from pathlib import Path
import time

import faiss
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from transformers import Dinov2Model
from sklearn.metrics import roc_auc_score


# -----------------------------------
# 1. Load model and memory bank
# -----------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = Dinov2Model.from_pretrained(
    "facebook/dinov2-small"
).to(device)

model.eval()

# Original PyTorch memory bank
# memory_bank = torch.load(
#     "outputs/features/coreset_5000.pt",
#     map_location=device,
#     weights_only=True
# )

# print("Memory bank:", memory_bank.shape)

# FAISS nearest-neighbor index
index = faiss.read_index(
    "outputs/features/screw_faiss.index"
)

print("FAISS index size:", index.ntotal)


# -----------------------------------
# 2. Image preprocessing
# -----------------------------------

preprocess = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------------
# 3. Prepare evaluation results
# -----------------------------------

image_labels = []
image_scores = []

pixel_labels = []
pixel_scores = []

test_dir = Path("data/screw/test")
ground_truth_dir = Path("data/screw/ground_truth")

# Track FAISS search time separately from model inference
total_faiss_time = 0.0


# -----------------------------------
# 4. Evaluate all test images
# -----------------------------------

for category_dir in sorted(test_dir.iterdir()):

    if not category_dir.is_dir():
        continue

    category = category_dir.name

    image_paths = sorted(category_dir.glob("*.png"))

    print(f"\nEvaluating category: {category}")

    for image_path in image_paths:

        # Load image
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        # Extract DINOv2 patch embeddings
        with torch.no_grad():
            outputs = model(pixel_values=image_tensor)

            patches = outputs.last_hidden_state[:, 1:, :]
            patches = patches.squeeze(0)

            # Find nearest normal embedding for each patch
            # distances = torch.cdist( # Compute pairwise distances between two sets of vectors using pytorch.cdist.
            #     patches.unsqueeze(0), #This function computes the Euclidean distance between each pair of vectors in the two input tensors.
            #     memory_bank.unsqueeze(0)
            # ).squeeze(0)

            # anomaly_scores = distances.min(dim=1).values

        # FAISS nearest-neighbor search
        # Move DINOv2 patch embeddings from GPU to CPU
        query_vectors = np.ascontiguousarray(
            patches.detach().cpu().numpy(),
            dtype=np.float32
        )

        # Find the closest normal embedding for every patch
        start_time = time.perf_counter()

        faiss_distances, faiss_indices = index.search(
            query_vectors, 1
        )

        total_faiss_time += time.perf_counter() - start_time

        # FAISS returns squared Euclidean distances
        # Convert to ordinary Euclidean distances
        anomaly_scores = torch.from_numpy(
            np.sqrt(np.maximum(faiss_distances[:, 0], 0))
        ).to(device)

        # Image-level anomaly score
        image_score = anomaly_scores.max().item()

        # Image label: 0 = good, 1 = defective
        image_label = 0 if category == "good" else 1

        image_labels.append(image_label)
        image_scores.append(image_score)

        # Convert patch scores into a pixel-level heatmap
        anomaly_map = anomaly_scores.reshape(1, 1, 37, 37)

        anomaly_map = F.interpolate(
            anomaly_map,
            size=(518, 518),
            mode="bilinear",
            align_corners=False
        )

        anomaly_map = anomaly_map.squeeze().cpu().numpy()

        # Load corresponding ground-truth mask
        if category == "good":

            # Good images contain no defective pixels
            mask = np.zeros((518, 518), dtype=np.uint8)

        else:

            mask_path = (
                ground_truth_dir
                / category
                / f"{image_path.stem}_mask.png"
            )

            mask_image = Image.open(mask_path).convert("L")

            mask_image = mask_image.resize(
                (518, 518),
                Image.Resampling.NEAREST
            )

            mask = (np.array(mask_image) > 0).astype(np.uint8)

        # Store pixel-level results
        pixel_labels.append(mask.flatten())
        pixel_scores.append(anomaly_map.flatten())

    print(f"Processed {len(image_paths)} images")


# -----------------------------------
# 5. Calculate AUROC
# -----------------------------------

print("\nCalculating evaluation metrics...")

image_auroc = roc_auc_score(
    image_labels,
    image_scores
)

# Combine all pixel labels and scores
pixel_labels = np.concatenate(pixel_labels)
pixel_scores = np.concatenate(pixel_scores)

pixel_auroc = roc_auc_score(
    pixel_labels,
    pixel_scores
)


# -----------------------------------
# 6. Print results
# -----------------------------------

print("\n" + "=" * 40)
print("DEFECTFORGE EVALUATION RESULTS")
print("=" * 40)

print(f"Total images evaluated: {len(image_labels)}")

print(f"Image-level AUROC: {image_auroc:.4f}")
print(f"Pixel-level AUROC: {pixel_auroc:.4f}")

print("=" * 40)

# -----------------------------------
# 7. FAISS search timing
# -----------------------------------

print("\nFAISS SEARCH PERFORMANCE")
print(f"Total FAISS search time: {total_faiss_time:.4f} seconds")
print(
    f"Average FAISS search time per image: "
    f"{total_faiss_time / len(image_labels) * 1000:.4f} ms"
)
