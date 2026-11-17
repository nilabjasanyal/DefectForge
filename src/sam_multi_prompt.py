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
# 5. Find top 5 separated anomaly peaks
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

    y1 = max(0, y - suppression_radius)
    y2 = min(518, y + suppression_radius + 1)
    x1 = max(0, x - suppression_radius)
    x2 = min(518, x + suppression_radius + 1)

    working_map[y1:y2, x1:x2] = -np.inf

print("Anomaly peaks:", peaks)


# -----------------------------------
# 6. Load ground truth for evaluation
# -----------------------------------

mask_path = (
    Path("data/screw/ground_truth")
    / image_path.parent.name
    / f"{image_path.stem}_mask.png"
)

ground_truth_image = Image.open(mask_path).convert("L")
ground_truth_image = ground_truth_image.resize(
    (518, 518),
    Image.Resampling.NEAREST
)

ground_truth = np.array(ground_truth_image) > 0


# -----------------------------------
# 7. Generate SAM masks for each peak
# -----------------------------------

predictor.set_image(image_array)

results = []

for peak_number, (x, y) in enumerate(peaks, start=1):

    point = np.array([[x, y]])
    label = np.array([1])

    half_size = 25

    box = np.array([
        max(0, x - half_size),
        max(0, y - half_size),
        min(517, x + half_size),
        min(517, y + half_size)
    ])

    with torch.no_grad():
        masks, scores, logits = predictor.predict(
            point_coords=point,
            point_labels=label,
            box=box,
            multimask_output=True
        )

    # Evaluate each SAM mask
    # Ground truth is used ONLY for evaluation
    for mask_number in range(3):

        predicted_mask = masks[mask_number].astype(bool)

        intersection = np.logical_and(
            predicted_mask, ground_truth
        ).sum()

        union = np.logical_or(
            predicted_mask, ground_truth
        ).sum()

        iou = intersection / union if union > 0 else 0.0

        
        # Calculate mean anomaly score inside SAM mask
        if predicted_mask.any():
            mean_anomaly = float(
                anomaly_map[predicted_mask].mean()
            )
        else:
            mean_anomaly = float("-inf")

        results.append({
            "peak": peak_number,
            "point": (x, y),
            "mask_number": mask_number + 1,
            "confidence": float(scores[mask_number]),
            "iou": float(iou),
            "mean_anomaly": mean_anomaly,
            "mask": predicted_mask.copy()
        })


    print(f"Processed peak {peak_number}: ({x}, {y})")


# -----------------------------------
# 8. Print results
# -----------------------------------

print("\n--- SAM MULTI-PROMPT RESULTS ---")

for result in results:
    print(
        f"Peak {result['peak']} | "
        f"Mask {result['mask_number']} | "
        f"Confidence: {result['confidence']:.3f} | "
        f"IoU: {result['iou']:.4f}"
    )

# Best mask by ground-truth IoU
# Diagnostic only: cannot be used at inference
best_result = max(results, key=lambda r: r["iou"])

print("\nBest diagnostic result:")
print("Peak:", best_result["peak"])
print("Point:", best_result["point"])
print("Mask:", best_result["mask_number"])
print(f"IoU: {best_result['iou']:.4f}")


# -----------------------------------
# 9. Visualize best diagnostic mask
# -----------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image_array)
axes[0].set_title("Original Screw")

axes[1].imshow(ground_truth, cmap="gray")
axes[1].set_title("Ground Truth")

axes[2].imshow(best_result["mask"], cmap="gray")
axes[2].set_title(
    f"Best SAM Mask (IoU: {best_result['iou']:.3f})"
)

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()


# -----------------------------------
# 10. Select best mask WITHOUT ground truth
# -----------------------------------

# Normalize anomaly scores to approximately 0-1
max_anomaly = float(anomaly_map.max())

for result in results:

    normalized_anomaly = (
        result["mean_anomaly"] / max_anomaly
        if max_anomaly > 0 else 0.0
    )

    # Combine anomaly strength and SAM confidence
    result["combined_score"] = (
        0.7 * normalized_anomaly
        + 0.3 * result["confidence"]
    )

# Select highest-scoring mask without ground truth
best_auto_result = max(
    results,
    key=lambda r: r["combined_score"]
)

print("\n--- AUTOMATIC CANDIDATE RANKING ---")

ranked_results = sorted(
    results,
    key=lambda r: r["combined_score"],
    reverse=True
)

for result in ranked_results[:5]:
    print(
        f"Peak {result['peak']} | "
        f"Mask {result['mask_number']} | "
        f"Combined: {result['combined_score']:.4f} | "
        f"IoU: {result['iou']:.4f}"
    )

print("\n--- AUTOMATIC MASK SELECTION ---")
print("Selected peak:", best_auto_result["peak"])
print("Selected point:", best_auto_result["point"])
print("Selected mask:", best_auto_result["mask_number"])

print(
    f"Mean anomaly score: "
    f"{best_auto_result['mean_anomaly']:.4f}"
)

# IoU is printed only to evaluate the selection
# It is NOT used to choose the mask
print(
    f"Selected mask IoU: "
    f"{best_auto_result['iou']:.4f}"
)

# Visualize automatic selection
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image_array)
axes[0].set_title("Original Screw")

axes[1].imshow(ground_truth, cmap="gray")
axes[1].set_title("Ground Truth (Evaluation Only)")

axes[2].imshow(best_auto_result["mask"], cmap="gray")
axes[2].set_title("Automatically Selected SAM Mask")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

