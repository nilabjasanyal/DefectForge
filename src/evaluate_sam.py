
import cv2
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from transformers import Dinov2Model
from segment_anything import sam_model_registry, SamPredictor


# -----------------------------------
# 1. Load models
# -----------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load pretrained DINOv2
dinov2 = Dinov2Model.from_pretrained(
    "facebook/dinov2-small"
).to(device)

dinov2.eval()

# Load FAISS normal memory bank index
index = faiss.read_index(
    "outputs/features/screw_faiss.index"
)

print("FAISS index size:", index.ntotal)

# Load Segment Anything Model
sam = sam_model_registry["vit_b"](
    checkpoint="models/sam_vit_b_01ec64.pth"
).to(device)

sam.eval()

predictor = SamPredictor(sam)

print("Models loaded successfully!")


# -----------------------------------
# 2. Image preprocessing
# -----------------------------------

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------------
# 3. Dataset paths
# -----------------------------------

test_dir = Path("data/screw/test")
ground_truth_dir = Path("data/screw/ground_truth")

defect_categories = [
    "manipulated_front",
    "scratch_head",
    "scratch_neck",
    "thread_side",
    "thread_top"
]


# -----------------------------------
# 4. Count defective test images
# -----------------------------------

total_images = 0

for category in defect_categories:

    category_dir = test_dir / category
    image_paths = sorted(category_dir.glob("*.png"))

    print(f"{category}: {len(image_paths)} images")

    total_images += len(image_paths)

print("\nTotal defective images:", total_images)


# -----------------------------------
# 5. Automatically segment one image
# -----------------------------------

# -----------------------------------
# Foreground detection
# -----------------------------------

def detect_foreground(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(mask, 8)
    )

    if num_labels > 1:
        largest = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )
        mask = (labels == largest).astype(np.uint8) * 255

    mask = cv2.dilate(
        mask,
        np.ones((7, 7), np.uint8),
        iterations=1
    )

    return mask > 0

def segment_defect(image_path):

    # Load and resize image
    if isinstance(image_path, Image.Image):
        image = image_path.convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
        
    image = image.resize((518, 518))
    image_array = np.array(image)

    # Extract DINOv2 patch embeddings
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = dinov2(pixel_values=image_tensor)
        patches = outputs.last_hidden_state[:, 1:, :].squeeze(0)

    # FAISS nearest-neighbor search
    query_vectors = np.ascontiguousarray(
        patches.detach().cpu().numpy(),
        dtype=np.float32
    )

    distances, indices = index.search(query_vectors, 1)

    anomaly_scores = torch.from_numpy(
        np.sqrt(np.maximum(distances[:, 0], 0))
    )

    # Convert patch scores into an anomaly heatmap
    anomaly_map = anomaly_scores.reshape(1, 1, 37, 37)

    anomaly_map = F.interpolate(
        anomaly_map,
        size=(518, 518),
        mode="bilinear",
        align_corners=False
    )

    anomaly_map = anomaly_map.squeeze().numpy()

    # Find top 5 separated anomaly peaks
    # Restrict peaks to the detected screw foreground
    foreground_mask = detect_foreground(image_array)

    working_map = anomaly_map.copy()
    working_map[~foreground_mask] = -np.inf

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

    # Generate SAM masks
    predictor.set_image(image_array)

    results = []
    max_anomaly = float(anomaly_map.max())

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

        for mask_number in range(3):

            predicted_mask = masks[mask_number].astype(bool)

            if predicted_mask.any():
                mean_anomaly = float(
                    anomaly_map[predicted_mask].mean()
                )
            else:
                mean_anomaly = float("-inf")

            normalized_anomaly = (
                mean_anomaly / max_anomaly
                if max_anomaly > 0 else 0.0
            )

            combined_score = (
                0.7 * normalized_anomaly
                + 0.3 * float(scores[mask_number])
            )

            results.append({
                "peak": peak_number,
                "mask_number": mask_number + 1,
                "combined_score": combined_score,
                "mask": predicted_mask.copy()
            })

    # Select mask WITHOUT ground truth
    best_result = max(
        results,
        key=lambda r: r["combined_score"]
    )
    # Keep all SAM candidates for diagnostic analysis.
    # Ground truth is NOT used for automatic selection.
    best_result["candidates"] = results

    return best_result["mask"], best_result, anomaly_map

if __name__ == "__main__":
    # -----------------------------------
    # 6. Evaluate SAM on all defective images
    # -----------------------------------

    all_ious = []
    category_ious = {}

    for category_dir in sorted(test_dir.iterdir()):

        if not category_dir.is_dir():
            continue

        category = category_dir.name

        if category == "good":
            continue

        image_paths = sorted(category_dir.glob("*.png"))
        current_ious = []

        print(f"\nEvaluating category: {category}")

        for image_number, image_path in enumerate(image_paths, start=1):

            # Generate automatic SAM prediction
            predicted_mask, result, anomaly_map = segment_defect(image_path)

            # Load corresponding ground-truth defect mask
            mask_path = (
                ground_truth_dir
                / category
                / f"{image_path.stem}_mask.png"
            )

            ground_truth = Image.open(mask_path).convert("L")

            ground_truth = ground_truth.resize(
                (518, 518),
                Image.Resampling.NEAREST
            )

            ground_truth_mask = np.array(ground_truth) > 0

            # Calculate Intersection over Union
            intersection = np.logical_and(
                predicted_mask,
                ground_truth_mask
            ).sum()

            union = np.logical_or(
                predicted_mask,
                ground_truth_mask
            ).sum()

            iou = intersection / union if union > 0 else 1.0

            current_ious.append(iou)
            all_ious.append(iou)

            print(
                f"  [{image_number}/{len(image_paths)}] "
                f"{image_path.name} | IoU: {iou:.4f}"
            )

        category_ious[category] = float(np.mean(current_ious))


    # -----------------------------------
    # 7. Print final SAM evaluation results
    # -----------------------------------

    print("\n" + "=" * 45)
    print("DEFECTFORGE SAM EVALUATION RESULTS")
    print("=" * 45)

    for category, mean_iou in category_ious.items():
        print(f"{category}: {mean_iou:.4f}")

    print("-" * 45)
    print(f"Total defective images: {len(all_ious)}")
    print(f"Overall Mean IoU: {np.mean(all_ious):.4f}")
    print(f"Median IoU: {np.median(all_ious):.4f}")
    print("=" * 45)
