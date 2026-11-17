
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Reuse the existing segmentation function
from evaluate_sam import segment_defect


# -----------------------------------
# 1. Select images for failure analysis
# -----------------------------------

test_images = [
    ("scratch_neck", "008.png"),
    ("thread_side", "000.png"),
    ("manipulated_front", "001.png")
]


# -----------------------------------
# 2. Visualize predictions
# -----------------------------------

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for row, (category, filename) in enumerate(test_images):

    image_path = Path("data/screw/test") / category / filename

    # Run automatic segmentation
    predicted_mask, result, anomaly_map = segment_defect(image_path)

    # Load original image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((518, 518))
    image_array = np.array(image)

    # Load ground-truth mask for evaluation only
    mask_path = (
        Path("data/screw/ground_truth")
        / category
        / f"{image_path.stem}_mask.png"
    )

    ground_truth = Image.open(mask_path).convert("L")
    ground_truth = ground_truth.resize(
        (518, 518),
        Image.Resampling.NEAREST
    )

    ground_truth_mask = np.array(ground_truth) > 0

    # Calculate IoU
    intersection = np.logical_and(
        predicted_mask, ground_truth_mask
    ).sum()

    union = np.logical_or(
        predicted_mask, ground_truth_mask
    ).sum()

    iou = intersection / union if union > 0 else 0.0

    print(
        f"\n{category}/{filename} | "
        f"Selected Peak: {result['peak']} | "
        f"Mask: {result['mask_number']} | "
        f"IoU: {iou:.4f}"
    )

    # -----------------------------------
    # Diagnostic: Find best possible SAM candidate
    # -----------------------------------

    candidate_ious = []

    for candidate in result["candidates"]:

        candidate_mask = candidate["mask"]

        candidate_intersection = np.logical_and(
            candidate_mask,
            ground_truth_mask
        ).sum()

        candidate_union = np.logical_or(
            candidate_mask,
            ground_truth_mask
        ).sum()

        candidate_iou = (
            candidate_intersection / candidate_union
            if candidate_union > 0 else 0.0
        )

        candidate_ious.append({
            "peak": candidate["peak"],
            "mask": candidate["mask_number"],
            "iou": candidate_iou,
            "score": candidate["combined_score"]
        })

    # Find best candidate AFTER evaluating all 15 masks
    best_possible = max(
        candidate_ious,
        key=lambda r: r["iou"]
    )

    print(
        f"  Best possible candidate: "
        f"Peak {best_possible['peak']}, "
        f"Mask {best_possible['mask']}, "
        f"IoU: {best_possible['iou']:.4f}"
    )

    print(
        f"  Selection gap: "
        f"{best_possible['iou'] - iou:.4f}"
    )

    # -----------------------------------
    # Plot results
    # -----------------------------------

    # Original image
    axes[row, 0].imshow(image_array)
    axes[row, 0].set_title(f"{category} - Original")

    # FAISS anomaly heatmap
    axes[row, 1].imshow(image_array)
    axes[row, 1].imshow(anomaly_map, cmap="jet", alpha=0.5)
    axes[row, 1].set_title("FAISS Heatmap")

    # Ground-truth defect mask
    axes[row, 2].imshow(ground_truth_mask, cmap="gray")
    axes[row, 2].set_title("Ground Truth")

    # Automatically selected SAM mask
    axes[row, 3].imshow(predicted_mask, cmap="gray")
    axes[row, 3].set_title(f"SAM Prediction (IoU: {iou:.3f})")

    for ax in axes[row]:
        ax.axis("off")

plt.tight_layout()
plt.show()
