
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Images to test
test_images = [
    "data/screw/test/scratch_neck/008.png",
    "data/screw/test/thread_side/000.png",
    "data/screw/test/manipulated_front/001.png"
]


def detect_foreground(image):
    """
    Detect the screw using grayscale thresholding.
    Returns a binary mask: 1 = screw, 0 = background.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Blur to reduce background noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Dark screw against light background
    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove isolated noise
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Keep the largest connected component
    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(mask, 8)
    )

    if num_labels > 1:
        largest = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        mask = (labels == largest).astype(np.uint8) * 255

    # Expand slightly to include screw edges
    mask = cv2.dilate(
        mask,
        np.ones((7, 7), np.uint8),
        iterations=1
    )

    return mask > 0


# Visualize results
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for row, image_path in enumerate(test_images):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (518, 518))

    foreground = detect_foreground(image)

    # Highlight detected foreground
    overlay = image.copy()
    overlay[foreground] = (
        0.5 * overlay[foreground]
        + 0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    axes[row, 0].imshow(image)
    axes[row, 0].set_title(Path(image_path).parent.name)

    axes[row, 1].imshow(foreground, cmap="gray")
    axes[row, 1].set_title("Detected Screw Mask")

    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title("Foreground Overlay")

    for ax in axes[row]:
        ax.axis("off")

plt.tight_layout()
plt.show()
