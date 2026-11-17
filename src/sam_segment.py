
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from segment_anything import sam_model_registry, SamPredictor


# -----------------------------------
# 1. Load SAM model
# -----------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

sam = sam_model_registry["vit_b"](
    checkpoint="models/sam_vit_b_01ec64.pth"
)

sam.to(device=device)
sam.eval()

predictor = SamPredictor(sam)

print("SAM model loaded successfully!")


# -----------------------------------
# 2. Load defective screw image
# -----------------------------------

image_path = Path(
    "data/screw/test/scratch_head/000.png"
)

image = Image.open(image_path).convert("RGB")
image = image.resize((518, 518))

image_array = np.array(image)

print("Image shape:", image_array.shape)


# -----------------------------------
# 3. Prepare image for SAM
# -----------------------------------

predictor.set_image(image_array)


# -----------------------------------
# 4. Provide a point prompt
# -----------------------------------

# Approximate location of the scratch
# Coordinates are (x, y)
input_point = np.array([[175, 140]])

# 1 = foreground point
input_label = np.array([1])


# -----------------------------------
# 5. Generate segmentation masks
# -----------------------------------

with torch.no_grad():
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

print("Generated masks:", masks.shape)
print("SAM confidence scores:", scores)


# -----------------------------------
# 6. Visualize SAM results
# -----------------------------------

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(image_array)
axes[0].scatter(
    input_point[:, 0],
    input_point[:, 1],
    color="red",
    marker="*",
    s=200
)
axes[0].set_title("Original Image + Prompt")

for i in range(3):
    axes[i + 1].imshow(image_array)
    axes[i + 1].imshow(
        masks[i],
        cmap="jet",
        alpha=0.4
    )
    axes[i + 1].set_title(
        f"SAM Mask {i + 1} (Score: {scores[i]:.3f})"
    )

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
