
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Project dataset paths
DATASET_DIR = Path("data")
CATEGORY = "screw"
DEFECT_TYPE = "scratch_head"

# Define image directories
normal_dir = DATASET_DIR / CATEGORY / "train" / "good" #instead of joining strings with + like ["screw" + "/" + "train"] we can use ["screw" / "train"]

defect_dir = (
    DATASET_DIR / CATEGORY / "test" / DEFECT_TYPE
)

mask_dir = (
    DATASET_DIR / CATEGORY / "ground_truth" / DEFECT_TYPE
)

# Find image files
normal_images = list(normal_dir.glob("*.png"))
defect_images = list(defect_dir.glob("*.png"))
mask_images = list(mask_dir.glob("*.png"))

# Display directory information
print("Normal images:", len(normal_images))
print("Defective images:", len(defect_images))
print("Ground-truth masks:", len(mask_images))

# Display one example filename
print("\nExample normal image:", normal_images[0].name)
print("Example defective image:", defect_images[0].name)

# ==============================Displaying a normal screw image==============================

# Select the first normal screw image
normal_image_path = sorted(normal_images)[0]

# Load the image
normal_image = Image.open(normal_image_path).convert("RGB")

# Print image information
print("\nImage size:", normal_image.size)
print("Image mode:", normal_image.mode)

# Display the image
plt.imshow(normal_image)
plt.title("Normal Screw")
plt.axis("off")
plt.show()


# Select the first defective screw image
defect_image_path = sorted(defect_images)[0]

# Find its corresponding ground-truth mask
mask_path = mask_dir / f"{defect_image_path.stem}_mask.png"

# Check that the mask exists
if not mask_path.exists():
    raise FileNotFoundError(f"Mask not found: {mask_path}")

# Load the defective image and mask
defect_image = Image.open(defect_image_path).convert("RGB")
mask_image = Image.open(mask_path).convert("L")

# Verify that their dimensions match
print("\nDefective image size:", defect_image.size)
print("Ground-truth mask size:", mask_image.size)

if defect_image.size != mask_image.size:
    raise ValueError("Image and mask dimensions do not match!")

# Display both images side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(defect_image)
axes[0].set_title("Defective Screw")
axes[0].axis("off")

axes[1].imshow(mask_image, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Ground-Truth Mask")
axes[1].axis("off")

plt.tight_layout()
plt.show()

# ============================Create an overlay of the defective image and its mask==============================

import numpy as np

# Convert images into NumPy arrays
defect_array = np.array(defect_image)
mask_array = np.array(mask_image)

# Create a copy of the defective image
overlay = defect_array.copy()

# Find pixels marked as defective
defect_pixels = mask_array > 0

# Highlight defective pixels in red
overlay[defect_pixels] = (
    0.5 * overlay[defect_pixels]
    + 0.5 * np.array([255, 0, 0])
).astype(np.uint8)

# Display the original and overlay
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(defect_image)
axes[0].set_title("Original Defective Screw")
axes[0].axis("off")

axes[1].imshow(overlay)
axes[1].set_title("Ground-Truth Overlay")
axes[1].axis("off")

plt.tight_layout()
plt.show()
