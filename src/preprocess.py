
from pathlib import Path
from PIL import Image
from torchvision import transforms

# Select one normal screw image
image_dir = Path("data/screw/train/good")
image_path = sorted(image_dir.glob("*.png"))[0]

# Load image
image = Image.open(image_path).convert("RGB")

# Define preprocessing operations
preprocess = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Apply preprocessing
image_tensor = preprocess(image)

# Inspect result
print("Original image size:", image.size)
print("Tensor shape:", image_tensor.shape)
print("Minimum value:", image_tensor.min().item())
print("Maximum value:", image_tensor.max().item())

# =======================================================

import torch
from torchvision.transforms import InterpolationMode

# Locate a defective screw
defect_dir = Path("data/screw/test/scratch_head")
mask_dir = Path("data/screw/ground_truth/scratch_head")

defect_path = sorted(defect_dir.glob("*.png"))[0]
mask_path = mask_dir / f"{defect_path.stem}_mask.png"

if not mask_path.exists():
    raise FileNotFoundError(f"Mask not found: {mask_path}")

# Load defective image and mask
defect_image = Image.open(defect_path).convert("RGB")
mask_image = Image.open(mask_path).convert("L")

# Ensure the original image and mask align in size
assert defect_image.size == mask_image.size

# Preprocess the defective image using our existing pipeline
defect_tensor = preprocess(defect_image)

# Resize the mask without introducing new pixel values
mask_transform = transforms.Compose([
    transforms.Resize(
        (518, 518),
        interpolation=InterpolationMode.NEAREST
    ),
    transforms.PILToTensor()
])

mask_tensor = mask_transform(mask_image)

# Convert mask into binary labels: 0 or 1
mask_tensor = (mask_tensor > 0).to(torch.uint8)

# Print results
print("\nDefective image shape:", defect_tensor.shape)
print("Mask shape:", mask_tensor.shape)
print("Unique mask values:", torch.unique(mask_tensor))
