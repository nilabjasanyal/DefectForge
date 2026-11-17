
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import Dinov2Model

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load pretrained DINOv2
model = Dinov2Model.from_pretrained(
    "facebook/dinov2-small"
).to(device)

model.eval()

# Same preprocessing as before
preprocess = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Find all normal training images
image_dir = Path("data/screw/train/good")
image_paths = sorted(image_dir.glob("*.png"))

print("Total training images:", len(image_paths))

all_embeddings = []

# Extract patch embeddings from every image
for i, image_path in enumerate(image_paths):

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=image_tensor)

    patches = outputs.last_hidden_state[:, 1:, :]
    patches = patches.squeeze(0).cpu()

    all_embeddings.append(patches)

    if (i + 1) % 20 == 0:
        print(f"Processed {i + 1}/{len(image_paths)} images")

# Combine all patch embeddings
combined_embeddings = torch.cat(all_embeddings, dim=0)

print("\nFull embeddings shape:", combined_embeddings.shape)

# Save all embeddings
save_dir = Path("outputs/features")
save_dir.mkdir(parents=True, exist_ok=True)

save_path = save_dir / "normal_screw_full.pt"

torch.save(combined_embeddings, save_path)

print("Saved to:", save_path)
