
# DINOv2: Extracts visual features (embeddings) from screw images.
import torch
from transformers import Dinov2Model

# Choose GPU if available
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# Load pretrained DINOv2 Small
model = Dinov2Model.from_pretrained(
    "facebook/dinov2-small"
)

# Move model to GPU or CPU
model = model.to(device)

# Set model to evaluation mode
model.eval()

print("DINOv2 loaded successfully!")
print("Hidden size:", model.config.hidden_size)
print("Patch size:", model.config.patch_size)
# ===========================
# extract features (patch embeddings) from a normal screw image

from pathlib import Path
from PIL import Image
from torchvision import transforms

# Select one normal screw image
image_dir = Path("data/screw/train/good")
image_path = sorted(image_dir.glob("*.png"))[0]

# Load the image
image = Image.open(image_path).convert("RGB")

# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image_tensor = preprocess(image)

# Add batch dimension: [3, 518, 518] -> [1, 3, 518, 518]
image_tensor = image_tensor.unsqueeze(0).to(device)

print("\nInput tensor shape:", image_tensor.shape)

# Extract features without training DINOv2
with torch.no_grad():
    outputs = model(pixel_values=image_tensor)

# All output tokens, including the CLS token
features = outputs.last_hidden_state

# Remove the CLS token to keep only patch embeddings
patch_embeddings = features[:, 1:, :]

print("All tokens shape:", features.shape)
print("Patch embeddings shape:", patch_embeddings.shape)

# =====================================================================
# milestone: Inspect an actual embedding
# Before building the PatchCore memory bank, let's understand what those 384 numbers look like.

# Select the first patch embedding
first_patch = patch_embeddings[0, 0, :]

print("\nFirst patch embedding shape:", first_patch.shape)

# Print the first 10 values
print("First 10 embedding values:")
print(first_patch[:10].cpu().numpy())

# Calculate basic statistics
print("\nEmbedding mean:", first_patch.mean().item())
print("Embedding standard deviation:", first_patch.std().item())

#===================================================================
# first, let's test this with just 5 normal images.
# Extract features from five images


# Select 5 normal screw images
sample_paths = sorted(image_dir.glob("*.png"))[:5]

all_embeddings = []

for path in sample_paths:
    # Load image
    img = Image.open(path).convert("RGB")

    # Preprocess and add batch dimension
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    # Extract DINOv2 features
    with torch.no_grad():
        output = model(pixel_values=img_tensor)

    # Remove CLS token and batch dimension
    patches = output.last_hidden_state[:, 1:, :].squeeze(0)

    # Move embeddings to CPU and store them
    all_embeddings.append(patches.cpu())

    print(f"Processed: {path.name}, shape: {patches.shape}")

# Combine embeddings from all images
combined_embeddings = torch.cat(all_embeddings, dim=0)

print("\nCombined embeddings shape:", combined_embeddings.shape)
# =====================================================
# First, save our five-image embeddings



# Create a directory for saved features
feature_dir = Path("outputs/features")
feature_dir.mkdir(parents=True, exist_ok=True)

# Save the embeddings
save_path = feature_dir / "normal_screw_5_images.pt"

torch.save(combined_embeddings, save_path)

print("\nEmbeddings saved to:", save_path)
print("Saved tensor shape:", combined_embeddings.shape)
 