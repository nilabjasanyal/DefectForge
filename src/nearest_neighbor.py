
from pathlib import Path
import torch

# Load the normal memory bank
memory_bank = torch.load(
    Path("outputs/features/coreset_100.pt"),
    map_location="cpu",
    weights_only=True
)

# Load the original normal embeddings
embeddings = torch.load(
    Path("outputs/features/normal_screw_5_images.pt"),
    map_location="cpu",
    weights_only=True
)

print("Memory bank shape:", memory_bank.shape)

# Select one patch embedding as a test query
test_patch = embeddings[500]

print("Test patch shape:", test_patch.shape)

# Calculate Euclidean distance to all memory embeddings
distances = torch.norm(
    memory_bank - test_patch,
    dim=1
)

# Find the nearest normal embedding
nearest_distance, nearest_index = torch.min(
    distances,
    dim=0
)

print("\nNearest memory index:", nearest_index.item())
print("Nearest distance:", nearest_distance.item())
