
from pathlib import Path
import torch

# Load embeddings from our 5 normal screw images
feature_path = Path("outputs/features/normal_screw_5_images.pt")

embeddings = torch.load(
    feature_path,
    map_location="cpu",
    weights_only=True
)

print("Original embeddings:", embeddings.shape)

# Select 100 representative embeddings
coreset_size = 100

# Begin with the first embedding
selected_indices = [0]

# Calculate squared distances from the first embedding
min_distances = torch.sum(
    (embeddings - embeddings[0]) ** 2,
    dim=1
)

# Greedy farthest-point selection
for i in range(1, coreset_size):

    # Find the patch farthest from its nearest selected patch
    next_index = torch.argmax(min_distances).item()

    # Add that patch to our selection
    selected_indices.append(next_index)

    # Calculate distances to the newly selected patch
    new_distances = torch.sum(
        (embeddings - embeddings[next_index]) ** 2,
        dim=1
    )

    # Keep the smallest distance for each patch
    min_distances = torch.minimum(
        min_distances,
        new_distances
    )

# Create our small memory bank
memory_bank = embeddings[selected_indices]

print("Selected coreset:", memory_bank.shape)

# Save the selected embeddings
save_path = Path("outputs/features/coreset_100.pt")
torch.save(memory_bank, save_path)

print("Memory bank saved to:", save_path)
