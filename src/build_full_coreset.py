
from pathlib import Path
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load all normal patch embeddings
embeddings = torch.load(
    "outputs/features/normal_screw_full.pt",
    map_location="cpu",
    weights_only=True
)

print("Original embeddings:", embeddings.shape)

# Reproducible random sampling
torch.manual_seed(42)

# First, sample 20,000 candidate embeddings
candidate_size = 20000
indices = torch.randperm(len(embeddings))[:candidate_size]

candidates = embeddings[indices].to(device)

print("Candidate embeddings:", candidates.shape)

# Select 5,000 diverse embeddings
coreset_size = 5000

# Start with the first candidate
selected_indices = [0]

# Track distance to the nearest selected embedding
min_distances = torch.sum(
    (candidates - candidates[0]) ** 2,
    dim=1
)

# Greedy farthest-point selection
for i in range(1, coreset_size):

    next_index = torch.argmax(min_distances).item()
    selected_indices.append(next_index)

    new_distances = torch.sum(
        (candidates - candidates[next_index]) ** 2,
        dim=1
    )

    min_distances = torch.minimum(
        min_distances,
        new_distances
    )

    if (i + 1) % 500 == 0:
        print(f"Selected {i + 1}/{coreset_size} embeddings")

# Create the final memory bank
memory_bank = candidates[selected_indices].cpu()

print("Final memory bank:", memory_bank.shape)

# Save
save_path = Path("outputs/features/coreset_5000.pt")
torch.save(memory_bank, save_path)

print("Saved to:", save_path)
