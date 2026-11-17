
from pathlib import Path
import time

import faiss
import numpy as np
import torch

# Load our existing memory bank
memory_bank = torch.load(
    "outputs/features/coreset_5000.pt",
    map_location="cpu",
    weights_only=True
)

# Convert to NumPy float32 for FAISS
memory_vectors = np.ascontiguousarray(
    memory_bank.numpy().astype("float32")
)

print("Memory bank shape:", memory_vectors.shape)

# Create an exact L2 nearest-neighbor index
dimension = memory_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add normal embeddings to the index
index.add(memory_vectors)

print("FAISS index size:", index.ntotal)

# Save index to disk
save_path = Path("outputs/features/screw_faiss.index")
faiss.write_index(index, str(save_path))

print("FAISS index saved to:", save_path)

# Load a sample query embedding
all_embeddings = torch.load(
    "outputs/features/normal_screw_full.pt",
    map_location="cpu",
    weights_only=True
)

test_patch = all_embeddings[500].unsqueeze(0)

# PyTorch nearest-neighbor search
start = time.perf_counter()

torch_distances = torch.cdist(
    test_patch,
    memory_bank
)

torch_distance, torch_index = torch_distances.min(dim=1)

torch_time = time.perf_counter() - start

# FAISS nearest-neighbor search
query = np.ascontiguousarray(
    test_patch.numpy().astype("float32")
)

start = time.perf_counter()

faiss_distances, faiss_indices = index.search(query, 1)

faiss_time = time.perf_counter() - start

print("\n--- PyTorch ---")
print("Nearest index:", torch_index.item())
print("Euclidean distance:", torch_distance.item())
print(f"Search time: {torch_time * 1000:.4f} ms")

print("\n--- FAISS ---")
print("Nearest index:", faiss_indices[0][0])

# FAISS returns squared Euclidean distances
faiss_distance = np.sqrt(faiss_distances[0][0])

print("Euclidean distance:", faiss_distance)
print(f"Search time: {faiss_time * 1000:.4f} ms")

print("\nDistance difference:",
      abs(torch_distance.item() - faiss_distance))
