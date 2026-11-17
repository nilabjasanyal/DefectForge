
from pathlib import Path

DATASET_DIR = Path("data")
CATEGORY = "screw"

category_dir = DATASET_DIR / CATEGORY

if not category_dir.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {category_dir.resolve()}"
    )

def count_images(folder):
    extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    return sum(
        1 for file in folder.rglob("*")
        if file.suffix.lower() in extensions
    )

print(f"\nCategory: {CATEGORY}")
print("-" * 40)

for split in ["train", "test", "ground_truth"]:
    split_dir = category_dir / split

    print(f"\n{split.upper()}")

    for folder in sorted(split_dir.iterdir()):
        if folder.is_dir():
            count = count_images(folder)
            print(f"  {folder.name}: {count} images")
