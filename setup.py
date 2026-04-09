import os
import shutil
import random

# Path to original dataset
SOURCE_DIR = "dataset/kvasir-dataset-v2"

# Target base directory
BASE_DIR = "data"

mapping = {
    "normal": ["normal-cecum", "normal-pylorus", "normal-z-line"],
    "polyp": ["polyps", "dyed-lifted-polyps", "dyed-resection-margins"],
    "ulcer": ["ulcerative-colitis"],
    "esophagitis": ["esophagitis"]
}

# Split ratio
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create folders
for split in ["train", "val", "test"]:
    for cls in mapping.keys():
        os.makedirs(os.path.join(BASE_DIR, split, cls), exist_ok=True)

# Process files
for new_class, folders in mapping.items():
    all_images = []

    for folder in folders:
        folder_path = os.path.join(SOURCE_DIR, folder)
        images = os.listdir(folder_path)
        images = [os.path.join(folder_path, img) for img in images]
        all_images.extend(images)

    random.shuffle(all_images)

    total = len(all_images)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    splits = {
        "train": all_images[:train_end],
        "val": all_images[train_end:val_end],
        "test": all_images[val_end:]
    }

    for split, files in splits.items():
        for file in files:
            filename = os.path.basename(file)
            dest = os.path.join(BASE_DIR, split, new_class, filename)
            shutil.copy(file, dest)

print("✅ Dataset ready!")