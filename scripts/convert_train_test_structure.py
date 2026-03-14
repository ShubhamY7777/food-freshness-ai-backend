import os, shutil
from pathlib import Path

# Your dataset
DATA_ROOT = Path("dataset")
SRC_TRAIN = DATA_ROOT / "Train"
SRC_TEST = DATA_ROOT / "Test"

# Destination folder for converted structure
DEST = Path("data/images")

# Mapping fruit folder names to clean labels
fruits_map = {
    "apples": "apple",
    "banana": "banana",
    "cucumber": "cucumber",
    "okra": "okra",
    "patato": "potato",
    "tomato": "tomato",
    "oranges": "orange"
}

def process_folder(split_path):
    for folder in split_path.iterdir():
        if not folder.is_dir():
            continue

        name = folder.name.lower()

        # Detect fresh/spoiled
        if name.startswith("fresh"):
            state = "fresh"
        elif name.startswith("rotten"):
            state = "spoiled"
        else:
            continue

        # Detect fruit type
        fruit_type = None
        for k, v in fruits_map.items():
            if k in name:
                fruit_type = v
                break

        if fruit_type is None:
            continue

        # Create destination path
        target = DEST / fruit_type / state
        target.mkdir(parents=True, exist_ok=True)

        # Copy all images
        for img in folder.glob("*.*"):
            shutil.copy(img, target)

# Run conversion for TRAIN and TEST
process_folder(SRC_TRAIN)
process_folder(SRC_TEST)

print("\n✔ DONE! Converted dataset saved inside: data/images/")
