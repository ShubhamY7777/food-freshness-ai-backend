import os
import pandas as pd

DATASET_DIR = "dataset/nonfood"
CSV_PATH = "dataset/nonfood_annotations.csv"
NONFOOD_LABEL = 14  # change if needed

rows = []

print("🔍 Scanning non-food images...\n")

for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, "dataset")
            rows.append([relative_path, 0, NONFOOD_LABEL])

df = pd.DataFrame(rows, columns=["image_path", "days_to_spoilage", "label_state"])
df.to_csv(CSV_PATH, index=False)

print("✅ Non-food CSV created successfully!")
print(f"📄 Total images: {len(rows)}")