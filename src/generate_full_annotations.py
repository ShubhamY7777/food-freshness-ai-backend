import os
import pandas as pd

DATASET_DIR = "dataset"
OUTPUT_CSV = "dataset/annotations_final.csv"

rows = []
label_map = {}
current_label = 0

print("🔍 Scanning dataset folders...\n")

# Scan Train, Test, and nonfood
for main_folder in ["Train", "Test", "nonfood"]:
    main_path = os.path.join(DATASET_DIR, main_folder)

    if not os.path.exists(main_path):
        continue

    # If nonfood (images directly inside)
    if main_folder == "nonfood":
        if "nonfood" not in label_map:
            label_map["nonfood"] = current_label
            current_label += 1

        for root, dirs, files in os.walk(main_path):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, DATASET_DIR)
                    rows.append([relative_path, 0, label_map["nonfood"]])

    else:
        # Train/Test contains class folders
        for class_folder in os.listdir(main_path):
            class_path = os.path.join(main_path, class_folder)

            if os.path.isdir(class_path):
                if class_folder not in label_map:
                    label_map[class_folder] = current_label
                    current_label += 1

                for root, dirs, files in os.walk(class_path):
                    for file in files:
                        if file.lower().endswith((".jpg", ".jpeg", ".png")):
                            full_path = os.path.join(root, file)
                            relative_path = os.path.relpath(full_path, DATASET_DIR)
                            rows.append([relative_path, 0, label_map[class_folder]])

df = pd.DataFrame(rows, columns=["image_path", "days_to_spoilage", "label_state"])
df.to_csv(OUTPUT_CSV, index=False)

print("✅ Full annotations CSV created!")
print(f"📄 Total images: {len(rows)}")
print(f"📊 Total classes: {len(label_map)}")
print("🧠 Label mapping:", label_map)