import os
import csv
from pathlib import Path

ROOT = Path("data")
IMG_ROOT = ROOT / "images"
OUT_CSV = ROOT / "annotations.csv"

rows = []

# mapping spoiled/fresh to numeric labels and days
state_to_label = {
    "fresh": (0, 5),     # label, days left
    "mid": (1, 2),
    "spoiled": (2, 0)
}

for fruit_dir in IMG_ROOT.iterdir():
    if not fruit_dir.is_dir():
        continue

    fruit = fruit_dir.name.lower()

    for state_dir in fruit_dir.iterdir():
        if not state_dir.is_dir():
            continue

        state = state_dir.name.lower()

        if state not in state_to_label:
            print("Skipping folder:", state)
            continue

        label_state, days = state_to_label[state]

        for img in state_dir.glob("*.*"):
            rows.append([
                str(img.relative_to(ROOT)).replace("\\", "/"),
                fruit,
                label_state,
                days
            ])

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "item_type", "label_state", "days_to_spoilage"])
    writer.writerows(rows)

print(f"✔ Wrote {len(rows)} rows to {OUT_CSV}")
