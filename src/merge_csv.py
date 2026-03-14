import pandas as pd

# Load existing food dataset
food_csv_path = "dataset/annotations.csv"   # change if different
nonfood_csv_path = "dataset/nonfood_annotations.csv"

food_df = pd.read_csv(food_csv_path)
nonfood_df = pd.read_csv(nonfood_csv_path)

# Merge
final_df = pd.concat([food_df, nonfood_df], ignore_index=True)

# Save
final_df.to_csv("dataset/annotations_final.csv", index=False)

print("✅ CSV merged successfully!")
print(f"📄 Total samples: {len(final_df)}")