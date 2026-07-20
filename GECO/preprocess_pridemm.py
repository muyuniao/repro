import os
import json
import pandas as pd

# Define paths
csv_path = "/home/duomeitinrfx/data/PrideMM/PrideMM.csv"
output_dir = "/home/duomeitinrfx/users/yunhe/reproduce/GECO"

# Load the dataset
df = pd.read_csv(csv_path)

def build_json_data(sub_df):
    data = []
    for _, row in sub_df.iterrows():
        # Get base filename without extension
        filename = row["name"]
        base_name, _ = os.path.splitext(filename)
        
        item = {
            "id": base_name,
            "hidden_state_file": f"llava/{base_name}.pt",
            "hidden_state_file2": f"qwen/{base_name}.pt",
            "hidden_state_file3": f"clip_image/{base_name}.pt",
            "hidden_state_file4": f"clip_text/{base_name}.pt",
            "hidden_state_file5": f"gemma/{base_name}.pt",
            "label": int(row["hate"]),
            "text": str(row["text"])  # Keep the raw text just in case for feature extraction
        }
        data.append(item)
    return data

# Split dataset
train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "val"]
test_df = df[df["split"] == "test"]

# Build json lists
train_json = build_json_data(train_df)
val_json = build_json_data(val_df)
test_json = build_json_data(test_df)

# Write to GECO dir
with open(os.path.join(output_dir, "train_PrideMM_with_hidden5.json"), "w", encoding="utf-8") as f:
    json.dump(train_json, f, indent=2, ensure_ascii=False)

with open(os.path.join(output_dir, "val_PrideMM_with_hidden5.json"), "w", encoding="utf-8") as f:
    json.dump(val_json, f, indent=2, ensure_ascii=False)

with open(os.path.join(output_dir, "test_PrideMM1_with_hidden5.json"), "w", encoding="utf-8") as f:
    json.dump(test_json, f, indent=2, ensure_ascii=False)

print(f"Generated JSON datasets under {output_dir}:")
print(f"  - train_PrideMM_with_hidden5.json: {len(train_json)} samples")
print(f"  - val_PrideMM_with_hidden5.json: {len(val_json)} samples")
print(f"  - test_PrideMM1_with_hidden5.json: {len(test_json)} samples")
