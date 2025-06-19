import os
import numpy as np

input_dir = "/Users/shaswatgarg/Documents/Job/ArenaX/Development/heirarchical_RL/dataset/FrankaGolfCourseEnv-v0"  # <-- change this
merged_data = {}

# List all .npz files in the folder
npz_files = [f for f in os.listdir(input_dir) if f.endswith(".npz")]
print(npz_files)
for filename in npz_files:
    filepath = os.path.join(input_dir, filename)
    data = np.load(filepath, allow_pickle=True)
    
    for key in data.files:
        arr = data[key]
        if key not in merged_data:
            merged_data[key] = [arr]
        else:
            merged_data[key].append(arr)

print(len(merged_data["observations"]))
# Concatenate arrays per key
for key in merged_data:

    if key == "terminals":
        merged_data[key] = np.concatenate(merged_data[key], axis=0, dtype=bool)
    else:
        merged_data[key] = np.concatenate(merged_data[key], axis=0, dtype=np.float32)
# Save to a single merged .npz file
print(len(merged_data["observations"]))
output_path = os.path.join(input_dir, "train/FrankaGolfCourseEnv-v0_train.npz")
np.savez_compressed(output_path, **merged_data)

print(f"Merged {len(npz_files)} files. Keys: {list(merged_data.keys())}")
