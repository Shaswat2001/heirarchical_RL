# import os
# import numpy as np

# base_dir = '/Users/shaswatgarg/Downloads/data/train/FrankaIkGolfCourseEnv-v0'
# combined_data = {}

# for folder in os.listdir(base_dir):
#     folder_path = os.path.join(base_dir, folder)
#     if os.path.isdir(folder_path):
#         npy_path = os.path.join(folder_path, 'episodes.npy')
#         if os.path.isfile(npy_path):
#             data = np.load(npy_path, allow_pickle=True)
#             if isinstance(data, np.ndarray):
#                 for ep in data:
#                     ep = ep.copy()
#                     if 'info' in ep and isinstance(ep['info'], list):
#                         # Replace 'info' with 'terminals' and convert to list of bools
#                         ep['terminals'] = [bool(d.get('success', False)) for d in ep['info']]
#                         del ep['info']

#                     for key, val in ep.items():
#                         if key not in combined_data:
#                             combined_data[key] = []
#                         combined_data[key].extend(val if isinstance(val, list) else list(val))
#             else:
#                 print(f"Warning: Unexpected format in {npy_path}")

# # Convert lists to numpy arrays, using dtype=bool for 'terminals'
# for key in combined_data:
#     if key == "terminals":
#         combined_data[key] = np.array(combined_data[key], dtype=bool)
#     else:
#         combined_data[key] = np.array(combined_data[key], dtype=np.float32)

# # Save to .npz
# output_path = "/Users/shaswatgarg/Documents/Job/ArenaX/Development/heirarchical_RL/dataset/FrankaIkGolfCourseEnv-v0/train/FrankaIkGolfCourseEnv-v0_train.npz"
# np.savez_compressed(output_path, **combined_data)

# # print(f"Saved concatenated data with keys {list(combined_data.keys())} to {output_path}")


import numpy as np

# data = np.load('/Users/shaswatgarg/.ogbench/data/humanoidmaze-medium-navigate-v0.npz', allow_pickle=True)

# Load the .npz file
data = np.load('/Users/shaswatgarg/Documents/Job/ArenaX/Development/heirarchical_RL/dataset/FrankaIkGolfCourseEnv-v0/val/FrankaIkGolfCourseEnv-v0_val.npz', allow_pickle=True)

# List all arrays stored in the file
print("Keys in the .npz file:", data.files)

# Access each array by key
for key in data.files:
    print(f"{key}: shape = {data[key].shape}, dtype = {data[key].dtype}")
    print(data[key])  # Print actual data if needed

for i in data["actions"]:
    print(i)