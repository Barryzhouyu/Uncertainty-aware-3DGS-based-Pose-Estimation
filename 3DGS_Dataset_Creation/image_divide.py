# import json
# import os
# import numpy as np
# import shutil

# # === Config ===
# json_path = "/home/roar3/Desktop/pool/transforms.json"
# image_dir = "/home/roar3/Desktop/pool/undistorted/images"
# split_json_out = "/home/roar3/Desktop/pool/undistorted/train_test_split.json"

# # === Load full JSON ===
# with open(json_path, "r") as f:
#     data = json.load(f)

# all_frames = data["frames"]

# # === Compute distance to (0, 0, 0) ===
# target = np.array([0, 0, 0])
# entries = []
# for frame in all_frames:
#     fname = os.path.basename(frame["file_path"])
#     matrix = np.array(frame["transform_matrix"])
#     pos = matrix[:3, 3]
#     dist = np.linalg.norm(pos - target)
#     entries.append((fname, dist, frame))

# # === Sort by distance ===
# entries.sort(key=lambda x: x[1])

# # === Partition as before ===
# part1 = entries[:57]
# part2 = entries[57:114]
# part3 = entries[114:]

# def split_part(part, ratio):
#     part = np.array(part, dtype=object)
#     np.random.shuffle(part)
#     N = int(len(part) * ratio)
#     return part[:N], part[N:]

# train1, test1 = split_part(part1, 0.95)
# train2, test2 = split_part(part2, 0.85)
# train3, test3 = split_part(part3, 0.75)

# train_all = np.concatenate([train1, train2, train3])
# test_all = np.concatenate([test1, test2, test3])

# # === Write train/test split JSON (by filenames) ===
# split_data = {
#     "train": [fname for fname, _, _ in train_all],
#     "test": [fname for fname, _, _ in test_all]
# }
# with open(split_json_out, "w") as f:
#     json.dump(split_data, f, indent=2)

# print(f"Saved {len(train_all)} train and {len(test_all)} test images.")
# print(f"JSON file: {split_json_out}")



import json

split_path = "/home/roar3/Desktop/pool/undistorted/train_test_split.json"
with open(split_path, "r") as f:
    data = json.load(f)

# Rename keys if needed
if "train" in data and "test" in data:
    data["train_files"] = data.pop("train")
    data["test_files"] = data.pop("test")
    with open(split_path, "w") as f:
        json.dump(data, f, indent=2)
    print("Renamed keys to 'train_files' and 'test_files'")
else:
    print("Already in correct format or keys missing!")

# import json
# import os
# import numpy as np
# import shutil

# # === Config ===
# json_path = "/home/roar3/Desktop/nb2/undistorted/transforms.json"
# image_dir = "/home/roar3/Desktop/nb2/undistorted/images"
# split_json_out = "/home/roar3/Desktop/nb2/undistorted/train_test_split.json"

# # === Load full JSON ===
# with open(json_path, "r") as f:
#     data = json.load(f)

# all_frames = data["frames"]

# # === Compute distance to (0, 0, 0) ===
# target = np.array([7.5, -2, 0])
# entries = []
# for frame in all_frames:
#     fname = os.path.basename(frame["file_path"])
#     matrix = np.array(frame["transform_matrix"])
#     pos = matrix[:3, 3]
#     dist = np.linalg.norm(pos - target)
#     entries.append((fname, dist, frame))

# # === Sort by distance ===
# entries.sort(key=lambda x: x[1])

# # === Partition as before ===
# # part1 = entries[:67]
# # part2 = entries[67:134]
# # part3 = entries[134:]

# part1 = entries[:50]
# part2 = entries[50:100]
# part3 = entries[100:150]
# part4 = entries[150:]

# def split_part(part, ratio):
#     part = np.array(part, dtype=object)
#     np.random.shuffle(part)
#     N = int(len(part) * ratio)
#     return part[:N], part[N:]

# train1, test1 = split_part(part1, 0.95)
# train2, test2 = split_part(part2, 0.85)
# train3, test3 = split_part(part3, 0.80)
# train4, test4 = split_part(part4, 0.70)

# train_all = np.concatenate([train1, train2, train3, train4])
# test_all = np.concatenate([test1, test2, test3, test4])


# # === Write train/test split JSON (by filenames) ===
# split_data = {
#     "train_files": [fname for fname, _, _ in train_all],
#     "test_files": [fname for fname, _, _ in test_all]
# }
# with open(split_json_out, "w") as f:
#     json.dump(split_data, f, indent=2)

# print(f"Saved {len(train_all)} train and {len(test_all)} test images.")
# print(f"JSON file: {split_json_out}")
