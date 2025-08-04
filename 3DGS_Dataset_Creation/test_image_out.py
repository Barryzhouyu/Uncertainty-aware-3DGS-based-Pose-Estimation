import os
import shutil
import json

src_dir = "/home/roar3/Desktop/pool/undistorted/images"
dst_dir = "/home/roar3/Desktop/pool_test_images"
os.makedirs(dst_dir, exist_ok=True)

json_path = "/home/roar3/Desktop/pool/undistorted/train_test_split.json"

with open(json_path, "r") as f:
    data = json.load(f)

test_files = data["test_files"]
for fname in test_files:
    src = os.path.join(src_dir, fname)
    dst = os.path.join(dst_dir, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied {fname}")
    else:
        print(f"File not found: {src}")

# import os

# src_dir = "/home/roar3/Desktop/3_test_images"
# txt_path = os.path.join(src_dir, "rename_pairs.txt")

# files = sorted([f for f in os.listdir(src_dir) if f.endswith('.png')])

# with open(txt_path, "w") as f:
#     for idx, fname in enumerate(files, 1):
#         new_name = f"frame_{idx:03d}.png"
#         src_path = os.path.join(src_dir, fname)
#         dst_path = os.path.join(src_dir, new_name)
#         os.rename(src_path, dst_path)
#         f.write(f"{new_name} <- {fname}\n")

# print("Done renaming and writing pairs to", txt_path)
