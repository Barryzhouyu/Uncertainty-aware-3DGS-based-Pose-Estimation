import os
import numpy as np
from glob import glob
from shutil import copy2

def parse_xyz_from_filename(filename):
    """Extract x, y, z from filename like ['23.81', '-35.50', '-300.68'].png"""
    coords = filename.replace(".png", "").strip("[]").split(",")
    return tuple(float(c.strip().strip("'")) for c in coords)

def match_pose_to_filename(filenames, poses):
    """Match each pose to its image by comparing x, y, z with filename"""
    matched = []
    used_indices = set()

    for fname in filenames:
        fx, fy, fz = parse_xyz_from_filename(fname)
        found = False
        for idx, pose in enumerate(poses):
            if idx in used_indices:
                continue
            px, py, pz = pose[:3, 3]
            if np.allclose([fx, fy, fz], [px, py, pz], atol=1e-2):
                matched.append((fname, pose))
                used_indices.add(idx)
                found = True
                break
        if not found:
            print(f"[WARN] No pose match for image: {fname}")
    return matched

# 🔧 Input folders
folders = [
    "/home/roar/Desktop/Holo_collected_images/48_-40_-294_better",
    "/home/roar/Desktop/Holo_collected_images/50_-42_-290_better",
    "/home/roar/Desktop/Holo_collected_images/52_-44_-285_better",
    "/home/roar/Desktop/Holo_collected_images/48_-40_-285_better"
]

# 🔧 Output folder
output_dir = "/home/roar/Desktop/holo_6"
os.makedirs(output_dir, exist_ok=True)

all_matched = []

# 🔁 Match poses for each folder
for folder in folders:
    pose_path = os.path.join(folder, "poses.npy")
    poses = np.load(pose_path)  # shape: [N, 4, 4]
    image_paths = sorted(glob(os.path.join(folder, "*.png")))

    filenames = [os.path.basename(p) for p in image_paths]
    matched = match_pose_to_filename(filenames, poses)
    all_matched.extend([(os.path.join(folder, fname), pose) for fname, pose in matched])

# ✅ Sort and save final renamed dataset
all_matched.sort()

pose_list = []
filename_mapping = []

for i, (src_img_path, pose) in enumerate(all_matched):
    new_name = f"frame_{i:05d}.png"
    dst_img_path = os.path.join(output_dir, new_name)
    copy2(src_img_path, dst_img_path)

    pose_list.append(pose)
    filename_mapping.append((os.path.basename(src_img_path), new_name))

# 💾 Save outputs
np.save(os.path.join(output_dir, "poses.npy"), np.stack(pose_list))

with open(os.path.join(output_dir, "image_filenames.txt"), "w") as f:
    for _, new_name in filename_mapping:
        f.write(f"{new_name}\n")

with open(os.path.join(output_dir, "filename_mapping.txt"), "w") as f:
    for old, new in filename_mapping:
        f.write(f"{old} -> {new}\n")

print(f"✅ Done! Saved {len(pose_list)} matched images and poses to: {output_dir}")

