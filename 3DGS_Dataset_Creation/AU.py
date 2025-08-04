# import os
# import cv2
# import numpy as np

# # === Config ===
# input_dir = "/home/roar/Desktop/test_images"
# output_root = "/home/roar/Desktop/test_images_noisy_multi"
# target_frames = {8, 14, 21, 54, 70, 96, 99, 136, 141, 142, 150, 175, 177, 193, 195}  # Frame IDs from EU list
# sigma_au = 0.05  # Noise std dev
# N = 10  # Number of samples

# # === Create output folders ===
# for i in range(N):
#     os.makedirs(os.path.join(output_root, f"sample_{i}"), exist_ok=True)

# # === Process only selected frames ===
# for frame_id in target_frames:
#     fname = f"frame_{frame_id:03d}.jpg"
#     img_path = os.path.join(input_dir, fname)

#     if not os.path.exists(img_path):
#         print(f"❌ Image not found: {img_path}")
#         continue

#     # Load and normalize image
#     img = cv2.imread(img_path).astype(np.float32) / 255.0

#     for i in range(N):
#         # Add Gaussian noise
#         noise = np.random.normal(0, sigma_au, img.shape).astype(np.float32)
#         noisy_img = np.clip(img + noise, 0, 1)
#         noisy_uint8 = (noisy_img * 255).astype(np.uint8)

#         # Save
#         out_path = os.path.join(output_root, f"sample_{i}", fname)
#         cv2.imwrite(out_path, noisy_uint8)

# print(f"✅ Injected Gaussian noise into {len(target_frames)} test images across {N} samples.")





import os
import json
import numpy as np
import cv2  # Don't forget to import OpenCV
from pathlib import Path

# === Config ===
root_dir = "/home/roar/Desktop/AU_results"
sample_dirs = [f"for_AU_{i}" for i in range(10)]

image_ids = [
    "f_008", "f_014", "f_021", "f_054", "f_070", "f_096",
    "f_099", "f_136", "f_141", "f_142", "f_150", "f_175",
    "f_177", "f_193", "f_195"
]

# === Store results ===
au_results = {}

for image_id in image_ids:
    t_all = []
    r_all = []

    for sample_dir in sample_dirs:
        pose_path = os.path.join(root_dir, sample_dir, f"{image_id}_AUpose", "pose_estimated.json")

        if not os.path.exists(pose_path):
            print(f"Missing: {pose_path}")
            continue

        with open(pose_path, "r") as f:
            data = json.load(f)

        transform = np.array(data["frames"][0]["transform_matrix"])
        t = transform[:3, 3]  # Translation vector
        R = transform[:3, :3]  # Rotation matrix

        rvec, _ = cv2.Rodrigues(R)  # Rotation matrix to angle-axis

        t_all.append(t)
        r_all.append(rvec.flatten())

    if len(t_all) < 2:
        print(f"Not enough data to compute AU for {image_id}")
        continue

    t_all = np.stack(t_all)
    r_all = np.stack(r_all)

    t_mean = np.mean(t_all, axis=0)
    r_mean = np.mean(r_all, axis=0)

    AU_trans = np.mean(np.linalg.norm(t_all - t_mean, axis=1) ** 2)
    AU_rot = np.mean(np.linalg.norm(r_all - r_mean, axis=1) ** 2)

    au_results[image_id] = {"AU_trans": AU_trans, "AU_rot": AU_rot}

# === Print results ===
for image_id, values in au_results.items():
    print(f"{image_id}: AU_trans={values['AU_trans']:.6f}, AU_rot={values['AU_rot']:.6f}")
