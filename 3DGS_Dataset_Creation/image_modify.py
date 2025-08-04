import os
import cv2
import numpy as np
from tqdm import tqdm

# === Parameters ===
image_dir     = "/home/roar/Desktop/test_nb"
output_root   = "/home/roar/Desktop/test_nb_noisy"
sample_count  = 10       # N = 10
sigma_au      = 0.1    # noise std
frame_list = [
    "frame_021.png",  # ID 21 → 0.000579
    "frame_019.png",  # ID 19 → 0.000445
    "frame_020.png",  # ID 20 → 0.000586
    "frame_016.png",  # ID 16 → 0.000749
    "frame_018.png",  # ID 18 → 0.003318
    "frame_023.png",  # ID 23 → 0.001894
    "frame_022.png",  # ID 22 → 0.038269
    "frame_012.png",  # ID 12 → 0.032927
    "frame_006.png",  # ID 6  → 0.346225
    "frame_024.png"   # ID 24 → 1.261227
]

os.makedirs(output_root, exist_ok=True)

# === Add noise and save ===
for sample_id in tqdm(range(sample_count), desc="Generating noisy samples"):
    sample_dir = os.path.join(output_root, f"sample_{sample_id}")
    os.makedirs(sample_dir, exist_ok=True)

    for fname in frame_list:
        img_path = os.path.join(image_dir, fname)
        img = cv2.imread(img_path).astype(np.float32) / 255.0

        noise = np.random.normal(loc=0.0, scale=sigma_au, size=img.shape)
        noisy_img = np.clip(img + noise, 0.0, 1.0)

        save_path = os.path.join(sample_dir, fname)
        cv2.imwrite(save_path, (noisy_img * 255).astype(np.uint8))

print(f"\n✅ Done. {sample_count} noisy versions saved in: {output_root}")

