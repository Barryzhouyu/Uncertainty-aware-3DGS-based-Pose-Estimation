import json
import os
import numpy as np

# === Load JSON ===
with open("/home/roar/Desktop/T3/holo_test_split.json", "r") as f:
    data = json.load(f)

gt_coords = []
colmap_coords = []

for frame in data["frames"]:
    # ✅ Extract just the filename
    fname = os.path.basename(frame["file_path"])  # e.g., '-344.94_-700.00_-3.00.png'
    
    try:
        x_gt, y_gt, _ = map(float, fname.replace(".png", "").split("_"))
    except ValueError:
        print(f"Skipping malformed filename: {fname}")
        continue

    transform = frame["transform_matrix"]
    x_colmap = transform[0][3]
    y_colmap = transform[1][3]

    gt_coords.append((x_gt, y_gt))
    colmap_coords.append((x_colmap, y_colmap))

# Convert to numpy arrays
gt_coords = np.array(gt_coords)
colmap_coords = np.array(colmap_coords)

# Solve for best-fit scale using least-squares
A = colmap_coords
b = gt_coords

scales, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
scale_x, scale_y = scales[0, 0], scales[1, 1]

print(f"Estimated scale_x: {scale_x:.6f}")
print(f"Estimated scale_y: {scale_y:.6f}")
