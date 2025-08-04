import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import shutil

# Paths
json_path = "/home/roar/Desktop/coke_test/undistorted/gt_coke.json"
image_path = "/home/roar/Desktop/coke_test/undistorted/images"
output_json_dir = "/home/roar/Desktop/coke_test/undistorted"
train_img_dir = os.path.join(output_json_dir, "images_train")
test_img_dir = os.path.join(output_json_dir, "images_test")

# Create output image directories
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(test_img_dir, exist_ok=True)

# Load camera data
with open(json_path, "r") as f:
    data = json.load(f)

frames = data["frames"]

# Extract camera positions
positions = []
for frame in frames:
    matrix = np.array(frame["transform_matrix"])
    position = matrix[:3, 3]
    positions.append(position)

positions = np.array(positions)

# Set custom origin and compute distances
origin = np.array([1.5, 0, -5])
distances = np.linalg.norm(positions - origin, axis=1)

# Sort frames by distance to origin
sorted_indices = np.argsort(distances)
frames_sorted = [frames[i] for i in sorted_indices]
positions_sorted = positions[sorted_indices]

# Divide into 4 parts and assign train/test splits
N = len(frames_sorted)
q = N // 4
ratios = [1, 0.7, 0.4, 0.1]

parts = [frames_sorted[:q], frames_sorted[q:2*q], frames_sorted[2*q:3*q], frames_sorted[3*q:]]
train_frames, test_frames = [], []

for part, ratio in zip(parts, ratios):
    split_idx = int(len(part) * ratio)
    train_frames.extend(part[:split_idx])
    test_frames.extend(part[split_idx:])

# Save new JSON files
with open(os.path.join(output_json_dir, "train_3dgs_split.json"), "w") as f:
    json.dump({**data, "frames": train_frames}, f)

with open(os.path.join(output_json_dir, "test_3dgs_split.json"), "w") as f:
    json.dump({**data, "frames": test_frames}, f)

# Copy corresponding images to new directories
def copy_images(frames, dst_dir):
    for frame in frames:
        img_name = os.path.basename(frame["file_path"])
        src_img_path = os.path.join(image_path, img_name)
        dst_img_path = os.path.join(dst_dir, img_name)
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
        else:
            print(f"Warning: {src_img_path} does not exist.")

copy_images(train_frames, train_img_dir)
copy_images(test_frames, test_img_dir)

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

train_pos = np.array([np.array(f["transform_matrix"])[:3, 3] for f in train_frames])
test_pos = np.array([np.array(f["transform_matrix"])[:3, 3] for f in test_frames])

ax.scatter(train_pos[:, 0], train_pos[:, 1], train_pos[:, 2], c='blue', label='Train')
ax.scatter(test_pos[:, 0], test_pos[:, 1], test_pos[:, 2], c='red', label='Test')
ax.scatter(*origin, c='orange', s=200, marker='*', label='Origin (-4, 0, -2)')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Train/Test Split by Distance (Farthest→Closest)")
plt.legend()
plt.show()

