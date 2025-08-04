import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load camera data
json_path = "/home/roar/Desktop/playroom/train_3dgs.json"
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

# Set custom origin
origin = np.array([-4, 0, -2])
distances = np.linalg.norm(positions - origin, axis=1)

# Sort frames by distance
sorted_indices = np.argsort(distances)
frames_sorted = [frames[i] for i in sorted_indices]
positions_sorted = positions[sorted_indices]

# Divide into 4 equal parts
N = len(frames_sorted)
q = N // 4
part1 = frames_sorted[:q]
part2 = frames_sorted[q:2*q]
part3 = frames_sorted[2*q:3*q]
part4 = frames_sorted[3*q:]

# Define sampling ratios
ratios = [0.9, 0.6, 0.3, 0.1]
parts = [part1, part2, part3, part4]

train_frames = []
test_frames = []

for part, ratio in zip(parts, ratios):
    split_idx = int(len(part) * ratio)
    train_frames.extend(part[:split_idx])
    test_frames.extend(part[split_idx:])

# Save results
with open("/home/roar/Desktop/playroom/train_3dgs_split.json", "w") as f:
    json.dump({**data, "frames": train_frames}, f)

with open("/home/roar/Desktop/playroom/test_3dgs_split.json", "w") as f:
    json.dump({**data, "frames": test_frames}, f)

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
