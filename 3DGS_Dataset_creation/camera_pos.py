import json
import numpy as np
import matplotlib.pyplot as plt

camera_json = "/home/roar/Desktop/holo/train_3dgs.json"

with open(camera_json, "r") as f:
    data = json.load(f)

frames = data["frames"]

# Extract camera positions from transform matrices (translation component)
positions = []
for frame in frames:
    matrix = np.array(frame["transform_matrix"])
    position = matrix[:3, 3]  # translation vector
    positions.append(position)

positions = np.array(positions)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='blue', s=30)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Camera Positions from 3DGS")
plt.tight_layout()
plt.show()

