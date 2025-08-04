import json
import numpy as np
import matplotlib.pyplot as plt

json_path = "/home/roar3/Desktop/pool/transforms.json"

# --- Load JSON ---
with open(json_path, "r") as f:
    data = json.load(f)

# --- Extract positions ---
positions = []
if "frames" in data:  
    frames = data["frames"]
    for frame in frames:
        mat = np.array(frame["transform_matrix"])
        pos = mat[:3, 3]  # X, Y, Z
        positions.append(pos)
else:
    print("No 'frames' key found!")

positions = np.array(positions)

# --- 3D scatter plot ---
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:,0], positions[:,1], positions[:,2], c='g', s=60)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera Positions from transforms.json')
plt.tight_layout()
plt.show()

# --- 2D scatter (XY plane) ---
plt.figure(figsize=(7,6))
plt.scatter(positions[:,0], positions[:,1], c='b', s=30)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Camera Positions (XY)')
plt.axis('equal')
plt.tight_layout()
plt.show()
