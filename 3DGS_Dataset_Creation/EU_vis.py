import os
import json
import numpy as np
import matplotlib.pyplot as plt

# === Paths ===
camera_json = "/home/roar/Desktop/rov/transforms.json"
translation_errors_file = "/home/roar/Desktop/epistemic_uncertainty_trans.txt"

# === Load ALL camera positions ===
with open(camera_json, "r") as f:
    data = json.load(f)

positions_all = []
filenames_all = []
frame_nums_all = []

for frame in data["frames"]:
    # fname = os.path.basename(frame["file_path"])  # e.g., 'frame_003.jpg'
    # matrix = np.array(frame["transform_matrix"])
    # position = matrix[:3, 3]
    # frame_num = int(fname.split("_")[1].split(".")[0])
    fname = frame["file_path"]  # e.g., 'images/0240.png'
    matrix = np.array(frame["transform_matrix"])
    position = matrix[:3, 3]
    position[2] = 0 
    position[0]*=2.5
    position[1]*=2.5
    frame_num = int(os.path.splitext(fname)[0].split("/")[-1])

    positions_all.append(position)
    filenames_all.append(fname)
    frame_nums_all.append(frame_num)

positions_all = np.array(positions_all)
frame_nums_all = np.array(frame_nums_all)

# === Load evaluated frame IDs + errors ===
results = np.loadtxt(translation_errors_file)
if results.ndim == 1:
    results = results.reshape(1, -1)
evaluated_frames = results[:, 0].astype(int)
translation_errors = results[:, 1]

# === Create frame → error map ===
frame_error_map = dict(zip(evaluated_frames, translation_errors))

# === Assign error to each frame (all frames) ===
errors = []
for frame_num in frame_nums_all:
    error = frame_error_map.get(frame_num, np.nan)
    errors.append(float(error))  # ensure scalar

errors = np.array(errors)

# # === Plot ===
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot all cameras as gray (background)
# ax.scatter(positions_all[:, 0], positions_all[:, 1], positions_all[:, 2],
#            color='gray', s=30, label='All Cameras')

# # Plot evaluated cameras as heatmap
# evaluated_mask = ~np.isnan(errors)
# p = ax.scatter(positions_all[evaluated_mask, 0],
#                positions_all[evaluated_mask, 1],
#                positions_all[evaluated_mask, 2],
#                c=errors[evaluated_mask],
#                cmap='turbo',
#                s=100,
#                label='Evaluated Cameras')

# #ax.scatter(-6, 0, -3, color='red', s=150, marker='*', label='Original Point')

# # Labels & layout
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title("Camera Positions\nAll Cameras + Evaluated Cameras Heatmap")
# fig.colorbar(p, ax=ax, label='Translation Error')
# ax.legend()
# plt.tight_layout()
# plt.show()
# === Create mask for frames that were evaluated ===
evaluated_mask = ~np.isnan(errors)

# === Clip errors only for visualization ===
errors_clipped = np.copy(errors)
errors_clipped[evaluated_mask] = np.minimum(errors_clipped[evaluated_mask], 1)

# === Plot ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot all cameras as gray (background)
ax.scatter(positions_all[:, 0], positions_all[:, 1], positions_all[:, 2],
           color='lightgray', s=40, label='All Cameras')

# Plot test image cameras as solid color points
p = ax.scatter(positions_all[evaluated_mask, 0],
               positions_all[evaluated_mask, 1],
               positions_all[evaluated_mask, 2],
               c=errors_clipped[evaluated_mask],
               cmap='turbo',
               s=100,
               marker='o',            # Solid circle
               linewidths=0,          # No edge
               alpha=1.0,
               label='Evaluated (Test) Cameras')

# Labels & layout
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Camera Positions\nAll Cameras + Evaluated Cameras (Solid Points)")
fig.colorbar(p, ax=ax, label='Translation Error (m²)')
ax.legend()
plt.tight_layout()
plt.show()