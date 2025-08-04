import os
import json
import numpy as np
import matplotlib.pyplot as plt

# === Paths ===
camera_json = "/home/roar/Desktop/coke_all/coke_gt.json"

# === Your precomputed AU results (can load from a file too) ===
au_results = {
    "f_013": {"AU_trans": 0.006914}, "f_014": {"AU_trans": 0.011397},
    "f_015": {"AU_trans": 0.002110}, "f_016": {"AU_trans": 0.024326},
    "f_151": {"AU_trans": 0.005470}, "f_152": {"AU_trans": 0.007497},
    "f_153": {"AU_trans": 0.007857}, "f_154": {"AU_trans": 0.004935},
    "f_155": {"AU_trans": 0.016246}, "f_156": {"AU_trans": 0.027048},
    "f_158": {"AU_trans": 0.007595}, "f_161": {"AU_trans": 0.007825},
    "f_164": {"AU_trans": 0.010152}
}

# === Load ALL camera positions ===
with open(camera_json, "r") as f:
    data = json.load(f)

positions_all = []
filenames_all = []
frame_nums_all = []
au_errors = []

for frame in data["frames"]:
    fname = os.path.basename(frame["file_path"])  # e.g., 'frame_013.jpg'
    matrix = np.array(frame["transform_matrix"])
    position = matrix[:3, 3]
    frame_num = int(fname.split("_")[1].split(".")[0])
    frame_key = f"f_{frame_num:03d}"

    positions_all.append(position)
    filenames_all.append(fname)
    frame_nums_all.append(frame_num)

    # Assign AU if available
    if frame_key in au_results:
        au_errors.append(au_results[frame_key]["AU_trans"])
    else:
        au_errors.append(np.nan)

positions_all = np.array(positions_all)
au_errors = np.array(au_errors)

# === Plot ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot all cameras as gray (background)
ax.scatter(positions_all[:, 0], positions_all[:, 1], positions_all[:, 2],
           color='gray', s=30, label='All Cameras')

# Plot evaluated AU as heatmap
evaluated_mask = ~np.isnan(au_errors)
p = ax.scatter(positions_all[evaluated_mask, 0],
               positions_all[evaluated_mask, 1],
               positions_all[evaluated_mask, 2],
               c=au_errors[evaluated_mask],
               cmap='turbo',
               s=100,
               label='AU Estimated Cameras')

# Labels & layout
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Aleatoric Uncertainty Heatmap\n(AU_trans per Camera Pose)")
fig.colorbar(p, ax=ax, label='Translation AU')
ax.legend()
plt.tight_layout()
plt.show()
