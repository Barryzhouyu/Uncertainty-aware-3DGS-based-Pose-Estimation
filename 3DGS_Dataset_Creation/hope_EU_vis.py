import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ── Paths ──────────────────────────────────────────────────────────────────────
camera_json            = "/home/roar3/Desktop/hope_2/undistorted/transforms.json"
translation_errors_txt = "/home/roar3/Desktop/4_test_images/epistemic_uncertainty_trans.txt"
mapping_path           = "/home/roar3/Desktop/4_test_images/rename_pairs.txt"  

# ── Load mapping (coordinate filename  →  integer frame_id) ────────────────────
coord_to_id = {}
with open(mapping_path, "r") as f:
    for line in f:
        if "<-" in line:
            new, old = line.strip().split(" <- ")
            frame_id = int(new.split("_")[1].split(".")[0])
            coord_to_id[old] = frame_id 

# ── Load *all* camera poses from JSON ──────────────────────────────────────────
positions_all   = []   # every pose (for gray dots)
frame_ids_all   = []   # matching frame-id  (-1 if no mapping)

with open(camera_json, "r") as f:
    for fr in json.load(f)["frames"]:
        coord_name = os.path.basename(fr["file_path"])      # filename in JSON
        fid        = coord_to_id.get(coord_name, -1)        # -1 ⇒ not evaluated

        mat  = np.array(fr["transform_matrix"])
        pos  = mat[:3, 3]                                   # (x,y,z)
        # pos[2] = -3                                          # flatten z
        # pos[1] *= 0.6                     # scale
        # pos[1] = pos[1] - 346
        positions_all.append(pos)
        frame_ids_all.append(fid)

positions_all = np.asarray(positions_all)
frame_ids_all = np.asarray(frame_ids_all)

# ── Load EU values for tested frames ───────────────────────────────────────────
eu_data        = np.loadtxt(translation_errors_txt, skiprows=1)   # [id, EU]
eu_dict        = {int(fid): val for fid, val in eu_data}

# ── Split into tested / untested for plotting ─────────────────────────────────
tested_mask    = np.isin(frame_ids_all, list(eu_dict.keys()))
untested_mask  = ~tested_mask

tested_pos     = positions_all[tested_mask]
tested_eu      = np.array([max(eu_dict[int(fid)], 1e-6) for fid in frame_ids_all[tested_mask]])
# ensure positive for LogNorm

# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection='3d')

# 1) untested → gray
ax.scatter(positions_all[untested_mask, 0],
           positions_all[untested_mask, 1],
           positions_all[untested_mask, 2],
           color='lightgray', s=30, label='Untested cameras')

# 2) tested  → coloured by EU
if tested_pos.size:
    sc = ax.scatter(tested_pos[:, 0], tested_pos[:, 1], tested_pos[:, 2],
                    c=tested_eu,
                    cmap='turbo',
                    norm=LogNorm(vmin=1e-5, vmax=1e5),
                    s=100, marker='o', linewidths=0,
                    label='Tested cameras (EU)')
    fig.colorbar(sc, ax=ax, label='Translation EU (log scale)')

# labels & layout
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Camera Poses gray = untested, colour = tested (EU)')
ax.legend()
plt.tight_layout();  plt.show()
