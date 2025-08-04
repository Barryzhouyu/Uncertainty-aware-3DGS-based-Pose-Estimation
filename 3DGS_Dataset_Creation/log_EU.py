# import pandas as pd
# import numpy as np

# # === Load the file ===
# df = pd.read_csv("/home/roar3/Desktop/hope_2/epistemic_uncertainty_trans.txt", sep='\s+')

# # === Compute log(EU_trans) and overwrite the column ===
# epsilon = 1e-10
# df['EU_trans'] = np.log(df['EU_trans'] + epsilon)

# # === Keep only the required columns and save ===
# df[['frame_id', 'EU_trans']].to_csv("/home/roar3/Desktop/hope_2/log_epistemic_uncertainty_trans.txt", index=False, sep='\t')

import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ── Paths ──────────────────────────────────────────────────────────────────────
camera_json            = "/home/roar3/Desktop/hope_2/undistorted/transforms.json"
translation_errors_txt = "/home/roar3/Desktop/hope_2/log_epistemic_uncertainty_trans.txt"
mapping_path           = "/home/roar3/Desktop/3_test_images/rename_pairs.txt" 

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

### ===== OPTION 1: Show Magnitude Only (use LogNorm + abs) =====
# tested_eu = np.array([max(abs(eu_dict[int(fid)]), 1e-5) for fid in frame_ids_all[tested_mask]])
# cmap = 'turbo'
# norm = LogNorm(vmin=1e-5, vmax=1e2)  # adjust vmax as needed
# colorbar_label = 'Translation EU Magnitude (log scale)'

### ===== OPTION 2: Show Signed EU (no LogNorm, use diverging colormap) =====
tested_eu = np.array([eu_dict[int(fid)] for fid in frame_ids_all[tested_mask]])
cmap = 'seismic'
abs_max = max(abs(min(tested_eu)), abs(max(tested_eu)))
norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)
colorbar_label = 'Signed Translation EU'

# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection='3d')

# 1) untested → gray
ax.scatter(positions_all[untested_mask, 0],
           positions_all[untested_mask, 1],
           positions_all[untested_mask, 2],
           color='lightgray', s=50, label='Untested poses')

# 2) tested → colored by signed EU
if tested_pos.size:
    sc = ax.scatter(tested_pos[:, 0], tested_pos[:, 1], tested_pos[:, 2],
                    c=tested_eu,
                    cmap=cmap,
                    norm=norm,
                    s=200, marker='o', linewidths=0,
                    label='Tested cameras (IG)')
    fig.colorbar(sc, ax=ax, label=colorbar_label)

# 3) Optional: mark origin or special point
ax.scatter(6, 0, 0,
           color='purple',
           marker='*',
           s=300,
           label='Origin (6,0,0)')

# labels & layout
ax.set_xlabel("X (m)", fontweight='bold')
ax.set_ylabel("Y (m)", fontweight='bold')
ax.set_zlabel("Z (m)", fontweight='bold')
ax.set_title('Camera Poses — Gray = Untested, Color = EU', fontweight='bold', fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()
