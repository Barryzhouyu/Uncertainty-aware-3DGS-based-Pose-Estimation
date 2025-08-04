import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# === Paths ===
est_poses_dir = "/home/roar/gaussian-splatting/output/holo_6/est_poses"
# gt_path = "/home/roar/Desktop/coke_all/coke_gt.json"
# test_txt = "/home/roar/Desktop/test_frames.txt"

gt_path = "/home/roar/Desktop/h_n/transforms.json"
test_txt = "/home/roar/Desktop/T2/holo_test_frames.txt"

# === Load test frame names ===
with open(test_txt, "r") as f:
    test_names = set(line.strip() for line in f)

# === Load Ground Truth Poses ===
with open(gt_path, "r") as f:
    gt_data = json.load(f)

gt_poses = {}
positions = []
filenames = []

for frame in gt_data["frames"]:
    fname = os.path.basename(frame["file_path"])
    matrix = np.array(frame["transform_matrix"])
    #idx = int(fname.split("_")[1].split(".")[0])
    idx = int(os.path.splitext(fname)[0])

    gt_poses[idx] = {
        "R": matrix[:3, :3],
        "t": matrix[:3, 3]
    }

    positions.append(matrix[:3, 3])
    filenames.append(fname)

positions = np.array(positions)
filenames = np.array(filenames)

# === Compute distance and region clusters ===
distances = np.linalg.norm(positions, axis=1)
sorted_indices = np.argsort(distances)
part1_idx = sorted_indices[:80]
part2_idx = sorted_indices[80:160]
part3_idx = sorted_indices[160:]

part1_frames = filenames[part1_idx]
part2_frames = filenames[part2_idx]
part3_frames = filenames[part3_idx]

# === Helper Functions ===
def load_estimated_poses(folder_path):
    pose_list = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".json"):
            with open(os.path.join(folder_path, fname), "r") as f:
                data = json.load(f)
                if "frames" not in data or not data["frames"]:
                    print(f"Skipping {fname}: missing 'frames'")
                    continue
                mat = np.array(data["frames"][0]["transform_matrix"])
                pose_list.append(mat)
    return pose_list


def pose_to_rt(matrix):
    Rmat = matrix[:3, :3]
    tvec = matrix[:3, 3]
    rvec = R.from_matrix(Rmat).as_rotvec()
    return rvec, tvec

# === Main Loop ===
results = []

for folder in sorted(os.listdir(est_poses_dir)):
    if not folder.startswith("f_") or not folder.endswith("_Epose"):
        continue

    frame_id = int(folder.split("_")[1])
    folder_path = os.path.join(est_poses_dir, folder)

    est_matrices = load_estimated_poses(folder_path)
    est_rvecs, est_tvecs = [], []

    for mat in est_matrices:
        rvec, tvec = pose_to_rt(mat)
        est_rvecs.append(rvec)
        est_tvecs.append(tvec)

    est_rvecs = np.stack(est_rvecs)
    est_tvecs = np.stack(est_tvecs)

    mean_rvec = np.mean(est_rvecs, axis=0)
    mean_tvec = np.mean(est_tvecs, axis=0)

    std_rvec = np.std(est_rvecs, axis=0)
    std_tvec = np.std(est_tvecs, axis=0)

    EU_rot = np.mean(np.linalg.norm(est_rvecs - mean_rvec, axis=1)**2)
    EU_trans = np.mean(np.linalg.norm(est_tvecs - mean_tvec, axis=1)**2)

    if frame_id in gt_poses:
        gt_rvec = R.from_matrix(gt_poses[frame_id]["R"]).as_rotvec()
        gt_tvec = gt_poses[frame_id]["t"]

        rot_error = np.linalg.norm(mean_rvec - gt_rvec) / 44
        trans_error = np.linalg.norm(mean_tvec - gt_tvec) / 44

        results.append({
            "frame": frame_id,
            "frame_name": f"frame_{frame_id:03d}.jpg",
            "rotation_error": rot_error,
            "translation_error": trans_error,
            "rotation_std": std_rvec.tolist(),
            "translation_std": std_tvec.tolist(),
            "EU_rot": EU_rot,
            "EU_trans": EU_trans
        })

# === Extract arrays for plotting ===
frame_ids = np.array([r["frame"] for r in results])
frame_names = [r["frame_name"] for r in results]
EU_trans_vals = np.array([r["EU_trans"] for r in results])

eu_trans_data = np.column_stack(([r["frame"] for r in results], [r["EU_trans"] for r in results]))
eu_rot_data = np.column_stack(([r["frame"] for r in results], [r["EU_rot"] for r in results]))

np.savetxt("/home/roar/Desktop/epistemic_uncertainty_trans.txt", eu_trans_data, fmt="%d %.6f", header="frame_id EU_trans")
np.savetxt("/home/roar/Desktop/epistemic_uncertainty_rot.txt", eu_rot_data, fmt="%d %.6f", header="frame_id EU_rot")
EU_trans_vals_clipped = np.minimum(EU_trans_vals, 1)

# === Identify 10 closest and 10 farthest test frames ===
# Load test frame list again to match names
with open(test_txt, "r") as f:
    test_names = set(line.strip() for line in f)

# Create a map: frame name → position
frame_name_to_pos = dict(zip(filenames, positions))

# Filter positions of test frames
test_positions = []
test_fnames = []
for name in test_names:
    if name in frame_name_to_pos:
        test_positions.append(frame_name_to_pos[name])
        test_fnames.append(name)

test_positions = np.array(test_positions)
test_distances = np.linalg.norm(test_positions, axis=1)
sorted_idx = np.argsort(test_distances)

# 10 closest and farthest test frame names
closest_test_frames = set(np.array(test_fnames)[sorted_idx[:10]])
farthest_test_frames = set(np.array(test_fnames)[sorted_idx[-10:]])

# === Assign colors for bar plot ===
colors = []
for fname in frame_names:
    if fname in closest_test_frames:
        colors.append("green")
    elif fname in farthest_test_frames:
        colors.append("red")
    elif fname in test_names:
        colors.append("gray")  # other test frame
    else:
        colors.append("lightgray")  # not a test frame at all

# === Plot EU_trans (clipped for visualization only) ===
plt.figure(figsize=(12, 6))
plt.bar(frame_names, EU_trans_vals_clipped, color=colors)
plt.xlabel("Frame Name")
plt.ylabel("Translation Epistemic Uncertainty (Variance)")
plt.title("Translation Epistemic Uncertainty per Frame (Capped at 1.0)")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.legend(handles=[
    plt.Rectangle((0,0),1,1,color='red', label='10 Closest Test Images'),
    plt.Rectangle((0,0),1,1,color='yellow', label='10 Farthest Test Images'),
    plt.Rectangle((0,0),1,1,color='gray', label='Other Test Images'),
    plt.Rectangle((0,0),1,1,color='lightgray', label='Non-Test Images')
])
plt.show()


