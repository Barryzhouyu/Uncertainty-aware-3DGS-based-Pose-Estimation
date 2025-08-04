import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

# === Path to estimated pose folders ===
#est_poses_dir = "/home/roar3/variational-3dgs/output/nb1_more/est_poses"
#est_poses_dir = "/home/roar3/variational-3dgs/output/pls/est_poses"
est_poses_dir = "/home/roar3/variational-3dgs/output/hope_2/est_poses"


def load_estimated_poses(folder_path):
    pose_list = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".json"):
            with open(os.path.join(folder_path, fname), "r") as f:
                data = json.load(f)
                if "frames" not in data or not data["frames"]:
                    continue
                mat = np.array(data["frames"][0]["transform_matrix"])
                pose_list.append(mat)
    return pose_list

def pose_to_rt(matrix):
    Rmat = matrix[:3, :3]
    tvec = matrix[:3, 3]
    rvec = R.from_matrix(Rmat).as_rotvec()
    return rvec, tvec

# === Compute EU
trans_results = []
rot_results = []

for folder in sorted(os.listdir(est_poses_dir)):
    if not folder.startswith("f_") or not folder.endswith("_Epose"):
        continue

    frame_id = int(folder.split("_")[1])  # e.g., '001' → 1
    folder_path = os.path.join(est_poses_dir, folder)

    est_matrices = load_estimated_poses(folder_path)
    if len(est_matrices) == 0:
        continue

    est_rvecs, est_tvecs = [], []
    for mat in est_matrices:
        rvec, tvec = pose_to_rt(mat)
        est_rvecs.append(rvec)
        est_tvecs.append(tvec)

    est_rvecs = np.stack(est_rvecs)
    est_tvecs = np.stack(est_tvecs)

    mean_rvec = np.mean(est_rvecs, axis=0)
    mean_tvec = np.mean(est_tvecs, axis=0)

    EU_rot = np.mean(np.linalg.norm(est_rvecs - mean_rvec, axis=1)**2)
    EU_trans = np.mean(np.linalg.norm(est_tvecs - mean_tvec, axis=1)**2)

    trans_results.append((frame_id, EU_trans))
    rot_results.append((frame_id, EU_rot))

# === Save translation EU
print("\n=== Translation Epistemic Uncertainty ===")
with open("/home/roar3/Desktop/hope_2/2_epistemic_uncertainty_trans.txt", "w") as f:
    f.write("frame_id EU_trans\n")
    for frame_id, eu_trans in sorted(trans_results):
        print(f"{frame_id} {eu_trans:.6f}")
        f.write(f"{frame_id} {eu_trans:.6f}\n")

# === Save rotation EU
print("\n=== Rotation Epistemic Uncertainty ===")
with open("/home/roar3/Desktop/hope_2/epistemic_uncertainty_rot.txt", "w") as f:
    f.write("frame_id EU_rot\n")
    for frame_id, eu_rot in sorted(rot_results):
        print(f"{frame_id} {eu_rot:.6f}")
        f.write(f"{frame_id} {eu_rot:.6f}\n")