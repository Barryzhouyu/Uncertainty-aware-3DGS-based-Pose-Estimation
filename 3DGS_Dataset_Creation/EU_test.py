import numpy as np
import json
from scipy.spatial.transform import Rotation as R
import os

def load_all_available_poses(pose_dir):
    """Load all pose files in the directory, regardless of naming/number"""
    translations = []
    rotations = []
    
    for filename in os.listdir(pose_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(pose_dir, filename)
            try:
                with open(file_path, "r") as f:
                    pose_data = json.load(f)
                    transform = np.array(pose_data["frames"][0]["transform_matrix"])
                    t = transform[:3, 3]  # Translation vector
                    R_mat = transform[:3, :3]  # Rotation matrix
                    translations.append(t)
                    rotations.append(R_mat)
            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")
                continue
    
    print(f"Loaded {len(translations)} valid poses from {pose_dir}")
    return np.array(translations), np.array(rotations)

def compute_eu_trans(translations):
    """EUtrans = mean Euclidean distance to centroid (Eq. 1)"""
    centroid = np.mean(translations, axis=0)
    return np.mean(np.linalg.norm(translations - centroid, axis=1)**2)

def compute_eu_rot(rotations):
    """EUrot = mean Frobenius norm to mean rotation (Eq. 2)"""
    quats = np.array([R.from_matrix(R_mat).as_quat() for R_mat in rotations])
    mean_R = R.from_quat(np.mean(quats, axis=0)).as_matrix()
    return np.mean([np.linalg.norm(R_mat - mean_R) for R_mat in rotations])

if __name__ == "__main__":
    pose_dir = "/home/roar/gaussian-splatting/output/coke3/upgraded_est_poses/f_096_Epose"
    translations, rotations = load_all_available_poses(pose_dir)
    
    if len(translations) == 0:
        print("No valid poses found in directory!")
    else:
        print("\n=== Euclidean Uncertainty (EU) ===")
        print(f"EUtrans (Translation): {compute_eu_trans(translations):.6f} units")
        print(f"EUrot (Rotation): {compute_eu_rot(rotations):.6f} (Frobenius norm)")