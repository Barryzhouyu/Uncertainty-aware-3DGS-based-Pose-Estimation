import json
import numpy as np

# Load data
with open("/home/roar/Desktop/train_3dgs_sorted_by_x.json", "r") as f:
    raw_data = json.load(f)

with open("/home/roar/Desktop/camera_poses.json", "r") as f:
    trained_data = json.load(f)

# Extract translations
raw_translations = []
trained_translations = []

for idx, trained_pose in enumerate(trained_data):
    if idx >= len(raw_data["frames"]):
        break
    raw_t = np.array(raw_data["frames"][idx]["transform_matrix"])[:3, 3]
    trained_t = np.array(trained_pose["t"])

    raw_translations.append(raw_t)
    trained_translations.append(trained_t)

raw_translations = np.array(raw_translations)
trained_translations = np.array(trained_translations)

# Convert to homogeneous coordinates
trained_homo = np.hstack([trained_translations, np.ones((trained_translations.shape[0], 1))])

# Solve for M such that: trained_homo @ M ≈ raw_translations
M_inv, _, _, _ = np.linalg.lstsq(trained_homo, raw_translations, rcond=None)

# Function to apply M_inv to a new trained pose
def map_trained_to_raw_using_M(tvec):
    t_homo = np.append(tvec, 1.0)
    return t_homo @ M_inv

# Example test
t_test = np.array([1.84312040, -1.61199776, -4.96307803])
t_recovered = map_trained_to_raw_using_M(t_test)

print("Recovered raw translation:", t_recovered)

