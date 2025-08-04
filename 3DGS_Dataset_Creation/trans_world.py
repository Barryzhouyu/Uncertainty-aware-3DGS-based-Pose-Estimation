import os
import json
import numpy as np

image_dir = "/home/roar/Desktop/holo3/images"
pose_dir = "/home/roar/Desktop/holo3"
pose_path = os.path.join(pose_dir, "poses.npy")
output_json_path = os.path.join(pose_dir, "train_3dgs.json")
# Intrinsics
width, height = 1278, 718
fx = fy = 639.39338914829489
cx, cy = 639, 359

# Load poses
poses = np.load(pose_path)
assert poses.shape[1:] == (4, 4), "poses.npy must be N x 4 x 4"
positions = np.array([pose[:3, 3] for pose in poses])

# Compute center and scale
center = np.mean(positions, axis=0)
scale = 1.0 / np.max(np.linalg.norm(positions - center, axis=1))
print(f"📍 Scene center: {center}")
print(f"📏 Scene scale: {scale}")

# Load image filenames
image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

def parse_coords(filename):
    stripped = filename.replace(".png", "").replace("[", "").replace("]", "").replace("'", "")
    return np.array(list(map(float, stripped.split(", "))))

matched_poses = []
unmatched_images = []
used_pose_indices = set()

for img_file in image_files:
    img_coords = parse_coords(img_file)
    dists = np.linalg.norm(positions - img_coords, axis=1)
    best_idx = np.argmin(dists)
    min_dist = dists[best_idx]

    if min_dist < 0.1 and best_idx not in used_pose_indices:
        used_pose_indices.add(best_idx)

        # Apply normalization
        pose_copy = np.copy(poses[best_idx])
        pose_copy[:3, 3] = scale * (pose_copy[:3, 3] - center)

        matched_poses.append({
            "file_path": f"images/{img_file}",
            "transform_matrix": pose_copy.tolist(),
            "camera_id": 1
        })
    else:
        unmatched_images.append((img_file, min_dist))

# Create JSON
final_json = {
    "camera_intrinsics": {
        "1": {
            "width": width,
            "height": height,
            "focal_length": [fx, fy],
            "principal_point": [cx, cy]
        }
    },
    "scene_center": center.tolist(),
    "scene_scale": scale,
    "frames": matched_poses
}

with open(output_json_path, "w") as f:
    json.dump(final_json, f, indent=4)

print(f"✅ Saved {len(matched_poses)} normalized poses to {output_json_path}")
if unmatched_images:
    print(f"⚠️ {len(unmatched_images)} unmatched images (min distance shown):")
    for img, dist in unmatched_images:
        print(f"{img} — dist: {dist:.4f}")
