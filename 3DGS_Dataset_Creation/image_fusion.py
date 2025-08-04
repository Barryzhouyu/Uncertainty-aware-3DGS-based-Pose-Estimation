import os
import json
import shutil

base_dir = '/home/roar3/Desktop/holo_image_3dgs'
out_dir = os.path.join(base_dir, 'hope')
os.makedirs(out_dir, exist_ok=True)

merged_json = []

for sub in sorted(os.listdir(base_dir)):
    subdir = os.path.join(base_dir, sub)
    if not os.path.isdir(subdir) or sub == 'merged':
        continue
    pose_json_path = os.path.join(subdir, 'pose_records.json')
    if not os.path.exists(pose_json_path):
        continue
    with open(pose_json_path, 'r') as f:
        poses = json.load(f)
    merged_json.extend(poses)
    # Copy images (do not rename)
    for fname in os.listdir(subdir):
        if fname.endswith('.png'):
            src = os.path.join(subdir, fname)
            dst = os.path.join(out_dir, fname)
            if os.path.exists(dst):
                print(f"Warning: {dst} already exists! Skipping.")
                continue
            shutil.copy(src, dst)

# Save merged pose_records.json
with open(os.path.join(out_dir, 'pose_records.json'), 'w') as f:
    json.dump(merged_json, f, indent=2)

print(f"Done! Merged images and JSON are in {out_dir}")
