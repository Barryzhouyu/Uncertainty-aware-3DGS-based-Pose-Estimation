import os
import shutil
import json

root_dir = "/home/roar/Desktop/holo_image_3dgs"
output_dir = "/home/roar/Desktop/rov"
image_out_dir = os.path.join(output_dir, "images")
os.makedirs(image_out_dir, exist_ok=True)

all_poses = {}
copied = set()

# Traverse numbered folders
for folder_name in sorted(os.listdir(root_dir), key=lambda x: int(x) if x.isdigit() else -1):
    folder_path = os.path.join(root_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    # Load pose_records.json if exists
    pose_path = os.path.join(folder_path, "pose_records.json")
    if os.path.exists(pose_path):
        with open(pose_path, "r") as f:
            poses = json.load(f)
            for p in poses:
                frame = p["frame"]
                if frame not in all_poses:
                    all_poses[frame] = p

    # Copy images and print matched info
    for file in os.listdir(folder_path):
        if file.endswith(".png") and file not in copied:
            shutil.copy(os.path.join(folder_path, file), os.path.join(image_out_dir, file))
            copied.add(file)
            if file in all_poses:
                print(f"✅ Matched: {file} ↔ pose at {all_poses[file]['location']}")
            else:
                print(f"⚠️  Warning: {file} has no matching pose entry!")

# Save merged pose JSON
with open(os.path.join(output_dir, "pose_records.json"), "w") as f:
    json.dump(list(all_poses.values()), f, indent=4)

print(f"\n✅ Total merged {len(copied)} images and {len(all_poses)} poses.")
