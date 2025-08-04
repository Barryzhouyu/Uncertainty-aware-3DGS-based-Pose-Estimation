import os
import json
import shutil

# Paths
base_dir = "/home/roar/Desktop/playroom_un"
#base_dir_1 = "/home/roar/Desktop/Dataset/playroom"
image_dir = os.path.join(base_dir, "images")
train_json = os.path.join(base_dir, "train_3dgs.json")
test_json = os.path.join(base_dir, "test_3dgs.json")

# Output folders
train_out = os.path.join(base_dir, "train_images")
test_out = os.path.join(base_dir, "test_images")
os.makedirs(train_out, exist_ok=True)
os.makedirs(test_out, exist_ok=True)

def copy_images(json_path, out_dir):
    with open(json_path, "r") as f:
        data = json.load(f)
    for frame in data["frames"]:
        src = frame["file_path"]
        if not os.path.isabs(src):  # Just in case paths are relative
            src = os.path.join(image_dir, src)
        filename = os.path.basename(src)
        dst = os.path.join(out_dir, filename)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Warning: {src} does not exist!")

# Do the split
copy_images(train_json, train_out)
copy_images(test_json, test_out)

print("Done splitting images.")
