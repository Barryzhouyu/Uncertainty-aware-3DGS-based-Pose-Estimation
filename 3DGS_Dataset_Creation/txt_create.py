import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

json_path = "/home/roar/Desktop/playroom_un/train_3dgs.json"
output_dir = "/home/roar/Desktop/playroom_un/sparse/0"
os.makedirs(output_dir, exist_ok=True)

with open(json_path, 'r') as f:
    data = json.load(f)

# Save cameras.txt
cam = data["camera_intrinsics"]["1"]
fx, fy = cam["focal_length"]
cx, cy = cam["principal_point"]
width, height = cam["width"], cam["height"]

with open(os.path.join(output_dir, "cameras.txt"), "w") as f:
    f.write("# Camera list with one line of data per camera:\n")
    f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    f.write(f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

# Save images.txt
with open(os.path.join(output_dir, "images.txt"), "w") as f:
    f.write("# Image list with one line of data per image:\n")
    f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
    f.write("# No 2D keypoints used here\n")

    for i, frame in enumerate(data["frames"]):
        T = np.array(frame["transform_matrix"])
        R_mat = T[:3, :3]
        t = T[:3, 3]
        rot = R.from_matrix(R_mat)
        q = rot.as_quat()  # (x, y, z, w)
        qw, qx, qy, qz = q[3], q[0], q[1], q[2]  # COLMAP: qw qx qy qz
        img_name = os.path.basename(frame["file_path"])
        f.write(f"{i+1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {img_name}\n\n")