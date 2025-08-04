import os
import numpy as np

# === CONFIG ===
image_dir = "/home/roar/Desktop/nb/images"
pose_file = "/home/roar/Desktop/nb/poses.npy"
output_dir = "/home/roar/Desktop/nb/sparse/0"
os.makedirs(output_dir, exist_ok=True)

# === INTRINSICS ===
W, H = 1280, 720
fx = fy = 1040.0
cx, cy = 640, 360

# === LOAD POSES ===
poses = np.load(pose_file)  # Shape: (N, 4, 4)
image_names = sorted([f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg"))])

assert len(poses) == len(image_names), "Mismatch between poses and image files."

# === SAVE cameras.txt ===
with open(os.path.join(output_dir, "cameras.txt"), 'w') as f:
    f.write("# Camera list with one line of data per camera:\n")
    f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    f.write(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")

# === SAVE images.txt ===
with open(os.path.join(output_dir, "images.txt"), 'w') as f:
    f.write("# Image list with two lines of data per image:\n")
    f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
    for i, (pose, name) in enumerate(zip(poses, image_names)):
        R = pose[:3, :3]
        t = pose[:3, 3]
        # Convert to quaternion
        from scipy.spatial.transform import Rotation as Rscipy
        q = Rscipy.from_matrix(R).as_quat()  # [x, y, z, w]
        qw, qx, qy, qz = q[3], q[0], q[1], q[2]
        f.write(f"{i+1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {name}\n")
        f.write("\n")

# === SAVE points3D.txt (dummy) ===
with open(os.path.join(output_dir, "points3D.txt"), 'w') as f:
    f.write("# Dummy file — not needed by 3DGS.\n")
    f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
