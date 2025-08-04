import os
import json
import numpy as np

# Set paths
colmap_path = "/home/roar/jsea/sparse/0"
images_path = "/home/roar/jsea/images_wb"
output_json = "/home/roar/jsea/train_3dgs.json"

def read_cameras(file):
    """Extract intrinsics from COLMAP cameras.txt."""
    with open(file, 'r') as f:
        lines = f.readlines()
    
    cameras = {}
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue
        data = line.split()
        cam_id = int(data[0])
        model = data[1]  # Camera model
        width, height = map(int, data[2:4])
        
        if model == "PINHOLE":
            fx = float(data[4])
            fy = float(data[5])
            cx = float(data[6])
            cy = float(data[7])
        elif model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            fx = fy = float(data[4])
            cx = float(data[5])
            cy = float(data[6])
        elif model in ["RADIAL", "OPENCV"]:
            fx = float(data[4])
            fy = float(data[5])
            cx = float(data[6])
            cy = float(data[7])
        else:
            raise ValueError(f"Unsupported camera model: {model}")

        cameras[cam_id] = {
            "width": width,
            "height": height,
            "focal_length": [fx, fy],
            "principal_point": [cx, cy]
        }
    return cameras

def read_images(file):
    """Extract poses from COLMAP images.txt."""
    with open(file, 'r') as f:
        lines = f.readlines()
    
    images = []
    for i in range(0, len(lines), 2):  # Every image has 2 lines
        if lines[i].startswith('#'):
            continue
        data = lines[i].split()
        img_id = int(data[0])
        qw, qx, qy, qz = map(float, data[1:5])  # Quaternion rotation
        tx, ty, tz = map(float, data[5:8])  # Translation
        cam_id = int(data[8])
        img_name = data[9]

        # Convert quaternion to rotation matrix
        q = np.array([qw, qx, qy, qz])
        R = quaternion_to_rotation_matrix(q)

        # Convert to 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3] = [tx, ty, tz]

        images.append({
            "file_path": os.path.join(images_path, img_name),
            "transform_matrix": transform_matrix.tolist(),
            "camera_id": cam_id
        })
    return images

def quaternion_to_rotation_matrix(q):
    """Convert quaternion (qw, qx, qy, qz) to a rotation matrix."""
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def generate_json(cameras, images, output_file):
    """Generate a JSON file for 3D Gaussian Splatting."""
    json_data = {
        "camera_intrinsics": cameras,
        "frames": images
    }
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"JSON saved to: {output_file}")

# File paths
cameras_file = os.path.join(colmap_path, "cameras.txt")
images_file = os.path.join(colmap_path, "images.txt")

# Read and process data
cameras = read_cameras(cameras_file)
images = read_images(images_file)

# Save JSON file
generate_json(cameras, images, output_json)

