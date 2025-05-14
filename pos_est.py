import json
import numpy as np
import cv2
import os
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


#ACTUAL_IMAGE_PATH = "/home/roar/Desktop/playroom_split_results/test_split/DSC05760.jpg"
#ACTUAL_IMAGE_PATH = "/home/roar/Desktop/coke/test_images/DSC05760.jpg"
#ACTUAL_IMAGE_PATH = "/home/roar/Desktop/coke_all/images/frame_039.jpg"
#ACTUAL_IMAGE_PATH = "/home/roar/Desktop/coke_all/images/frame_118.jpg"
#ACTUAL_IMAGE_PATH = "/home/roar/Desktop/coke_all/images/frame_003.jpg"
#ACTUAL_IMAGE_PATH = "/home/roar/Desktop/coke_all/images/frame_013.jpg"
#ACTUAL_IMAGE_PATH = "/home/roar/Desktop/coke_all/images/frame_118.jpg"
#ACTUAL_IMAGE_PATH = "/home/roar/Desktop/coke_all/images/frame_108.jpg"
#ACTUAL_IMAGE_PATH = "/home/roar/Desktop/coke_all/images/frame_107.jpg"
#ACTUAL_IMAGE_PATH = "/home/roar/Desktop/coke_all/images/frame_075.jpg"
#ACTUAL_IMAGE_PATH = "/home/roar/Desktop/coke_all/images/frame_154.jpg"
#ACTUAL_IMAGE_PATH = "/home/roar/Desktop/coke_all/images/frame_153.jpg"
#ACTUAL_IMAGE_PATH = "/home/roar/Desktop/coke_all/images/frame_156.jpg"
#ACTUAL_IMAGE_PATH = "/home/roar/Desktop/coke_all/images/frame_165.jpg"
#ACTUAL_IMAGE_PATH = "/home/roar/Desktop/coke_all/images/frame_157.jpg"
#ACTUAL_IMAGE_PATH = "/home/roar/Desktop/coke_all/images/frame_160.jpg"
#ACTUAL_IMAGE_PATH = "/home/roar/Desktop/coke_all/images/frame_162.jpg"
#ACTUAL_IMAGE_PATH = "/home/roar/Desktop/coke_all/images/frame_078.jpg"
ACTUAL_IMAGE_PATH = "/home/roar/Desktop/coke_all/images/frame_138.jpg"
RENDERED_IMAGE_PATH = "/home/roar/gaussian-splatting/output/coke/custom/ours_20000/renders/custom_render.png"
POSE_JSON_PATH = "/home/roar/Desktop/guessed_pose.json"
POINT_CLOUD_PATH = "/home/roar/gaussian-splatting/output/coke/point_cloud/iteration_20000/point_cloud.ply"
MODEL_PATH = "/home/roar/gaussian-splatting/output/coke/"


###################coke##################
camera_matrix = np.array([
    [340.82879583167528, 0, 622],     # fx, 0, cx
    [0, 340.82879583167528, 342.5],   # 0, fy, cy
    [0, 0, 1]
], dtype=np.float32)



# 🔹 Load Pose from JSON
def load_pose(json_path):
    """Load rvec & tvec from guessed pose JSON file."""
    with open(json_path, "r") as f:
        pose_data = json.load(f)

    transform_matrix = np.array(pose_data["frames"][0]["transform_matrix"])
    R = transform_matrix[:3, :3]
    tvec = transform_matrix[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)  # Convert rotation matrix to vector

    return rvec, tvec, pose_data

# 🔹 Load 3D Point Cloud
def load_point_cloud(ply_path):
    """Loads a point cloud from a PLY file and returns an Nx3 NumPy array."""
    pcd = o3d.io.read_point_cloud(ply_path)
    return np.asarray(pcd.points)  # Extracts the 3D points as a NumPy array

# 🔹 Project 3D Point Cloud to 2D
def project_3d_to_2d(pts_3d, camera_matrix, rvec, tvec):
    """Projects 3D points into 2D image space using camera parameters."""
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, camera_matrix, None)
    return pts_2d.squeeze()  # Remove extra dimensions

# 🔹 Extract 2D-2D Correspondences Using Feature Matching
def match_features():
    """Match features between actual and rendered images using ORB."""
    actual_img = cv2.imread(ACTUAL_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    rendered_img = cv2.imread(RENDERED_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

    # Initialize ORB
    orb = cv2.ORB_create(nfeatures=2000)  # Increased features
    kp1, des1 = orb.detectAndCompute(actual_img, None)
    kp2, des2 = orb.detectAndCompute(rendered_img, None)

    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)[:500]  # More matches

    # Extract corresponding 2D points
    pts_actual = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts_rendered = np.float32([kp2[m.trainIdx].pt for m in matches])

    print(f"Matched {len(pts_actual)} feature points")

    return pts_actual, pts_rendered

# 🔹 Convert 2D-2D Matches to 2D-3D Correspondences
def get_2d_3d_correspondences(pts_actual, pts_rendered, camera_matrix, rvec, tvec, point_cloud):
    """Converts 2D-2D correspondences into 2D-3D pairs using the point cloud."""
    projected_2d = project_3d_to_2d(point_cloud, camera_matrix, rvec, tvec)

    pts_3d = []
    pts_2d = []
    seen_3d_points = set()  # Prevent duplicate 3D matches

    for i in range(len(pts_rendered)):
        distances = np.linalg.norm(projected_2d - pts_rendered[i], axis=1)
        nearest_idx = np.argmin(distances)
        nearest_3d = tuple(point_cloud[nearest_idx])  # Convert to tuple for set operations

        if nearest_3d not in seen_3d_points:
            seen_3d_points.add(nearest_3d)
            pts_3d.append(point_cloud[nearest_idx])
            pts_2d.append(pts_actual[i])

    return np.array(pts_2d, dtype=np.float32), np.array(pts_3d, dtype=np.float32)

def compute_reprojection_error(pts_2d, pts_3d, camera_matrix, rvec, tvec):
    """
    Computes the reprojection error by projecting 3D points into 2D and
    measuring the difference with actual 2D matches.
    """
    projected_pts, _ = cv2.projectPoints(pts_3d, rvec, tvec, camera_matrix, None)
    projected_pts = projected_pts.squeeze()

    # Compute Euclidean distance between projected and actual 2D points
    error = np.linalg.norm(pts_2d - projected_pts, axis=1)
    mean_error = np.mean(error)

    return mean_error

def plot_pose_error_trajectory(rvec_list, tvec_list):
    """
    Plots Euler angle errors (roll, pitch, yaw) and translation L2 error over iterations.
    """

    # === Ground truth rotation matrix and translation vector ===
    # R_gt = np.array([
    #     [0.9912667460752744,  0.04539799482941835,  -0.12381138958434183],
    #     [-0.03090563015007028,  1.08382176549242,   0.1319257653529341],
    #     [0.1286621277214873,  -0.1271951363441222,  0.9834907033512148]
    # ])
    # t_gt = np.array([2.6583017666251947, 1.2040469601821748, 4.53092570245018])
    
    # R_gt = np.array([
    #     [ 0.9997651874074769,  -0.01223078357134774,   0.017887923868694708],
    #     [ 0.017243190569452765,  0.9489776094171887,  -0.3148716710087194],
    #     [-0.013124116239893558,  0.3151061801330919,   0.9489656752564095]
    # ])

    # t_gt = np.array([
    #    -1.25437111168831167,
    #    -0.3542506714700255,
    #     3.791449111719658
    # ])
    
    #########################playroom split#############################

    # R_gt = np.array([
    #     [ 0.9912667460752744,  0.04539799482941835,  -0.12381138958434183],
    #     [-0.028900563015007028,  0.9908382176549242,   0.1319256756352941],
    #     [ 0.1286662177214873,  -0.1271953163441222,   0.9834970035121438]
    # ])

    # t_gt = np.array([2.6583017666251947, 1.1840469061821748, 4.63092570245018])
    
    
    #########################coke003#############################
    R_gt = np.array([
        [0.9995760301076153, -0.0019346787482901427, -0.029051971575793496],
        [0.002143888367292467, 0.9999719841343944,    0.00717179816998596],
        [0.02903728253415575, -0.007231041727395881,  0.9995521738551563]
    ])

    t_gt = np.array([
        3.729789058561073,
        1.2698616033619443,
    -0.4110934986660983
    ])

    # Convert GT rotation matrix to Euler angles
    euler_gt = R.from_matrix(R_gt).as_euler('xyz', degrees=True)

    # === Containers for error over time ===
    roll_errors = []
    pitch_errors = []
    yaw_errors = []
    translation_errors = []

    for rvec, tvec in zip(rvec_list, tvec_list):
        # Rotation matrix from rvec
        R_est, _ = cv2.Rodrigues(rvec)
        euler_est = R.from_matrix(R_est).as_euler('xyz', degrees=True)
        angle_error = euler_est - euler_gt

        # Absolute angle error per axis
        roll_errors.append(abs(angle_error[0]))
        pitch_errors.append(abs(angle_error[1]))
        yaw_errors.append(abs(angle_error[2]))

        # Translation error
        t_est = np.array(tvec).flatten()
        trans_error = np.linalg.norm(t_est - t_gt)
        translation_errors.append(trans_error)

    # === Plotting ===
    iterations = range(1, len(rvec_list) + 1)

    plt.figure(figsize=(8, 6))

    # Rotation error subplot
    plt.subplot(2, 1, 1)
    plt.plot(iterations, roll_errors, label='Roll Error (°)', marker='o', color='orange', linewidth=3, markersize=8)
    plt.plot(iterations, pitch_errors, label='Pitch Error (°)', marker='o', color='green', linewidth=3, markersize=8)
    plt.plot(iterations, yaw_errors, label='Yaw Error (°)', marker='o', color='blue', linewidth=3, markersize=8)
    plt.ylabel("Euler Angle Error (degrees)")
    plt.title("Rotation Errors")
    plt.legend()
    plt.grid(True)

    # Translation error subplot
    plt.subplot(2, 1, 2)
    plt.plot(iterations, translation_errors, label='Translation Error (L2)', marker='o', color='hotpink', linewidth=3, markersize=8)
    plt.xlabel("Iteration")
    plt.ylabel("Translation Error")
    plt.title("Translation Error over Iterations")
    plt.grid(True)

    plt.tight_layout()

    # === Save figure ===
    save_path = f"/home/roar/Desktop/norm_plot_138/pose_error_custom_sample_{perturb_id}.png"
    plt.savefig(save_path)
    print(f"Saved pose error plot to {save_path}")

    #plt.show()

    
def optimize_pose_pnp(perturb_id, iterations=50):
    """Optimize pose using PnP with a fixed SH perturbation."""
    rvec, tvec, pose_data = load_pose(POSE_JSON_PATH)
    point_cloud = load_point_cloud(POINT_CLOUD_PATH)

    loss_history = []
    prev_loss = None
    no_improvement_count = 0
    LOSS_STOP_THRESHOLD = 3
    MIN_ITERATIONS = 3
    rvec_list = []
    tvec_list = []

    RENDER_CMD_TEMPLATE = (
        "python /home/roar/gaussian-splatting/render_cus_2.py "
        "--model_path {model_path} "
        "--iteration 20000 "
        "--pose {pose_json} "
        "--features_dc {features_dc}"
    )

    features_dc_path = os.path.join(
        MODEL_PATH, f"custom_sample_{perturb_id}", "features_dc.pt"
    )

    RENDER_CMD = RENDER_CMD_TEMPLATE.format(
        model_path=MODEL_PATH,
        pose_json=POSE_JSON_PATH,
        features_dc=features_dc_path
    )

    for i in range(iterations):
        print(f"\n Iteration {i+1}/{iterations}: Rendering with custom_sample_{perturb_id}")
        os.system(RENDER_CMD)

        pts_actual, pts_rendered = match_features()
        pts_2d, pts_3d = get_2d_3d_correspondences(
            pts_actual, pts_rendered, camera_matrix, rvec, tvec, point_cloud
        )

        if len(pts_2d) < 4:
            print("Not enough correspondences. Skipping iteration.")
            continue

        success, rvec_new, tvec_new, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, camera_matrix, None, reprojectionError=15.0, iterationsCount=100
        )

        if not success:
            print("RANSAC Failed. Trying normal PnP...")
            success, rvec_new, tvec_new = cv2.solvePnP(
                pts_3d, pts_2d, camera_matrix, None
            )

        if success:
            # Gradual update for smoothing
            alpha = 0.1
            rvec = (1 - alpha) * rvec + alpha * rvec_new
            tvec = (1 - alpha) * tvec + alpha * tvec_new

            rvec_list.append(rvec.flatten())
            tvec_list.append(tvec.flatten())

            # Calculate reprojection error
            loss = compute_reprojection_error(pts_2d, pts_3d, camera_matrix, rvec, tvec)
            loss_history.append(loss)
            print(f" Reprojection Loss: {loss:.4f} pixels")

            if prev_loss is not None and i >= MIN_ITERATIONS:
                loss_delta = abs(prev_loss - loss)
                if loss_delta < LOSS_STOP_THRESHOLD:
                    no_improvement_count += 1
                    print(f"Stable loss for {no_improvement_count} iterations")
                else:
                    no_improvement_count = 0

                if no_improvement_count >= 3:
                    print(" Early stopping: Loss stabilized")
                    break

            prev_loss = loss

            # Save pose to JSON
            R_mat, _ = cv2.Rodrigues(rvec)
            transform = np.eye(4)
            transform[:3, :3] = R_mat
            transform[:3, 3] = tvec.flatten()
            pose_data["frames"][0]["transform_matrix"] = transform.tolist()

            with open(POSE_JSON_PATH, "w") as f:
                json.dump(pose_data, f, indent=4)
            
            output_pose_path = os.path.join(MODEL_PATH, f"pose_{perturb_id}.json")
            with open(output_pose_path, "w") as f:
                json.dump(pose_data, f, indent=4)
            print(f"Saved final pose to {output_pose_path}")

        else:
            print("PnP Failed")

    plot_loss(loss_history)
    plot_pose_error_trajectory(rvec_list, tvec_list)
    print(f"Optimization complete for custom_sample_{perturb_id}")



def plot_loss(loss_history):
    """Plots the reprojection loss vs. iteration."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-', color='b')
    plt.xlabel("Iteration")
    plt.ylabel("Reprojection Loss (pixels)")
    plt.title("Reprojection Loss vs Iteration")
    plt.grid(True)
    #plt.show()

for perturb_id in range(1, 11):
    print(f"\n==============================")
    print(f"Starting pose optimization with custom_sample_{perturb_id}")
    print(f"==============================")
    os.system("python /home/roar/Desktop/3DGS_Dataset_creation/guessed_pose_json_create.py")
    optimize_pose_pnp(perturb_id=perturb_id, iterations=50)



