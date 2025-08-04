from scipy.spatial.transform import Rotation as R
import numpy as np

poses = np.load("/home/roar/Desktop/straight_line_600/timesteps.npy")

for i, pose in enumerate(poses):
    print(pose)
    # position = pose[:3, 3]
    # rotation_matrix = pose[:3, :3]
    # euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)

    # print(f"Frame {i+1}")
    # print(f"  Position (x, y, z): {position}")
    # print(f"  Orientation (roll, pitch, yaw) in degrees: {euler_angles}")


        #perturb_gaussians(gaussians, sigma_position=0.003217, sigma_color=0.044893, sigma_opacity=0.0, sigma_scale=0.0)
        #perturb_gaussians(gaussians, sigma_position=0.005474, sigma_color=0.019684, sigma_opacity=0.0, sigma_scale=0.0)
        #perturb_gaussians(gaussians, sigma_position=0.001686, sigma_color=0.015804, sigma_opacity=0.0, sigma_scale=0.0)
        #perturb_gaussians(gaussians, sigma_position=0.003691, sigma_color=0.023162, sigma_opacity=0.0, sigma_scale=0.0)
        #perturb_gaussians(gaussians, sigma_position=0.000896, sigma_color=0.030654, sigma_opacity=0.0, sigma_scale=0.0)
        #perturb_gaussians(gaussians, sigma_position=0.002332, sigma_color=0.031388, sigma_opacity=0.0, sigma_scale=0.0)
        #perturb_gaussians(gaussians, sigma_position=0.001985, sigma_color=0.025175, sigma_opacity=0.0, sigma_scale=0.0)
        #perturb_gaussians(gaussians, sigma_position=0.002413, sigma_color=0.021796, sigma_opacity=0.0, sigma_scale=0.0)
        #perturb_gaussians(gaussians, sigma_position=0.002957, sigma_color=0.027382, sigma_opacity=0.0, sigma_scale=0.0)
        #perturb_gaussians(gaussians, sigma_position=0.001354, sigma_color=0.040245, sigma_opacity=0.0, sigma_scale=0.0)