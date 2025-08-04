import open3d as o3d
import numpy as np

# Load the point cloud
pcd = o3d.io.read_point_cloud("/home/roar/gaussian-splatting/output/holo_o/point_cloud/iteration_20000/point_cloud.ply")
points = np.asarray(pcd.points)

print("Loaded", len(points), "points.")
print("First few points:\n", points[:5])
print("Range of Z:", np.min(points[:, 2]), "to", np.max(points[:, 2]))
