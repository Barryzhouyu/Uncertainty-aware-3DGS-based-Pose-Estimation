import json
import numpy as np
import matplotlib.pyplot as plt
import os

# === Paths ===
camera_json = "/home/roar3/Desktop/hope_2/undistorted/transforms.json"
test_txt = "/home/roar3/Desktop/hope_2/test_frame.txt"
mapping_path = "/home/roar3/Desktop/3_test_images/rename_pairs.txt"

coord_to_new = {}
with open(mapping_path, "r") as f:
    for line in f:
        if "<-" in line:
            new, old= line.strip().split(" <- ")
            coord_to_new[old] = new

# coord_to_new = {}  # "-341.00_-698.50_-3.00.png" -> "frame_001.png"
# with open(mapping_path, "r") as f:
#     for line in f:
#         if "Renamed:" in line:
#             old, new = line.strip().split("Renamed: ")[1].split(" -> ")
#             coord_to_new[old] = new

# === Load test frame names ===
with open(test_txt, "r") as f:
    test_names = set(coord_to_new.get(line.strip(), line.strip()) for line in f)

# === Load poses ===
with open(camera_json, "r") as f:
    data = json.load(f)

frames = data["frames"]
positions = []
filenames = []

for frame in frames:
    fname = os.path.basename(frame["file_path"])  # e.g., '-341.00_-698.50_-3.00.png'
    matrix = np.array(frame["transform_matrix"])
    position = matrix[:3, 3]
    #position[2] = -3                                          # flatten z
    # position[1] *= 0.6                     # scale
    # position[1] = position[1] - 346
    # position[0] = position[0] - 700
    positions.append(position)
    renamed = coord_to_new.get(fname, fname)
    filenames.append(renamed)

positions = np.array(positions)
filenames = np.array(filenames)

# === Compute distances to (0, 0, 0)
#target = np.array([-700, -346, -3])
target = np.array([6, 0, 0])
# 
distances = np.linalg.norm(positions - target, axis=1)
sorted_indices = np.argsort(distances)

# === Split into three parts
part1_idx = sorted_indices[:67]
part2_idx = sorted_indices[67:134]
part3_idx = sorted_indices[134:]

# === Extract test positions ===
test_mask = np.array([fname in test_names for fname in filenames])
test_positions = positions[test_mask]

# === Plot ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# === Mask test from part1/2/3
part1_mask = np.array([f not in test_names for f in filenames[part1_idx]])
part2_mask = np.array([f not in test_names for f in filenames[part2_idx]])
part3_mask = np.array([f not in test_names for f in filenames[part3_idx]])

light_blue = "#8FDAF3"   # Light Blue
standard_blue = '#4169E1'   # Regular blue
tiffany_blue = '#0ABAB5' # Tiffany Blue

# === Plot only training images in color
ax.scatter(positions[part1_idx][part1_mask, 0], positions[part1_idx][part1_mask, 1], positions[part1_idx][part1_mask, 2],
          c=standard_blue, s=100, label='Part 1 (Dense)')
ax.scatter(positions[part2_idx][part2_mask, 0], positions[part2_idx][part2_mask, 1], positions[part2_idx][part2_mask, 2],
           c=tiffany_blue, s=100, label='Part 2 (Mid)')
ax.scatter(positions[part3_idx][part3_mask, 0], positions[part3_idx][part3_mask, 1], positions[part3_idx][part3_mask, 2],
           c=light_blue, s=100, label='Part 3 (Sparse)')

# === Plot test points only in gray
ax.scatter(test_positions[:, 0], test_positions[:, 1], test_positions[:, 2],
            c='gray', s=100, label='Test Set', alpha=1.0)

# ax.set_xlabel("Y (m)")  
# ax.set_ylabel("X (m)")
# ax.set_zlabel("Z (m)")

ax.set_xlabel("X (m)", fontweight='bold')
ax.set_ylabel("Y (m)", fontweight='bold')
ax.set_zlabel("Z (m)", fontweight='bold')

# Make tick labels bold
for tick in ax.xaxis.get_ticklabels():
    tick.set_fontweight('bold')
for tick in ax.yaxis.get_ticklabels():
    tick.set_fontweight('bold')
for tick in ax.zaxis.get_ticklabels():
    tick.set_fontweight('bold')

ax.set_title("Training and Test Images Spliting by Distance (Scene 2)", fontweight='bold', fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()

# === Get frames in each region
part1_frames = filenames[part1_idx]
part2_frames = filenames[part2_idx]
part3_frames = filenames[part3_idx]

# === Get test frames in each region
test_in_part1 = sorted([f for f in part1_frames if f in test_names])
test_in_part2 = sorted([f for f in part2_frames if f in test_names])
test_in_part3 = sorted([f for f in part3_frames if f in test_names])

# === Print results
print("\n=== Test Frames in Part 1 (Closest to origin) ===")
for f in test_in_part1:
    print(f)

print("\n=== Test Frames in Part 2 (Mid) ===")
for f in test_in_part2:
    print(f)

print("\n=== Test Frames in Part 3 (Farthest) ===")
for f in test_in_part3:
    print(f)

# === Summary
print(f"\nSummary: {len(test_in_part1)} in Part1 | {len(test_in_part2)} in Part2 | {len(test_in_part3)} in Part3")
# === Load coordinate → renamed-frame mapping ===

# coord_to_new = {}
# with open(mapping_path, "r") as f:
#     for line in f:
#         if "Renamed:" in line:
#             old, new = line.strip().split("Renamed: ")[1].split(" -> ")
#             coord_to_new[old] = new

# # === Load test frame names ===
# with open(test_txt, "r") as f:
#     test_names = set(coord_to_new.get(line.strip(), line.strip()) for line in f)

# # === Load poses ===
# with open(camera_json, "r") as f:
#     data = json.load(f)

# frames = data["frames"]
# positions = []
# filenames = []

# for frame in frames:
#     fname = os.path.basename(frame["file_path"])
#     matrix = np.array(frame["transform_matrix"])
#     position = matrix[:3, 3]
#     position[2] = -3  # flatten z
#     positions.append(position)
#     renamed = coord_to_new.get(fname, fname)
#     filenames.append(renamed)

# positions = np.array(positions)
# filenames = np.array(filenames)

# # === Define the bottom-left reference point ===
# bottom_left = np.array([1.4304, -7.3024, -0.00098])
# target = bottom_left.copy()
# target[2] = -3

# # === Compute distances to bottom-left ===
# distances = np.linalg.norm(positions - target, axis=1)

# # === Split using distance percentiles ===
# percentiles = np.percentile(distances, [25, 50, 75])
# part1_idx = np.where(distances <= percentiles[0])[0]
# part2_idx = np.where((distances > percentiles[0]) & (distances <= percentiles[1]))[0]
# part3_idx = np.where((distances > percentiles[1]) & (distances <= percentiles[2]))[0]
# part4_idx = np.where(distances > percentiles[2])[0]

# # === Define function to split train/test in each region ===
# def split_part(indices, train_ratio):
#     np.random.seed(42)
#     indices = np.array(indices)
#     np.random.shuffle(indices)
#     split_idx = int(len(indices) * train_ratio)
#     return indices[:split_idx], indices[split_idx:]

# # === Apply train/test density based on distance ===
# train1_idx, test1_idx = split_part(part1_idx, 0.95)
# train2_idx, test2_idx = split_part(part2_idx, 0.85)
# train3_idx, test3_idx = split_part(part3_idx, 0.75)
# train4_idx, test4_idx = split_part(part4_idx, 0.65)

# # === Combine all test indices and positions ===
# all_test_idx = np.concatenate([test1_idx, test2_idx, test3_idx, test4_idx])
# test_mask = np.zeros(len(positions), dtype=bool)
# test_mask[all_test_idx] = True
# test_positions = positions[test_mask]

# # === Training masks for plotting ===
# part1_mask = np.isin(part1_idx, train1_idx)
# part2_mask = np.isin(part2_idx, train2_idx)
# part3_mask = np.isin(part3_idx, train3_idx)
# part4_mask = np.isin(part4_idx, train4_idx)

# # === Plot ===
# fig = plt.figure(figsize=(12, 10))
# ax = fig.add_subplot(111, projection='3d')

# # Define colors
# standard_blue = '#4169E1'
# tiffany_blue = '#0ABAB5'
# navy_blue = "#000080"
# sky_blue = "#87CEEB"

# # Plot each part (training only)
# ax.scatter(positions[part1_idx][part1_mask, 0], positions[part1_idx][part1_mask, 1], positions[part1_idx][part1_mask, 2],
#            c=standard_blue, s=60, label='Part 1 (Closest)')
# ax.scatter(positions[part2_idx][part2_mask, 0], positions[part2_idx][part2_mask, 1], positions[part2_idx][part2_mask, 2],
#            c=tiffany_blue, s=60, label='Part 2')
# ax.scatter(positions[part3_idx][part3_mask, 0], positions[part3_idx][part3_mask, 1], positions[part3_idx][part3_mask, 2],
#            c=navy_blue, s=60, label='Part 3')
# ax.scatter(positions[part4_idx][part4_mask, 0], positions[part4_idx][part4_mask, 1], positions[part4_idx][part4_mask, 2],
#            c=sky_blue, s=60, label='Part 4 (Farthest)')

# # Plot test points in gray
# ax.scatter(test_positions[:, 0], test_positions[:, 1], test_positions[:, 2],
#            c='gray', s=60, label='Test Set', alpha=1.0)

# # Labels and formatting
# ax.set_xlabel("Y (m)", fontweight='bold')
# ax.set_ylabel("X (m)", fontweight='bold')
# ax.set_zlabel("Z (m)", fontweight='bold')

# for tick in ax.xaxis.get_ticklabels(): tick.set_fontweight('bold')
# for tick in ax.yaxis.get_ticklabels(): tick.set_fontweight('bold')
# for tick in ax.zaxis.get_ticklabels(): tick.set_fontweight('bold')

# ax.set_title("Training and Test Images Split by Distance from Bottom-Left (Scene 1)", fontweight='bold', fontsize=14)
# ax.legend()
# plt.tight_layout()
# plt.show()

# # === Summary ===
# print(f"\nSummary: {len(test1_idx)} in Part1 | {len(test2_idx)} in Part2 | {len(test3_idx)} in Part3 | {len(test4_idx)} in Part4")