import json
import numpy as np
from pathlib import Path

def normalize_poses(input_path, output_path, keep_filenames=True):
    """
    Normalize poses and save scene_center and scene_scale into output JSON.

    Args:
        input_path: Path to original train_3dgs.json
        output_path: Path to save normalized JSON
        keep_filenames: Preserve original filenames
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    frames = data['frames']
    flip_mat = np.eye(4)
    flip_mat[1:3] *= -1  # Flip Y and Z to convert OpenCV to OpenGL

    translations = []
    original_filenames = []

    # Convert and collect
    for frame in frames:
        c2w = np.array(frame['transform_matrix'])
        c2w = c2w @ flip_mat
        translations.append(c2w[:3, 3])
        original_filenames.append(frame['file_path'])
        frame['transform_matrix'] = c2w.tolist()

    translations = np.array(translations)
    scene_center = np.mean(translations, axis=0)
    translations -= scene_center
    scene_scale = np.max(np.abs(translations))
    translations /= scene_scale

    # Update frames with normalized translations
    for i, frame in enumerate(frames):
        c2w = np.array(frame['transform_matrix'])
        c2w[:3, 3] = translations[i]
        frame['transform_matrix'] = c2w.tolist()
        if keep_filenames:
            frame['file_path'] = original_filenames[i]

    # Save normalization params
    data['scene_center'] = scene_center.tolist()
    data['scene_scale'] = float(scene_scale)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"✅ Normalized {len(frames)} poses.")
    print(f"📍 scene_center: {scene_center}")
    print(f"📏 scene_scale: {scene_scale:.6f}")
    print(f"💾 Saved to: {output_path}")


def estimate_pose_norm_params(normalized_json_path):
    """Load and print scene_center and scene_scale for reference"""
    with open(normalized_json_path, 'r') as f:
        data = json.load(f)
    
    return {
        'scene_center': np.array(data['scene_center']),
        'scene_scale': float(data['scene_scale'])
    }


def denormalize_pose(pose_matrix, norm_params):
    """
    Convert a normalized pose back to world coordinates.

    Args:
        pose_matrix: 4x4 numpy array
        norm_params: dict with keys 'scene_center' and 'scene_scale'

    Returns:
        Denormalized 4x4 pose
    """
    pose_matrix = np.array(pose_matrix)
    pose_matrix[:3, 3] = pose_matrix[:3, 3] * norm_params['scene_scale'] + norm_params['scene_center']
    return pose_matrix


if __name__ == "__main__":
    # === Example usage ===
    input_json = "/home/roar/Desktop/holo3/train_3dgs.json"
    output_json = "/home/roar/Desktop/holo3/train_3dgs_normalized.json"

    normalize_poses(input_json, output_json, keep_filenames=True)

    norm = estimate_pose_norm_params(output_json)
    print("\nNormalization parameters:")
    print(f"scene_center = {norm['scene_center']}")
    print(f"scene_scale  = {norm['scene_scale']:.6f}")