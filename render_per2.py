import torch
import os
import json
import math
import numpy as np
import torchvision.utils
from PIL import Image
from argparse import ArgumentParser
from os import makedirs

from scene import Scene
from scene.cameras_initial import Camera
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state, PILtoTorch

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def perturb_color_only(gaussians, sigma_color):
    with torch.no_grad():
        if sigma_color > 0.0 and hasattr(gaussians, "_features_dc"):
            noise_color_dc = torch.normal(mean=0, std=sigma_color, size=gaussians._features_dc.shape, device=gaussians._features_dc.device)
            gaussians._features_dc = torch.clamp(gaussians._features_dc + noise_color_dc, 0, 1)
            print("Mean Color Change:", noise_color_dc.abs().mean().item())


def save_perturbed_features(gaussians, output_dir):
    makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "features_dc.pt")
    torch.save(gaussians._features_dc.cpu(), path)
    print(f"Saved perturbed features_dc to: {path}")


def render_set(model_path, sample_name, iteration, gaussians, pipeline, background, train_test_exp, separate_sh, custom_pose):
    render_path = os.path.join(model_path, sample_name, f"ours_{iteration}", "renders")
    makedirs(render_path, exist_ok=True)

    frame = custom_pose["frames"][0]
    image_path = frame["file_path"]
    if not os.path.exists(image_path):
        print(f"❌ Error: Image path does not exist -> {image_path}")
        return

    try:
        pil_image = Image.open(image_path).convert("RGB")
        width = custom_pose["camera_intrinsics"]["1"]["width"]
        height = custom_pose["camera_intrinsics"]["1"]["height"]
        pil_image = pil_image.resize((width, height), Image.LANCZOS)
        image_tensor = PILtoTorch(pil_image, (width, height))
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    transform_matrix = np.array(frame["transform_matrix"])
    transform_matrix = np.linalg.inv(transform_matrix)
    R = torch.tensor(transform_matrix[:3, :3], dtype=torch.float32)
    T = torch.tensor(transform_matrix[:3, 3] * -1.0, dtype=torch.float32)

    if torch.cuda.is_available():
        R, T, image_tensor = R.cuda(), T.cuda(), image_tensor.cuda()

    fx, fy = custom_pose["camera_intrinsics"]["1"]["focal_length"]
    FoVx = 2 * math.atan(width / (2 * fx))
    FoVy = 2 * math.atan(height / (2 * fy))

    custom_camera = Camera(
        resolution=(width, height),
        colmap_id=frame["camera_id"],
        R=R,
        T=T,
        FoVx=FoVx,
        FoVy=FoVy,
        depth_params=None,
        image=image_tensor,
        invdepthmap=None,
        image_name=frame["file_path"],
        uid=frame["camera_id"],
        train_test_exp=False,
        is_test_dataset=False,
        is_test_view=False
    )

    print("Rendering...")
    rendering = render(custom_camera, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
    output_path = os.path.join(render_path, "custom_render.png")
    torchvision.utils.save_image(rendering, output_path)
    print(f"Custom rendered image saved at {output_path}")


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, separate_sh: bool, custom_pose_file=None):
    with torch.no_grad():
        background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

        if custom_pose_file is None:
            raise ValueError("❌ Error: Custom pose file must be provided!")

        with open(custom_pose_file, "r") as f:
            custom_pose = json.load(f)

        if "frames" not in custom_pose or "camera_intrinsics" not in custom_pose:
            raise ValueError("❌ Error: Custom pose file missing 'frames' or 'camera_intrinsics' keys!")

        for i in range(10):
            sample_name = f"custom_sample_{i+1}"
            print(f"\n==> Rendering sample {i+1}/10 with color perturbation")

            gaussians = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

            perturb_color_only(gaussians, sigma_color=0.3)

            sample_dir = os.path.join(dataset.model_path, sample_name)
            save_perturbed_features(gaussians, sample_dir)

            render_set(dataset.model_path, sample_name, scene.loaded_iter, gaussians, pipeline, background, dataset.train_test_exp, separate_sh, custom_pose)


if __name__ == "__main__":
    parser = ArgumentParser(description="Gaussian Splatting Rendering Script with Color Perturbation")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int, help="Iteration to load the model from")
    parser.add_argument("--pose", type=str, default=None, help="Path to JSON file containing custom camera pose")

    args = get_combined_args(parser)
    print("Rendering from model path: " + args.model_path)

    safe_state(False)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        separate_sh=SPARSE_ADAM_AVAILABLE,
        custom_pose_file=args.pose
    )
