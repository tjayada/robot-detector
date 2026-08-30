"""
Slim loader for the CtRNet Baxter real-world dataset.

Directory layout:
    <data_folder>/
        pose_0/ ... pose_N/    PNG images (multiple views per pose)
        ground_truth_data      pickle: {"pose_N": {"joints": [...], ...}}

Only extracts what detection needs: image, joint angles, frame ID.
Camera intrinsics are fixed constants from the CtRNet paper.
"""
import glob
import os
import pickle
from typing import Iterator, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Full-resolution intrinsics (Azure Kinect, CtRNet paper).
# Detection runs at native resolution so bbox/mask coordinates carry full precision.
_FX, _FY   = 960.41357421875, 960.22314453125
_CX, _CY   = 1021.7171020507812, 776.2381591796875
_W_FULL, _H_FULL = 2048, 1536

_to_tensor = T.ToTensor()


def camera_info(data_folder: str) -> Tuple[np.ndarray, int, int]:
    """Return (K_3x3_float32, image_H, image_W) at native resolution."""
    K = np.array(
        [[_FX, 0.,  _CX],
         [0.,  _FY, _CY],
         [0.,  0.,  1. ]], dtype=np.float32
    )
    return K, _H_FULL, _W_FULL


def _pose_dirs_and_gt(data_folder: str):
    """Return (sorted pose directories, ground-truth dict) for the dataset root."""
    gt_path = os.path.join(data_folder, "ground_truth_data")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"ground_truth_data not found in {data_folder}")

    with open(gt_path, "rb") as f:
        ground_truth = pickle.load(f)

    pose_dirs = sorted(
        [d for d in os.listdir(data_folder)
         if d.startswith("pose_") and os.path.isdir(os.path.join(data_folder, d))],
        key=lambda d: int(d.split("_")[1]),
    )
    return pose_dirs, ground_truth


def count_frames(data_folder: str) -> int:
    """Count the frames iter_frames will yield, without loading any image."""
    data_folder = os.path.expanduser(data_folder)
    pose_dirs, ground_truth = _pose_dirs_and_gt(data_folder)
    return sum(
        len(glob.glob(os.path.join(data_folder, pose_dir, "*.png")))
        for pose_dir in pose_dirs if pose_dir in ground_truth
    )


def iter_frames(
    data_folder: str,
) -> Iterator[Tuple[torch.Tensor, np.ndarray, str]]:
    """
    Yield (image_chw, joint_angles_float64, frame_id) for every image.

    joint_angles: (7,) radians, left arm.
    frame_id:     "pose_3_0042" (pose_key + zero-padded image index within pose).
    """
    data_folder = os.path.expanduser(data_folder)
    pose_dirs, ground_truth = _pose_dirs_and_gt(data_folder)

    for pose_dir in pose_dirs:
        pose_key = pose_dir
        if pose_key not in ground_truth:
            continue

        joints = np.array(ground_truth[pose_key]["joints"], dtype=np.float64)
        img_paths = sorted(glob.glob(os.path.join(data_folder, pose_dir, "*.png")))

        for img_idx, img_path in enumerate(img_paths):
            pil = Image.open(img_path).convert("RGB")
            image     = _to_tensor(pil)                          # (3, H, W) float32
            frame_id  = f"{pose_key}_{img_idx:04d}"
            yield image, joints, frame_id
