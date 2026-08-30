"""
Slim loader for the Hydra ICP benchmark (hydra_eval/ layout).

Directory layout (one robot root passed as --data_folder):
    <data_folder>/
        measurement_0/
            T_cam_base.npy      (4,4) float64  GT camera pose (not needed for detection)
            camera_K.npy        (3,3) float64  RealSense intrinsics, native 1280x720
            images/             000.png ... 014.png
            ground_truth.json   {"0": {"joints": [...]}, "1": ...}
        measurement_1/
        measurement_2/

Only extracts what detection needs: image, joint angles, frame ID.
K is read once from measurement_0/camera_K.npy (static camera across the split).
Images are yielded at native resolution.

frame_id is emitted as "meas{M}_{F:03d}" (e.g. "meas0_000").

Joint angles are stored in ground_truth.json in radians (7 values for lbr/xarm,
6 for meca): no degree conversion or sign flip needed.
"""
import json
import os
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

_to_tensor = T.ToTensor()


def _measurement_dirs(data_folder: Path) -> list[Path]:
    """Return sorted measurement_N directories under the robot root."""
    dirs = sorted(
        (p for p in data_folder.iterdir()
         if p.is_dir() and p.name.startswith("measurement_")),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not dirs:
        raise FileNotFoundError(
            f"No measurement_N/ directories found under {data_folder}"
        )
    return dirs


def camera_info(data_folder: str) -> Tuple[np.ndarray, int, int]:
    """Return (K_3x3_float32, image_H, image_W) from measurement_0/camera_K.npy."""
    data_folder = Path(os.path.expanduser(data_folder))
    meas0 = _measurement_dirs(data_folder)[0]

    K = np.load(meas0 / "camera_K.npy").astype(np.float32)

    # Read H, W from the first image (avoids hardcoding the sensor resolution).
    first_img = sorted((meas0 / "images").glob("*.png"))[0]
    with Image.open(first_img) as im:
        W, H = im.size
    return K, H, W


def count_frames(data_folder: str) -> int:
    """Count the frames iter_frames will yield, without loading any image."""
    data_folder = Path(os.path.expanduser(data_folder))
    total = 0
    for meas_dir in _measurement_dirs(data_folder):
        with open(meas_dir / "ground_truth.json") as fh:
            total += len(json.load(fh))
    return total


def iter_frames(
    data_folder: str,
) -> Iterator[Tuple[torch.Tensor, np.ndarray, str]]:
    """
    Yield (image_chw, joint_angles_float64, frame_id) for every frame across all
    measurements in the robot root.

    joint_angles: (7,) lbr/xarm or (6,) meca, radians, as stored in ground_truth.json.
    frame_id:     "meas{M}_{F:03d}", e.g. "meas0_000".
    """
    data_folder = Path(os.path.expanduser(data_folder))

    for meas_dir in _measurement_dirs(data_folder):
        meas_idx  = int(meas_dir.name.split("_")[1])
        image_dir = meas_dir / "images"

        with open(meas_dir / "ground_truth.json") as fh:
            gt_raw = json.load(fh)

        # JSON keys are strings; iterate in numeric frame order.
        for frame_idx in sorted(int(k) for k in gt_raw):
            joints = np.array(gt_raw[str(frame_idx)]["joints"], dtype=np.float64)

            img_path = image_dir / f"{frame_idx:03d}.png"
            pil      = Image.open(img_path).convert("RGB")
            image    = _to_tensor(pil)   # (3, H, W) float32

            yield image, joints, f"meas{meas_idx}_{frame_idx:03d}"
