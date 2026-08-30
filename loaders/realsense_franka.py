"""
Slim loader for the RealSense-Franka dataset (CtRNet real Franka, D415 views).

Directory layout (one root passed as --data_folder):
    <data_folder>/
        1_D415_front_0/
            000000.json  000001.json ...  per-frame NDDS-style annotations
            000000.png ...                RGB images
            _camera_settings.json         camera intrinsics
        2_D415_front_1/
        6_D415_left_1/
        12_D415_right_1/

Multi-view root like loaders/hydra.py, per-view NDDS reading like loaders/dream.py.
Differs from dream.py: the per-frame JSON is a bare list (annotation at [0]), the
image is "<stem>.png" (no ".rgb" infix), and views share intrinsics.

frame_id is emitted as "<view_dir>/<stem>" (e.g. "1_D415_front_0/000042") so the
four views do not collide on their shared 000000.. stems.
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


def _view_dirs(data_folder: Path) -> list[Path]:
    """Return the view subdirectories (those carrying a _camera_settings.json)."""
    dirs = sorted(
        p for p in data_folder.iterdir()
        if p.is_dir() and (p / "_camera_settings.json").exists()
    )
    if not dirs:
        raise FileNotFoundError(
            f"No view subdirectories with _camera_settings.json under {data_folder}"
        )
    return dirs


def _annotation_paths(view_dir: Path) -> list:
    """Return the sorted per-frame annotation JSONs (digit-prefixed stems)."""
    return sorted(
        p for p in view_dir.iterdir()
        if p.suffix == ".json" and p.stem[0].isdigit()
    )


def camera_info(data_folder: str) -> Tuple[np.ndarray, int, int]:
    """Return (K_3x3_float32, image_H, image_W) from view 0's _camera_settings.json + first image."""
    data_folder = Path(os.path.expanduser(data_folder))
    view0 = _view_dirs(data_folder)[0]

    with open(view0 / "_camera_settings.json") as f:
        data = json.load(f)
    s = data["camera_settings"][0]["intrinsic_settings"]
    fx, fy, cx, cy = s["fx"], s["fy"], s["cx"], s["cy"]
    K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], dtype=np.float32)

    # Read H, W from the first image (avoids hardcoding the sensor resolution).
    first_img = sorted(view0.glob("*.png"))[0]
    with Image.open(first_img) as im:
        W, H = im.size
    return K, H, W


def count_frames(data_folder: str) -> int:
    """Count the frames iter_frames will yield, without loading any image."""
    data_folder = Path(os.path.expanduser(data_folder))
    return sum(len(_annotation_paths(v)) for v in _view_dirs(data_folder))


def iter_frames(
    data_folder: str,
) -> Iterator[Tuple[torch.Tensor, np.ndarray, str]]:
    """
    Yield (image_chw, joint_angles_float64, frame_id) for every frame across all views.

    joint_angles: (7,) radians, from the annotation's joints list.
    frame_id:     "<view_dir>/<stem>", e.g. "1_D415_front_0/000042".
    """
    data_folder = Path(os.path.expanduser(data_folder))

    for view_dir in _view_dirs(data_folder):
        for json_path in _annotation_paths(view_dir):
            with open(json_path) as f:
                data = json.load(f)

            # RealSense-Franka annotation is a single-element list; take the first 7
            # joints (index 7 is the finger, which the Panda FK adapter does not accept).
            joints_raw = data[0]["joints"]
            joints = np.array(
                [joints_raw[i]["position"] for i in range(7)],
                dtype=np.float64,
            )

            img_path = view_dir / f"{json_path.stem}.png"
            pil   = Image.open(img_path).convert("RGB")
            image = _to_tensor(pil)   # (3, H, W) float32

            yield image, joints, f"{view_dir.name}/{json_path.stem}"
