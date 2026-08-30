"""
Slim loader for DREAM/NDDS panda-orb datasets.

Directory layout:
    <data_folder>/
        000000.json  000001.json ...    per-frame NDDS annotations
        000000.rgb.jpg ...              RGB images
        _camera_settings.json           camera intrinsics

Only extracts what detection needs: image, joint angles, frame ID.
K is read once from _camera_settings.json; H/W from the first image.
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


def camera_info(data_folder: str) -> Tuple[np.ndarray, int, int]:
    """Return (K_3x3_float32, image_H, image_W) from _camera_settings.json + first image."""
    data_folder = os.path.expanduser(data_folder)
    cam_path = os.path.join(data_folder, "_camera_settings.json")
    if not os.path.exists(cam_path):
        raise FileNotFoundError(f"_camera_settings.json not found in {data_folder}")

    with open(cam_path) as f:
        data = json.load(f)
    s = data["camera_settings"][0]["intrinsic_settings"]
    fx, fy, cx, cy = s["fx"], s["fy"], s["cx"], s["cy"]
    K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], dtype=np.float32)

    # Read H, W from first RGB image (avoids hardcoding resolution per dataset).
    first_img = _find_first_rgb(data_folder)
    with Image.open(first_img) as im:
        W, H = im.size
    return K, H, W


def _annotation_paths(data_folder: str) -> list:
    """Return the sorted per-frame annotation JSONs (digit-prefixed stems)."""
    json_paths = sorted(
        p for p in Path(data_folder).iterdir()
        if p.suffix == ".json" and p.stem[0].isdigit()
    )
    if not json_paths:
        raise FileNotFoundError(f"No digit-prefixed .json files found in {data_folder}")
    return json_paths


def count_frames(data_folder: str) -> int:
    """Count the frames iter_frames will yield, without loading any image."""
    return len(_annotation_paths(os.path.expanduser(data_folder)))


def iter_frames(
    data_folder: str,
) -> Iterator[Tuple[torch.Tensor, np.ndarray, str]]:
    """
    Yield (image_chw, joint_angles_float64, frame_id) for every frame.

    joint_angles: (7,) radians, from sim_state.joints.
    frame_id:     digit-prefixed filename stem, e.g. "000042".
    """
    data_folder = os.path.expanduser(data_folder)
    json_paths = _annotation_paths(data_folder)

    for json_path in json_paths:
        with open(json_path) as f:
            data = json.load(f)

        joints_raw = data["sim_state"]["joints"]
        # Take only the first 7: NDDS/DREAM JSONs include 2 finger joints
        # at indices 7-8 which the Panda FK adapter does not accept.
        joints = np.array(
            [joints_raw[i]["position"] for i in range(7)],
            dtype=np.float64,
        )

        # Derive RGB image path: stem + ".rgb.jpg" (or ".rgb.png").
        stem = json_path.stem
        rgb_path = _find_rgb(data_folder, stem)
        pil   = Image.open(rgb_path).convert("RGB")
        image = _to_tensor(pil)   # (3, H, W) float32

        yield image, joints, stem


def _find_first_rgb(data_folder: str) -> str:
    for ext in (".rgb.jpg", ".rgb.png"):
        candidates = sorted(Path(data_folder).glob(f"*{ext}"))
        if candidates:
            return str(candidates[0])
    raise FileNotFoundError(f"No .rgb.jpg / .rgb.png images in {data_folder}")


def _find_rgb(data_folder: str, stem: str) -> str:
    for ext in (".rgb.jpg", ".rgb.png"):
        p = Path(data_folder) / f"{stem}{ext}"
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"No RGB image for frame {stem} in {data_folder}")
