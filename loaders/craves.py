"""
Slim loader for the CRAVES-Lab OWI-535 dataset.

Directory layout (test_20181024/):
    FusionCameraActor3_2/
        lit/        {frame_id}.jpg      RGB images
        caminfo/    {frame_id}.json     per-frame camera info (K + FoV)
    angles/         {frame_id}.json     joint angles in degrees
    joint/          {frame_id}.json     (not needed for detection)
    vertex/         (not needed for detection)

Only extracts what detection needs: image, joint angles, frame ID.
K is loaded from the first frame's caminfo (camera is fixed across the split).

Joint angles: 4 values, converted degrees to radians, with q[0] *= -1 sign flip
to match URDF convention.
"""
import json
import math
import os
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

_to_tensor = T.ToTensor()


def camera_info(data_folder: str) -> Tuple[np.ndarray, int, int]:
    """Return (K_3x3_float32, image_H, image_W) from the first frame's caminfo."""
    data_folder = Path(os.path.expanduser(data_folder))
    caminfo_dir = data_folder / "FusionCameraActor3_2" / "caminfo"
    first_cam   = sorted(caminfo_dir.glob("*.json"))[0]

    cam = json.loads(first_cam.read_text())
    W   = cam["FilmWidth"]
    H   = cam["FilmHeight"]
    fov = cam.get("Fov", 90.0)
    f   = W / (2.0 * math.tan(math.radians(fov / 2.0)))
    K   = np.array(
        [[f,   0., W / 2.0],
         [0.,  f,  H / 2.0],
         [0.,  0., 1.     ]], dtype=np.float32
    )
    return K, H, W


def _frame_ids(data_folder: Path) -> list:
    """Return the sorted frame IDs (jpg filename stems) for the split."""
    img_dir = data_folder / "FusionCameraActor3_2" / "lit"
    frame_ids = sorted(p.stem for p in img_dir.glob("*.jpg"))
    if not frame_ids:
        raise FileNotFoundError(f"No .jpg images found under {img_dir}")
    return frame_ids


def count_frames(data_folder: str) -> int:
    """Count the frames iter_frames will yield, without loading any image."""
    return len(_frame_ids(Path(os.path.expanduser(data_folder))))


def iter_frames(
    data_folder: str,
) -> Iterator[Tuple[torch.Tensor, np.ndarray, str]]:
    """
    Yield (image_chw, joint_angles_float64, frame_id) for every frame.

    joint_angles: (4,) radians, q[0] sign-flipped to match URDF convention.
    frame_id:     filename stem, e.g. "00000037".
    """
    data_folder = Path(os.path.expanduser(data_folder))
    img_dir     = data_folder / "FusionCameraActor3_2" / "lit"
    angles_dir  = data_folder / "angles"

    for frame_id in _frame_ids(data_folder):
        # Joint angles: degrees to radians, sign-flip q[0] for URDF convention.
        angles_raw = np.array(
            json.loads((angles_dir / f"{frame_id}.json").read_text()),
            dtype=np.float64,
        )
        joints    = angles_raw[:4] * (math.pi / 180.0)
        joints[0] *= -1.0

        pil   = Image.open(img_dir / f"{frame_id}.jpg").convert("RGB")
        image = _to_tensor(pil)   # (3, H, W) float32

        yield image, joints, frame_id
