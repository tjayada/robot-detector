"""Pure helper functions shared by run_detect.py, visualize.py, tools/ and the
unit tests.

This module only needs numpy and the standard library, so the unit tests can
exercise the K-scaling and frame-selection math without torch, the cnos
environment, GPU weights, or any dataset.
"""

import gzip
import json

import numpy as np


def load_detections(path) -> dict:
    """Read a detections file written by run_detect.py, gzipped or plain."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def scale_K_to_render(K: np.ndarray, render_size: int, img_H: int, img_W: int) -> np.ndarray:
    """Scale a 3x3 intrinsic matrix from image resolution to the square render
    resolution, using center-crop semantics: one uniform scale factor
    min(H, W) -> render_size, with the principal point shifted by the crop.
    A single factor keeps fx == fy for square-pixel cameras, so templates are
    not rendered with a squished aspect ratio on non-square images.
    """
    K = K.astype(np.float32).copy()
    s = min(img_H, img_W)
    crop_x = (img_W - s) // 2
    crop_y = (img_H - s) // 2
    scale = render_size / s
    K[0, 0] *= scale
    K[0, 2] = (K[0, 2] - crop_x) * scale
    K[1, 1] *= scale
    K[1, 2] = (K[1, 2] - crop_y) * scale
    return K


def select_frame_indices(n_total: int, num_frames) -> list:
    """Pick evenly spaced frame indices out of n_total frames.

    num_frames=None (or >= n_total): every frame, in order.
    num_frames=N < n_total: N evenly spaced indices spanning [0, n_total-1]
        inclusive (np.linspace, rounded to int, duplicates removed, sorted).
    """
    if n_total < 0:
        raise ValueError(f"n_total must be non-negative, got {n_total}")
    if num_frames is not None and num_frames < 1:
        raise ValueError(f"num_frames must be at least 1, got {num_frames}")
    if num_frames is None or num_frames >= n_total:
        return list(range(n_total))
    idx = np.linspace(0, n_total - 1, num_frames).round().astype(int)
    return sorted(set(idx.tolist()))


def validate_frame_indices(frame_indices, n_total: int) -> list[int]:
    """Validate explicit 0-based frame indices and return them sorted."""
    indices = list(frame_indices)
    seen, duplicates = set(), set()
    for idx in indices:
        if idx in seen:
            duplicates.add(idx)
        seen.add(idx)
    if duplicates:
        raise ValueError(f"duplicate frame indices: {sorted(duplicates)}")

    invalid = sorted(idx for idx in indices if idx < 0 or idx >= n_total)
    if invalid:
        valid_range = f"0..{n_total - 1}" if n_total else "empty"
        raise ValueError(
            f"frame indices out of range: {invalid}; valid range is {valid_range}"
        )
    return sorted(indices)
