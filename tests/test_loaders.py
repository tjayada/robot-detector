"""Synthetic mini-dataset tests for all five loaders.

Each test builds a tiny fake dataset in tmp_path (a few small images made with
PIL) and checks the loader contract: frame_id formats, joint array shapes,
image tensor shapes, camera_info values, and count_frames == number of frames
iter_frames yields.

Needs numpy, PIL, torch/torchvision (for the image-to-tensor path). No GPU,
no weights, no real dataset.
"""
import json
import math
import pickle

import numpy as np
import pytest
from PIL import Image


def _save_png(path, w, h, color=(120, 30, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), color).save(path)


def _save_jpg(path, w, h, color=(120, 30, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), color).save(path, quality=90)


# ---------------------------------------------------------------------------
# Baxter
# ---------------------------------------------------------------------------

@pytest.fixture
def baxter_root(tmp_path):
    gt = {
        "pose_0":  {"joints": [0.1] * 7},
        "pose_2":  {"joints": [0.2] * 7},
        "pose_10": {"joints": [0.3] * 7},
        # no entry for pose_1: the loader must skip that directory
    }
    with open(tmp_path / "ground_truth_data", "wb") as f:
        pickle.dump(gt, f)
    _save_png(tmp_path / "pose_0" / "img_a.png", 8, 6)
    _save_png(tmp_path / "pose_0" / "img_b.png", 8, 6)
    _save_png(tmp_path / "pose_1" / "img_a.png", 8, 6)   # skipped (no GT)
    _save_png(tmp_path / "pose_2" / "img_a.png", 8, 6)
    _save_png(tmp_path / "pose_10" / "img_a.png", 8, 6)
    return tmp_path


def test_baxter_loader(baxter_root):
    from loaders import baxter

    K, H, W = baxter.camera_info(str(baxter_root))
    assert (H, W) == (1536, 2048), "Baxter camera_info must report native resolution"
    assert K.dtype == np.float32
    assert np.isclose(K[0, 0], 960.41357421875)
    assert np.isclose(K[1, 2], 776.2381591796875)

    frames = list(baxter.iter_frames(str(baxter_root)))
    # Numeric pose ordering (pose_2 before pose_10) and pose_1 skipped.
    assert [fid for _, _, fid in frames] == [
        "pose_0_0000", "pose_0_0001", "pose_2_0000", "pose_10_0000",
    ]
    assert baxter.count_frames(str(baxter_root)) == len(frames)

    image, joints, _ = frames[0]
    assert image.shape == (3, 6, 8) and image.dtype.is_floating_point
    assert joints.shape == (7,) and joints.dtype == np.float64
    assert np.allclose(joints, 0.1)


# ---------------------------------------------------------------------------
# DREAM (panda-orb)
# ---------------------------------------------------------------------------

@pytest.fixture
def dream_root(tmp_path):
    cam = {"camera_settings": [
        {"intrinsic_settings": {"fx": 320.0, "fy": 321.0, "cx": 160.0, "cy": 120.0}}
    ]}
    (tmp_path / "_camera_settings.json").write_text(json.dumps(cam))
    for i, base in enumerate((0.1, 0.2)):
        stem = f"{i:06d}"
        # 9 joints like real NDDS files: 7 arm + 2 finger joints.
        ann = {"sim_state": {"joints": [{"position": base + j} for j in range(9)]}}
        (tmp_path / f"{stem}.json").write_text(json.dumps(ann))
        _save_jpg(tmp_path / f"{stem}.rgb.jpg", 10, 8)
    return tmp_path


def test_dream_loader(dream_root):
    from loaders import dream

    K, H, W = dream.camera_info(str(dream_root))
    assert (H, W) == (8, 10)
    assert np.isclose(K[0, 0], 320.0) and np.isclose(K[1, 1], 321.0)

    frames = list(dream.iter_frames(str(dream_root)))
    assert [fid for _, _, fid in frames] == ["000000", "000001"]
    assert dream.count_frames(str(dream_root)) == len(frames)

    image, joints, _ = frames[0]
    assert image.shape == (3, 8, 10)
    assert joints.shape == (7,), "finger joints (indices 7-8) must be dropped"
    assert np.allclose(joints, 0.1 + np.arange(7))


# ---------------------------------------------------------------------------
# CRAVES
# ---------------------------------------------------------------------------

@pytest.fixture
def craves_root(tmp_path):
    cam = {"FilmWidth": 12, "FilmHeight": 10, "Fov": 90.0}
    for stem in ("00000003", "00000007"):
        _save_jpg(tmp_path / "FusionCameraActor3_2" / "lit" / f"{stem}.jpg", 12, 10)
        cam_path = tmp_path / "FusionCameraActor3_2" / "caminfo" / f"{stem}.json"
        cam_path.parent.mkdir(parents=True, exist_ok=True)
        cam_path.write_text(json.dumps(cam))
        angles_path = tmp_path / "angles" / f"{stem}.json"
        angles_path.parent.mkdir(parents=True, exist_ok=True)
        angles_path.write_text(json.dumps([90.0, -45.0, 30.0, 0.0]))
    return tmp_path


def test_craves_loader(craves_root):
    from loaders import craves

    K, H, W = craves.camera_info(str(craves_root))
    assert (H, W) == (10, 12)
    # FoV math: f = W / (2 tan(fov/2)); for fov=90 that is W/2.
    f = 12 / (2.0 * math.tan(math.radians(45.0)))
    assert np.isclose(K[0, 0], f) and np.isclose(K[1, 1], f)
    assert np.isclose(K[0, 2], 6.0) and np.isclose(K[1, 2], 5.0)

    frames = list(craves.iter_frames(str(craves_root)))
    assert [fid for _, _, fid in frames] == ["00000003", "00000007"]
    assert craves.count_frames(str(craves_root)) == len(frames)

    image, joints, _ = frames[0]
    assert image.shape == (3, 10, 12)
    assert joints.shape == (4,)
    # Degrees to radians with q[0] sign-flipped for URDF convention.
    assert np.allclose(joints, [-math.pi / 2, -math.pi / 4, math.pi / 6, 0.0])


# ---------------------------------------------------------------------------
# Hydra (lbr / xarm / meca share this layout)
# ---------------------------------------------------------------------------

@pytest.fixture
def hydra_root(tmp_path):
    K = np.array([[400.0, 0.0, 6.0], [0.0, 401.0, 4.0], [0.0, 0.0, 1.0]])
    for m, n_frames in ((0, 2), (1, 1)):
        meas = tmp_path / f"measurement_{m}"
        (meas / "images").mkdir(parents=True)
        np.save(meas / "camera_K.npy", K)
        gt = {str(i): {"joints": [0.1 * (m + 1)] * 7} for i in range(n_frames)}
        (meas / "ground_truth.json").write_text(json.dumps(gt))
        for i in range(n_frames):
            _save_png(meas / "images" / f"{i:03d}.png", 12, 8)
    return tmp_path


def test_hydra_loader(hydra_root):
    from loaders import hydra

    K, H, W = hydra.camera_info(str(hydra_root))
    assert (H, W) == (8, 12)
    assert np.isclose(K[0, 0], 400.0) and np.isclose(K[1, 1], 401.0)

    frames = list(hydra.iter_frames(str(hydra_root)))
    assert [fid for _, _, fid in frames] == ["meas0_000", "meas0_001", "meas1_000"]
    assert hydra.count_frames(str(hydra_root)) == len(frames)

    image, joints, _ = frames[2]
    assert image.shape == (3, 8, 12)
    assert joints.shape == (7,)
    assert np.allclose(joints, 0.2)


# ---------------------------------------------------------------------------
# RealSense-Franka (multi-view D415, NDDS-style, list-JSON)
# ---------------------------------------------------------------------------

@pytest.fixture
def realsense_franka_root(tmp_path):
    cam = {"camera_settings": [
        {"intrinsic_settings": {"fx": 320.0, "fy": 321.0, "cx": 160.0, "cy": 120.0}}
    ]}
    # Two views; sorted view order and shared 000000.. stems.
    for view in ("1_D415_front_0", "2_D415_front_1"):
        vdir = tmp_path / view
        vdir.mkdir()
        (vdir / "_camera_settings.json").write_text(json.dumps(cam))
        for i, base in enumerate((0.1, 0.2)):
            stem = f"{i:06d}"
            # Single-element list; 8 joints (7 arm + 1 finger).
            ann = [{"joints": [{"position": base + j} for j in range(8)]}]
            (vdir / f"{stem}.json").write_text(json.dumps(ann))
            _save_png(vdir / f"{stem}.png", 10, 8)
    return tmp_path


def test_realsense_franka_loader(realsense_franka_root):
    from loaders import realsense_franka

    K, H, W = realsense_franka.camera_info(str(realsense_franka_root))
    assert (H, W) == (8, 10)
    assert np.isclose(K[0, 0], 320.0) and np.isclose(K[1, 1], 321.0)

    frames = list(realsense_franka.iter_frames(str(realsense_franka_root)))
    # View-prefixed ids; views share their 000000.. stems.
    assert [fid for _, _, fid in frames] == [
        "1_D415_front_0/000000", "1_D415_front_0/000001",
        "2_D415_front_1/000000", "2_D415_front_1/000001",
    ]
    assert realsense_franka.count_frames(str(realsense_franka_root)) == len(frames)

    image, joints, _ = frames[0]
    assert image.shape == (3, 8, 10)
    assert joints.shape == (7,), "finger joint (index 7) must be dropped"
    assert np.allclose(joints, 0.1 + np.arange(7))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_loaders_registry_exports_triples():
    from loaders import LOADERS

    assert set(LOADERS) == {"baxter_real", "panda_orb", "craves", "hydra", "realsense_franka"}
    for name, entry in LOADERS.items():
        assert len(entry) == 3, f"{name}: expected (camera_info, iter_frames, count_frames)"
        assert all(callable(fn) for fn in entry)
