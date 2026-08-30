"""Lightweight smoke test: no weights, no GPU, no dataset needed.

Covers the pieces that can drift silently:
  - every shipped config parses, carries the required keys, and stays
    consistent with the documented defaults (segmentor: sam)
  - every config's dataset name resolves to a registered loader
  - detect_utils.scale_K_to_render keeps center-crop semantics
  - DetectionResult found/not-found contract

Run from the repo root:  python -m pytest tests/ -v
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIGS = sorted((REPO / "configs").glob("*.yaml"))


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def test_configs_found():
    assert len(CONFIGS) == 6, f"expected 6 dataset configs, found {len(CONFIGS)}"


@pytest.mark.parametrize("cfg_path", CONFIGS, ids=lambda p: p.stem)
def test_config_required_keys(cfg_path):
    cfg = _load(cfg_path)
    assert cfg["dataset"]["name"]
    assert cfg["robot"]["name"]
    r = cfg["renderer"]
    for key in ("render_size", "num_views", "sphere_distance_factor", "elevation_range"):
        assert key in r, f"{cfg_path.name}: renderer.{key} missing"
    # fill_frame must never be set in detector configs (see the note in
    # run_detect.py); the detector does its own tight crop.
    assert "fill_frame" not in r, f"{cfg_path.name}: fill_frame must not be exposed here"


@pytest.mark.parametrize("cfg_path", CONFIGS, ids=lambda p: p.stem)
def test_config_segmentor_is_shipped_default(cfg_path):
    # The README promises all shipped configs default to SAM; FastSAM is
    # selected per-run via --segmentor. This test is what catches the
    # README-vs-configs drift.
    cfg = _load(cfg_path)
    assert cfg["detector"]["segmentor"] == "sam", (
        f"{cfg_path.name}: shipped segmentor default must be 'sam' "
        "(use --segmentor fastsam at run time instead)"
    )


@pytest.mark.parametrize("cfg_path", CONFIGS, ids=lambda p: p.stem)
def test_config_dataset_has_loader(cfg_path):
    from loaders import LOADERS

    cfg = _load(cfg_path)
    name = cfg["dataset"]["name"]
    assert name in LOADERS, f"{cfg_path.name}: dataset '{name}' not in LOADERS"


def test_scale_k_square_image_is_pure_scale():
    from detect_utils import scale_K_to_render

    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    K_r = scale_K_to_render(K, render_size=448, img_H=640, img_W=640)
    s = 448 / 640
    assert np.allclose(K_r[0, 0], 600.0 * s)
    assert np.allclose(K_r[1, 1], 600.0 * s)
    assert np.allclose(K_r[0, 2], 320.0 * s)
    assert np.allclose(K_r[1, 2], 240.0 * s)


def test_scale_k_16x9_keeps_fx_eq_fy():
    # The whole point of center-crop semantics: square pixels must survive
    # a non-square image (fx == fy stays true), and a centered principal
    # point must land at the render center.
    from detect_utils import scale_K_to_render

    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    K_r = scale_K_to_render(K, render_size=448, img_H=720, img_W=1280)
    assert np.allclose(K_r[0, 0], K_r[1, 1])
    assert np.allclose(K_r[0, 2], 224.0)  # (640 - 280) * 448/720
    assert np.allclose(K_r[1, 2], 224.0)


def test_detection_result_contract():
    from detector import DetectionResult

    missed = DetectionResult(bbox_xyxy=None, mask=None)
    assert not missed.found and missed.score == 0.0

    hit = DetectionResult(
        bbox_xyxy=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        mask=np.zeros((4, 4), dtype=bool),
        score=0.9,
    )
    assert hit.found


def test_template_foreground_boxes_use_exclusive_upper_bounds():
    from detector import RobotDetector

    zbuf = torch.full((2, 5, 7), -1.0)
    zbuf[0, 1:4, 2:6] = 1.0
    zbuf[1, 0, 0] = 1.0

    boxes = RobotDetector._template_fg_boxes(zbuf, H=5, W=7)
    assert boxes.tolist() == [
        [2, 1, 6, 4],
        [0, 0, 1, 1],
    ]


def test_template_cache_keeps_only_the_last_joint_state():
    from detector import RobotDetector

    class FakeRenderer:
        calls = 0

        def render_templates(self, _q):
            self.calls += 1
            view = SimpleNamespace(image=torch.zeros(3, 4, 4))
            # The real renderer always returns a per-view depth buffer
            # (background = -1); give one foreground pixel so the tight-crop
            # path runs.
            zbuf = torch.full((1, 4, 4), -1.0)
            zbuf[0, 1, 1] = 1.0
            return SimpleNamespace(views=[view], zbuf=zbuf)

    class FakeDino:
        @staticmethod
        def compute_features(images, token_name):
            assert token_name == "x_norm_clstoken"
            return torch.ones(len(images), 2)

    detector = object.__new__(RobotDetector)
    detector.renderer = FakeRenderer()
    detector.device = torch.device("cpu")
    detector._crop_resize = lambda images, _boxes: images
    detector._normalise = lambda image: image
    detector._dino = FakeDino()
    detector._last_joint_key = None
    detector._last_template_feats = None

    q1 = np.array([0.0, 1.0])
    q2 = np.array([0.0, 2.0])
    detector._get_template_features(q1)
    detector._get_template_features(q1)
    detector._get_template_features(q2)

    assert detector.renderer.calls == 2
    assert detector._last_joint_key == q2.tobytes()
