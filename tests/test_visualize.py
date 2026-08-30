"""--max_width path of visualize.render_detection, on synthetic images."""
import numpy as np
from PIL import Image

from visualize import render_detection

from test_rle import mask_to_rle


def _det(mask: np.ndarray, bbox, score=0.5) -> dict:
    return {
        "found": True,
        "score": score,
        "bbox_xyxy": list(bbox),
        "segmentation": mask_to_rle(mask.astype(np.uint8)),
    }


def _image(w: int, h: int) -> Image.Image:
    # Flat grey, so tint and outline are the only coloured pixels.
    return Image.new("RGB", (w, h), (128, 128, 128))


def test_max_width_scales_output_and_keeps_aspect():
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:300, 200:400] = True
    vis = render_detection(_image(640, 480), _det(mask, (200, 100, 400, 300)), max_width=320)
    assert vis.size == (320, 240)


def test_max_width_does_not_upscale():
    mask = np.zeros((60, 80), dtype=bool)
    mask[10:30, 20:50] = True
    vis = render_detection(_image(80, 60), _det(mask, (20, 10, 50, 30)), max_width=400)
    assert vis.size == (80, 60)


def test_mask_and_bbox_follow_the_downscale():
    # Blob in the top-left quadrant: tint and outline must not drift out of it.
    mask = np.zeros((400, 400), dtype=bool)
    mask[20:180, 20:180] = True
    vis = render_detection(_image(400, 400), _det(mask, (20, 20, 180, 180)), max_width=100)
    assert vis.size == (100, 100)

    arr = np.array(vis).astype(int)
    greenish = (arr[:, :, 1] > arr[:, :, 2] + 20) & (arr[:, :, 1] > 60)
    assert greenish[5:40, 5:40].any(), "tint missing from the scaled blob"
    assert not greenish[60:, 60:].any(), "tint leaked outside the scaled blob"

    orange = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 140) & (arr[:, :, 2] < 80)
    ys, xs = np.nonzero(orange)
    assert len(xs), "bbox outline missing"
    # 160 px box at 400 px scaled to 100 px wide -> corners near 5 and 45.
    assert abs(xs.min() - 5) <= 3 and abs(xs.max() - 45) <= 3
    assert abs(ys.min() - 5) <= 3 and abs(ys.max() - 45) <= 3


def test_side_by_side_uses_the_scaled_size():
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:300, 200:400] = True
    vis = render_detection(_image(640, 480), _det(mask, (200, 100, 400, 300)),
                           side_by_side=True, max_width=320)
    assert vis.size == (320 * 2 + 4, 240)


def test_not_found_still_scales():
    det = {"found": False, "score": None, "bbox_xyxy": None, "segmentation": None}
    vis = render_detection(_image(640, 480), det, max_width=160)
    assert vis.size == (160, 120)


def test_without_max_width_output_is_native_size():
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:300, 200:400] = True
    vis = render_detection(_image(640, 480), _det(mask, (200, 100, 400, 300)))
    assert vis.size == (640, 480)
