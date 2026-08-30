"""RLE round-trip test: CNOS's encoder against visualize.py's decoder.

run_detect.py encodes masks with CNOS's mask_to_rle (column-major counts,
starting with the run of zeros). The encoder below is copied verbatim from
external/cnos/src/model/utils.py so this test runs without the cnos submodule
or its dependencies.
"""
import numpy as np

from visualize import rle_to_mask


def mask_to_rle(binary_mask):
    # Copied verbatim from external/cnos/src/model/utils.py (reference encoder).
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")

    last_elem = 0
    running_length = 0

    for i, elem in enumerate(binary_mask.ravel(order="F")):  # noqa: B007 (verbatim copy)
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1

    counts.append(running_length)

    return rle


def _roundtrip(mask: np.ndarray) -> np.ndarray:
    return rle_to_mask(mask_to_rle(mask.astype(np.uint8)))


def test_roundtrip_random_masks():
    rng = np.random.default_rng(0)
    for shape in ((7, 5), (32, 48), (60, 33)):
        mask = rng.random(shape) > 0.5
        assert np.array_equal(_roundtrip(mask), mask)


def test_roundtrip_all_zero():
    mask = np.zeros((16, 9), dtype=bool)
    assert np.array_equal(_roundtrip(mask), mask)


def test_roundtrip_all_one():
    mask = np.ones((16, 9), dtype=bool)
    assert np.array_equal(_roundtrip(mask), mask)


def test_roundtrip_single_pixel():
    for y, x in ((0, 0), (5, 3), (15, 8)):
        mask = np.zeros((16, 9), dtype=bool)
        mask[y, x] = True
        assert np.array_equal(_roundtrip(mask), mask)


def test_counts_start_with_zero_run():
    # Column-major convention: counts[0] is the number of leading zeros in
    # F-order. A mask whose first pixel (top-left) is set must encode a
    # leading zero-run of length 0.
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    rle = mask_to_rle(mask.astype(np.uint8))
    assert rle["counts"][0] == 0
    assert rle["counts"][1] == 1

    # And a blob in the second column starts after a full first column of
    # zeros (4 pixels), confirming column-major (not row-major) order.
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 1] = True
    rle = mask_to_rle(mask.astype(np.uint8))
    assert rle["counts"][0] == 4
