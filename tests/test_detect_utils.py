"""Tests for detect_utils.select_frame_indices.

The selection must be deterministic and evenly spaced: np.linspace over
[0, n_total-1], rounded to int, duplicates removed, sorted.
"""
import numpy as np

import pytest

from detect_utils import select_frame_indices, validate_frame_indices


def test_none_returns_all_frames():
    assert select_frame_indices(5, None) == [0, 1, 2, 3, 4]


def test_n_equal_total_returns_all_frames():
    assert select_frame_indices(5, 5) == [0, 1, 2, 3, 4]


def test_n_greater_than_total_returns_all_frames():
    assert select_frame_indices(5, 100) == [0, 1, 2, 3, 4]


def test_subsample_matches_linspace_semantics():
    n_total, n = 32000, 1000
    got = select_frame_indices(n_total, n)
    expected = sorted(set(
        np.linspace(0, n_total - 1, n).round().astype(int).tolist()
    ))
    assert got == expected


def test_subsample_spans_full_range():
    got = select_frame_indices(1000, 10)
    assert got[0] == 0
    assert got[-1] == 999
    assert len(got) == 10


def test_result_is_sorted_unique_ints():
    # The dedup in select_frame_indices is defensive (for N < n_total the
    # linspace spacing is > 1, so rounded indices cannot collide), but the
    # output contract must hold regardless: unique, sorted, in range.
    for n_total, n in ((317, 50), (10, 9), (4, 3)):
        got = select_frame_indices(n_total, n)
        assert got == sorted(set(got))
        assert all(isinstance(i, int) for i in got)
        assert got[0] == 0 and got[-1] == n_total - 1
        assert len(got) == n


@pytest.mark.parametrize("num_frames", [0, -1])
def test_subsample_must_be_positive(num_frames):
    with pytest.raises(ValueError, match="at least 1"):
        select_frame_indices(100, num_frames)


def test_explicit_frames_are_validated_and_sorted():
    assert validate_frame_indices([9, 0, 4], 10) == [0, 4, 9]

    with pytest.raises(ValueError, match="duplicate"):
        validate_frame_indices([0, 4, 4], 10)
    with pytest.raises(ValueError, match="out of range"):
        validate_frame_indices([-1, 10], 10)
