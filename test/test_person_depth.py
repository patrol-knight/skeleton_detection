"""Tests for the per-person depth estimator.

Pure numpy: ``skeleton_detection/inference/depth_estimation.py`` imports
neither ROS nor pyrealsense2, so this runs on the host as well as inside the
container::

    python3 -m pytest test/test_person_depth.py -v

Coverage:

    1  two-stage median: 3x3 neighbourhood median, then median over joints
    2  zero / NaN / inf samples are rejected
    3  a keypoint with no valid neighbourhood is dropped, not counted as 0
    4  a person with no usable keypoint gets NaN, never 0
    5  keypoints below the score threshold do not vote
    6  image borders are clipped, not wrapped
    7  raw Z16 + depth_scale is converted to meters exactly once
    8  optional min/max metric bounds
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skeleton_detection.inference.depth_estimation import (  # noqa: E402
    NO_DEPTH,
    DepthParams,
    compute_joint_z,
    compute_person_z,
    format_depth,
    is_valid_depth,
)

METERS = DepthParams(depth_scale=1.0, keypoint_score_threshold=0.3)


def _visible(count):
    return np.ones(count, dtype=np.float32)


def test_joint_depth_is_the_neighbourhood_median():
    depth = np.zeros((10, 10), dtype=np.float32)
    # A 3x3 patch of 9 values; the median is the 5th smallest, 5.0.
    depth[4:7, 4:7] = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
    assert compute_joint_z(depth, 5, 5, METERS) == 5.0


def test_person_depth_is_the_median_of_joint_depths():
    depth = np.zeros((20, 20), dtype=np.float32)
    # Three isolated, uniform 3x3 blobs -> joint depths 2.0, 3.0, 10.0.
    for (row, column), value in (((3, 3), 2.0), ((9, 9), 3.0), ((15, 15), 10.0)):
        depth[row - 1: row + 2, column - 1: column + 2] = value

    keypoints = np.array([[3, 3], [9, 9], [15, 15]], dtype=np.float32)
    # Median of [2, 3, 10] is 3.0 -- a mean would have given 5.0.
    assert compute_person_z(depth, keypoints, _visible(3), METERS) == 3.0


def test_zero_nan_and_inf_samples_are_rejected():
    depth = np.full((9, 9), np.nan, dtype=np.float32)
    depth[3, 3] = 0.0
    depth[3, 4] = np.inf
    depth[4, 3] = -np.inf
    depth[4, 4] = 4.0  # the only valid sample in the patch
    assert compute_joint_z(depth, 4, 4, METERS) == 4.0


def test_keypoint_without_valid_depth_is_dropped_not_zero():
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[2:5, 2:5] = 6.0          # only this keypoint has data
    keypoints = np.array([[3, 3], [15, 15]], dtype=np.float32)
    # If the empty keypoint contributed 0.0, the median would be 3.0.
    assert compute_person_z(depth, keypoints, _visible(2), METERS) == 6.0


def test_person_without_any_valid_depth_is_nan():
    depth = np.zeros((20, 20), dtype=np.float32)
    keypoints = np.array([[5, 5], [10, 10]], dtype=np.float32)
    result = compute_person_z(depth, keypoints, _visible(2), METERS)
    assert math.isnan(result)
    assert result != 0.0


def test_missing_depth_image_is_nan():
    keypoints = np.array([[5, 5]], dtype=np.float32)
    assert math.isnan(compute_person_z(None, keypoints, _visible(1), METERS))


def test_low_score_keypoints_do_not_vote():
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[2:5, 2:5] = 2.0
    depth[9:12, 9:12] = 8.0
    keypoints = np.array([[3, 3], [10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.05], dtype=np.float32)  # second joint invisible
    assert compute_person_z(depth, keypoints, scores, METERS) == 2.0


def test_border_keypoint_is_clipped_not_wrapped():
    depth = np.zeros((10, 10), dtype=np.float32)
    depth[0, 0] = 1.0
    depth[0, 1] = 3.0
    depth[1, 0] = 5.0
    depth[1, 1] = 7.0
    # The 3x3 window around (0, 0) clips to the 2x2 top-left corner; the median
    # of [1, 3, 5, 7] is 4.0. Nothing from the opposite edge may appear.
    assert compute_joint_z(depth, 0, 0, METERS) == 4.0


def test_out_of_image_keypoint_yields_no_depth():
    depth = np.ones((10, 10), dtype=np.float32)
    assert math.isnan(compute_joint_z(depth, 50, 5, METERS))
    assert math.isnan(compute_joint_z(depth, 5, -3, METERS))
    assert math.isnan(compute_joint_z(depth, np.nan, 5, METERS))


def test_raw_z16_is_scaled_to_meters_once():
    # 2000 raw units * 0.001 m/unit = 2.0 m.
    depth = np.full((10, 10), 2000, dtype=np.uint16)
    params = DepthParams(depth_scale=0.001)
    keypoints = np.array([[5, 5]], dtype=np.float32)
    assert compute_person_z(depth, keypoints, _visible(1), params) == 2.0


def test_metric_bounds_reject_out_of_range_samples():
    depth = np.full((10, 10), 12.0, dtype=np.float32)
    bounded = DepthParams(max_depth_m=8.0)
    keypoints = np.array([[5, 5]], dtype=np.float32)
    assert math.isnan(compute_person_z(depth, keypoints, _visible(1), bounded))
    # With the bound disabled (0.0) the same sample is accepted.
    assert compute_person_z(depth, keypoints, _visible(1), METERS) == 12.0


def test_format_depth_and_validity():
    assert format_depth(3.2449) == "3.24 m"
    assert format_depth(NO_DEPTH) == "N/A"
    assert format_depth(float("inf")) == "N/A"
    assert not is_valid_depth(0.0)
    assert is_valid_depth(1.5)
