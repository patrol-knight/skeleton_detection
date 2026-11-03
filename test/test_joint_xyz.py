"""Tests for the V2.1 per-joint XYZ (``compute_joint_depths``).

Pure numpy, no ROS::

    python3 -m pytest test/test_joint_xyz.py -v

Coverage:

    1  depth gate: below / equal / above the threshold, non-finite scores,
       and no depth sampling at all below it
    2  XYZ against an independent pinhole calculation at known intrinsics,
       deprojecting each joint's OWN pixel
    3  3x3 median: eight 2 m samples and one 6 m outlier -> 2 m
    4  image boundaries (clipped patch) and joints outside the image
    5  all-invalid depth, range bounds, and NaN XYZ with the original
       confidence kept in all 17 slots
    6  RealSense 16UC1 millimetres (scale 0.001) and ZED 32FC1 meters
    7  legacy person position from the same joint Z equals the original
       ``compute_person_position``
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skeleton_detection.inference import depth_estimation  # noqa: E402
from skeleton_detection.inference.depth_estimation import (  # noqa: E402
    CameraIntrinsics,
    DepthParams,
    compute_joint_depths,
    compute_person_position,
    person_position_from_joint_z,
)
from skeleton_detection.input.ros_camera_subscriber import (  # noqa: E402
    depth_scale_for_encoding,
)

WIDTH, HEIGHT = 848, 480
INTRINSICS = CameraIntrinsics(
    fx=600.0, fy=610.0, cx=424.5, cy=239.5, width=WIDTH, height=HEIGHT
)
METERS = DepthParams(depth_scale=1.0, keypoint_score_threshold=0.30)


def flat_depth(value_m, dtype=np.float32):
    return np.full((HEIGHT, WIDTH), value_m, dtype=dtype)


def one_joint(u, v, score, n=17):
    """Joint 0 at (u, v) with ``score``; the other 16 below any threshold."""
    keypoints = np.zeros((n, 2), dtype=np.float32)
    keypoints[0] = (u, v)
    scores = np.zeros(n, dtype=np.float32)
    scores[0] = score
    return keypoints, scores


def joint(xyzc, index):
    return xyzc[4 * index: 4 * index + 4]


def expected_xyz(u, v, z, k=INTRINSICS):
    return ((u - k.cx) * z / k.fx, (v - k.cy) * z / k.fy, z)


# ----------------------------------------------------------------------
# 1 - depth gate
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "score, sampled",
    [(np.float32(0.29), False), (np.float32(0.30), True), (np.float32(0.31), True)],
)
def test_default_gate_below_equal_above(score, sampled):
    keypoints, scores = one_joint(100.0, 200.0, score)
    _, xyzc = compute_joint_depths(
        flat_depth(2.0), keypoints, scores, INTRINSICS, METERS
    )
    j0 = joint(xyzc, 0)
    assert j0[3] == score
    assert bool(np.all(np.isfinite(j0[:3]))) is sampled


@pytest.mark.parametrize("threshold", [0.30, 0.5, 0.7, 0.85])
def test_score_exactly_at_threshold_is_sampled(threshold):
    keypoints, scores = one_joint(100.0, 200.0, np.float32(threshold))
    params = DepthParams(depth_scale=1.0, keypoint_score_threshold=threshold)
    _, xyzc = compute_joint_depths(flat_depth(2.0), keypoints, scores, INTRINSICS, params)
    assert np.all(np.isfinite(joint(xyzc, 0)[:3]))


def test_no_depth_sampling_below_threshold(monkeypatch):
    calls = []
    real = depth_estimation.compute_joint_z

    def counting(image, u, v, params):
        calls.append((u, v))
        return real(image, u, v, params)

    monkeypatch.setattr(depth_estimation, "compute_joint_z", counting)
    keypoints = np.tile(np.array([[50.0, 60.0]], dtype=np.float32), (17, 1))
    scores = np.full(17, 0.2999, dtype=np.float32)
    scores[[3, 9]] = (0.30, 0.95)
    _, xyzc = compute_joint_depths(flat_depth(2.0), keypoints, scores, INTRINSICS, METERS)

    assert len(calls) == 2
    finite = [i for i in range(17) if np.all(np.isfinite(joint(xyzc, i)[:3]))]
    assert finite == [3, 9]


def test_non_finite_score_is_not_sampled(monkeypatch):
    monkeypatch.setattr(
        depth_estimation,
        "compute_joint_z",
        lambda *a: pytest.fail("depth sampled for a non-finite score"),
    )
    keypoints, scores = one_joint(100.0, 200.0, np.nan)
    _, xyzc = compute_joint_depths(flat_depth(2.0), keypoints, scores, INTRINSICS, METERS)
    assert np.all(np.isnan(joint(xyzc, 0)))


# ----------------------------------------------------------------------
# 2 - XYZ against an independent calculation
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "u, v", [(424.5, 239.5), (100.0, 50.0), (800.0, 470.0), (10.25, 300.75)]
)
def test_xyz_matches_pinhole_at_joint_pixel(u, v):
    keypoints, scores = one_joint(u, v, 0.9)
    _, xyzc = compute_joint_depths(flat_depth(2.5), keypoints, scores, INTRINSICS, METERS)
    x, y, z, c = joint(xyzc, 0)
    ex, ey, ez = expected_xyz(u, v, 2.5)
    assert (x, y, z) == pytest.approx((ex, ey, ez), abs=1e-5)
    assert c == pytest.approx(0.9)


def test_each_joint_uses_its_own_pixel_not_the_bbox_centre():
    keypoints = np.array([[100.0 + 20 * i, 50.0 + 10 * i] for i in range(17)], np.float32)
    scores = np.full(17, 0.8, dtype=np.float32)
    _, xyzc = compute_joint_depths(flat_depth(3.0), keypoints, scores, INTRINSICS, METERS)
    for i in range(17):
        assert tuple(joint(xyzc, i)[:3]) == pytest.approx(
            expected_xyz(*keypoints[i], 3.0), abs=1e-5
        )


# ----------------------------------------------------------------------
# 3 - 3x3 median with an outlier
# ----------------------------------------------------------------------
def test_three_by_three_median_rejects_one_outlier():
    depth = flat_depth(9.0)
    depth[199:202, 99:102] = 2.0
    depth[201, 101] = 6.0  # one outlier in the 3x3 patch
    keypoints, scores = one_joint(100.0, 200.0, 0.9)
    joint_z, xyzc = compute_joint_depths(depth, keypoints, scores, INTRINSICS, METERS)
    assert joint_z[0] == pytest.approx(2.0)
    assert tuple(joint(xyzc, 0)[:3]) == pytest.approx(
        expected_xyz(100.0, 200.0, 2.0), abs=1e-5
    )


# ----------------------------------------------------------------------
# 4 - boundaries
# ----------------------------------------------------------------------
def test_corner_joint_uses_clipped_patch():
    depth = flat_depth(9.0)
    depth[0:2, 0:2] = (1.0, 1.0), (1.0, 5.0)  # 4 in-bounds samples, median 1.0
    keypoints, scores = one_joint(0.0, 0.0, 0.9)
    joint_z, _ = compute_joint_depths(depth, keypoints, scores, INTRINSICS, METERS)
    assert joint_z[0] == pytest.approx(1.0)


def test_last_pixel_and_outside_image():
    depth = flat_depth(2.0)
    keypoints, scores = one_joint(WIDTH - 1, HEIGHT - 1, 0.9)
    keypoints[1] = (WIDTH + 3, 10)
    keypoints[2] = (-5, 10)
    scores[1] = scores[2] = 0.9
    joint_z, xyzc = compute_joint_depths(depth, keypoints, scores, INTRINSICS, METERS)
    assert joint_z[0] == pytest.approx(2.0)
    for i in (1, 2):
        assert np.all(np.isnan(joint(xyzc, i)[:3]))
        assert joint(xyzc, i)[3] == pytest.approx(0.9)


# ----------------------------------------------------------------------
# 5 - missing values keep their slot and confidence
# ----------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0.0, np.nan, np.inf, -np.inf])
def test_all_invalid_patch_gives_nan_xyz_and_original_confidence(bad):
    keypoints, scores = one_joint(100.0, 200.0, 0.77)
    _, xyzc = compute_joint_depths(flat_depth(bad), keypoints, scores, INTRINSICS, METERS)
    assert xyzc.shape == (68,) and xyzc.dtype == np.float32
    assert np.all(np.isnan(joint(xyzc, 0)[:3]))
    assert joint(xyzc, 0)[3] == np.float32(0.77)


def test_range_bounds_apply():
    keypoints, scores = one_joint(100.0, 200.0, 0.9)
    params = DepthParams(depth_scale=1.0, min_depth_m=0.5, max_depth_m=1.5)
    _, xyzc = compute_joint_depths(flat_depth(2.0), keypoints, scores, INTRINSICS, params)
    assert np.all(np.isnan(joint(xyzc, 0)[:3]))


def test_no_depth_or_intrinsics_keeps_all_slots():
    keypoints = np.zeros((17, 2), np.float32)
    scores = np.linspace(0.1, 0.9, 17).astype(np.float32)
    for depth, k in ((None, INTRINSICS), (flat_depth(2.0), None)):
        joint_z, xyzc = compute_joint_depths(depth, keypoints, scores, k, METERS)
        assert xyzc.shape == (68,)
        assert np.all(np.isnan(xyzc.reshape(17, 4)[:, :3]))
        assert np.array_equal(xyzc.reshape(17, 4)[:, 3], scores)
        assert np.all(np.isnan(joint_z))


def test_misaligned_depth_raises():
    keypoints, scores = one_joint(10.0, 10.0, 0.9)
    with pytest.raises(ValueError):
        compute_joint_depths(
            np.ones((720, 1280), np.float32), keypoints, scores, INTRINSICS, METERS
        )


# ----------------------------------------------------------------------
# 6 - depth units
# ----------------------------------------------------------------------
def test_realsense_16uc1_millimetres():
    depth = flat_depth(2000, dtype=np.uint16)
    depth[200, 100] = 0  # RealSense "no measurement" inside the patch
    params = DepthParams(depth_scale=depth_scale_for_encoding("16UC1"))
    keypoints, scores = one_joint(100.0, 200.0, 0.9)
    _, xyzc = compute_joint_depths(depth, keypoints, scores, INTRINSICS, params)
    assert tuple(joint(xyzc, 0)[:3]) == pytest.approx(
        expected_xyz(100.0, 200.0, 2.0), abs=1e-5
    )


def test_zed_32fc1_meters_with_nan_and_inf():
    depth = flat_depth(1.75)
    depth[199, 99] = np.nan
    depth[199, 100] = np.inf
    depth[199, 101] = -np.inf
    params = DepthParams(depth_scale=depth_scale_for_encoding("32FC1"))
    keypoints, scores = one_joint(100.0, 200.0, 0.9)
    joint_z, _ = compute_joint_depths(depth, keypoints, scores, INTRINSICS, params)
    assert joint_z[0] == pytest.approx(1.75)


# ----------------------------------------------------------------------
# 7 - legacy position reuses the joint Z
# ----------------------------------------------------------------------
def test_legacy_position_from_joint_z_matches_original():
    rng = np.random.default_rng(7)
    depth = rng.uniform(1.0, 4.0, size=(HEIGHT, WIDTH)).astype(np.float32)
    depth[rng.random((HEIGHT, WIDTH)) < 0.2] = 0.0
    keypoints = rng.uniform([0, 0], [WIDTH, HEIGHT], size=(17, 2)).astype(np.float32)
    scores = rng.uniform(0.0, 1.0, size=17).astype(np.float32)
    bbox = np.array([200.0, 100.0, 400.0, 460.0], np.float32)

    joint_z, _ = compute_joint_depths(depth, keypoints, scores, INTRINSICS, METERS)
    reused = person_position_from_joint_z(joint_z, bbox, INTRINSICS)
    original = compute_person_position(depth, keypoints, scores, bbox, INTRINSICS, METERS)
    assert (reused.x, reused.y, reused.z) == pytest.approx(
        (original.x, original.y, original.z)
    )
    assert math.isfinite(reused.z)
