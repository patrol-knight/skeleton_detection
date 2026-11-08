"""Tests for method D (person-depth cluster scoring) in depth_estimation.

Pure numpy, no ROS. Synthetic scenes: a person plane at 2.0 m, background at
6.0 m (outside z_center +- PERSON_DEPTH_RADIUS_M + sigma tail).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skeleton_detection.inference import depth_estimation as de  # noqa: E402
from skeleton_detection.inference.depth_estimation import (  # noqa: E402
    CameraIntrinsics,
    DepthParams,
    compute_joint_depths,
    compute_person_cluster_joint_depths,
    compute_person_joint_depths,
    person_reference_z,
    select_person_cluster_z,
)

WIDTH, HEIGHT = 848, 480
K = CameraIntrinsics(fx=600.0, fy=610.0, cx=424.5, cy=239.5, width=WIDTH, height=HEIGHT)
METERS = DepthParams(depth_scale=1.0, keypoint_score_threshold=0.30)
BODY, WALL = 2.0, 6.0

# 17 joints spread over a 2 m "person" region, all above threshold
GRID = np.array([[300 + 20 * (i % 5), 150 + 30 * (i // 5)] for i in range(17)], dtype=np.float32)


def scene(body=BODY, wall=WALL):
    depth = np.full((HEIGHT, WIDTH), wall, dtype=np.float32)
    depth[100:300, 250:450] = body
    return depth


def scores(value=0.9):
    return np.full(17, value, dtype=np.float32)


def xyz(packed):
    return np.asarray(packed, dtype=np.float32).reshape(17, 4)


def test_median_mode_is_bit_identical_to_baseline():
    depth = scene()
    depth[150:153, 300:302] = np.nan
    kp, sc = GRID.copy(), scores()
    sc[3] = 0.1
    a = compute_joint_depths(depth, kp, sc, K, METERS)
    b = compute_person_joint_depths(depth, kp, sc, K, METERS, method="median")
    assert np.array_equal(a[0], b[0], equal_nan=True)
    assert np.array_equal(a[1], b[1], equal_nan=True)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        compute_person_joint_depths(scene(), GRID, scores(), K, METERS, method="mean")


def test_all_depth_invalid_gives_all_nan_and_keeps_confidence():
    depth = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
    z, packed = compute_person_cluster_joint_depths(depth, GRID, scores(0.8), K, METERS)
    p = xyz(packed)
    assert np.all(np.isnan(p[:, :3])) and np.all(np.isnan(z))
    assert np.allclose(p[:, 3], 0.8)


def test_fewer_than_min_reference_joints_falls_back_to_baseline():
    # only 4 gated joints; one of them sits at a body/wall edge where D would differ
    depth = scene()
    kp, sc = GRID.copy(), scores(0.0)
    sc[:4] = 0.9
    kp[0] = (450.0, 150.0)          # baseline 3x3 at the edge -> median of mixed values
    base = compute_joint_depths(depth, kp, sc, K, METERS)
    d = compute_person_cluster_joint_depths(depth, kp, sc, K, METERS)
    assert de.MIN_REFERENCE_JOINTS == 5
    assert np.array_equal(base[1], d[1], equal_nan=True)
    assert np.array_equal(base[0], d[0], equal_nan=True)


def test_below_confidence_threshold_is_never_sampled():
    kp, sc = GRID.copy(), scores()
    sc[5] = np.float32(0.2999)
    p = xyz(compute_person_cluster_joint_depths(scene(), kp, sc, K, METERS)[1])
    assert np.all(np.isnan(p[5, :3])) and p[5, 3] == np.float32(0.2999)


def test_d_recovers_joint_whose_baseline_3x3_is_all_invalid():
    depth = scene()
    depth[198:203, 298:303] = np.nan    # 5x5 hole around joint at (300, 200)
    kp = GRID.copy()
    kp[0] = (300.0, 200.0)
    base_z, _ = compute_joint_depths(depth, kp, scores(), K, METERS)
    z, packed = compute_person_cluster_joint_depths(depth, kp, scores(), K, METERS)
    assert np.isnan(base_z[0])
    assert z[0] == pytest.approx(BODY)
    x, y, zz, _ = xyz(packed)[0]
    assert (x, y) == pytest.approx(((300.0 - K.cx) * BODY / K.fx, (200.0 - K.cy) * BODY / K.fy), rel=1e-6)


def test_patch_at_image_border_is_clipped():
    depth = np.full((HEIGHT, WIDTH), BODY, dtype=np.float32)
    assert select_person_cluster_z(depth, 0.0, 0.0, BODY, METERS) == pytest.approx(BODY)
    assert select_person_cluster_z(depth, WIDTH - 1, HEIGHT - 1, BODY, METERS) == pytest.approx(BODY)
    assert np.isnan(select_person_cluster_z(depth, WIDTH + 2.0, 5.0, BODY, METERS))


def test_single_pixel_cluster_is_not_eligible():
    depth = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
    depth[200, 300] = BODY          # only one valid pixel in the 7x7
    assert de.MIN_CLUSTER_PIXELS == 2
    assert np.isnan(select_person_cluster_z(depth, 300.0, 200.0, BODY, METERS))
    depth[200, 301] = BODY + 0.05   # second pixel within tolerance -> one 2-pixel cluster
    assert select_person_cluster_z(depth, 300.0, 200.0, BODY, METERS) == pytest.approx(BODY + 0.025)


def test_all_clusters_ineligible_gives_missing_not_baseline():
    # joint whose whole 7x7 is wall, person elsewhere at 2 m
    depth = scene()
    kp = GRID.copy()
    kp[0] = (600.0, 400.0)          # pure wall
    base_z, _ = compute_joint_depths(depth, kp, scores(), K, METERS)
    z, packed = compute_person_cluster_joint_depths(depth, kp, scores(), K, METERS)
    assert base_z[0] == pytest.approx(WALL)
    assert np.isnan(z[0]) and np.all(np.isnan(xyz(packed)[0, :3]))
    assert xyz(packed)[0, 3] == pytest.approx(0.9)


def test_edge_joint_picks_person_cluster_over_wall():
    depth = scene()
    kp = GRID.copy()
    kp[0] = (450.0, 150.0)          # right edge of the person: half body, half wall
    z, _ = compute_person_cluster_joint_depths(depth, kp, scores(), K, METERS)
    assert z[0] == pytest.approx(BODY)


def test_persons_are_independent():
    depth = np.full((HEIGHT, WIDTH), WALL, dtype=np.float32)
    depth[100:300, 50:250] = 2.0
    depth[100:300, 500:700] = 4.5
    a = np.array([[60 + 20 * (i % 5), 150 + 30 * (i // 5)] for i in range(17)], np.float32)
    b = a + np.array([450, 0], np.float32)
    za, _ = compute_person_cluster_joint_depths(depth, a, scores(), K, METERS)
    zb, _ = compute_person_cluster_joint_depths(depth, b, scores(), K, METERS)
    alone = depth.copy()
    alone[100:300, 500:700] = WALL
    za_alone, _ = compute_person_cluster_joint_depths(alone, a, scores(), K, METERS)
    assert np.allclose(za, 2.0) and np.allclose(zb, 4.5)
    assert np.array_equal(za, za_alone)


def test_exact_tie_is_deterministic_more_pixels_then_smaller_depth():
    # two 2-pixel clusters mirrored about the joint: a + b == b + a exactly,
    # so S and support tie and the smaller depth must win every time
    depth = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
    depth[200, 298:300] = 2.0
    depth[200, 301:303] = 2.5
    for _ in range(3):
        assert select_person_cluster_z(depth, 300.0, 200.0, 2.2, METERS) == pytest.approx(2.0)


def test_reference_comes_from_baseline_snapshot_only(monkeypatch):
    depth = scene()
    kp = GRID.copy()
    kp[0] = (450.0, 150.0)
    base_z, _ = compute_joint_depths(depth, kp, scores(), K, METERS)
    expected = person_reference_z(base_z)
    seen = []
    real = de.select_person_cluster_z

    def spy(image, u, v, z_center, params):
        seen.append(z_center)
        return real(image, u, v, z_center, params)

    monkeypatch.setattr(de, "select_person_cluster_z", spy)
    compute_person_cluster_joint_depths(depth, kp, scores(), K, METERS)
    assert len(seen) == 17 and all(s == expected for s in seen)


def test_person_reference_requires_min_joints():
    assert np.isnan(person_reference_z([2.0, 2.0, 2.0, 2.0, np.nan]))
    assert person_reference_z([1.0, 2.0, 3.0, 4.0, 5.0]) == 3.0


def test_depth_scale_and_bounds_follow_baseline_rules():
    depth_mm = (scene() * 1000).astype(np.uint16)
    params_mm = DepthParams(depth_scale=0.001, keypoint_score_threshold=0.30)
    z, _ = compute_person_cluster_joint_depths(depth_mm, GRID, scores(), K, params_mm)
    assert np.allclose(z, BODY)
    bounded = DepthParams(depth_scale=1.0, keypoint_score_threshold=0.30, max_depth_m=1.5)
    z, _ = compute_person_cluster_joint_depths(scene(), GRID, scores(), K, bounded)
    assert np.all(np.isnan(z))


@pytest.mark.parametrize("name, value", [
    ("DEPTH_SAMPLING_METHOD", "mean"), ("DEPTH_PATCH_SIZE", 6), ("DEPTH_PATCH_SIZE", 0),
    ("DEPTH_PIXEL_SIGMA_PX", 0.0), ("PERSON_DEPTH_RADIUS_M", -1.0), ("PERSON_DEPTH_SIGMA_M", float("nan")),
    ("MIN_PERSON_DEPTH_SCORE", 0.0), ("DEPTH_CLUSTER_TOLERANCE_M", 0.0), ("MIN_CLUSTER_PIXELS", 0),
    ("MIN_REFERENCE_JOINTS", 0),
])
def test_invalid_constants_are_rejected(monkeypatch, name, value):
    monkeypatch.setattr(de, name, value)
    with pytest.raises(ValueError):
        de._validate_depth_sampling_config()
