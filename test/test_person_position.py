"""Tests for the camera-frame person position and Euclidean depth.

Pure numpy, no camera: synthetic intrinsics and depth images only::

    python3 -m pytest test/test_person_position.py -v

Coverage:

    1  centre pixel          -> X = Y = 0, depth = Z
    2  horizontal offset     -> X = (u-cx)*Z/fx, Y = 0, depth = sqrt(X^2+Z^2)
    3  general 3D case       -> depth = sqrt(X^2+Y^2+Z^2)
    4  invalid Z / no intrinsics / no bbox -> position and depth all NaN
    5  representative pixel is the centre of the bbox clipped to the image
    6  end-to-end: two-stage-median Z + bbox centre + intrinsics
    7  a depth image not matching the colour intrinsics is a loud error
    8  PersonDetection.depth is the Euclidean distance of its position
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skeleton_detection.inference.depth_estimation import (  # noqa: E402
    NO_POSITION,
    CameraIntrinsics,
    CameraPoint,
    DepthParams,
    bbox_center_pixel,
    compute_person_position,
    deproject_pixel_to_point,
)
from skeleton_detection.inference.rtmo_inference import (  # noqa: E402
    PersonDetection,
)

# Synthetic colour intrinsics for a 848x480 stream (fx != fy on purpose, so a
# swapped axis would be caught).
INTRINSICS = CameraIntrinsics(
    fx=420.0, fy=410.0, cx=424.0, cy=240.0, width=848, height=480
)
METERS = DepthParams(depth_scale=1.0, keypoint_score_threshold=0.3)
# A person box whose centre pixel is (634, 240).
PERSON_BBOX = [560.0, 100.0, 708.0, 380.0]


def _assert_all_nan(point):
    assert math.isnan(point.x) and math.isnan(point.y) and math.isnan(point.z)
    assert math.isnan(point.distance)


def test_center_pixel_lies_on_the_optical_axis():
    point = deproject_pixel_to_point(INTRINSICS.cx, INTRINSICS.cy, 3.0, INTRINSICS)
    assert point.x == 0.0
    assert point.y == 0.0
    assert point.z == 3.0
    assert point.distance == 3.0


def test_horizontal_offset():
    u, z = 634.0, 3.0  # 210 px right of cx
    point = deproject_pixel_to_point(u, INTRINSICS.cy, z, INTRINSICS)
    expected_x = (u - INTRINSICS.cx) * z / INTRINSICS.fx  # 1.5 m
    assert point.x == pytest.approx(expected_x)
    assert point.x == pytest.approx(1.5)
    assert point.y == 0.0
    assert point.z == z
    assert point.distance == pytest.approx(math.sqrt(expected_x**2 + z**2))
    # Off-centre the Euclidean distance is strictly larger than the Z-depth.
    assert point.distance > z


def test_general_3d_case_uses_the_right_axes():
    u, v, z = 200.0, 400.0, 2.5  # left of and below the principal point
    point = deproject_pixel_to_point(u, v, z, INTRINSICS)
    expected_x = (u - INTRINSICS.cx) * z / INTRINSICS.fx
    expected_y = (v - INTRINSICS.cy) * z / INTRINSICS.fy
    assert point.x == pytest.approx(expected_x) and point.x < 0.0  # image left
    assert point.y == pytest.approx(expected_y) and point.y > 0.0  # image down
    assert point.z == z
    assert point.distance == pytest.approx(
        math.sqrt(expected_x**2 + expected_y**2 + z**2)
    )


@pytest.mark.parametrize("z", [float("nan"), 0.0, -1.0, float("inf")])
def test_invalid_z_gives_nan_position_and_depth(z):
    _assert_all_nan(deproject_pixel_to_point(100.0, 100.0, z, INTRINSICS))


def test_missing_or_invalid_intrinsics_give_nan():
    _assert_all_nan(deproject_pixel_to_point(100.0, 100.0, 2.0, None))
    bad = CameraIntrinsics(fx=0.0, fy=410.0, cx=424.0, cy=240.0, width=848, height=480)
    _assert_all_nan(deproject_pixel_to_point(100.0, 100.0, 2.0, bad))


def test_no_position_is_nan_not_zero():
    _assert_all_nan(NO_POSITION)
    assert CameraPoint(0.0, 0.0, 0.0).distance == 0.0  # zero is a real value


def test_bbox_center_pixel():
    assert bbox_center_pixel([100, 50, 300, 450], 848, 480) == (200.0, 250.0)


def test_bbox_center_uses_the_visible_part_of_an_unclipped_box():
    # RTMO boxes are not clipped: this one runs off the left and bottom edges.
    assert bbox_center_pixel([-100, 200, 100, 600], 848, 480) == (50.0, 340.0)


def test_bbox_center_rejects_unusable_boxes():
    assert bbox_center_pixel([900, 10, 1000, 100], 848, 480) is None  # off image
    assert bbox_center_pixel([np.nan, 0, 10, 10], 848, 480) is None
    assert bbox_center_pixel(None, 848, 480) is None


def _person_on_uniform_depth(z, dtype=np.float32):
    depth = np.full((INTRINSICS.height, INTRINSICS.width), z, dtype=dtype)
    keypoints = np.array([[600, 200], [620, 260], [640, 320]], dtype=np.float32)
    scores = np.ones(3, dtype=np.float32)
    return depth, keypoints, scores


def test_end_to_end_position_from_depth_image():
    depth = np.zeros((INTRINSICS.height, INTRINSICS.width), dtype=np.float32)
    # Joint Z values 2.0, 3.0, 9.0 -> person Z is the median, 3.0.
    keypoints = np.array([[600, 200], [620, 260], [640, 320]], dtype=np.float32)
    for (u, v), value in zip(keypoints.astype(int), (2.0, 3.0, 9.0)):
        depth[v - 1: v + 2, u - 1: u + 2] = value
    scores = np.ones(3, dtype=np.float32)

    point = compute_person_position(
        depth, keypoints, scores, PERSON_BBOX, INTRINSICS, METERS
    )

    assert point.z == 3.0  # robust Z, not the bbox-centre pixel's depth (0)
    assert point.x == pytest.approx((634.0 - INTRINSICS.cx) * 3.0 / INTRINSICS.fx)
    assert point.y == 0.0
    assert point.distance == pytest.approx(math.sqrt(point.x**2 + 9.0))


def test_end_to_end_raw_z16_is_scaled_once():
    # 2000 raw units = 2.0 m at 0.001 m/unit
    depth, keypoints, scores = _person_on_uniform_depth(2000, dtype=np.uint16)
    bbox = [INTRINSICS.cx - 50, INTRINSICS.cy - 50, INTRINSICS.cx + 50, INTRINSICS.cy + 50]
    point = compute_person_position(
        depth, keypoints, scores, bbox, INTRINSICS, DepthParams(depth_scale=0.001)
    )
    assert point.z == pytest.approx(2.0)
    assert point.distance == pytest.approx(2.0)


def test_invalid_person_z_gives_nan_position_and_depth():
    depth, keypoints, scores = _person_on_uniform_depth(0.0)  # all holes
    _assert_all_nan(
        compute_person_position(
            depth, keypoints, scores, PERSON_BBOX, INTRINSICS, METERS
        )
    )


def test_missing_inputs_give_nan():
    depth, keypoints, scores = _person_on_uniform_depth(3.0)
    _assert_all_nan(
        compute_person_position(None, keypoints, scores, PERSON_BBOX, INTRINSICS)
    )
    _assert_all_nan(compute_person_position(depth, keypoints, scores, PERSON_BBOX, None))
    _assert_all_nan(
        compute_person_position(depth, keypoints, scores, None, INTRINSICS, METERS)
    )


def test_depth_image_not_matching_colour_intrinsics_raises():
    depth = np.full((240, 424), 3.0, dtype=np.float32)  # wrong size
    keypoints = np.array([[100, 100]], dtype=np.float32)
    with pytest.raises(ValueError):
        compute_person_position(
            depth, keypoints, np.ones(1), [0, 0, 200, 200], INTRINSICS, METERS
        )


def test_person_detection_depth_is_euclidean():
    detection = PersonDetection(
        bbox_xyxy=np.zeros(4, dtype=np.float32),
        score=0.9,
        keypoints_xy=np.zeros((17, 2), dtype=np.float32),
        keypoint_scores=np.zeros(17, dtype=np.float32),
        detection_index=0,
    )
    assert math.isnan(detection.depth)  # default: unavailable, not 0
    detection.position = CameraPoint(x=1.0, y=2.0, z=2.0)
    assert detection.depth == pytest.approx(3.0)  # sqrt(1 + 4 + 4)
