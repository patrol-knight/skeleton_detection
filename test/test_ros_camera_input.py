"""Tests for the ``input_mode='ros_camera'`` RGBD input path.

Pure Python, no camera, no ROS driver and no realsense2_camera_msgs: the
conversion helpers take duck-typed stand-ins for the message fields, which is
exactly why they live in their own module instead of inline in the node::

    python3 -m pytest test/test_ros_camera_input.py -v

Coverage:

    1  CameraInfo.k -> CameraIntrinsics (fx, fy, cx, cy, width, height)
    2  a degenerate / uncalibrated CameraInfo is a loud error, not NaN output
    3  depth scale comes from the ENCODING: 16UC1 -> 0.001, 32FC1 -> 1.0
    4  an unknown depth encoding is refused rather than guessed
    5  RGBD message -> the existing (H, W) depth representation
    6  a field-name change upstream fails with a readable error
    7  a driver without align_depth.enable is caught once, at startup
    8  end-to-end: RGBD-derived intrinsics + depth scale reproduce a known Z

The node itself (rclpy, cv_bridge, RGBD) is NOT imported here. The node's
input_mode / rgbd_topic parameters are covered by ``test_node_config.py``.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skeleton_detection.inference.depth_estimation import (  # noqa: E402
    CameraIntrinsics,
    DepthParams,
    deproject_pixel_to_point,
)
from skeleton_detection.input.ros_camera_subscriber import (  # noqa: E402
    ROS_DEPTH_SCALE_16UC1,
    ROS_DEPTH_SCALE_32FC1,
    RGBD_QUEUE_DEPTH,
    RGBDInputError,
    depth_array_from_msg,
    depth_scale_for_encoding,
    intrinsics_from_camera_info,
    rgbd_parts,
    validate_depth_alignment,
)


# ----------------------------------------------------------------------
# Minimal stand-ins for the ROS messages. Only the fields the code reads.
# ----------------------------------------------------------------------
class FakeCameraInfo:
    def __init__(self, k, width=848, height=480):
        self.k = k
        self.width = width
        self.height = height


class FakeImage:
    def __init__(self, encoding, array=None, frame_id="camera_color_optical_frame"):
        self.encoding = encoding
        self.array = array
        self.header = type("H", (), {"frame_id": frame_id, "stamp": None})()


class FakeRGBD:
    """Shaped like realsense2_camera_msgs/msg/RGBD."""

    def __init__(self, rgb, depth, rgb_camera_info):
        self.rgb = rgb
        self.depth = depth
        self.rgb_camera_info = rgb_camera_info
        self.depth_camera_info = rgb_camera_info


class FakeBridge:
    """Stands in for CvBridge: hands back the array the message carries."""

    def imgmsg_to_cv2(self, msg, desired_encoding="passthrough"):
        return msg.array


# The D456 colour stream at 848x480, in CameraInfo row-major k order.
FX, FY, CX, CY = 600.0, 600.5, 424.0, 240.0
K_848x480 = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]


# ----------------------------------------------------------------------
# 1 - CameraInfo.k -> CameraIntrinsics
# ----------------------------------------------------------------------
def test_camera_info_k_maps_to_camera_intrinsics():
    intrinsics = intrinsics_from_camera_info(FakeCameraInfo(K_848x480))

    assert isinstance(intrinsics, CameraIntrinsics)
    # k = [fx, 0, cx, 0, fy, cy, 0, 0, 1] -- indices 0, 4, 2, 5.
    assert intrinsics.fx == pytest.approx(FX)
    assert intrinsics.fy == pytest.approx(FY)
    assert intrinsics.cx == pytest.approx(CX)
    assert intrinsics.cy == pytest.approx(CY)
    assert (intrinsics.width, intrinsics.height) == (848, 480)
    assert intrinsics.is_valid()


def test_camera_info_accepts_a_numpy_k_array():
    """rclpy hands k over as an array, not a list."""
    intrinsics = intrinsics_from_camera_info(
        FakeCameraInfo(np.array(K_848x480, dtype=np.float64))
    )
    assert intrinsics.fx == pytest.approx(FX)
    assert intrinsics.cy == pytest.approx(CY)


# ----------------------------------------------------------------------
# 2 - a useless CameraInfo must be loud
# ----------------------------------------------------------------------
def test_uncalibrated_camera_info_raises():
    """An all-zero k is what an uncalibrated driver publishes."""
    with pytest.raises(RGBDInputError, match="degenerate intrinsics"):
        intrinsics_from_camera_info(FakeCameraInfo([0.0] * 9))


def test_wrong_length_k_raises():
    with pytest.raises(RGBDInputError, match="9 numbers"):
        intrinsics_from_camera_info(FakeCameraInfo([1.0, 2.0, 3.0]))


def test_zero_size_camera_info_raises():
    with pytest.raises(RGBDInputError, match="degenerate intrinsics"):
        intrinsics_from_camera_info(FakeCameraInfo(K_848x480, width=0, height=0))


# ----------------------------------------------------------------------
# 3 / 4 - depth scale is decided by the encoding
# ----------------------------------------------------------------------
def test_16uc1_is_millimetres():
    assert depth_scale_for_encoding("16UC1") == pytest.approx(0.001)
    assert ROS_DEPTH_SCALE_16UC1 == pytest.approx(0.001)


def test_32fc1_is_already_metres():
    assert depth_scale_for_encoding("32FC1") == pytest.approx(1.0)
    assert ROS_DEPTH_SCALE_32FC1 == pytest.approx(1.0)


def test_encoding_match_is_case_insensitive():
    assert depth_scale_for_encoding("16uc1") == pytest.approx(0.001)


def test_unknown_encoding_is_refused():
    """Guessing a scale here would publish silently wrong distances."""
    with pytest.raises(RGBDInputError, match="Unsupported depth encoding"):
        depth_scale_for_encoding("rgb8")


# ----------------------------------------------------------------------
# 5 - RGBD -> the existing (H, W) depth representation
# ----------------------------------------------------------------------
def test_depth_message_becomes_a_2d_raw_array():
    raw = np.full((480, 848), 1500, dtype=np.uint16)  # 1.5 m in millimetres
    array = depth_array_from_msg(FakeImage("16UC1", raw), FakeBridge())

    assert array.shape == (480, 848)
    assert array.ndim == 2
    # Raw units are preserved: scaling is the depth module's job.
    assert array[0, 0] == 1500
    assert array.dtype == np.uint16


def test_multichannel_depth_is_rejected():
    rgb_like = np.zeros((480, 848, 3), dtype=np.uint8)
    with pytest.raises(RGBDInputError, match=r"single-channel"):
        depth_array_from_msg(FakeImage("16UC1", rgb_like), FakeBridge())


# ----------------------------------------------------------------------
# 6 - the RGBD field contract
# ----------------------------------------------------------------------
def test_rgbd_parts_extracts_rgb_depth_and_colour_info():
    info = FakeCameraInfo(K_848x480)
    msg = FakeRGBD(FakeImage("rgb8"), FakeImage("16UC1"), info)

    rgb, depth, camera_info = rgbd_parts(msg)

    assert rgb.encoding == "rgb8"
    assert depth.encoding == "16UC1"
    # The COLOUR CameraInfo, not the depth one: keypoints are colour pixels.
    assert camera_info is info


def test_missing_rgbd_field_names_what_is_wrong():
    incomplete = type("Partial", (), {"rgb": FakeImage("rgb8")})()
    with pytest.raises(RGBDInputError, match="missing"):
        rgbd_parts(incomplete)


# ----------------------------------------------------------------------
# 7 - unaligned depth is caught once, not once per frame
# ----------------------------------------------------------------------
def test_aligned_depth_passes_validation():
    intrinsics = intrinsics_from_camera_info(FakeCameraInfo(K_848x480))
    # Same grid as colour: exactly what align_depth.enable produces.
    assert validate_depth_alignment(848, 480, intrinsics) is None


def test_unaligned_depth_names_align_depth_enable():
    """A driver started without align_depth.enable must fail loudly."""
    intrinsics = intrinsics_from_camera_info(FakeCameraInfo(K_848x480))
    with pytest.raises(RGBDInputError, match="align_depth.enable"):
        validate_depth_alignment(1280, 720, intrinsics)


def test_rgbd_subscription_queue_depth_is_one():
    """Keep only the newest frame: a deeper queue would build up latency."""
    assert RGBD_QUEUE_DEPTH == 1


# ----------------------------------------------------------------------
# 8 - end to end, against the untouched depth module
# ----------------------------------------------------------------------
def test_rgbd_intrinsics_and_scale_reproduce_a_known_point():
    """A 2.0 m point at a known pixel deprojects correctly from RGBD inputs.

    Exercises the whole conversion chain the callback performs, then hands the
    result to the EXISTING depth-estimation interface unchanged.
    """
    info = FakeCameraInfo(K_848x480)
    depth_mm = np.full((480, 848), 2000, dtype=np.uint16)
    msg = FakeRGBD(FakeImage("rgb8"), FakeImage("16UC1", depth_mm), info)

    rgb, depth_msg, camera_info = rgbd_parts(msg)
    intrinsics = intrinsics_from_camera_info(camera_info)
    validate_depth_alignment(848, 480, intrinsics)
    depth_image = depth_array_from_msg(depth_msg, FakeBridge())
    params = DepthParams(depth_scale=depth_scale_for_encoding(depth_msg.encoding))

    # Raw millimetres * 0.001 = 2.0 m.
    z_m = float(depth_image[240, 424]) * params.depth_scale
    assert z_m == pytest.approx(2.0)

    # At the principal point, X and Y are 0 and the distance is exactly Z.
    point = deproject_pixel_to_point(CX, CY, z_m, intrinsics)
    assert point.x == pytest.approx(0.0)
    assert point.y == pytest.approx(0.0)
    assert point.z == pytest.approx(2.0)
    assert point.distance == pytest.approx(2.0)

    # 60 px right of centre: X = (u - cx) * Z / fx.
    offset = deproject_pixel_to_point(CX + 60.0, CY, z_m, intrinsics)
    assert offset.x == pytest.approx(60.0 * 2.0 / FX)
    assert offset.z == pytest.approx(2.0)
