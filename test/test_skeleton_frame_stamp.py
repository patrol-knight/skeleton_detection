"""Tests for the published ``SkeletonFrame`` contract: header and fields.

``header.stamp`` is the ONLY timestamp on ``SkeletonFrame``. In
``input_mode='ros_camera'`` it must be the stamp of the RealSense driver's
colour image, carried unchanged through RTMO, tracking, depth and message
construction -- never replaced by the node's current time. This holds for
empty frames (zero persons) too.

Needs a sourced ROS 2 workspace with ``patrolknight_msgs`` and ``cv_bridge``
(i.e. run inside the container); skipped otherwise::

    python3 -m pytest test/test_skeleton_frame_stamp.py -v

Coverage:

    1  SkeletonFrame has no separate ``timestamp`` field
    2  ros_camera: the RGB image stamp reaches the published header.stamp
    3  the same for an empty frame (zero detections), which is still published
    4  header.frame_id is the driver's; camera_frame_id is only a fallback
       for an empty one, and frame_index counts published frames
    5  PersonSkeleton wire format: bbox [x, y, w, h], joints [x, y, conf] * 17,
       the 19 COCO connections, and NaN (never 0) for unavailable position

The RGBD path is driven through the node's real ``_on_rgbd`` and
``_process_frame`` on a harness object, so no model, camera, driver or
``realsense2_camera_msgs`` is needed: RTMO is replaced by a stub returning
fixed detections, and the publisher records what it is given.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("rclpy")
pytest.importorskip("cv_bridge")
pytest.importorskip("patrolknight_msgs.msg")

from builtin_interfaces.msg import Time  # noqa: E402
from cv_bridge import CvBridge  # noqa: E402
from patrolknight_msgs.msg import SkeletonFrame  # noqa: E402
from std_msgs.msg import Header  # noqa: E402

from skeleton_detection.inference.depth_estimation import CameraPoint  # noqa: E402
from skeleton_detection.inference.rtmo_inference import PersonDetection  # noqa: E402
from skeleton_detection.iot_node import RTMONode  # noqa: E402
from skeleton_detection.output.message_builder import (  # noqa: E402
    build_skeleton_frame,
)
from skeleton_detection.utils.coco_keypoints import COCO_CONNECTIONS_FLAT  # noqa: E402

# Deliberately far from "now", with a non-zero nanosecond part, so a
# substituted current time or a float round trip cannot pass by accident.
SOURCE_STAMP = Time(sec=1_700_000_123, nanosec=456_789_012)
WIDTH, HEIGHT = 64, 48


class FakeRGBD:
    """Shaped like realsense2_camera_msgs/msg/RGBD; the parts are real msgs."""

    def __init__(self, rgb, depth, rgb_camera_info):
        self.header = rgb.header
        self.rgb = rgb
        self.depth = depth
        self.rgb_camera_info = rgb_camera_info
        self.depth_camera_info = rgb_camera_info


class StubInference:
    def __init__(self, detections):
        self.detections = detections

    def infer(self, frame_bgr):
        return list(self.detections)


class RecordingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class StubStats:
    def record_frame(self, **kwargs):
        pass


class HarnessNode:
    """Just the state the RGBD -> SkeletonFrame path reads, plus its real methods."""

    _on_rgbd = RTMONode._on_rgbd
    _process_frame = RTMONode._process_frame
    _attach_person_position = RTMONode._attach_person_position

    def __init__(self, detections):
        self.bridge = CvBridge()
        self.camera_frame_id = "camera_color_optical_frame"
        self._rgbd_configured = True
        self.camera_intrinsics = None
        self.inference = StubInference(detections)
        self.tracker = None
        self.frame_publisher = RecordingPublisher()
        self.stats = StubStats()
        self.log_every_frame = False
        self.frame_index = 0

    def _handle_visualization(self, frame_bgr, persons, header):
        pass

    def get_logger(self):
        return FailingLogger()


class FailingLogger:
    """The RGBD path only logs when it swallowed an error: surface it instead."""

    def _fail(self, message):
        raise AssertionError(f"RGBD path logged: {message}")

    error = fatal = warning = _fail


def make_rgbd(frame_id="camera_color_optical_frame"):
    bridge = CvBridge()
    rgb = bridge.cv2_to_imgmsg(
        np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8), encoding="bgr8"
    )
    rgb.header.stamp = Time(sec=SOURCE_STAMP.sec, nanosec=SOURCE_STAMP.nanosec)
    rgb.header.frame_id = frame_id
    depth = bridge.cv2_to_imgmsg(
        np.zeros((HEIGHT, WIDTH), dtype=np.uint16), encoding="16UC1"
    )
    depth.header = rgb.header
    return FakeRGBD(rgb, depth, rgb_camera_info=None)


def make_detection():
    return PersonDetection(
        bbox_xyxy=np.array([10.0, 5.0, 30.0, 40.0], dtype=np.float32),
        score=0.9,
        keypoints_xy=np.full((17, 2), 20.0, dtype=np.float32),
        keypoint_scores=np.full(17, 0.8, dtype=np.float32),
        detection_index=0,
    )


def run_one_frame(detections, frame_id="camera_color_optical_frame"):
    node = HarnessNode(detections)
    node._on_rgbd(make_rgbd(frame_id))
    assert len(node.frame_publisher.messages) == 1
    return node.frame_publisher.messages[0]


# ----------------------------------------------------------------------
# 1 - no separate timestamp field
# ----------------------------------------------------------------------
def test_skeleton_frame_has_no_timestamp_field():
    fields = SkeletonFrame.get_fields_and_field_types()
    assert "timestamp" not in fields
    assert fields["header"] == "std_msgs/Header"


# ----------------------------------------------------------------------
# 2 - ros_camera source stamp is preserved
# ----------------------------------------------------------------------
def test_ros_camera_stamp_reaches_published_frame():
    frame = run_one_frame([make_detection()])

    assert isinstance(frame, SkeletonFrame)
    assert len(frame.persons) == 1
    assert frame.header.stamp.sec == SOURCE_STAMP.sec
    assert frame.header.stamp.nanosec == SOURCE_STAMP.nanosec
    assert frame.header.frame_id == "camera_color_optical_frame"


# ----------------------------------------------------------------------
# 3 - empty frames carry the same source stamp
# ----------------------------------------------------------------------
def test_empty_frame_is_published_with_source_stamp():
    frame = run_one_frame([])

    assert len(frame.persons) == 0
    assert frame.header.stamp.sec == SOURCE_STAMP.sec
    assert frame.header.stamp.nanosec == SOURCE_STAMP.nanosec


# ----------------------------------------------------------------------
# 4 - frame_id and frame_index
# ----------------------------------------------------------------------
def test_driver_frame_id_is_kept():
    frame = run_one_frame([], frame_id="some_other_optical_frame")
    assert frame.header.frame_id == "some_other_optical_frame"


def test_empty_driver_frame_id_falls_back_to_camera_frame_id():
    frame = run_one_frame([], frame_id="")
    assert frame.header.frame_id == HarnessNode([]).camera_frame_id


def test_frame_index_counts_published_frames():
    node = HarnessNode([])
    node._on_rgbd(make_rgbd())
    node._on_rgbd(make_rgbd())
    assert [m.frame_index for m in node.frame_publisher.messages] == [0, 1]


# ----------------------------------------------------------------------
# 5 - PersonSkeleton wire format
# ----------------------------------------------------------------------
def test_person_fields_use_the_published_conventions():
    detection = make_detection()
    detection.keypoints_xy = np.arange(34, dtype=np.float32).reshape(17, 2)
    detection.keypoint_scores = np.linspace(0.1, 0.9, 17, dtype=np.float32)
    detection.position = CameraPoint(0.5, -0.25, 3.0)

    (person,) = build_skeleton_frame(Header(), 0, [detection]).persons

    # internal xyxy [10, 5, 30, 40] -> published [x, y, width, height]
    assert list(person.bbox) == pytest.approx([10.0, 5.0, 20.0, 35.0])
    assert person.score == pytest.approx(0.9)
    # [x0, y0, conf0, x1, y1, conf1, ...] in keypoint order
    assert len(person.joints) == 51
    assert list(person.joints[:6]) == pytest.approx(
        [0.0, 1.0, detection.keypoint_scores[0], 2.0, 3.0, detection.keypoint_scores[1]]
    )
    assert list(person.connections) == list(COCO_CONNECTIONS_FLAT)
    assert len(person.connections) == 38
    assert (person.position.x, person.position.y, person.position.z) == pytest.approx(
        (0.5, -0.25, 3.0)
    )
    # depth is the Euclidean distance, not the Z value
    assert person.depth == pytest.approx(math.sqrt(0.25 + 0.0625 + 9.0))


def test_unavailable_position_is_published_as_nan_not_zero():
    (person,) = build_skeleton_frame(Header(), 0, [make_detection()]).persons
    for value in (person.position.x, person.position.y, person.position.z, person.depth):
        assert math.isnan(value)
