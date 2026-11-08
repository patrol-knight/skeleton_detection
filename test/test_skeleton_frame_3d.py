"""Tests for the V2.1 outputs: official SkeletonFrame3D + legacy debug frame.

Needs a sourced ROS 2 workspace with ``patrolknight_msgs`` (V2.1 messages)
and ``cv_bridge`` (i.e. run inside the container); skipped otherwise::

    python3 -m pytest test/test_skeleton_frame_3d.py -v

Coverage:

    1  message definitions: field names, order and types of both 3D messages,
       and the legacy messages still importable
    2  one RTMO + tracking pass per frame feeds every output
    3  official persons: 68-float joints_xyzc with original confidences,
       XYZ from the aligned depth, NaN where gated or unusable
    4  negative person IDs are omitted from the official frame only; a tracked
       person with no XYZ at all is still published
    5  empty frames: official and debug both published, capture stamp kept
    6  debug frame and overlay keep the legacy fields (bbox, joints, position,
       depth, ids) and can be switched off without touching the official one
    7  depth_keypoint_score_threshold from the node YAML / an override
"""

import math
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("cv_bridge")
pytest.importorskip("patrolknight_msgs.msg")
yaml = pytest.importorskip("yaml")

from builtin_interfaces.msg import Time  # noqa: E402
from cv_bridge import CvBridge  # noqa: E402
from patrolknight_msgs.msg import (  # noqa: E402
    PersonSkeleton,
    PersonSkeleton3D,
    SkeletonFrame,
    SkeletonFrame3D,
)
from rclpy.node import Node  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from sensor_msgs.msg import CameraInfo  # noqa: E402

from skeleton_detection import iot_node  # noqa: E402
from skeleton_detection.inference.depth_estimation import DepthParams  # noqa: E402
from skeleton_detection.inference.rtmo_inference import PersonDetection  # noqa: E402
from skeleton_detection.iot_node import RTMONode  # noqa: E402
from skeleton_detection.output.visualization import draw_skeleton_overlay  # noqa: E402
from skeleton_detection.utils.coco_keypoints import COCO_CONNECTIONS_FLAT  # noqa: E402

STAMP = Time(sec=1_700_000_321, nanosec=987_654_321)
WIDTH, HEIGHT = 848, 480
FX, FY, CX, CY = 600.0, 605.0, 423.0, 241.0
NODE_YAML = os.path.join(REPO_ROOT, "config", "skeleton_detection_node.yaml")


# ----------------------------------------------------------------------
# harness
# ----------------------------------------------------------------------
class RecordingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class CountingInference:
    def __init__(self, detections):
        self.detections = detections
        self.calls = 0

    def infer(self, frame_bgr):
        self.calls += 1
        return [d for d in self.detections]


class CountingTracker:
    active_tracks = 0

    def __init__(self):
        self.calls = 0

    def update(self, detections, frame_bgr, frame_index, timestamp):
        self.calls += 1
        return detections


class StubStats:
    def record_frame(self, **kwargs):
        pass


class FailingLogger:
    def _fail(self, message):
        raise AssertionError(f"RGBD path logged: {message}")

    error = fatal = warning = _fail

    def info(self, message):
        pass

    debug = info


class FakeRGBD:
    def __init__(self, rgb, depth, info):
        self.header = rgb.header
        self.rgb = rgb
        self.depth = depth
        self.rgb_camera_info = info
        self.depth_camera_info = info


class HarnessNode:
    _on_rgbd = RTMONode._on_rgbd
    _configure_from_rgbd = RTMONode._configure_from_rgbd
    _process_frame = RTMONode._process_frame
    _attach_joint_depths = RTMONode._attach_joint_depths

    def __init__(self, detections, debug=True, threshold=0.30):
        self.bridge = CvBridge()
        self.camera_frame_id = "camera_color_optical_frame"
        self._rgbd_configured = False
        self.camera_intrinsics = None
        self.depth_params = DepthParams(keypoint_score_threshold=threshold)
        self.inference = CountingInference(detections)
        self.tracker = CountingTracker()
        self.frame_publisher = RecordingPublisher()
        self.debug_frame_publisher = RecordingPublisher() if debug else None
        self.overlay_persons = []
        self.stats = StubStats()
        self.log_every_frame = False
        self.frame_index = 0

    def _handle_visualization(self, frame_bgr, persons, header):
        self.overlay_persons.append(persons)

    def get_logger(self):
        return FailingLogger()


def camera_info():
    info = CameraInfo()
    info.width, info.height = WIDTH, HEIGHT
    info.k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
    return info


def make_rgbd(depth_array, encoding):
    bridge = CvBridge()
    rgb = bridge.cv2_to_imgmsg(np.zeros((HEIGHT, WIDTH, 3), np.uint8), encoding="bgr8")
    rgb.header.stamp = Time(sec=STAMP.sec, nanosec=STAMP.nanosec)
    rgb.header.frame_id = "camera_color_optical_frame"
    depth = bridge.cv2_to_imgmsg(depth_array, encoding=encoding)
    depth.header = rgb.header
    return FakeRGBD(rgb, depth, camera_info())


def zed_depth(z=2.0):
    return np.full((HEIGHT, WIDTH), z, dtype=np.float32)


def detection(track_id, scores=None, u0=300.0, v0=150.0):
    keypoints = np.array([[u0 + 10 * i, v0 + 12 * i] for i in range(17)], np.float32)
    if scores is None:
        scores = np.linspace(0.1, 0.95, 17).astype(np.float32)
    return PersonDetection(
        bbox_xyxy=np.array([280.0, 130.0, 480.0, 400.0], np.float32),
        score=0.88,
        keypoints_xy=keypoints,
        keypoint_scores=np.asarray(scores, np.float32),
        detection_index=0,
        track_id=track_id,
    )


def run(detections, depth_array=None, encoding="32FC1", **kwargs):
    node = HarnessNode(detections, **kwargs)
    node._on_rgbd(make_rgbd(zed_depth() if depth_array is None else depth_array, encoding))
    return node


# ----------------------------------------------------------------------
# 1 - message definitions
# ----------------------------------------------------------------------
def test_person_skeleton_3d_fields_and_order():
    fields = PersonSkeleton3D.get_fields_and_field_types()
    assert list(fields.items()) == [
        ("person_id", "int32"),
        ("score", "float"),
        ("joints_xyzc", "float[68]"),
        ("connections", "sequence<int32>"),
    ]


def test_skeleton_frame_3d_fields_and_order():
    fields = SkeletonFrame3D.get_fields_and_field_types()
    assert list(fields.items()) == [
        ("header", "std_msgs/Header"),
        ("frame_index", "int32"),
        ("persons", "sequence<patrolknight_msgs/PersonSkeleton3D>"),
    ]
    legacy = SkeletonFrame.get_fields_and_field_types()
    assert legacy["header"] == fields["header"]
    assert legacy["frame_index"] == fields["frame_index"]


def test_legacy_messages_unchanged():
    assert list(PersonSkeleton.get_fields_and_field_types()) == [
        "person_id", "score", "bbox", "joints", "connections", "position", "depth",
    ]
    assert list(SkeletonFrame.get_fields_and_field_types()) == [
        "header", "frame_index", "persons",
    ]


def test_joints_xyzc_rejects_wrong_length():
    message = PersonSkeleton3D()
    assert len(message.joints_xyzc) == 68
    with pytest.raises(Exception):
        message.joints_xyzc = [0.0] * 67


# ----------------------------------------------------------------------
# 2 - one pass feeds every output
# ----------------------------------------------------------------------
def test_one_inference_and_tracking_pass_per_frame():
    node = run([detection(4)])
    assert node.inference.calls == 1
    assert node.tracker.calls == 1
    assert len(node.frame_publisher.messages) == 1
    assert len(node.debug_frame_publisher.messages) == 1
    assert len(node.overlay_persons) == 1


# ----------------------------------------------------------------------
# 3 - official joints
# ----------------------------------------------------------------------
def test_official_person_joints_from_zed_meters():
    det = detection(4)
    node = run([det], depth_array=zed_depth(2.0))
    (frame,) = node.frame_publisher.messages
    assert isinstance(frame, SkeletonFrame3D)
    (person,) = frame.persons
    assert person.person_id == 4
    assert person.score == pytest.approx(0.88)
    assert list(person.connections) == list(COCO_CONNECTIONS_FLAT)
    packed = np.asarray(person.joints_xyzc, dtype=np.float32).reshape(17, 4)
    assert packed.shape == (17, 4)
    # confidence is the original float32 score in every slot
    assert np.array_equal(packed[:, 3], det.keypoint_scores)
    for i in range(17):
        u, v = det.keypoints_xy[i]
        if det.keypoint_scores[i] >= np.float32(0.30):
            assert tuple(packed[i, :3]) == pytest.approx(
                ((u - CX) * 2.0 / FX, (v - CY) * 2.0 / FY, 2.0), abs=1e-5
            )
        else:
            assert np.all(np.isnan(packed[i, :3]))


def test_official_person_joints_from_realsense_millimetres():
    det = detection(1, scores=np.full(17, 0.9, np.float32))
    node = run([det], depth_array=np.full((HEIGHT, WIDTH), 1500, np.uint16), encoding="16UC1")
    packed = np.asarray(node.frame_publisher.messages[0].persons[0].joints_xyzc).reshape(17, 4)
    assert np.allclose(packed[:, 2], 1.5)


# ----------------------------------------------------------------------
# 4 - person ids
# ----------------------------------------------------------------------
def test_negative_ids_omitted_from_official_only():
    node = run([detection(-1), detection(None), detection(7)])
    official = node.frame_publisher.messages[0]
    debug = node.debug_frame_publisher.messages[0]
    assert [p.person_id for p in official.persons] == [7]
    assert [p.person_id for p in debug.persons] == [-1, -1, 7]


def test_tracked_person_without_any_xyz_is_kept():
    node = run([detection(2)], depth_array=zed_depth(np.nan))
    (person,) = node.frame_publisher.messages[0].persons
    packed = np.asarray(person.joints_xyzc).reshape(17, 4)
    assert person.person_id == 2
    assert np.all(np.isnan(packed[:, :3]))
    assert np.all(np.isfinite(packed[:, 3]))


# ----------------------------------------------------------------------
# 5 - empty frames and stamps
# ----------------------------------------------------------------------
@pytest.mark.parametrize("detections", [[], [detection(None)]])
def test_empty_official_frame_keeps_capture_stamp(detections):
    node = run(detections)
    official = node.frame_publisher.messages[0]
    debug = node.debug_frame_publisher.messages[0]
    assert len(official.persons) == 0
    for frame in (official, debug):
        assert (frame.header.stamp.sec, frame.header.stamp.nanosec) == (STAMP.sec, STAMP.nanosec)
        assert frame.frame_index == 0


# ----------------------------------------------------------------------
# 6 - legacy debug and overlay
# ----------------------------------------------------------------------
def test_debug_frame_keeps_legacy_fields():
    det = detection(5)
    node = run([det], depth_array=zed_depth(3.0))
    (person,) = node.debug_frame_publisher.messages[0].persons
    assert isinstance(person, PersonSkeleton)
    assert list(person.bbox) == pytest.approx([280.0, 130.0, 200.0, 270.0])
    assert len(person.joints) == 51
    assert list(person.joints[:3]) == pytest.approx(
        [det.keypoints_xy[0, 0], det.keypoints_xy[0, 1], det.keypoint_scores[0]]
    )
    # bbox centre (380, 265) at the person Z
    assert (person.position.x, person.position.y, person.position.z) == pytest.approx(
        ((380.0 - CX) * 3.0 / FX, (265.0 - CY) * 3.0 / FY, 3.0), abs=1e-5
    )
    assert person.depth == pytest.approx(
        math.sqrt(person.position.x ** 2 + person.position.y ** 2 + 9.0)
    )


def test_overlay_receives_legacy_persons_and_renders():
    node = run([detection(5)])
    (persons,) = node.overlay_persons
    assert isinstance(persons[0], PersonSkeleton)
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    overlay = draw_skeleton_overlay(
        image, persons, joint_score_threshold=0.3, draw_joint_scores=False,
        title="t", legend=None, draw_position_xyz=True,
    )
    assert overlay.shape == image.shape and overlay.any()


def test_debug_disabled_keeps_official_and_overlay():
    node = run([detection(5)], debug=False)
    assert len(node.frame_publisher.messages) == 1
    assert len(node.overlay_persons) == 1


# ----------------------------------------------------------------------
# 7 - threshold parameter
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


def read_node(overrides):
    node = Node("rtmo_node_v21_test", parameter_overrides=overrides)
    node.get_logger().set_level(50)
    RTMONode._declare_parameters(node)
    RTMONode._read_parameters(node)
    return node


def test_threshold_default_and_debug_defaults(ros_context):
    node = read_node([])
    try:
        assert node.depth_params.keypoint_score_threshold == pytest.approx(0.30)
        assert node.publish_debug_frame is True
        assert node.debug_output_topic == "/skeleton_detection/debug_frame"
        assert node.output_topic == "/skeleton_detection/frame"
    finally:
        node.destroy_node()


def test_threshold_in_node_yaml_and_override(ros_context):
    with open(NODE_YAML) as handle:
        params = yaml.safe_load(handle)["rtmo_node"]["ros__parameters"]
    assert params["depth_keypoint_score_threshold"] == pytest.approx(0.30)

    node = read_node([Parameter("depth_keypoint_score_threshold", value=0.55)])
    try:
        assert node.depth_params.keypoint_score_threshold == pytest.approx(0.55)
    finally:
        node.destroy_node()

    det = detection(3, scores=np.full(17, 0.5, np.float32))
    packed = np.asarray(
        run([det], threshold=0.55).frame_publisher.messages[0].persons[0].joints_xyzc
    ).reshape(17, 4)
    assert np.all(np.isnan(packed[:, :3]))
    packed = np.asarray(
        run([det], threshold=0.5).frame_publisher.messages[0].persons[0].joints_xyzc
    ).reshape(17, 4)
    assert np.all(np.isfinite(packed[:, :3]))


def test_iot_node_imports_both_types():
    assert iot_node.SkeletonFrame3D is SkeletonFrame3D
    assert iot_node.SkeletonFrame is SkeletonFrame
