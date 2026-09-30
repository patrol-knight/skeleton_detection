"""Tests for rtmo_node's parameters, the packaged configs and the launch default.

Runs the node's REAL ``_declare_parameters`` / ``_read_parameters`` on a plain
rclpy node, so no model, camera or GPU is involved. Needs a sourced ROS 2
environment with ``patrolknight_msgs`` (i.e. the Docker image); skipped
otherwise::

    python3 -m pytest test/test_node_config.py -v

Coverage:

    1  the three supported input modes and the rgbd_topic default
    2  invalid values fail at start-up with a readable error
    3  derived settings: tracking_frame_rate fallback, cmc_method 'none'
    4  every packaged rtmo_node config loads: each key is a declared parameter
       of the right type (rclpy silently IGNORES an undeclared key, so a typo
       would otherwise disable a setting without any error), and each file
       selects its intended input_mode
    5  the bringup launch file exposes only ``config`` and defaults to
       config/rtmo_node.yaml
"""

import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("cv_bridge")
pytest.importorskip("patrolknight_msgs.msg")
yaml = pytest.importorskip("yaml")

from rclpy.exceptions import InvalidParameterTypeException  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402

from skeleton_detection import iot_node  # noqa: E402
from skeleton_detection.input.ros_camera_subscriber import (  # noqa: E402
    DEFAULT_RGBD_TOPIC,
)

CONFIG_DIR = os.path.join(REPO_ROOT, "config")
LAUNCH_FILE = os.path.join(REPO_ROOT, "launch", "skeleton_detection_bringup.launch.py")


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def make_node():
    """Declare + read rtmo_node's parameters with the given overrides."""
    nodes = []

    def build(**overrides):
        node = Node(
            "rtmo_node",
            parameter_overrides=[Parameter(k, value=v) for k, v in overrides.items()],
        )
        nodes.append(node)
        iot_node.RTMONode._declare_parameters(node)
        iot_node.RTMONode._read_parameters(node)
        return node

    yield build
    for node in nodes:
        node.destroy_node()


def load_rtmo_config(filename):
    with open(os.path.join(CONFIG_DIR, filename)) as stream:
        return yaml.safe_load(stream)["rtmo_node"]["ros__parameters"]


# ----------------------------------------------------------------------
# 1 - input modes
# ----------------------------------------------------------------------
def test_supported_input_modes():
    assert set(iot_node.INPUT_MODES) == {"realsense", "ros_topic", "ros_camera"}


def test_rgbd_topic_default(make_node):
    node = make_node(input_mode="ros_camera")
    assert DEFAULT_RGBD_TOPIC == "/camera/camera/rgbd"
    assert node.rgbd_topic == DEFAULT_RGBD_TOPIC


# ----------------------------------------------------------------------
# 2 - validation
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"input_mode": "usb_camera"}, "input_mode must be one of"),
        ({"visualization_reliability": "sometimes"}, "visualization_reliability"),
        ({"visualization_width": 0}, "visualization_width/height must be >= 1"),
        ({"visualization_height": 0}, "visualization_width/height must be >= 1"),
        ({"normal_bbox_history_size": 0}, "normal_bbox_history_size must be >= 1"),
        ({"min_normal_width_samples": 0}, "min_normal_width_samples must be >= 1"),
    ],
)
def test_invalid_values_are_rejected_at_startup(make_node, overrides, message):
    with pytest.raises(RuntimeError, match=message):
        make_node(**overrides)


def test_input_mode_is_case_insensitive(make_node):
    assert make_node(input_mode="ROS_CAMERA").input_mode == "ros_camera"


def test_integer_for_a_double_parameter_is_rejected(make_node):
    """Why the docs insist on ``run_duration_sec: 30.0`` rather than ``30``."""
    with pytest.raises(InvalidParameterTypeException):
        make_node(run_duration_sec=30)


# ----------------------------------------------------------------------
# 3 - derived settings
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "configured,effective",
    [(0.0, iot_node.DEFAULT_TRACKING_FRAME_RATE), (-1.0, 55), (30.0, 30)],
)
def test_tracking_frame_rate_falls_back_to_the_pipeline_rate(
    make_node, configured, effective
):
    assert make_node(tracking_frame_rate=configured).tracking_frame_rate == effective


@pytest.mark.parametrize("configured,effective", [("none", None), ("", None), ("ECC", "ecc")])
def test_cmc_method_none_maps_to_python_none(make_node, configured, effective):
    # BoxMOT accepts None but raises on the string "none".
    assert make_node(cmc_method=configured).cmc_method == effective


# ----------------------------------------------------------------------
# 4 - packaged configs
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "filename,input_mode",
    [
        ("rtmo_node.yaml", "ros_camera"),
        ("rtmo_node_direct_realsense.yaml", "realsense"),
        ("offline_rtmo_node.yaml", "ros_topic"),
    ],
)
def test_packaged_config_loads_and_selects_its_input_mode(make_node, filename, input_mode):
    values = load_rtmo_config(filename)
    node = make_node(**values)  # a wrongly typed value raises here

    undeclared = sorted(name for name in values if not node.has_parameter(name))
    assert undeclared == [], f"{filename} sets undeclared parameters: {undeclared}"
    assert node.input_mode == input_mode


def test_default_config_is_the_ros_camera_deployment(make_node):
    node = make_node(**load_rtmo_config("rtmo_node.yaml"))
    assert node.input_mode == "ros_camera"
    assert node.rgbd_topic == "/camera/camera/rgbd"
    assert node.enable_tracking and node.with_reid and node.occlusion_aware_tracking
    assert node.publish_visualization_image and node.draw_person_xyz


# ----------------------------------------------------------------------
# 5 - launch default
# ----------------------------------------------------------------------
def test_launch_file_exposes_only_config_defaulting_to_rtmo_node_yaml():
    pytest.importorskip("launch")
    from ament_index_python.packages import (
        PackageNotFoundError,
        get_package_share_directory,
    )
    from launch.actions import DeclareLaunchArgument

    try:
        get_package_share_directory("skeleton_detection")
    except PackageNotFoundError:
        pytest.skip("skeleton_detection is not installed in this environment")

    spec = importlib.util.spec_from_file_location("bringup_launch", LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    description = module.generate_launch_description()

    arguments = [e for e in description.entities if isinstance(e, DeclareLaunchArgument)]
    assert [argument.name for argument in arguments] == ["config"]
    default = "".join(part.text for part in arguments[0].default_value)
    assert default.endswith(os.path.join("config", "rtmo_node.yaml"))
