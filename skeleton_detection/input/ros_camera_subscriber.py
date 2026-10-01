"""External RealSense ROS 2 driver input: one ``RGBD`` topic, no SDK.

This is the ``input_mode='ros_camera'`` path.  It does NOT open a camera and
does NOT launch ``realsense2_camera``: it assumes an **already running**
external driver (typically in another container) started roughly with::

    enable_rgbd:=true enable_sync:=true align_depth.enable:=true \\
        enable_color:=true enable_depth:=true

Why the RGBD message and not three topics
-----------------------------------------
``realsense2_camera_msgs/msg/RGBD`` carries the colour image, the
ALREADY-ALIGNED depth image and both ``CameraInfo`` messages **in one
message**::

    std_msgs/Header         header
    sensor_msgs/CameraInfo  rgb_camera_info
    sensor_msgs/CameraInfo  depth_camera_info
    sensor_msgs/Image       rgb
    sensor_msgs/Image       depth

So the driver has already done the two jobs this package would otherwise have
to do itself:

* **synchronization** — ``enable_sync:=true`` pairs colour and depth upstream,
  so there is no ``message_filters`` synchronizer here and no chance of a
  colour frame being matched with the wrong depth frame;
* **depth-to-colour alignment** — ``align_depth.enable:=true`` makes the
  driver run the equivalent of ``rs.align(rs.stream.color)``, so ``depth[v, u]``
  is already the distance at colour pixel ``(u, v)``.

``rs.align`` is therefore never called on this path, and
:mod:`skeleton_detection.inference.depth_estimation` gets exactly the input
contract it already documents.  Compare
:mod:`skeleton_detection.input.realsense_capture`, which does both jobs
in-process for ``input_mode='realsense'``.

Units
-----
The RealSense ROS driver publishes aligned depth as ``16UC1`` in
MILLIMETERS, so the depth scale on this path is ``0.001`` m/unit rather than
a value read from the device.  ``32FC1`` (already meters) is accepted too and
maps to ``1.0``; see :func:`depth_scale_for_encoding`.

Dependencies
------------
Only the message package ``realsense2_camera_msgs`` is needed — the message
definition, not the driver node.  It is imported lazily by
:func:`import_rgbd_message` so this module (and its tests) can be imported
without ROS present.
"""

from typing import Any, Tuple

import numpy as np

from ..inference.depth_estimation import CameraIntrinsics

#: Default topic of an external driver launched as the ``camera`` node inside
#: the ``camera`` namespace, which is what ``rs_launch.py`` produces.
DEFAULT_RGBD_TOPIC = "/camera/camera/rgbd"

#: Meters per raw unit for the driver's ``16UC1`` millimetre depth image.
ROS_DEPTH_SCALE_16UC1 = 0.001

#: ``32FC1`` depth is already in meters, so it must NOT be rescaled.
ROS_DEPTH_SCALE_32FC1 = 1.0

#: Queue depth of the RGBD subscription. 1, deliberately: inference runs in
#: the callback, so a deeper queue would only buffer stale frames. Dropping
#: them keeps latency bounded, matching the latest-frame-wins behaviour of the
#: direct capture path.
RGBD_QUEUE_DEPTH = 1


class RGBDInputError(RuntimeError):
    """Raised when the RGBD topic cannot be used as an input source."""


def import_rgbd_message():
    """Import ``realsense2_camera_msgs.msg.RGBD``, or explain why it failed.

    Lazy on purpose: ``input_mode='realsense'`` and ``input_mode='ros_topic'``
    must keep working in an environment where this message package is not
    installed.
    """
    try:
        from realsense2_camera_msgs.msg import RGBD
    except ImportError as exc:
        raise RGBDInputError(
            "realsense2_camera_msgs is not available in this environment. It "
            "provides the RGBD message required by input_mode='ros_camera'. "
            "Install it with `apt install ros-humble-realsense2-camera-msgs` "
            "(it is already pulled in by ros-humble-realsense2-camera) and "
            "re-source the ROS setup files. Only the MESSAGE package is "
            "needed; this node never launches the driver."
        ) from exc
    return RGBD


def rgbd_qos():
    """QoS matching the external driver's RGBD publisher.

    ``realsense2_camera`` publishes its image/RGBD topics with sensor-data QoS
    (BEST_EFFORT, KEEP_LAST, VOLATILE).  A RELIABLE subscription would simply
    **never match** such a publisher and the callback would stay silent, so
    the reliability is pinned to BEST_EFFORT here rather than left at rclpy's
    RELIABLE default.  BEST_EFFORT also matches a RELIABLE publisher, so this
    profile works whichever way the driver is configured.

    Depth is :data:`RGBD_QUEUE_DEPTH` (1) so a slow inference step drops stale
    frames instead of accumulating a backlog.
    """
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=RGBD_QUEUE_DEPTH,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


def intrinsics_from_camera_info(camera_info: Any) -> CameraIntrinsics:
    """``sensor_msgs/CameraInfo`` -> :class:`CameraIntrinsics`.

    Reads the 3x3 row-major intrinsic matrix ``k``::

        k = [fx,  0, cx,
              0, fy, cy,
              0,  0,  1]

    ``k`` is used rather than ``p``: ``p`` is the projection matrix of a
    *rectified* frame, while the colour stream here is unrectified and its
    keypoints are raw colour pixels.  Distortion (``d``) is ignored, which is
    the same pure-pinhole assumption the direct capture path already makes.

    Args:
        camera_info: anything exposing ``k`` (9 numbers), ``width`` and
            ``height`` — the ``rgb_camera_info`` field of an RGBD message.

    Raises:
        RGBDInputError: if ``k`` is the wrong length, or the resulting
            intrinsics are degenerate (non-finite, or zero focal length/size),
            which is what an uncalibrated ``CameraInfo`` of all zeros looks
            like.
    """
    matrix = np.asarray(getattr(camera_info, "k", []), dtype=np.float64).ravel()
    if matrix.size != 9:
        raise RGBDInputError(
            f"CameraInfo.k must hold 9 numbers, got {matrix.size}. The RGBD "
            "message did not carry a usable colour CameraInfo."
        )

    intrinsics = CameraIntrinsics(
        fx=float(matrix[0]),
        fy=float(matrix[4]),
        cx=float(matrix[2]),
        cy=float(matrix[5]),
        width=int(getattr(camera_info, "width", 0)),
        height=int(getattr(camera_info, "height", 0)),
    )
    if not intrinsics.is_valid():
        raise RGBDInputError(
            f"CameraInfo carries degenerate intrinsics (fx={intrinsics.fx}, "
            f"fy={intrinsics.fy}, cx={intrinsics.cx}, cy={intrinsics.cy}, "
            f"{intrinsics.width}x{intrinsics.height}). The external driver "
            "published an uncalibrated colour CameraInfo; no 3D position can "
            "be computed from it."
        )
    return intrinsics


def depth_scale_for_encoding(encoding: str) -> float:
    """Meters per raw unit for a depth image encoding.

    ``16UC1`` is the RealSense ROS driver's aligned-depth encoding and is in
    millimeters, hence :data:`ROS_DEPTH_SCALE_16UC1`. ``32FC1`` is already in
    meters and must not be rescaled.

    Raises:
        RGBDInputError: for any other encoding, rather than guessing a scale
            and silently publishing wrong distances.
    """
    normalized = str(encoding).strip().lower()
    if normalized == "16uc1":
        return ROS_DEPTH_SCALE_16UC1
    if normalized == "32fc1":
        return ROS_DEPTH_SCALE_32FC1
    raise RGBDInputError(
        f"Unsupported depth encoding '{encoding}'. Expected '16UC1' "
        "(millimeters, what realsense2_camera publishes for aligned depth) "
        "or '32FC1' (meters). Refusing to guess a depth scale."
    )


def rgbd_parts(msg: Any) -> Tuple[Any, Any, Any]:
    """Pull ``(rgb, depth, rgb_camera_info)`` out of an RGBD message.

    Isolated so a field-name change upstream fails here with a readable error
    listing what the message actually carries, instead of an ``AttributeError``
    from somewhere inside the callback.
    """
    missing = [
        name for name in ("rgb", "depth", "rgb_camera_info") if not hasattr(msg, name)
    ]
    if missing:
        available = [name for name in dir(msg) if not name.startswith("_")]
        raise RGBDInputError(
            f"RGBD message is missing {missing}. Expected "
            "realsense2_camera_msgs/msg/RGBD with fields rgb, depth, "
            f"rgb_camera_info. Message exposes: {available}"
        )
    return msg.rgb, msg.depth, msg.rgb_camera_info


def validate_depth_alignment(
    depth_width: int, depth_height: int, intrinsics: CameraIntrinsics
) -> None:
    """Check once that the depth image sits on the COLOUR pixel grid.

    ``compute_person_position`` already refuses a depth image whose shape does
    not match the colour intrinsics -- but it does so per person, per frame.
    Checking here means a driver started WITHOUT ``align_depth.enable:=true``
    fails once, at startup, with a message naming the actual cause, instead of
    flooding the log at the camera's frame rate.

    Raises:
        RGBDInputError: when the depth image is not the size of the colour
            image the intrinsics describe.
    """
    if (int(depth_width), int(depth_height)) == (intrinsics.width, intrinsics.height):
        return
    raise RGBDInputError(
        f"Depth image is {depth_width}x{depth_height} but the colour stream "
        f"is {intrinsics.width}x{intrinsics.height}. The RGBD message is not "
        "carrying depth aligned to colour, so RTMO's colour-space keypoints "
        "would index the wrong pixels. Restart the external driver with "
        "align_depth.enable:=true (and enable_rgbd:=true enable_sync:=true)."
    )


def depth_array_from_msg(depth_msg: Any, bridge: Any) -> np.ndarray:
    """``sensor_msgs/Image`` -> the ``(H, W)`` depth array the pipeline wants.

    Converted with ``passthrough`` so the RAW units survive: the metric
    conversion is the depth module's job, driven by
    :func:`depth_scale_for_encoding`, exactly as the direct capture path hands
    over raw Z16 plus a device scale.

    Args:
        depth_msg: the ``depth`` field of an RGBD message, already aligned to
            colour by the external driver.
        bridge: a ``cv_bridge.CvBridge``; injected so this module does not
            depend on cv_bridge at import time.
    """
    array = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
    array = np.asarray(array)
    if array.ndim != 2:
        raise RGBDInputError(
            f"Expected a single-channel (H, W) depth image, got shape "
            f"{array.shape} from encoding '{depth_msg.encoding}'."
        )
    return array
