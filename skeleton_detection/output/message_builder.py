"""Convert internal detections into the published ROS messages.

Single owner of the internal -> ROS conversion, so the wire conventions live
in exactly one place.

Official V2.1 output, ``SkeletonFrame3D`` (:func:`build_skeleton_frame_3d`):

* ``header`` / ``frame_index``: as below, shared with the debug frame.
* ``persons``: only detections with a tracked ``person_id >= 0``; an
  untracked (``-1``) detection is omitted. Zero persons is a valid frame.
* ``joints_xyzc``: 17 COCO joints * ``[X, Y, Z, conf]`` = 68 floats, meters
  in the colour optical frame (x right, y down, z forward). XYZ is NaN when
  unavailable; ``conf`` is always the original RTMO joint score.
* ``connections``: the same flattened COCO topology as ``PersonSkeleton``.

Legacy debug output, ``SkeletonFrame`` (:func:`build_skeleton_frame`):

* ``header``: the input frame's header, passed through unchanged.
  ``header.stamp`` is the only timestamp on the message; it is never replaced
  by the current time.
* ``bbox``: internal **xyxy** becomes published ``[x, y, width, height]``
  (top-left + size), in original source image pixels, unclipped.
* ``joints``: 17 COCO keypoints flattened to 51 floats,
  ``[x0, y0, conf0, x1, y1, conf1, ...]``.
* ``connections``: the fixed 19-edge COCO topology flattened to 38 ints.
* ``person_id``: the persistent BoT-SORT track id (``>= 0``) when tracking is
  enabled and the tracker returned a track for this detection; otherwise
  ``-1`` (no persistent identity). The person is still published with its
  skeleton, bbox and position. See :func:`person_id_semantics`.
* ``position``: the person's 3D point in METERS in the colour camera OPTICAL
  frame (``header.frame_id``, default ``camera_color_optical_frame``): x toward
  image right, y toward image down, z forward along the optical axis. Computed
  by :func:`skeleton_detection.inference.depth_estimation.compute_person_position`
  and
  attached to the detection by the node. All NaN when unavailable.
* ``depth``: the EUCLIDEAN camera-to-person distance in METERS,
  ``sqrt(x^2 + y^2 + z^2)`` of ``position``. This is NOT the RealSense depth
  value -- that is ``position.z``. ``NaN`` when unavailable -- never ``0``.
"""

from typing import List, Optional, Sequence

from geometry_msgs.msg import Point
from std_msgs.msg import Header

import numpy as np

from patrolknight_msgs.msg import (
    PersonSkeleton,
    PersonSkeleton3D,
    SkeletonFrame,
    SkeletonFrame3D,
)

from ..utils.coco_keypoints import COCO_CONNECTIONS_FLAT, NUM_COCO_KEYPOINTS
from ..inference.rtmo_inference import PersonDetection


def person_id_semantics(tracking_enabled: bool) -> str:
    """One-line description of what ``person_id`` means right now."""
    if tracking_enabled:
        return (
            "person_id = persistent BoT-SORT track id (stable across frames), "
            "-1 while a detection has no confirmed track"
        )
    return "person_id = -1 for every detection (no tracker identity)"


def build_person_skeleton(detection: PersonDetection) -> PersonSkeleton:
    """Convert one detection into a PersonSkeleton message."""
    x_min, y_min, x_max, y_max = (float(value) for value in detection.bbox_xyxy)

    joints: List[float] = []
    for index in range(NUM_COCO_KEYPOINTS):
        joints.append(float(detection.keypoints_xy[index, 0]))
        joints.append(float(detection.keypoints_xy[index, 1]))
        joints.append(float(detection.keypoint_scores[index]))

    message = PersonSkeleton()
    message.person_id = int(detection.person_id)
    message.score = float(detection.score)
    message.bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
    message.joints = joints
    message.connections = list(COCO_CONNECTIONS_FLAT)
    # Colour optical frame, meters; estimated upstream (iot_node ->
    # depth_estimation). NaN on every axis = unavailable.
    position = detection.position
    message.position = Point(
        x=float(position.x), y=float(position.y), z=float(position.z)
    )
    # Euclidean distance sqrt(x^2+y^2+z^2), NOT the RealSense Z-depth.
    message.depth = float(position.distance)
    return message


def build_skeleton_frame(
    header: Header,
    frame_index: int,
    detections: Sequence[PersonDetection],
) -> SkeletonFrame:
    """Assemble the frame message published on /skeleton_detection/frame."""
    frame = SkeletonFrame()
    frame.header = header
    frame.frame_index = int(frame_index)
    frame.persons = [build_person_skeleton(detection) for detection in detections]
    return frame


JOINTS_XYZC_LENGTH = NUM_COCO_KEYPOINTS * 4


def build_person_skeleton_3d(
    detection: PersonDetection, joints_xyzc: Sequence[float]
) -> PersonSkeleton3D:
    """One tracked detection + its packed joints -> PersonSkeleton3D."""
    packed = np.asarray(joints_xyzc, dtype=np.float32).ravel()
    if packed.size != JOINTS_XYZC_LENGTH:
        raise ValueError(
            f"joints_xyzc must hold {JOINTS_XYZC_LENGTH} floats, got {packed.size}"
        )
    message = PersonSkeleton3D()
    message.person_id = int(detection.person_id)
    message.score = float(detection.score)
    message.joints_xyzc = packed
    message.connections = list(COCO_CONNECTIONS_FLAT)
    return message


def build_skeleton_frame_3d(
    header: Header,
    frame_index: int,
    detections: Sequence[PersonDetection],
    joints_xyzc: Sequence[Optional[Sequence[float]]],
) -> SkeletonFrame3D:
    """Official frame: tracked persons only, one packed array per detection.

    ``joints_xyzc[i]`` belongs to ``detections[i]``.
    """
    if len(joints_xyzc) != len(detections):
        raise ValueError("one joints_xyzc entry per detection is required")
    frame = SkeletonFrame3D()
    frame.header = header
    frame.frame_index = int(frame_index)
    frame.persons = [
        build_person_skeleton_3d(detection, packed)
        for detection, packed in zip(detections, joints_xyzc)
        if int(detection.person_id) >= 0
    ]
    return frame
