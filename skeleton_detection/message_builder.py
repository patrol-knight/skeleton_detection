"""Convert internal detections into the published ROS messages.

Single owner of the internal -> ROS conversion, so the wire conventions live
in exactly one place:

* ``bbox``: internal **xyxy** becomes published ``[x, y, width, height]``
  (top-left + size), in original source image pixels, unclipped.
* ``joints``: 17 COCO keypoints flattened to 51 floats,
  ``[x0, y0, conf0, x1, y1, conf1, ...]``.
* ``connections``: the fixed 19-edge COCO topology flattened to 38 ints.
* ``person_id``: the persistent BoT-SORT track id when tracking is enabled and
  the tracker returned a track for this detection; otherwise the frame-local
  detection index. See :func:`person_id_semantics`.
* ``position``: still ``(0, 0, 0)`` -- depth/3D is a later milestone.
"""

from typing import List, Sequence

from geometry_msgs.msg import Point
from std_msgs.msg import Header

from skeleton_detection.msg import PersonSkeleton, SkeletonFrame

from .coco_keypoints import COCO_CONNECTIONS_FLAT, NUM_COCO_KEYPOINTS
from .rtmo_inference import PersonDetection


def person_id_semantics(tracking_enabled: bool) -> str:
    """One-line description of what ``person_id`` means right now."""
    if tracking_enabled:
        return "person_id = persistent BoT-SORT track id (stable across frames)"
    return "person_id = per-frame detection index (NOT stable across frames)"


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
    # No depth in this milestone: position is always zero and must be treated
    # as invalid by consumers.
    message.position = Point(x=0.0, y=0.0, z=0.0)
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
    frame.timestamp = header.stamp.sec + header.stamp.nanosec / 1e9
    frame.persons = [build_person_skeleton(detection) for detection in detections]
    return frame
