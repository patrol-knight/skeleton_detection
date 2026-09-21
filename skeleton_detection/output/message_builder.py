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

from typing import List, Sequence

from geometry_msgs.msg import Point
from std_msgs.msg import Header

from skeleton_detection.msg import PersonSkeleton, SkeletonFrame

from ..utils.coco_keypoints import COCO_CONNECTIONS_FLAT, NUM_COCO_KEYPOINTS
from ..inference.rtmo_inference import PersonDetection


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
    frame.timestamp = header.stamp.sec + header.stamp.nanosec / 1e9
    frame.persons = [build_person_skeleton(detection) for detection in detections]
    return frame
