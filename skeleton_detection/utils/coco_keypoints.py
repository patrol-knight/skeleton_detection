"""Canonical COCO 17-keypoint topology for this package.
"""

from typing import List

NUM_COCO_KEYPOINTS = 17

COCO_KEYPOINT_NAMES: List[str] = [
    "nose",             # 0
    "left_eye",         # 1
    "right_eye",        # 2
    "left_ear",         # 3
    "right_ear",        # 4
    "left_shoulder",    # 5
    "right_shoulder",   # 6
    "left_elbow",       # 7
    "right_elbow",      # 8
    "left_wrist",       # 9
    "right_wrist",      # 10
    "left_hip",         # 11
    "right_hip",        # 12
    "left_knee",        # 13
    "right_knee",       # 14
    "left_ankle",       # 15
    "right_ankle",      # 16
]

COCO_CONNECTIONS: List[List[int]] = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6],
    [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
    [3, 5], [4, 6],
]

# Flattened form published in PersonSkeleton.connections (int32[]):
# [a0, b0, a1, b1, ...] -> 19 edges * 2 = 38 values.
COCO_CONNECTIONS_FLAT: List[int] = [
    index for connection in COCO_CONNECTIONS for index in connection
]
