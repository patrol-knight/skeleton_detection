from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np


BBox = List[float]


@dataclass
class Track:
    track_id: int
    bbox: BBox
    hits: int = 1
    missed_frames: int = 0


def _normalize_bbox(bbox: Sequence[float]) -> BBox:
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with 4 values, got {bbox!r}")

    x, y, width, height = [float(value) for value in bbox]
    return [x, y, max(0.0, width), max(0.0, height)]


def iou(bbox_a: Sequence[float], bbox_b: Sequence[float]) -> float:
    ax, ay, aw, ah = _normalize_bbox(bbox_a)
    bx, by, bw, bh = _normalize_bbox(bbox_b)

    if aw <= 0.0 or ah <= 0.0 or bw <= 0.0 or bh <= 0.0:
        return 0.0

    a_right = ax + aw
    a_bottom = ay + ah
    b_right = bx + bw
    b_bottom = by + bh

    inter_left = max(ax, bx)
    inter_top = max(ay, by)
    inter_right = min(a_right, b_right)
    inter_bottom = min(a_bottom, b_bottom)

    inter_width = max(0.0, inter_right - inter_left)
    inter_height = max(0.0, inter_bottom - inter_top)
    inter_area = inter_width * inter_height

    if inter_area <= 0.0:
        return 0.0

    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def extract_bbox(annotation) -> Optional[BBox]:
    bbox_value = None

    bbox_attr = getattr(annotation, "bbox", None)
    if callable(bbox_attr):
        bbox_value = bbox_attr()
    elif bbox_attr is not None:
        bbox_value = bbox_attr

    if bbox_value is not None:
        try:
            return _normalize_bbox(bbox_value)
        except (TypeError, ValueError):
            pass

    keypoints = getattr(annotation, "data", None)
    if keypoints is None:
        return None

    keypoints = np.asarray(keypoints, dtype=float)
    if keypoints.ndim != 2 or keypoints.shape[1] < 3:
        return None

    visible = keypoints[:, 2] > 0.0
    if not np.any(visible):
        return None

    visible_points = keypoints[visible, :2]
    min_xy = visible_points.min(axis=0)
    max_xy = visible_points.max(axis=0)
    width_height = np.maximum(max_xy - min_xy, 0.0)

    return [
        float(min_xy[0]),
        float(min_xy[1]),
        float(width_height[0]),
        float(width_height[1]),
    ]


class IOUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed_frames: int = 10):
        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = int(max_missed_frames)
        self._next_track_id = 0
        self._tracks: List[Track] = []

    def update(self, detections: Iterable[Optional[Sequence[float]]]) -> List[int]:
        detection_bboxes = [
            None if bbox is None else _normalize_bbox(bbox)
            for bbox in detections
        ]
        assigned_track_ids = [-1] * len(detection_bboxes)

        candidate_matches = []
        for track_index, track in enumerate(self._tracks):
            for detection_index, bbox in enumerate(detection_bboxes):
                if bbox is None:
                    continue

                overlap = iou(track.bbox, bbox)
                if overlap >= self.iou_threshold:
                    candidate_matches.append((overlap, track_index, detection_index))

        candidate_matches.sort(reverse=True)

        matched_track_indices = set()
        matched_detection_indices = set()

        for _overlap, track_index, detection_index in candidate_matches:
            if track_index in matched_track_indices:
                continue
            if detection_index in matched_detection_indices:
                continue

            track = self._tracks[track_index]
            track.bbox = detection_bboxes[detection_index]
            track.hits += 1
            track.missed_frames = 0

            assigned_track_ids[detection_index] = track.track_id
            matched_track_indices.add(track_index)
            matched_detection_indices.add(detection_index)

        active_tracks: List[Track] = []
        for track_index, track in enumerate(self._tracks):
            if track_index not in matched_track_indices:
                track.missed_frames += 1

            if track.missed_frames <= self.max_missed_frames:
                active_tracks.append(track)

        self._tracks = active_tracks

        for detection_index, bbox in enumerate(detection_bboxes):
            if detection_index in matched_detection_indices or bbox is None:
                continue

            track = Track(track_id=self._next_track_id, bbox=bbox)
            self._next_track_id += 1
            self._tracks.append(track)
            assigned_track_ids[detection_index] = track.track_id

        return assigned_track_ids
