"""BoT-SORT + OSNet ReID tracking for RTMO detections (no ROS code).

BoxMOT gets the frame because ReID crops each detection from it. ``det_ind``
in its output indexes the detections we passed, so each track maps back to
the exact skeleton that produced it.
"""

import os.path as osp
import time
from typing import List, Optional, Sequence

import numpy as np

from .occlusion_tracking import OcclusionParams, compute_detection_visibility
from .rtmo_inference import PersonDetection
from ..utils.model_paths import DEFAULT_REID_CHECKPOINT

PERSON_CLASS_ID = 0
# [x1, y1, x2, y2, conf, cls]
DETECTION_COLUMNS = 6


class TrackerInitError(RuntimeError):
    """Raised when the tracker or its ReID model cannot be constructed."""


# Max cosine distance: a track without a ReID feature yet is matched on IoU alone.
NO_APPEARANCE_COST = 2.0


def install_featureless_guard(botsort_module) -> None:
    """Make ``embedding_distance`` tolerate tracks born on a ReID-skipped frame.

    BoxMOT 19.0.0 doesn't None-check ``smooth_feat`` and would crash. Idempotent.
    """
    current = getattr(botsort_module, "embedding_distance", None)
    if current is None:
        raise TrackerInitError(
            "boxmot.trackers.botsort.botsort no longer exposes "
            "embedding_distance; reid_interval cannot guard featureless tracks"
        )
    stock = getattr(current, "featureless_guard_stock", current)

    def embedding_distance(tracks, detections, metric="cosine"):
        if all(t.smooth_feat is not None for t in tracks):
            return stock(tracks, detections, metric)
        cost = np.full((len(tracks), len(detections)), NO_APPEARANCE_COST)
        rows = [i for i, t in enumerate(tracks) if t.smooth_feat is not None]
        if rows and len(detections):
            cost[rows] = stock([tracks[i] for i in rows], detections, metric)
        return cost

    embedding_distance.featureless_guard_stock = stock
    botsort_module.embedding_distance = embedding_distance


class SkeletonTracker:
    """One BoT-SORT instance and its ReID model, built once."""

    def __init__(
        self,
        reid_checkpoint: str = DEFAULT_REID_CHECKPOINT,
        device: str = "cuda:0",
        with_reid: bool = True,
        frame_rate: int = 55,
        cmc_method: Optional[str] = None,
        half: bool = False,
        track_high_thresh: float = 0.5,
        new_track_thresh: float = 0.6,
        track_buffer: int = 90,
        match_thresh: float = 0.8,
        appearance_thresh: float = 0.25,
        proximity_thresh: float = 0.7,
        logger=None,
        occlusion_aware_tracking: bool = False,
        occlusion_params: Optional[OcclusionParams] = None,
        # ReID on every Nth frame; frames between match on IoU/Kalman only
        reid_interval: int = 2,
    ) -> None:
        if int(reid_interval) < 1:
            raise TrackerInitError(f"reid_interval must be >= 1, got {reid_interval}")
        self.reid_interval = int(reid_interval)
        self.reid_checkpoint = reid_checkpoint
        self.device = device
        self.with_reid = bool(with_reid)
        self.frame_rate = int(frame_rate)
        self.cmc_method = cmc_method
        self.half = bool(half)
        self._logger = logger
        self.reid_load_seconds = 0.0
        self.track_buffer = int(track_buffer)

        self.occlusion_aware_tracking = bool(occlusion_aware_tracking)
        self.occlusion_params = occlusion_params or OcclusionParams()
        self.occlusion_logger = None
        self._visibility_fallbacks = 0
        self._visibility_fallback_logged = 0

        reid_model = self._build_reid() if self.with_reid else None

        try:
            import boxmot.trackers.botsort.botsort as botsort_module
            from boxmot.trackers.botsort.botsort import BotSort
        except ImportError as exc:
            raise TrackerInitError(
                "boxmot is not installed. Install the pinned version: "
                "pip install -c /etc/pip-constraints.txt boxmot==19.0.0"
            ) from exc
        if self.with_reid:
            install_featureless_guard(botsort_module)

        botsort_kwargs = dict(
            reid_model=reid_model,
            with_reid=self.with_reid,
            cmc_method=self.cmc_method,
            frame_rate=self.frame_rate,
            track_high_thresh=track_high_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            appearance_thresh=appearance_thresh,
            proximity_thresh=proximity_thresh,
        )

        try:
            base_cls = BotSort
            if self.occlusion_aware_tracking:
                from .occlusion_tracking import make_occlusion_aware_botsort

                base_cls = make_occlusion_aware_botsort(
                    BotSort, self.occlusion_params, self.occlusion_logger
                )

            self.tracker = base_cls(**botsort_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise TrackerInitError(f"Failed to construct BoT-SORT: {exc}") from exc

        if self.occlusion_aware_tracking:
            params = self.occlusion_params
            self._log(
                "debug",
                "Occlusion-aware tracking ENABLED: an OCCLUDED matched "
                "detection skips the Kalman measurement update and freezes the "
                "ReID feature (prediction and track lifetime continue). "
                f"keypoint_visibility_threshold="
                f"{params.keypoint_visibility_threshold}, "
                f"visible_ratio_threshold={params.visible_ratio_threshold}. "
                "visible_ratio is the ONLY classification signal; bbox width "
                "is recorded for the debug log but classifies nothing "
                f"(normal_bbox_history_size={params.normal_bbox_history_size}, "
                f"min_normal_width_samples={params.min_normal_width_samples}).",
            )

        features = [
            f"ReID every {self.reid_interval} frames" if self.with_reid else "ReID off"
        ]
        if self.occlusion_aware_tracking:
            features.append("occlusion-aware (experimental)")
        self._log("info", f"BoT-SORT ready ({', '.join(features)})")
        self._log(
            "debug",
            f"BoT-SORT with_reid={self.with_reid}, "
            f"reid_interval={self.reid_interval}, "
            f"cmc_method={self.cmc_method}, frame_rate={self.frame_rate}, "
            f"device={self.device}, track_buffer={self.track_buffer}, "
            f"max_time_lost={self.max_time_lost} frames "
            f"~{self.max_time_lost / max(self.frame_rate, 1):.2f}s",
        )
        self._log(
            "debug",
            "Association gates: "
            f"proximity_thresh={self.tracker.proximity_thresh:.2f} "
            f"(ReID eligible for IoU >= "
            f"{1.0 - self.tracker.proximity_thresh:.2f}; the mask is "
            "iou_dist > proximity_thresh and iou_dist = 1 - IoU), "
            f"appearance_thresh={self.tracker.appearance_thresh:.2f}, "
            f"match_thresh={self.tracker.match_thresh:.2f}, "
            f"track_high_thresh={self.tracker.track_high_thresh:.2f}, "
            f"new_track_thresh={self.tracker.new_track_thresh:.2f}",
        )
        self.last_track_count = 0

    @property
    def max_time_lost(self) -> int:
        """BoxMOT's internal lost-track lifetime, in frames.

        BoxMOT 19.0.0 derives it once in ``BotSort.__init__`` as
        ``int(frame_rate / 30.0 * track_buffer)``; this reads the value the
        tracker actually ended up with rather than recomputing it.
        """
        return int(self.tracker.max_time_lost)

    def _log(self, level: str, message: str) -> None:
        if self._logger is None:
            print(f"[{level}] {message}", flush=True)
            return
        try:
            getattr(self._logger, level)(message)
        except Exception:  # noqa: BLE001
            print(f"[{level}] {message}", flush=True)

    def _build_reid(self):
        """Load the OSNet ReID model."""
        if not osp.isfile(self.reid_checkpoint):
            raise TrackerInitError(
                f"ReID checkpoint not found: {self.reid_checkpoint}. "
                "Run docker/fetch_models.py or point 'reid_checkpoint' at a real file."
            )

        try:
            from boxmot.reid import ReID
        except ImportError as exc:
            raise TrackerInitError(
                "boxmot is not installed; cannot build the ReID model."
            ) from exc

        from pathlib import Path

        start = time.time()
        try:
            reid = ReID(
                path=Path(self.reid_checkpoint), device=self.device, half=self.half
            )
        except Exception as exc:  # noqa: BLE001
            raise TrackerInitError(
                f"Failed to load ReID model from {self.reid_checkpoint}: {exc}"
            ) from exc
        self.reid_load_seconds = time.time() - start

        self._log("info", f"ReID loaded in {self.reid_load_seconds:.2f}s")
        self._log(
            "debug",
            f"ReID {self.reid_checkpoint} (backend={type(reid.model).__name__}, "
            f"device={self.device}, half={self.half})",
        )
        return reid.model

    @staticmethod
    def to_boxmot_detections(detections: Sequence[PersonDetection]) -> np.ndarray:
        """Build the ``(N, 6)`` ``[x1, y1, x2, y2, conf, cls]`` array.

        Boxes are taken straight from ``PersonDetection.bbox_xyxy`` -- they are
        never round-tripped through the ROS ``[x, y, w, h]`` form.
        """
        if not detections:
            # Shape matters: BoxMOT infers the detection layout from the column
            # count, and an empty update must still age existing tracks.
            return np.empty((0, DETECTION_COLUMNS), dtype=np.float32)

        boxes = np.stack([d.bbox_xyxy for d in detections]).astype(np.float32)
        scores = np.asarray([d.score for d in detections], dtype=np.float32)
        classes = np.full((len(detections), 1), PERSON_CLASS_ID, dtype=np.float32)
        return np.concatenate([boxes, scores.reshape(-1, 1), classes], axis=1)

    def _frame_visibility(self, detections: Sequence[PersonDetection]):
        """One :class:`DetectionVisibility` per detection, same order."""
        visibility = [
            compute_detection_visibility(
                detection.keypoint_scores,
                self.occlusion_params.keypoint_visibility_threshold,
            )
            for detection in detections
        ]
        unusable = sum(1 for v in visibility if not v.available)
        partial = sum(1 for v in visibility if v.available and v.note)
        if unusable or partial:
            self._visibility_fallbacks += unusable + partial
            # Throttled: one line the first time, then on each power-of-ten.
            if self._visibility_fallbacks >= self._visibility_fallback_logged * 10:
                self._visibility_fallback_logged = max(
                    1, self._visibility_fallbacks
                )
                notes = sorted({v.note for v in visibility if v.note})
                self._log(
                    "warning",
                    "Occlusion detection fell back on "
                    f"{self._visibility_fallbacks} detection(s) so far: "
                    f"{', '.join(notes)}. The visible-ratio criterion is "
                    "SKIPPED for those; missing keypoint scores are never "
                    "treated as zero confidence.",
                )
        return visibility

    def update(
        self,
        detections: List[PersonDetection],
        frame_bgr: np.ndarray,
        frame_index: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> List[PersonDetection]:
        """Assign persistent ``track_id`` to ``detections`` in place.

        Always calls the tracker, including on frames with zero detections, so
        BoT-SORT can age and retire its tracks correctly (verified: a track
        survives empty frames and is re-associated when the person returns).

        Detections the tracker does not return a track for (e.g. a new,
        not yet confirmed track, or an unmatched detection) keep
        ``track_id=None`` and are published with ``person_id=-1``.
        """
        dets = self.to_boxmot_detections(detections)
        if self.occlusion_aware_tracking:
            # OCCLUSION-AWARE TRACKING: the visible-joint ratio is computed
            # here, from the RTMO per-keypoint scores, and handed to the
            # tracker keyed by position in `detections` -- which is exactly
            # BoxMOT's det_ind.
            self.tracker.set_frame_visibility(
                self._frame_visibility(detections),
                frame_index=frame_index,
                timestamp=timestamp,
            )
        if self.with_reid:
            # frame_count is BoxMOT's processed-frame counter, read before
            # update() increments it, so the first frame runs ReID.
            self.tracker.with_reid = (
                self.tracker.frame_count % self.reid_interval == 0
            )
        tracks = np.asarray(self.tracker.update(dets, frame_bgr))
        self.last_track_count = int(tracks.shape[0]) if tracks.size else 0

        if tracks.size == 0:
            return detections

        for row in tracks:
            # [x1, y1, x2, y2, track_id, conf, cls, det_ind]
            track_id = int(row[4])
            det_index = int(row[7])
            if 0 <= det_index < len(detections):
                detections[det_index].track_id = track_id
            else:
                self._log(
                    "warning",
                    f"BoT-SORT returned det_ind={det_index} outside the "
                    f"{len(detections)} detections sent; dropping that track id",
                )
        return detections

    @property
    def active_tracks(self) -> int:
        return self.last_track_count
