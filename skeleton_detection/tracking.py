"""In-process BoT-SORT + OSNet ReID tracking for RTMO detections.

This module deliberately contains no ROS code. It receives the detections
RTMO produced and the SAME in-memory BGR frame RTMO ran on, and hands back
persistent track ids. Nothing is serialized, published or copied between
processes: the frame is passed by reference from the capture thread through
inference into the tracker.

Why BoxMOT gets the frame
-------------------------
With ``with_reid=True`` BoT-SORT crops each detection out of the frame itself
and runs the ReID backbone on those crops (``model.get_features(boxes, img)``).
That is why ``update()`` takes the image: it is an appearance-model input, not
an image transport.

Detection contract (verified against BoxMOT 19.0.0
``boxmot/trackers/detection_layout.py``)::

    in : (N, 6) float32  [x1, y1, x2, y2, conf, cls]
    out: (M, 8) float32  [x1, y1, x2, y2, track_id, conf, cls, det_ind]

``det_ind`` indexes back into the array we passed in, so a track maps to the
exact detection -- and therefore the exact 17 keypoints -- that produced it. No
second IoU matching step is needed or performed.

Frame-rate note
---------------
The camera runs at 60 FPS but the capture buffer is latest-frame-wins, so the
tracker sees only the ~40-55 FPS the pipeline actually processes, with
occasional gaps where stale frames were dropped. ``frame_rate`` is therefore
configured from the measured pipeline rate, not from the camera rate; it scales
BoT-SORT's internal track buffer (how long a lost track survives), so passing
60 would make tracks live ~10% longer than intended in wall-clock terms.

Concretely, BoxMOT 19.0.0 derives the lost-track lifetime as::

    max_time_lost = int(frame_rate / 30.0 * track_buffer)

so ``frame_rate=55`` with ``track_buffer=90`` gives ``max_time_lost=165``
frames, i.e. about 3.0 s of wall clock at the processed rate.
"""

import os
import os.path as osp
import time
from typing import List, Optional, Sequence

import numpy as np

from .rtmo_inference import PersonDetection

# BoxMOT's person class id. Every RTMO detection is a person.
PERSON_CLASS_ID = 0

# Columns of the (N, 6) detection array BoxMOT expects.
DETECTION_COLUMNS = 6

DEFAULT_REID_CHECKPOINT = os.environ.get(
    "REID_CHECKPOINT", "/opt/models/reid/osnet_x0_25_msmt17.pt"
)


class TrackerInitError(RuntimeError):
    """Raised when the tracker or its ReID model cannot be constructed."""


class SkeletonTracker:
    """Owns one BoT-SORT instance (and its ReID model) for the node's lifetime.

    Both the ReID backbone and the tracker are built once in ``__init__``;
    nothing is constructed per frame.
    """

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
        # 90 (not BoxMOT's default 30) for the ID-switch investigation: it
        # makes buffer expiry an unlikely explanation for an ID switch after a
        # short occlusion. max_time_lost = int(frame_rate / 30.0 * 90).
        track_buffer: int = 90,
        match_thresh: float = 0.8,
        appearance_thresh: float = 0.25,
        proximity_thresh: float = 0.5,
        logger=None,
        # TEMPORARY TRACKING DEBUG -- see tracking_debug.py; delete with it.
        debug_enabled: bool = False,
        debug_path: str = "",
    ) -> None:
        self.reid_checkpoint = reid_checkpoint
        self.device = device
        self.with_reid = bool(with_reid)
        self.frame_rate = int(frame_rate)
        self.cmc_method = cmc_method
        self.half = bool(half)
        self._logger = logger
        self.reid_load_seconds = 0.0
        self.track_buffer = int(track_buffer)

        reid_model = self._build_reid() if self.with_reid else None

        # TEMPORARY TRACKING DEBUG ---------------------------------------
        # When off, a stock BotSort is constructed and nothing below runs, so
        # the per-frame cost is exactly zero.
        self.debug_writer = None
        if debug_enabled:
            from .tracking_debug import TrackingDebugWriter

            self.debug_writer = TrackingDebugWriter(debug_path, logger=logger)
        # ---------------------------------------------------------------

        try:
            from boxmot.trackers.botsort.botsort import BotSort
        except ImportError as exc:
            raise TrackerInitError(
                "boxmot is not installed. Install the pinned version: "
                "pip install -c /etc/pip-constraints.txt boxmot==19.0.0"
            ) from exc

        botsort_kwargs = dict(
            reid_model=reid_model,
            with_reid=self.with_reid,
            # BoxMOT accepts None to disable camera-motion compensation but
            # rejects the string "none"; the node maps its parameter for us.
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
            if self.debug_writer is not None:
                # TEMPORARY TRACKING DEBUG: a recording subclass, same math.
                from .tracking_debug import build_instrumented_botsort

                self.tracker = build_instrumented_botsort(
                    self.debug_writer, **botsort_kwargs
                )
            else:
                self.tracker = BotSort(**botsort_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise TrackerInitError(f"Failed to construct BoT-SORT: {exc}") from exc

        self._log(
            "info",
            f"BoT-SORT ready (with_reid={self.with_reid}, "
            f"cmc_method={self.cmc_method}, frame_rate={self.frame_rate}, "
            f"device={self.device}, track_buffer={self.track_buffer}, "
            f"max_time_lost={self.max_time_lost} frames "
            f"~{self.max_time_lost / max(self.frame_rate, 1):.2f}s)",
        )
        if self.debug_writer is not None:
            # TEMPORARY TRACKING DEBUG
            self._log(
                "warning",
                "TEMPORARY tracking debug is ENABLED; new-track diagnostics "
                f"are being written to {self.debug_writer.path} (truncated at "
                "startup). Turn it off with tracking_debug_enabled:=false.",
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

    # ------------------------------------------------------------------
    def _log(self, level: str, message: str) -> None:
        if self._logger is None:
            print(f"[{level}] {message}", flush=True)
            return
        try:
            getattr(self._logger, level)(message)
        except Exception:  # noqa: BLE001
            print(f"[{level}] {message}", flush=True)

    def _build_reid(self):
        """Load the OSNet ReID model once, from the baked image path."""
        if not osp.isfile(self.reid_checkpoint):
            raise TrackerInitError(
                f"ReID checkpoint not found: {self.reid_checkpoint}\n"
                "It is baked into the Docker image at "
                "/opt/models/reid/osnet_x0_25_msmt17.pt by docker/fetch_models.py. "
                "Rebuild the image or point 'reid_checkpoint' at a real file. "
                "Refusing to fall back to a runtime download."
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

        self._log(
            "info",
            f"ReID loaded in {self.reid_load_seconds:.2f}s from "
            f"{self.reid_checkpoint} (backend={type(reid.model).__name__}, "
            f"device={self.device}, half={self.half})",
        )
        return reid.model

    # ------------------------------------------------------------------
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

        Detections the tracker does not return a track for keep
        ``track_id=None``; the message builder then falls back to the
        frame-local index for that person.
        """
        dets = self.to_boxmot_detections(detections)
        if self.debug_writer is not None:
            # TEMPORARY TRACKING DEBUG: only so the report can quote the node's
            # own frame counter and wall-clock time. Read, never acted upon.
            self.tracker.debug_frame_index = frame_index
            self.tracker.debug_timestamp = timestamp
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
