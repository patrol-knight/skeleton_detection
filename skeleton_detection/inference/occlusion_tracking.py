"""Occlusion-aware state protection for BoT-SORT (project-local, experimental).

The failure mode this addresses
-------------------------------
A stationary person is slowly occluded by someone walking past. RTMO keeps
emitting a person box, but it shrinks onto whatever is still visible -- a leg,
half a torso -- so its centre and width drift sideways. BoT-SORT feeds those
partial boxes to the Kalman filter as ordinary measurements, which teaches the
motion model a velocity the person never had, and crops the ReID embedding from
a half-body (or from the occluder), which poisons ``smooth_feat``. By the time
the person is fully hidden the tracker is predicting away from them, and when
they reappear the IoU is near zero and the appearance distance is large, so a
new id is allocated.

What this module does
---------------------
Classify each matched observation as exactly one of two states::

    NORMAL      trust it: stock BoT-SORT behaviour, unchanged
    OCCLUDED    keep the track alive, but do not let the observation move the
                Kalman motion state or the ReID appearance state

There are deliberately only two states, two signals and one OR. This is an
experiment, not a model.

The occlusion signal (see :func:`classify_occlusion`)
-----------------------------------------------------
**Skeleton visible ratio, and nothing else** -- the fraction of RTMO's 17
per-keypoint confidences at or above ``keypoint_visibility_threshold``. This is
the raw ``PersonDetection.keypoint_scores`` array, NOT the aggregate person
score.

Why bbox width is no longer a signal
------------------------------------
An earlier version also marked a detection OCCLUDED when its width fell below
``0.60`` of that track's recent median. That produced false positives whenever
somebody simply turned sideways: 16/17 keypoints visible, nothing occluding
them, but the box narrows to roughly half its frontal width. A person's own
pose changes their box width as much as an occluder does, so width cannot
separate the two. The width history is still maintained and still printed in
the debug log, but it is **informational only** and has zero effect on
classification or on any tracker state.

How it hooks into BoxMOT 19.0.0
-------------------------------
Nothing under ``/usr/local/lib/python3.10/dist-packages/boxmot`` is modified.
BoxMOT builds every ``STrack`` through the module-level name
``boxmot.trackers.botsort.botsort.STrack`` (in ``_create_detections`` and in
``_second_association``), and a track *is* the detection ``STrack`` that
``activate()`` promoted -- so replacing that one module global for the duration
of one ``_update_impl`` call is enough to make every track and every detection
an :class:`OcclusionAwareSTrack`. The association code (IoU distance, proximity
gate, ReID distance, appearance gate, Hungarian assignment, track buffer, the
four stages) is untouched; only ``STrack.update`` / ``STrack.re_activate`` /
``STrack.activate`` are overridden.

Turning it off
--------------
``occlusion_aware_tracking:=false`` never builds any of this: ``tracking.py``
constructs a stock ``BotSort`` and the per-frame cost is exactly zero.

Removing it
-----------
1. delete this file
2. delete the ``OCCLUSION-AWARE TRACKING`` blocks in ``tracking.py``,
   ``iot_node.py``, ``launch/skeleton_detection_bringup.launch.py`` and
   ``config/rtmo_node_realsense.yaml``
3. delete ``test/test_occlusion_tracking.py``
"""

import datetime as _dt
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

NORMAL = "NORMAL"
OCCLUDED = "OCCLUDED"

# Why a detection was classified OCCLUDED. There is exactly one reason: the
# rule has exactly one signal. (BBOX_WIDTH_RATIO_FAIL was removed deliberately
# -- see "Why bbox width is no longer a signal" above.)
REASON_VISIBLE_RATIO = "VISIBLE_RATIO_FAIL"

# Why the visible-ratio signal could not be evaluated.
UNAVAILABLE_NONE = "NO_KEYPOINT_SCORES"
UNAVAILABLE_EMPTY = "EMPTY_KEYPOINT_SCORES"
UNAVAILABLE_NON_FINITE = "NON_FINITE_KEYPOINT_SCORES"
PARTIAL_SCORES = "PARTIAL_KEYPOINT_SCORES"


@dataclass(frozen=True)
class OcclusionParams:
    """Everything tunable, mirrored one-for-one by ROS parameters."""

    keypoint_visibility_threshold: float = 0.30
    visible_ratio_threshold: float = 0.50
    # The two below size the INFORMATIONAL bbox-width history only. There is no
    # bbox_width_ratio_threshold any more: width does not classify anything.
    normal_bbox_history_size: int = 15
    # Below this many NORMAL widths the median is not worth printing, so the
    # debug line reports the ratio as unavailable rather than guessing one.
    min_normal_width_samples: int = 5


@dataclass(frozen=True)
class DetectionVisibility:
    """Result of reading one detection's RTMO per-keypoint confidences.

    ``available`` is False when the scores were missing or unusable. In that
    case ``visible_ratio`` is None and the visible-ratio criterion is skipped
    entirely -- missing keypoints are never silently treated as zero
    confidence.
    """

    available: bool
    visible_count: int = 0
    total_count: int = 0
    visible_ratio: Optional[float] = None
    note: Optional[str] = None


def compute_detection_visibility(
    keypoint_scores, keypoint_visibility_threshold: float
) -> DetectionVisibility:
    """visible_ratio = (# keypoints with score >= threshold) / (# keypoints).

    Only finite scores are counted, in the numerator and in the denominator.
    RTMO always emits 17 finite floats, so the partial path is defensive; when
    it does fire the detection carries ``note=PARTIAL_KEYPOINT_SCORES`` so the
    caller can log it instead of quietly shrinking the denominator.
    """
    if keypoint_scores is None:
        return DetectionVisibility(available=False, note=UNAVAILABLE_NONE)

    scores = np.asarray(keypoint_scores, dtype=np.float64).ravel()
    if scores.size == 0:
        return DetectionVisibility(available=False, note=UNAVAILABLE_EMPTY)

    finite = np.isfinite(scores)
    total = int(finite.sum())
    if total == 0:
        return DetectionVisibility(available=False, note=UNAVAILABLE_NON_FINITE)

    visible = int((scores[finite] >= float(keypoint_visibility_threshold)).sum())
    return DetectionVisibility(
        available=True,
        visible_count=visible,
        total_count=total,
        visible_ratio=visible / total,
        note=None if total == scores.size else PARTIAL_SCORES,
    )


@dataclass(frozen=True)
class OcclusionAssessment:
    """The binary verdict, plus diagnostics.

    ``state``, ``visibility`` and ``reasons`` are the decision. Everything from
    ``current_width`` down is INFORMATIONAL ONLY: it is carried so the debug
    log can show what the box was doing, and it is read by nothing that
    classifies, gates, matches or updates.
    """

    state: str
    visibility: DetectionVisibility
    reasons: List[str] = field(default_factory=list)

    # ---- informational only, never used for classification ----
    current_width: float = 0.0
    reference_width: Optional[float] = None
    width_ratio: Optional[float] = None
    width_ratio_available: bool = False
    normal_samples: int = 0

    @property
    def is_occluded(self) -> bool:
        return self.state == OCCLUDED


def classify_occlusion(
    visibility: DetectionVisibility,
    params: OcclusionParams,
    current_width: float = 0.0,
    reference_width: Optional[float] = None,
    normal_samples: int = 0,
) -> OcclusionAssessment:
    """The whole occlusion rule. One signal, one comparison::

        is_occluded = visibility.available
                      and visible_ratio < visible_ratio_threshold

    A signal that cannot be evaluated never votes for OCCLUDED, so a detection
    with unusable keypoint scores stays NORMAL.

    ``current_width`` / ``reference_width`` / ``normal_samples`` are accepted
    only to be echoed into :class:`OcclusionAssessment` for the debug log. They
    do NOT influence the returned ``state`` -- deliberately, because a person
    turning sideways halves their box width while remaining fully visible.
    """
    is_occluded = (
        visibility.available
        and visibility.visible_ratio is not None
        and visibility.visible_ratio < params.visible_ratio_threshold
    )

    # Informational only.
    width = float(current_width)
    width_available = reference_width is not None and float(reference_width) > 0.0
    width_ratio = width / float(reference_width) if width_available else None

    return OcclusionAssessment(
        state=OCCLUDED if is_occluded else NORMAL,
        visibility=visibility,
        reasons=[REASON_VISIBLE_RATIO] if is_occluded else [],
        current_width=width,
        reference_width=float(reference_width) if width_available else None,
        width_ratio=width_ratio,
        width_ratio_available=width_available,
        normal_samples=int(normal_samples),
    )


# ----------------------------------------------------------------------
# per-frame context shared between the tracker subclass and the STrack subclass
# ----------------------------------------------------------------------
class _FrameContext:
    """Carries this frame's per-detection visibility into ``STrack.__init__``.

    ``STrack`` objects are constructed deep inside BoxMOT with no reference to
    the tracker, so the visibility computed from the RTMO keypoints is parked
    here (keyed by ``det_ind``, which is the index into the ``PersonDetection``
    list the node handed in) and read back during construction.
    """

    def __init__(self) -> None:
        self.visibility: Sequence[DetectionVisibility] = ()
        self.frame_count: int = 0
        self.frame_index = None
        self.timestamp = None

    def begin(self, visibility, frame_count, frame_index, timestamp) -> None:
        self.visibility = visibility or ()
        self.frame_count = int(frame_count)
        self.frame_index = frame_index
        self.timestamp = timestamp

    def end(self) -> None:
        self.visibility = ()

    def for_det_ind(self, det_ind: int) -> DetectionVisibility:
        if 0 <= det_ind < len(self.visibility):
            return self.visibility[det_ind]
        # Low-confidence second-association detections are real detections and
        # do have keypoints; an out-of-range index means the node did not
        # supply visibility at all, which must not be read as "occluded".
        return DetectionVisibility(available=False, note=UNAVAILABLE_NONE)


# ----------------------------------------------------------------------
# the debug log
# ----------------------------------------------------------------------
SEPARATOR = "-" * 60


class OcclusionDebugLogger:
    """Renders occlusion events into any object exposing ``write_block(text)``.

    No writer is currently supplied: the temporary new-track instrumentation
    that used to provide one has been removed, so ``SkeletonTracker`` passes
    ``None`` and no occlusion log is written. The dependency is one duck-typed
    method, so any object exposing ``write_block`` can be wired back in without
    touching the classification logic.
    """

    def __init__(self, writer) -> None:
        self.writer = writer
        self.events = 0

    def transition(self, track_id: int, previous: str, current: str) -> None:
        self.writer.write_block(f"Track {track_id}: {previous} -> {current}")
        self.events += 1

    def created(self, track_id: int, state: str) -> None:
        self.writer.write_block(f"Track {track_id}: created {state}")
        self.events += 1

    def observation(
        self,
        track,
        assessment: OcclusionAssessment,
        context: _FrameContext,
        params: OcclusionParams,
        entry_point: str,
    ) -> None:
        self.writer.write_block(
            _render_observation(track, assessment, context, params, entry_point)
        )
        self.events += 1


def _fmt(value, spec="{:.4f}") -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return spec.format(value) if np.isfinite(value) else "inf"


def _fmt_timestamp(value) -> str:
    if value is None:
        return "n/a"
    try:
        return _dt.datetime.fromtimestamp(float(value)).isoformat(
            timespec="milliseconds"
        )
    except (TypeError, ValueError, OSError):
        return str(value)


def _fmt_box(box) -> str:
    if box is None:
        return "n/a"
    b = np.asarray(box, dtype=float).ravel()
    if b.size < 4:
        return "n/a"
    return "[{:.1f}, {:.1f}, {:.1f}, {:.1f}]".format(*b[:4])


def _render_observation(track, assessment, context, params, entry_point) -> str:
    occluded = assessment.is_occluded
    visibility = assessment.visibility
    mean = getattr(track, "mean", None)
    velocity = (
        "n/a"
        if mean is None or np.asarray(mean).size < 8
        else "vx={:+.4f} vy={:+.4f} vw={:+.4f} vh={:+.4f}".format(
            *np.asarray(mean, dtype=float)[4:8]
        )
    )
    if visibility.available:
        keypoints = f"{visibility.visible_count} / {visibility.total_count}"
        ratio = _fmt(visibility.visible_ratio)
    else:
        keypoints = f"unavailable ({visibility.note})"
        ratio = "n/a  (criterion skipped)"

    if assessment.width_ratio_available:
        reference = _fmt(assessment.reference_width, "{:.1f}")
        width_ratio = _fmt(assessment.width_ratio, "{:.3f}")
    else:
        reference = (
            f"unavailable ({assessment.normal_samples} NORMAL widths, "
            f"need {params.min_normal_width_samples})"
        )
        width_ratio = "unavailable"

    lines = [
        SEPARATOR,
        f"Track ID: {int(getattr(track, 'id', -1))}",
        f"frame: {context.frame_count}"
        f"   (node frame_index {context.frame_index}, "
        f"timestamp {_fmt_timestamp(context.timestamp)})",
        f"entry point: STrack.{entry_point}()",
        "",
        f"visibility state: {assessment.state}",
        "",
        f"keypoints visible: {keypoints}",
        f"visible_ratio: {ratio}",
        f"visible_ratio_threshold: {_fmt(params.visible_ratio_threshold, '{:.2f}')}"
        f"   (keypoint_visibility_threshold "
        f"{_fmt(params.keypoint_visibility_threshold, '{:.2f}')})",
        "",
        "occlusion reasons: "
        + (", ".join(assessment.reasons) if assessment.reasons else "(none)"),
        "classification rule: visible_ratio < visible_ratio_threshold"
        "   (the ONLY signal)",
        "",
        "-- bbox diagnostics ------------------------------------------",
        "classification use: INFORMATIONAL ONLY",
        f"current bbox: {_fmt_box(getattr(track, '_occ_observed_xyxy', None))}",
        f"current bbox width: {_fmt(assessment.current_width, '{:.1f}')}",
        f"reference NORMAL bbox width: {reference}",
        f"bbox width ratio: {width_ratio}"
        "   (not used for classification; a person turning sideways narrows "
        "their box while fully visible)",
        "--------------------------------------------------------------",
        "",
        f"Kalman measurement update: {'SKIPPED' if occluded else 'APPLIED'}",
        "Kalman prediction: CONTINUED",
        f"ReID feature update: {'FROZEN' if occluded else 'APPLIED'}",
        f"NORMAL width history: {'NOT UPDATED' if occluded else 'UPDATED'}"
        f"  (size {len(getattr(track, 'normal_widths', ()))}"
        f"/{params.normal_bbox_history_size})",
        "",
        f"current predicted bbox: {_fmt_box(getattr(track, 'xyxy', None))}",
        ("velocity preserved from trusted state: " if occluded else "velocity: ")
        + velocity,
        SEPARATOR,
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# the BoxMOT overrides
# ----------------------------------------------------------------------
def make_occlusion_aware_botsort(
    base_cls,
    params: OcclusionParams,
    debug_logger: Optional[OcclusionDebugLogger] = None,
):
    """Build a ``base_cls`` subclass whose STracks protect their own state.

    ``base_cls`` is normally ``boxmot.trackers.botsort.botsort.BotSort``. The
    returned class can itself be used as the base of another subclass.
    """
    import boxmot.trackers.botsort.botsort as botsort_module
    from boxmot.trackers.botsort.basetrack import TrackState
    from boxmot.trackers.botsort.botsort_track import STrack

    context = _FrameContext()

    class OcclusionAwareSTrack(STrack):
        """STrack that refuses to learn from an OCCLUDED observation.

        Only three methods differ from BoxMOT 19.0.0. Everything else --
        ``predict``, ``multi_predict``, ``multi_gmc``, ``xyxy``, the Kalman
        filter itself -- is inherited untouched, so prediction always
        continues and velocity is never reset or zeroed by this class.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Read at construction: this object is a DETECTION right now, and
            # it is the same object that becomes a TRACK if activate() runs.
            self.det_visibility = context.for_det_ind(int(self.det_ind))
            self.normal_widths = deque(maxlen=params.normal_bbox_history_size)
            self.visibility_state = NORMAL
            self.last_assessment: Optional[OcclusionAssessment] = None
            # Raw measurement box of the last matched detection, for the log.
            self._occ_observed_xyxy = None

        # -- the reference width -------------------------------------
        def normal_reference_width(self) -> Optional[float]:
            """Median of the last N widths seen on NORMAL frames, or None.

            INFORMATIONAL ONLY. Nothing branches on this value: it exists so
            the debug log can show whether the box was narrowing, which is
            useful when reading a crossing but is NOT evidence of occlusion (a
            person turning sideways narrows their box just as much).

            Per-track by construction: the deque lives on the STrack object
            that carries the BoT-SORT id, so it can never jump between people.
            """
            if len(self.normal_widths) < params.min_normal_width_samples:
                return None
            return float(np.median(np.asarray(self.normal_widths, dtype=np.float64)))

        def _assess(self, new_track) -> OcclusionAssessment:
            # Only `visibility` decides. The width arguments are diagnostics.
            return classify_occlusion(
                visibility=getattr(
                    new_track,
                    "det_visibility",
                    DetectionVisibility(available=False, note=UNAVAILABLE_NONE),
                ),
                params=params,
                current_width=float(np.asarray(new_track.xywh, dtype=float)[2]),
                reference_width=self.normal_reference_width(),
                normal_samples=len(self.normal_widths),
            )

        # -- the OCCLUDED path ---------------------------------------
        def _occluded_bookkeeping(self, new_track, frame_id, reactivating: bool):
            """Everything ``update``/``re_activate`` do EXCEPT the two writes
            that would let a partial box teach the tracker something wrong:

              * ``self.kalman_filter.update(...)``  -- skipped, so ``mean`` and
                ``covariance`` stay exactly as ``predict()`` left them. The
                position keeps advancing on the last trusted velocity and
                ``mean[4:8]`` is not touched at all.
              * ``self.update_features(...)``       -- skipped, so ``curr_feat``,
                ``smooth_feat`` and ``features`` keep their last NORMAL values.

            The lifetime bookkeeping is kept verbatim so BoT-SORT still counts
            the track as seen this frame: ``frame_id`` (which drives
            ``end_frame`` and therefore ``max_time_lost``), ``tracklet_len``,
            ``state``, ``is_activated``, ``conf``, ``cls``, ``det_ind`` and the
            class histogram.
            """
            self.frame_id = frame_id
            if reactivating:
                self.tracklet_len = 0
            else:
                self.tracklet_len += 1
                if not self.is_obb:
                    self.history_observations.append(self.xyxy)

            self.state = TrackState.Tracked
            self.is_activated = True
            self.conf = new_track.conf
            self.cls = new_track.cls
            self.det_ind = new_track.det_ind
            self.update_cls(new_track.cls, new_track.conf)
            if self.is_obb:
                self.history_observations.append(self._state_obb_for_plot())

        # -- BoxMOT entry points -------------------------------------
        def activate(self, kalman_filter, frame_id):
            super().activate(kalman_filter, frame_id)
            assessment = classify_occlusion(
                visibility=self.det_visibility,
                params=params,
                current_width=float(np.asarray(self.xywh, dtype=float)[2]),
                reference_width=None,
                normal_samples=0,
            )
            self._occ_observed_xyxy = np.asarray(self.xyxy, dtype=float).copy()
            self._finish(assessment, "activate", seed_width=self.xywh[2])

        def update(self, new_track, frame_id):
            assessment = self._assess(new_track)
            self._occ_observed_xyxy = np.asarray(new_track.xyxy, dtype=float).copy()
            if assessment.is_occluded:
                self._occluded_bookkeeping(new_track, frame_id, reactivating=False)
            else:
                super().update(new_track, frame_id)
            self.det_visibility = getattr(
                new_track, "det_visibility", self.det_visibility
            )
            self._finish(assessment, "update", seed_width=new_track.xywh[2])

        def re_activate(self, new_track, frame_id, new_id=False):
            assessment = self._assess(new_track)
            self._occ_observed_xyxy = np.asarray(new_track.xyxy, dtype=float).copy()
            if assessment.is_occluded:
                # Same protection as ``update``. A lost track recovered on a
                # partial box keeps its last trusted motion state and
                # appearance; the detection still gets this id, and the next
                # NORMAL frame snaps the filter onto the real measurement.
                self._occluded_bookkeeping(new_track, frame_id, reactivating=True)
                if new_id:
                    self.id = self.next_id()
            else:
                super().re_activate(new_track, frame_id, new_id=new_id)
            self.det_visibility = getattr(
                new_track, "det_visibility", self.det_visibility
            )
            self._finish(assessment, "re_activate", seed_width=new_track.xywh[2])

        # -- state transition + logging ------------------------------
        def _finish(self, assessment, entry_point, seed_width):
            previous = self.visibility_state
            self.visibility_state = assessment.state
            self.last_assessment = assessment
            if not assessment.is_occluded:
                # OCCLUDED widths still never enter the history, so the
                # informational median stays a "this is what NORMAL looks
                # like" reference rather than an average of both states.
                self.normal_widths.append(float(seed_width))

            if debug_logger is None:
                return
            track_id = int(getattr(self, "id", -1))
            born = entry_point == "activate"
            changed = (not born) and previous != assessment.state
            if born:
                debug_logger.created(track_id, assessment.state)
            elif changed:
                debug_logger.transition(track_id, previous, assessment.state)
            # Every OCCLUDED frame, plus the frame that returns to NORMAL:
            # exactly the window around a transition, without flooding steady
            # NORMAL tracking.
            if assessment.is_occluded or changed:
                debug_logger.observation(
                    self, assessment, context, params, entry_point
                )

    class OcclusionAwareBotSort(base_cls):
        """BoT-SORT with the STrack class swapped for the duration of a frame.

        No association logic is overridden: IoU distance, the proximity gate,
        ReID distance, the appearance gate, the Hungarian assignment, the four
        stages and the track buffer are all stock BoxMOT.
        """

        strack_cls = OcclusionAwareSTrack
        occlusion_params = params
        occlusion_context = context

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Set by SkeletonTracker before each update().
            self.pending_visibility: Sequence[DetectionVisibility] = ()
            self.occlusion_frame_index = None
            self.occlusion_timestamp = None

        def set_frame_visibility(
            self, visibility, frame_index=None, timestamp=None
        ) -> None:
            self.pending_visibility = visibility
            self.occlusion_frame_index = frame_index
            self.occlusion_timestamp = timestamp

        def _update_impl(self, dets, img, embs=None):
            context.begin(
                self.pending_visibility,
                # super() increments frame_count as its first act.
                self.frame_count + 1,
                self.occlusion_frame_index,
                self.occlusion_timestamp,
            )
            saved = botsort_module.STrack
            botsort_module.STrack = OcclusionAwareSTrack
            try:
                return super()._update_impl(dets, img, embs)
            finally:
                botsort_module.STrack = saved
                context.end()

    return OcclusionAwareBotSort
