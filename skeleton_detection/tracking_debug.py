"""TEMPORARY TRACKING DEBUG -- BoT-SORT new-track diagnostics.

# TEMPORARY TRACKING DEBUG
This entire module exists to answer ONE question during the ID-switch
investigation::

    when BoT-SORT allocates a genuinely NEW persistent track id, why did the
    detection that caused it fail to associate with an existing tracked/lost
    track?

It is meant to be deleted in one piece once that question is answered. See
"Removing this instrumentation" at the bottom of this docstring.

Why a subclass instead of patching site-packages
------------------------------------------------
BoxMOT 19.0.0 computes the interesting values -- the raw IoU distance matrix,
the raw (pre-mask) embedding distance matrix, the fused/masked cost matrix and
the Hungarian assignment -- as locals inside
``BotSort._first_association`` / ``_second_association`` /
``_handle_unconfirmed_tracks``. They are gone by the time ``update()`` returns,
so nothing useful can be reconstructed from the wrapper level.

``InstrumentedBotSort`` therefore does two cheap, side-effect-free things:

1. It overrides the four stage methods purely to *label* the stage and to
   snapshot the match results after delegating to ``super()``. The overrides
   never compute an association themselves.
2. For the duration of one ``_update_impl`` call it swaps the four module-level
   helpers that ``boxmot/trackers/botsort/botsort.py`` imported
   (``iou_distance``, ``embedding_distance``, ``fuse_score``,
   ``linear_assignment``) for pass-through proxies that record a *copy* of the
   arguments and the return value and then hand back the real object,
   unmodified. The swap is restored in a ``finally``.

Because the proxies return exactly what the real functions return, and because
the stage overrides only read state that ``super()`` already produced, the
tracker's arithmetic and its output are bit-identical to stock BoxMOT. Nothing
under ``/usr/local/lib/python3.10/dist-packages/boxmot`` is modified.

The copies matter: BoxMOT mutates the embedding matrix in place
(``emb_dists[emb_dists > appearance_thresh] = 1.0`` and
``emb_dists[ious_dists_mask] = 1.0``), so the RAW appearance distance only
exists between ``embedding_distance`` returning and the next statement running.
That is precisely the value this module captures.

BoxMOT 19.0.0 association flow (verified by reading the installed source)::

    _update_impl
      dets split by confidence:
          dets_first  = conf >  track_high_thresh   (get ReID features)
          dets_second = track_low_thresh < conf < track_high_thresh
      strack_pool = active(Tracked) tracks + lost tracks
      stage 1  _first_association(strack_pool, dets_first)
          STrack.multi_predict(strack_pool)          <- Kalman prediction
          ious_dists      = iou_distance(pool, dets)
          ious_dists_mask = ious_dists > proximity_thresh
          emb_dists       = embedding_distance(pool, dets)   <- RAW
          emb_dists[emb_dists > appearance_thresh] = 1.0
          emb_dists[ious_dists_mask]               = 1.0     <- proximity kills ReID
          dists = minimum(ious_dists, emb_dists)
          linear_assignment(dists, thresh=match_thresh)
          matched Tracked -> update(); matched Lost -> re_activate(new_id=False)
      stage 2  _second_association(dets_second)
          ONLY tracks left over from stage 1 whose state is Tracked.
          A LOST track can never be recovered here. IoU only, thresh 0.5.
          Leftovers here are mark_lost().
      stage 3  _handle_unconfirmed_tracks(dets_first leftovers)
          unconfirmed = active tracks that are not yet is_activated.
          IoU (fused with det score) + ReID, same masking, thresh 0.7.
          Leftovers here are mark_removed().
      stage 4  _initialize_new_tracks(stage-3 leftover dets)
          conf >= new_track_thresh -> STrack.activate() -> NEW PERSISTENT ID.
      _update_track_states: lost tracks with
          frame_count - end_frame > max_time_lost  are removed for good.

``max_time_lost = int(frame_rate / 30.0 * track_buffer)``.

Removing this instrumentation
-----------------------------
1. delete this file
2. delete the ``# TEMPORARY TRACKING DEBUG`` blocks in ``tracking.py``,
   ``rtmo_node.py``, ``launch/milestone2_realsense_rtmo.launch.py`` and
   ``config/rtmo_node_realsense.yaml``
3. delete ``test/test_tracking_debug.py``
"""

import datetime as _dt
import os
from typing import Any, Dict, List, Optional

import numpy as np

SEPARATOR = "=" * 60

# boxmot.trackers.botsort.basetrack.TrackState, spelled out so this module can
# be imported (and unit tested) without boxmot present.
_STATE_NAMES = {0: "NEW", 1: "TRACKED", 2: "LOST", 3: "LONG_LOST", 4: "REMOVED"}

# Above this many rows/columns the full matrices are omitted; the per-candidate
# rows always carry the same numbers anyway.
MAX_MATRIX_DIM = 8

STAGE_FIRST = "first"
STAGE_SECOND = "second"
STAGE_UNCONFIRMED = "unconfirmed"

_STAGE_TITLES = {
    STAGE_FIRST: "stage 1 - first association (high-confidence detections)",
    STAGE_SECOND: "stage 2 - second association (low-confidence detections)",
    STAGE_UNCONFIRMED: "stage 3 - unconfirmed-track association",
}


# ----------------------------------------------------------------------
# formatting helpers
# ----------------------------------------------------------------------
def _fmt_box(box) -> str:
    if box is None:
        return "n/a"
    b = np.asarray(box, dtype=float).ravel()
    if b.size < 4:
        return "n/a"
    return "[{:8.1f}, {:8.1f}, {:8.1f}, {:8.1f}]".format(*b[:4])


def _fmt_center(box) -> str:
    if box is None:
        return "n/a"
    b = np.asarray(box, dtype=float).ravel()
    if b.size < 4:
        return "n/a"
    return "({:.1f}, {:.1f})".format((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _fmt_wh(box) -> str:
    if box is None:
        return "n/a"
    b = np.asarray(box, dtype=float).ravel()
    if b.size < 4:
        return "n/a"
    return "{:.1f} x {:.1f}".format(b[2] - b[0], b[3] - b[1])


def _fmt_num(value, spec: str = "{:.4f}") -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(value):
        return "inf"
    return spec.format(value)


def _state_name(state) -> str:
    return _STATE_NAMES.get(int(state), f"UNKNOWN({state})")


def _fmt_matrix(matrix, row_labels: List[str], col_labels: List[str]) -> List[str]:
    """Render a small cost matrix with track ids on rows, detections on cols."""
    if matrix is None:
        return ["    (not computed this frame)"]
    m = np.asarray(matrix, dtype=float)
    if m.size == 0:
        return ["    (empty)"]
    if m.shape[0] > MAX_MATRIX_DIM or m.shape[1] > MAX_MATRIX_DIM:
        return [
            f"    (omitted: {m.shape[0]}x{m.shape[1]} exceeds "
            f"{MAX_MATRIX_DIM}x{MAX_MATRIX_DIM}; see the per-candidate rows)"
        ]
    width = max(9, max(len(c) for c in col_labels) + 1) if col_labels else 9
    head = " " * 14 + "".join(f"{c:>{width}}" for c in col_labels)
    lines = [head]
    for r, label in enumerate(row_labels):
        cells = "".join(f"{m[r, c]:>{width}.4f}" for c in range(m.shape[1]))
        lines.append(f"    {label:<10}{cells}")
    return lines


# ----------------------------------------------------------------------
# per-frame capture
# ----------------------------------------------------------------------
class _StageRecord:
    """Everything captured for one association stage of one frame."""

    __slots__ = (
        "name",
        "tracks",
        "dets",
        "det_obj_ids",
        "iou_dists",
        "fused_iou",
        "emb_raw",
        "final_cost",
        "assign_thresh",
        "matches",
        "u_track",
        "u_det",
        "ran",
    )

    def __init__(self, name: str) -> None:
        self.name = name
        self.tracks: List[Dict[str, Any]] = []
        self.dets: List[Dict[str, Any]] = []
        self.det_obj_ids: List[int] = []
        self.iou_dists = None
        self.fused_iou = None
        self.emb_raw = None
        self.final_cost = None
        self.assign_thresh = None
        self.matches = None
        self.u_track = None
        self.u_det = None
        self.ran = False

    # -- lookups -------------------------------------------------------
    def det_column(self, det_obj) -> Optional[int]:
        try:
            return self.det_obj_ids.index(id(det_obj))
        except ValueError:
            return None

    def track_row(self, track_id: int) -> Optional[int]:
        for row, snap in enumerate(self.tracks):
            if snap["id"] == track_id:
                return row
        return None

    def match_for_track(self, row: int) -> Optional[int]:
        if self.matches is None:
            return None
        for itracked, idet in self.matches:
            if int(itracked) == row:
                return int(idet)
        return None

    def match_for_det(self, col: int) -> Optional[int]:
        if self.matches is None:
            return None
        for itracked, idet in self.matches:
            if int(idet) == col:
                return int(itracked)
        return None


class _FrameRecord:
    """All association diagnostics for the frame currently being processed.

    Held in memory only; written to disk exclusively when a new id is created.
    """

    def __init__(self, frame_count: int, frame_index, timestamp) -> None:
        self.frame_count = frame_count
        self.frame_index = frame_index
        self.timestamp = timestamp
        self.stage: Optional[str] = None
        self.stages: Dict[str, _StageRecord] = {
            STAGE_FIRST: _StageRecord(STAGE_FIRST),
            STAGE_SECOND: _StageRecord(STAGE_SECOND),
            STAGE_UNCONFIRMED: _StageRecord(STAGE_UNCONFIRMED),
        }

    def current(self) -> Optional[_StageRecord]:
        if self.stage is None:
            return None
        return self.stages[self.stage]


def _snapshot_track(track) -> Dict[str, Any]:
    """Freeze the parts of an STrack the report needs.

    Taken while the proxies run, i.e. after ``STrack.multi_predict`` and before
    any matched track is ``update()``d, so ``pred_xyxy`` really is the Kalman
    prediction the association saw.
    """
    mean = getattr(track, "mean", None)
    return {
        "id": int(getattr(track, "id", -1)),
        "state": int(getattr(track, "state", -1)),
        "is_activated": bool(getattr(track, "is_activated", False)),
        "frame_id": int(getattr(track, "frame_id", -1)),
        "start_frame": int(getattr(track, "start_frame", -1)),
        "tracklet_len": int(getattr(track, "tracklet_len", -1)),
        "conf": float(getattr(track, "conf", float("nan"))),
        "pred_xyxy": np.asarray(track.xyxy, dtype=float).copy(),
        "mean": None if mean is None else np.asarray(mean, dtype=float).copy(),
    }


def _snapshot_det(det) -> Dict[str, Any]:
    return {
        "det_ind": int(getattr(det, "det_ind", -1)),
        "conf": float(getattr(det, "conf", float("nan"))),
        "xyxy": np.asarray(det.xyxy, dtype=float).copy(),
    }


# ----------------------------------------------------------------------
# the debug file
# ----------------------------------------------------------------------
class TrackingDebugWriter:
    """Truncate-on-start, append-per-event plain-text debug file.

    Truncation happens in ``__init__``, which the node calls exactly once at
    startup, so every node launch begins with an empty file and runs never
    append to each other.
    """

    def __init__(self, path: str, logger=None) -> None:
        self.path = path
        self._logger = logger
        self.events = 0
        self.failed = False

        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        # "w" truncates: a fresh, empty file for every node launch.
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write(
                f"{SEPARATOR}\n"
                "TEMPORARY TRACKING DEBUG - BoT-SORT new-track diagnostics\n"
                f"opened: {_dt.datetime.now().isoformat(timespec='seconds')}\n"
                "one section per NEWLY CREATED persistent track id\n"
                f"{SEPARATOR}\n"
            )

    def write_block(self, text: str) -> None:
        if self.failed:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(text)
                if not text.endswith("\n"):
                    handle.write("\n")
            self.events += 1
        except OSError as exc:  # noqa: BLE001
            # A debug sink must never take the pipeline down.
            self.failed = True
            if self._logger is not None:
                self._logger.error(
                    f"tracking debug file {self.path} became unwritable: {exc}; "
                    "diagnostics disabled for the rest of this run"
                )


# ----------------------------------------------------------------------
# the instrumented tracker
# ----------------------------------------------------------------------
def build_instrumented_botsort(debug_writer: TrackingDebugWriter, **botsort_kwargs):
    """Construct ``InstrumentedBotSort``.

    Imported lazily so ``tracking.py`` can keep raising ``TrackerInitError``
    with its own message when boxmot is missing.
    """
    from boxmot.trackers.botsort.botsort import BotSort

    import boxmot.trackers.botsort.botsort as botsort_module
    from boxmot.utils import matching as matching_module

    class InstrumentedBotSort(BotSort):
        """BoT-SORT that records, but never changes, its own association math.

        Every override delegates to ``super()`` for the actual work. The only
        additional effect is bookkeeping into ``self._dbg_frame``.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._dbg_writer = debug_writer
            self._dbg_frame: Optional[_FrameRecord] = None
            # track id -> the raw detection that last updated it
            self._dbg_last_observed: Dict[int, Dict[str, Any]] = {}
            # track id -> frame_count at which BoxMOT retired it
            self._dbg_removed_at: Dict[int, int] = {}
            # set by SkeletonTracker before each update() so the report can
            # quote the node's own frame counter and wall-clock timestamp
            self.debug_frame_index = None
            self.debug_timestamp = None

        # -- stage labelling; no association logic lives here -----------
        def _update_impl(self, dets, img, embs=None):
            self._dbg_frame = _FrameRecord(
                frame_count=self.frame_count + 1,  # super() increments it
                frame_index=self.debug_frame_index,
                timestamp=self.debug_timestamp,
            )
            saved = (
                botsort_module.iou_distance,
                botsort_module.embedding_distance,
                botsort_module.fuse_score,
                botsort_module.linear_assignment,
            )
            botsort_module.iou_distance = self._dbg_iou_distance
            botsort_module.embedding_distance = self._dbg_embedding_distance
            botsort_module.fuse_score = self._dbg_fuse_score
            botsort_module.linear_assignment = self._dbg_linear_assignment
            try:
                return super()._update_impl(dets, img, embs)
            finally:
                (
                    botsort_module.iou_distance,
                    botsort_module.embedding_distance,
                    botsort_module.fuse_score,
                    botsort_module.linear_assignment,
                ) = saved
                self._dbg_frame = None

        # -- pass-through recording proxies ----------------------------
        # Each returns EXACTLY what boxmot's own helper returned. The copies
        # exist because boxmot masks the embedding matrix in place afterwards.
        def _dbg_iou_distance(self, atracks, btracks, *args, **kwargs):
            result = matching_module.iou_distance(atracks, btracks, *args, **kwargs)
            stage = self._dbg_frame.current() if self._dbg_frame else None
            if stage is not None:
                stage.ran = True
                stage.tracks = [_snapshot_track(t) for t in atracks]
                stage.dets = [_snapshot_det(d) for d in btracks]
                stage.det_obj_ids = [id(d) for d in btracks]
                stage.iou_dists = np.asarray(result, dtype=float).copy()
            return result

        def _dbg_embedding_distance(self, tracks, detections, *args, **kwargs):
            result = matching_module.embedding_distance(
                tracks, detections, *args, **kwargs
            )
            stage = self._dbg_frame.current() if self._dbg_frame else None
            if stage is not None:
                # RAW appearance distance, before
                #   emb_dists[emb_dists > appearance_thresh] = 1.0
                #   emb_dists[ious_dists_mask]               = 1.0
                stage.emb_raw = np.asarray(result, dtype=float).copy()
            return result

        def _dbg_fuse_score(self, cost_matrix, detections, *args, **kwargs):
            result = matching_module.fuse_score(
                cost_matrix, detections, *args, **kwargs
            )
            stage = self._dbg_frame.current() if self._dbg_frame else None
            if stage is not None:
                stage.fused_iou = np.asarray(result, dtype=float).copy()
            return result

        def _dbg_linear_assignment(self, cost_matrix, *args, **kwargs):
            result = matching_module.linear_assignment(
                cost_matrix, *args, **kwargs
            )
            thresh = kwargs.get("thresh", args[0] if args else None)
            stage = self._dbg_frame.current() if self._dbg_frame else None
            if stage is not None:
                matches, u_track, u_det = result
                stage.final_cost = np.asarray(cost_matrix, dtype=float).copy()
                stage.assign_thresh = thresh
                stage.matches = np.asarray(matches).reshape(-1, 2).copy()
                stage.u_track = np.asarray(u_track).ravel().copy()
                stage.u_det = np.asarray(u_det).ravel().copy()
            return result

        # -- stage overrides: label, delegate, then read the outcome ----
        def _first_association(
            self,
            dets,
            dets_first,
            active_tracks,
            unconfirmed,
            img,
            detections,
            activated_stracks,
            refind_stracks,
            strack_pool,
        ):
            if self._dbg_frame is not None:
                self._dbg_frame.stage = STAGE_FIRST
            result = super()._first_association(
                dets,
                dets_first,
                active_tracks,
                unconfirmed,
                img,
                detections,
                activated_stracks,
                refind_stracks,
                strack_pool,
            )
            if self._dbg_frame is not None:
                self._dbg_note_observations(strack_pool, detections, result[0])
                self._dbg_frame.stage = None
            return result

        def _second_association(
            self,
            dets_second,
            activated_stracks,
            lost_stracks,
            refind_stracks,
            u_track_first,
            strack_pool,
        ):
            if self._dbg_frame is not None:
                self._dbg_frame.stage = STAGE_SECOND
            result = super()._second_association(
                dets_second,
                activated_stracks,
                lost_stracks,
                refind_stracks,
                u_track_first,
                strack_pool,
            )
            if self._dbg_frame is not None:
                stage = self._dbg_frame.stages[STAGE_SECOND]
                self._dbg_note_observations_from_snapshots(stage, result[0])
                self._dbg_frame.stage = None
            return result

        def _handle_unconfirmed_tracks(
            self,
            u_detection,
            detections,
            activated_stracks,
            removed_stracks,
            unconfirmed,
        ):
            if self._dbg_frame is not None:
                self._dbg_frame.stage = STAGE_UNCONFIRMED
            before = len(removed_stracks)
            result = super()._handle_unconfirmed_tracks(
                u_detection, detections, activated_stracks, removed_stracks, unconfirmed
            )
            if self._dbg_frame is not None:
                stage = self._dbg_frame.stages[STAGE_UNCONFIRMED]
                self._dbg_note_observations_from_snapshots(stage, result[0])
                for track in removed_stracks[before:]:
                    self._dbg_removed_at.setdefault(
                        int(getattr(track, "id", -1)), self.frame_count
                    )
                self._dbg_frame.stage = None
            return result

        def _update_track_states(self, removed_stracks):
            before = len(removed_stracks)
            super()._update_track_states(removed_stracks)
            for track in removed_stracks[before:]:
                # Retired because frame_count - end_frame > max_time_lost.
                self._dbg_removed_at[int(getattr(track, "id", -1))] = self.frame_count

        # -- the only place a NEW persistent id can be born -------------
        def _initialize_new_tracks(self, u_detections, activated_stracks, detections):
            if self._dbg_frame is None or self._dbg_writer is None:
                return super()._initialize_new_tracks(
                    u_detections, activated_stracks, detections
                )

            # STrack has no ``id`` attribute until activate() assigns one, so
            # "gained an id" is an exact test for "a new persistent id was
            # allocated". Tracks that are merely updated or re-activated
            # (re_activate(..., new_id=False)) never pass through here at all.
            pending = []
            for index in u_detections:
                index = int(index)
                det = detections[index]
                pending.append((index, det, getattr(det, "id", None)))

            result = super()._initialize_new_tracks(
                u_detections, activated_stracks, detections
            )

            for index, det, previous_id in pending:
                new_id = getattr(det, "id", None)
                if new_id is None or new_id == previous_id:
                    continue  # below new_track_thresh -> no id was allocated
                self._dbg_last_observed[int(new_id)] = {
                    "frame": self.frame_count,
                    "xyxy": np.asarray(det.xyxy, dtype=float).copy(),
                    "conf": float(det.conf),
                    "det_ind": int(det.det_ind),
                }
                try:
                    self._dbg_writer.write_block(
                        self._dbg_render_new_track(int(new_id), det, index)
                    )
                except Exception as exc:  # noqa: BLE001
                    # Diagnostics must never change tracker behaviour, and that
                    # includes never raising out of it.
                    self._dbg_writer.write_block(
                        f"{SEPARATOR}\nNEW TRACK CREATED: ID {new_id}\n"
                        f"(report generation failed: {type(exc).__name__}: {exc})\n"
                        f"{SEPARATOR}\n"
                    )
            return result

        # -- bookkeeping helpers ---------------------------------------
        def _dbg_note_observations(self, tracks, detections, matches):
            """Remember the raw detection box that last updated each track."""
            if matches is None:
                return
            for itracked, idet in np.asarray(matches).reshape(-1, 2):
                track = tracks[int(itracked)]
                det = detections[int(idet)]
                self._dbg_last_observed[int(getattr(track, "id", -1))] = {
                    "frame": self.frame_count,
                    "xyxy": np.asarray(det.xyxy, dtype=float).copy(),
                    "conf": float(det.conf),
                    "det_ind": int(getattr(det, "det_ind", -1)),
                }

        def _dbg_note_observations_from_snapshots(self, stage, matches):
            """Same, using the snapshots taken by the iou_distance proxy.

            Stages 2 and 3 build their track/detection lists internally, so the
            proxy snapshot is the only handle on them.
            """
            if matches is None or not stage.tracks:
                return
            for itracked, idet in np.asarray(matches).reshape(-1, 2):
                track_snap = stage.tracks[int(itracked)]
                det_snap = stage.dets[int(idet)]
                self._dbg_last_observed[track_snap["id"]] = {
                    "frame": self.frame_count,
                    "xyxy": det_snap["xyxy"].copy(),
                    "conf": det_snap["conf"],
                    "det_ind": det_snap["det_ind"],
                }

        def _dbg_prune(self) -> None:
            """Keep the two bookkeeping dicts bounded on long runs."""
            for store in (self._dbg_last_observed, self._dbg_removed_at):
                excess = len(store) - 2000
                if excess > 0:
                    for key in sorted(store)[:excess]:  # ids increase over time
                        store.pop(key, None)

        # -- report rendering ------------------------------------------
        def _dbg_render_new_track(self, new_id, det, unconfirmed_index):
            self._dbg_prune()
            frame = self._dbg_frame
            stage1 = frame.stages[STAGE_FIRST]
            stage2 = frame.stages[STAGE_SECOND]
            stage3 = frame.stages[STAGE_UNCONFIRMED]

            det_box = np.asarray(det.xyxy, dtype=float).copy()
            det_conf = float(det.conf)
            det_ind = int(det.det_ind)
            col1 = stage1.det_column(det)
            col3 = stage3.det_column(det)

            out: List[str] = [SEPARATOR, f"NEW TRACK CREATED: ID {new_id}", ""]

            out += [
                "FRAME INFORMATION",
                f"  node frame_index           : {frame.frame_index}",
                f"  boxmot frame_count         : {self.frame_count}",
                f"  timestamp                  : {_fmt_timestamp(frame.timestamp)}",
                "",
                "CURRENT DETECTION",
                f"  detection_index (det_ind)  : {det_ind}",
                f"  stage-1 matrix column      : {col1}",
                f"  bbox_xyxy                  : {_fmt_box(det_box)}",
                f"  bbox center                : {_fmt_center(det_box)}",
                f"  bbox width/height          : {_fmt_wh(det_box)}",
                f"  detection confidence       : {_fmt_num(det_conf)}",
                "",
                "TRACKER CONFIG",
                f"  frame_rate                 : {self._dbg_frame_rate}",
                f"  track_buffer               : {self._dbg_track_buffer}",
                f"  internal max_time_lost     : {self.max_time_lost}"
                "   (= int(frame_rate / 30.0 * track_buffer))",
                f"  track_high_thresh          : {_fmt_num(self.track_high_thresh)}",
                f"  track_low_thresh           : {_fmt_num(self.track_low_thresh)}",
                f"  new_track_thresh           : {_fmt_num(self.new_track_thresh)}",
                f"  match_thresh               : {_fmt_num(self.match_thresh)}",
                f"  proximity_thresh           : {_fmt_num(self.proximity_thresh)}",
                f"  appearance_thresh          : {_fmt_num(self.appearance_thresh)}",
                f"  with_reid                  : {self.with_reid}",
                f"  fuse_first_associate       : {self.fuse_first_associate}",
                f"  cmc                        : "
                f"{type(self.cmc).__name__ if self.cmc is not None else 'disabled'}",
                "",
            ]

            # ---------------- stage-1 candidates ----------------------
            out.append(
                f"CANDIDATE EXISTING TRACKS - stage-1 pool "
                f"(active + lost): {len(stage1.tracks)}"
            )
            if not stage1.tracks:
                out.append(
                    "  (empty: BoT-SORT had no tracked or lost track to compare "
                    "this detection against, so a new id was unavoidable)"
                )
            for row, snap in enumerate(stage1.tracks):
                out += self._dbg_render_candidate(
                    snap, stage1, row, col1, det_box, det
                )
            out.append("")

            # ---------------- stage-3 candidates ----------------------
            out.append(
                f"CANDIDATE EXISTING TRACKS - stage-3 unconfirmed: "
                f"{len(stage3.tracks)}"
            )
            if not stage3.tracks:
                out.append("  (none)")
            for row, snap in enumerate(stage3.tracks):
                out += self._dbg_render_candidate(
                    snap, stage3, row, col3, det_box, det, fused=True
                )
            out.append("")

            # ---------------- already-removed tracks ------------------
            out += self._dbg_render_removed(det_box, det)
            out.append("")

            # ---------------- assignment ------------------------------
            out += self._dbg_render_assignment(
                new_id, det_ind, col1, col3, stage1, stage2, stage3
            )
            out.append("")

            # ---------------- stage trace -----------------------------
            out += self._dbg_render_stage_trace(
                new_id, det_conf, col1, col3, stage1, stage2, stage3
            )

            # trailing rule + blank lines so consecutive events stay legible
            out += ["", SEPARATOR, "", ""]
            return "\n".join(out)

        def _dbg_render_candidate(
            self, snap, stage, row, col, det_box, det, fused=False
        ):
            """One candidate track vs the detection that became the new id."""
            track_id = snap["id"]
            lost_for = self.frame_count - snap["frame_id"]
            expired = lost_for > self.max_time_lost
            observed = self._dbg_last_observed.get(track_id)
            mean = snap["mean"]
            velocity = (
                "n/a"
                if mean is None or mean.size < 8
                else "vx={:+.2f} vy={:+.2f} vw={:+.2f} vh={:+.2f}".format(*mean[4:8])
            )

            lines = [
                "",
                f"  Candidate ID {track_id}",
                f"    state                      : {_state_name(snap['state'])}"
                f" (is_activated={snap['is_activated']})",
                f"    eligible for this stage    : YES (row {row} of the "
                f"{stage.name}-association matrix)",
                f"    last updated at frame      : {snap['frame_id']}"
                f"   (start_frame {snap['start_frame']}, "
                f"tracklet_len {snap['tracklet_len']})",
                f"    frames since last update   : {lost_for}",
                f"    exceeded max_time_lost     : "
                f"{'YES' if expired else 'NO'} ({lost_for} vs {self.max_time_lost})",
            ]
            if observed is not None:
                lines += [
                    f"    last observed det bbox     : {_fmt_box(observed['xyxy'])}"
                    f"  (frame {observed['frame']}, conf "
                    f"{_fmt_num(observed['conf'])})",
                    f"    last observed center       : "
                    f"{_fmt_center(observed['xyxy'])}",
                ]
            else:
                lines.append(
                    "    last observed det bbox     : n/a "
                    "(not seen since debug started)"
                )
            lines += [
                f"    kalman predicted bbox      : {_fmt_box(snap['pred_xyxy'])}",
                f"    predicted center           : {_fmt_center(snap['pred_xyxy'])}",
                f"    predicted width/height     : {_fmt_wh(snap['pred_xyxy'])}",
                f"    kalman velocity            : {velocity}",
                f"    current detection bbox     : {_fmt_box(det_box)}",
            ]

            if col is None:
                lines.append(
                    "    (this detection did not take part in this stage; no "
                    "cost-matrix entry)"
                )
                return lines

            iou_dist = _cell(stage.iou_dists, row, col)
            raw_iou = None if iou_dist is None else 1.0 - iou_dist
            prox_pass = None if iou_dist is None else iou_dist <= self.proximity_thresh
            lines += [
                "    ---- IoU gate ----",
                f"    raw IoU                    : {_fmt_num(raw_iou)}",
                f"    IoU distance (1 - IoU)     : {_fmt_num(iou_dist)}",
                f"    proximity_thresh           : "
                f"{_fmt_num(self.proximity_thresh)}",
                f"    proximity gate             : "
                f"{'PASS' if prox_pass else 'FAIL'}"
                f"   (mask is iou_dist > proximity_thresh)",
            ]
            if fused:
                lines.append(
                    f"    IoU cost after fuse_score  : "
                    f"{_fmt_num(_cell(stage.fused_iou, row, col))}"
                    "   (1 - (1-iou_dist) * det_conf)"
                )

            if self.with_reid and stage.emb_raw is not None:
                emb_raw = _cell(stage.emb_raw, row, col)
                scale = 2.0 if fused else 1.0  # stage 3 divides by 2.0
                emb_scaled = None if emb_raw is None else emb_raw / scale
                app_pass = (
                    None
                    if emb_scaled is None
                    else emb_scaled <= self.appearance_thresh
                )
                masked = bool(prox_pass is False)
                if emb_scaled is None:
                    emb_after = None
                elif masked or not app_pass:
                    emb_after = 1.0
                else:
                    emb_after = emb_scaled
                lines += [
                    "    ---- ReID (appearance) ----",
                    f"    RAW ReID distance          : {_fmt_num(emb_raw)}"
                    "   <-- before any masking",
                ]
                if fused:
                    lines.append(
                        f"    after boxmot's  / 2.0      : {_fmt_num(emb_scaled)}"
                    )
                lines += [
                    f"    appearance_thresh          : "
                    f"{_fmt_num(self.appearance_thresh)}",
                    f"    appearance gate            : "
                    f"{'PASS' if app_pass else 'FAIL'}",
                    f"    masked by proximity gate   : "
                    f"{'YES' if masked else 'NO'}",
                    f"    ReID distance after mask   : {_fmt_num(emb_after)}",
                ]
                if masked and app_pass:
                    lines.append(
                        "    >>> STRONG APPEARANCE MATCH DISCARDED BY THE IoU "
                        "PROXIMITY GATE <<<"
                    )
            elif self.with_reid:
                lines.append(
                    "    ---- ReID: not computed for this stage this frame ----"
                )

            final = _cell(stage.final_cost, row, col)
            assigned_col = stage.match_for_track(row)
            if assigned_col is None:
                assignment = "UNMATCHED"
            elif assigned_col == col:
                assignment = (
                    "matched to THIS detection "
                    "(unexpected for a new-id event; see ASSIGNMENT RESULT)"
                )
            else:
                other = (
                    stage.dets[assigned_col]
                    if assigned_col < len(stage.dets)
                    else None
                )
                assignment = (
                    f"matched to a DIFFERENT detection: column {assigned_col}"
                    + (f" (det_ind {other['det_ind']})" if other else "")
                )
            thresh = stage.assign_thresh
            lines += [
                "    ---- final association ----",
                f"    final cost used by BoT-SORT: {_fmt_num(final)}"
                "   (= min(IoU cost, masked ReID cost))",
                f"    assignment threshold       : {_fmt_num(thresh)}",
                f"    below threshold            : "
                + (
                    "n/a"
                    if final is None or thresh is None
                    else ("YES" if final < thresh else "NO")
                ),
                f"    assignment result          : {assignment}",
            ]
            return lines

        def _dbg_render_removed(self, det_box, det):
            """Tracks BoxMOT already retired: they cannot match by construction.

            This is what separates "the buffer expired, ID 13 was gone" from
            "ID 13 was still alive but association failed".
            """
            lines = [
                "RECENTLY REMOVED TRACKS (retired by the track buffer; NOT "
                "eligible for any association)"
            ]
            recent = []
            horizon = max(3 * self.max_time_lost, 300)
            for track in getattr(self, "removed_stracks", []):
                track_id = int(getattr(track, "id", -1))
                removed_at = self._dbg_removed_at.get(track_id)
                if removed_at is None or self.frame_count - removed_at > horizon:
                    continue
                recent.append((removed_at, track_id, track))
            recent.sort(reverse=True)
            if not recent:
                lines.append("  (none within the last "
                             f"{horizon} frames)")
                return lines

            for removed_at, track_id, track in recent[:10]:
                last_box = np.asarray(track.xyxy, dtype=float)
                observed = self._dbg_last_observed.get(track_id)
                iou = _plain_iou(last_box, det_box)
                reid = _cosine_distance(
                    getattr(track, "smooth_feat", None),
                    getattr(det, "curr_feat", None),
                )
                lines += [
                    "",
                    f"  Removed ID {track_id}",
                    f"    removed at frame           : {removed_at}"
                    f"   ({self.frame_count - removed_at} frames ago)",
                    f"    last updated at frame      : "
                    f"{int(getattr(track, 'frame_id', -1))}",
                    f"    eligible for association   : NO (already removed; "
                    "it is not in strack_pool)",
                    f"    last known bbox            : {_fmt_box(last_box)}"
                    "   (stale: no Kalman prediction has run since removal)",
                ]
                if observed is not None:
                    lines.append(
                        f"    last observed det bbox     : "
                        f"{_fmt_box(observed['xyxy'])}  (frame "
                        f"{observed['frame']}, conf {_fmt_num(observed['conf'])})"
                    )
                lines += [
                    f"    informational raw IoU      : {_fmt_num(iou)}"
                    "   (vs the stale box; not used by the tracker)",
                    f"    informational RAW ReID     : {_fmt_num(reid)}"
                    "   (cosine distance, not used by the tracker)",
                ]
            if len(recent) > 10:
                lines.append(f"  ... and {len(recent) - 10} more")
            return lines

        def _dbg_render_assignment(
            self, new_id, det_ind, col1, col3, stage1, stage2, stage3
        ):
            lines = [
                "ASSIGNMENT RESULT",
                "",
                f"Detection det_ind {det_ind} (stage-1 column {col1}) "
                f"eventually became new ID {new_id}.",
                "",
                f"{_STAGE_TITLES[STAGE_FIRST]}  "
                f"[thresh {_fmt_num(stage1.assign_thresh)}]",
            ]
            if not stage1.tracks:
                lines.append("  no candidate tracks")
            for row, snap in enumerate(stage1.tracks):
                final = _cell(stage1.final_cost, row, col1)
                assigned = stage1.match_for_track(row)
                if assigned is None:
                    verdict = "unmatched"
                elif assigned == col1:
                    verdict = "matched to THIS detection"
                else:
                    other = (
                        stage1.dets[assigned]
                        if assigned < len(stage1.dets)
                        else None
                    )
                    verdict = (
                        f"matched to detection column {assigned}"
                        + (f" (det_ind {other['det_ind']})" if other else "")
                    )
                lines.append(
                    f"  Candidate ID {snap['id']:<5} "
                    f"final cost vs this detection: {_fmt_num(final)}   "
                    f"assignment: {verdict}"
                )
            det_owner = stage1.match_for_det(col1) if col1 is not None else None
            lines.append(
                f"  Current detection column {col1}: "
                + (
                    "unmatched after the first-stage linear assignment"
                    if det_owner is None
                    else f"matched to row {det_owner}"
                )
            )

            lines += [
                "",
                f"{_STAGE_TITLES[STAGE_SECOND]}  "
                f"[thresh {_fmt_num(stage2.assign_thresh)}]",
                "  Only stage-1 leftover tracks whose state is TRACKED enter "
                "here; a LOST track can never be recovered in stage 2.",
                "  eligible tracks: "
                + (
                    ", ".join(str(s["id"]) for s in stage2.tracks)
                    if stage2.tracks
                    else "(none)"
                ),
                "  low-confidence detections: "
                + (
                    ", ".join(
                        f"det_ind {d['det_ind']} (conf {_fmt_num(d['conf'])})"
                        for d in stage2.dets
                    )
                    if stage2.dets
                    else "(none)"
                ),
            ]
            for row, snap in enumerate(stage2.tracks):
                assigned = stage2.match_for_track(row)
                if assigned is None:
                    continue
                other = stage2.dets[assigned] if assigned < len(stage2.dets) else None
                lines.append(
                    f"  Candidate ID {snap['id']} was matched in stage 2 to "
                    + (f"det_ind {other['det_ind']}" if other else f"column {assigned}")
                )

            lines += [
                "",
                f"{_STAGE_TITLES[STAGE_UNCONFIRMED]}  "
                f"[thresh {_fmt_num(stage3.assign_thresh)}]",
                "  unconfirmed tracks: "
                + (
                    ", ".join(str(s["id"]) for s in stage3.tracks)
                    if stage3.tracks
                    else "(none)"
                ),
            ]
            if col3 is not None:
                owner3 = stage3.match_for_det(col3)
                lines.append(
                    f"  Current detection (stage-3 column {col3}): "
                    + (
                        "unmatched"
                        if owner3 is None
                        else f"matched to unconfirmed row {owner3}"
                    )
                )

            lines += [
                "",
                "COST MATRICES (rows = candidate track ids, "
                "cols = detections by det_ind)",
                "  stage-1 raw IoU distance (1 - IoU):",
            ]
            row_labels = [f"ID {s['id']}" for s in stage1.tracks]
            col_labels = [f"det{d['det_ind']}" for d in stage1.dets]
            lines += _fmt_matrix(stage1.iou_dists, row_labels, col_labels)
            if self.with_reid:
                lines.append("  stage-1 RAW ReID distance (before masking):")
                lines += _fmt_matrix(stage1.emb_raw, row_labels, col_labels)
            lines.append("  stage-1 final cost handed to the Hungarian solver:")
            lines += _fmt_matrix(stage1.final_cost, row_labels, col_labels)
            return lines

        def _dbg_render_stage_trace(
            self, new_id, det_conf, col1, col3, stage1, stage2, stage3
        ):
            unmatched1 = (
                stage1.u_det is not None and col1 is not None and col1 in stage1.u_det
            )
            unmatched3 = (
                stage3.u_det is not None and col3 is not None and col3 in stage3.u_det
            )
            lines = [
                "STAGE TRACE - where this detection went",
                f"  stage 1 (first association, high-conf dets, thresh "
                f"{_fmt_num(stage1.assign_thresh)}): "
                + ("UNMATCHED" if unmatched1 else "see ASSIGNMENT RESULT above"),
                f"  stage 2 (second association, low-conf dets, thresh "
                f"{_fmt_num(stage2.assign_thresh)}): NOT ELIGIBLE - this "
                f"detection's conf {_fmt_num(det_conf)} > track_high_thresh "
                f"{_fmt_num(self.track_high_thresh)}, so it is a first-stage "
                "detection and never enters the second association",
                f"  stage 3 (unconfirmed tracks, thresh "
                f"{_fmt_num(stage3.assign_thresh)}): "
                + (
                    "UNMATCHED"
                    if unmatched3
                    else (
                        "no unconfirmed track to match"
                        if not stage3.tracks
                        else "see ASSIGNMENT RESULT above"
                    )
                ),
                f"  stage 4 (new track init): conf {_fmt_num(det_conf)} >= "
                f"new_track_thresh {_fmt_num(self.new_track_thresh)} "
                f"-> STrack.activate() -> NEW ID {new_id}",
            ]
            return lines

    tracker = InstrumentedBotSort(**botsort_kwargs)
    # Recorded for the report: BotSort keeps only the derived buffer_size.
    tracker._dbg_frame_rate = botsort_kwargs.get("frame_rate", 30)
    tracker._dbg_track_buffer = botsort_kwargs.get("track_buffer", 30)
    return tracker


# ----------------------------------------------------------------------
# small numeric helpers (module level so the report code stays readable)
# ----------------------------------------------------------------------
def _cell(matrix, row, col):
    if matrix is None or row is None or col is None:
        return None
    m = np.asarray(matrix)
    if row >= m.shape[0] or col >= m.shape[1]:
        return None
    return float(m[row, col])


def _plain_iou(box_a, box_b) -> Optional[float]:
    """Plain xyxy IoU, used only for tracks boxmot already removed."""
    if box_a is None or box_b is None:
        return None
    a = np.asarray(box_a, dtype=float).ravel()
    b = np.asarray(box_b, dtype=float).ravel()
    if a.size < 4 or b.size < 4:
        return None
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _cosine_distance(feat_a, feat_b) -> Optional[float]:
    """Cosine distance between two already-L2-normalised ReID vectors."""
    if feat_a is None or feat_b is None:
        return None
    a = np.asarray(feat_a, dtype=float).ravel()
    b = np.asarray(feat_b, dtype=float).ravel()
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return None
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return None
    return float(max(0.0, 1.0 - float(np.dot(a, b)) / norm))


def _fmt_timestamp(value) -> str:
    if value is None:
        return "n/a"
    try:
        return _dt.datetime.fromtimestamp(float(value)).isoformat(
            timespec="milliseconds"
        )
    except (TypeError, ValueError, OSError):
        return str(value)
