"""Focused tests for the experimental occlusion-aware state protection.

Delete this file together with
``skeleton_detection/inference/occlusion_tracking.py``.

Needs ``boxmot``, which exists only inside the Docker image; skipped
elsewhere. No GPU and no ReID model: embeddings are passed in directly.

Coverage, matching the experiment's acceptance list:

    A  a clean NORMAL sequence is bit-identical to stock BoT-SORT
    B  an OCCLUDED observation does not change smooth_feat / curr_feat
    C  laterally shifting, narrowing OCCLUDED boxes do not drive the Kalman
       velocity, and prediction continues from the last trusted state
    D  NORMAL -> OCCLUDED -> OCCLUDED -> NORMAL resumes both updates
    E  OCCLUDED widths never enter the (now informational) width history

plus, for the visible-ratio-only rule and the relaxed proximity gate:

    1  a sideways person (16/17 joints, box at 0.45x width) stays NORMAL
    2  a low visible ratio (6/17) is OCCLUDED
    3  the gate masks ReID exactly when iou_dist > 0.70, probed on the real
       stage-1 cost matrix from IoU 0.45 down to 0.10 (so IoU 0.35 with ReID
       0.10 reaches the solver as 0.10, IoU 0.20 is masked)
    4  the SkeletonTracker defaults for the association thresholds
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("boxmot", reason="boxmot only exists inside the Docker image")

from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState  # noqa: E402
from boxmot.trackers.botsort.botsort import BotSort  # noqa: E402

from skeleton_detection.inference.occlusion_tracking import (  # noqa: E402
    NORMAL,
    OCCLUDED,
    REASON_VISIBLE_RATIO,
    OcclusionParams,
    classify_occlusion,
    compute_detection_visibility,
    make_occlusion_aware_botsort,
)

IMG = np.zeros((480, 848, 3), dtype=np.uint8)
EMB_DIM = 8
NUM_KEYPOINTS = 17

TRACKER_KWARGS = dict(
    reid_model=None,
    with_reid=True,
    cmc_method=None,
    frame_rate=55,
    track_high_thresh=0.5,
    new_track_thresh=0.6,
    track_buffer=90,
    match_thresh=0.8,
    appearance_thresh=0.25,
    # 0.7, matching the node default: the gate is on IoU DISTANCE, so this
    # keeps ReID eligible down to raw IoU 0.30.
    proximity_thresh=0.7,
)

PARAMS = OcclusionParams()


def _emb(seed: int) -> np.ndarray:
    vector = np.zeros(EMB_DIM, dtype=np.float32)
    vector[seed % EMB_DIM] = 1.0
    vector[(seed + 3) % EMB_DIM] = 0.25
    return vector / np.linalg.norm(vector)


def _box(cx, cy, w=60.0, h=160.0, conf=0.9):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf, 0.0]


def _scores(n_visible, value=0.9, low=0.05):
    """A COCO-17 keypoint-score row with exactly ``n_visible`` joints above
    the default keypoint_visibility_threshold of 0.30."""
    scores = np.full(NUM_KEYPOINTS, low, dtype=np.float32)
    scores[:n_visible] = value
    return scores


def _visibility(n_visible):
    return compute_detection_visibility(
        _scores(n_visible), PARAMS.keypoint_visibility_threshold
    )


def _build(params=PARAMS, **overrides):
    BaseTrack.clear_count()
    cls = make_occlusion_aware_botsort(BotSort, params, None)
    return cls(**dict(TRACKER_KWARGS, **overrides))


def _step(tracker, boxes, embs, visibility):
    tracker.set_frame_visibility(visibility, frame_index=tracker.frame_count)
    return np.asarray(
        tracker.update(
            np.asarray(boxes, dtype=np.float32),
            IMG,
            np.asarray(embs, dtype=np.float32),
        ),
        dtype=np.float64,
    ).copy()


def _stock_step(tracker, boxes, embs):
    """One update of an unmodified BotSort, for the control experiments."""
    return np.asarray(
        tracker.update(
            np.asarray(boxes, dtype=np.float32), IMG, np.asarray(embs, dtype=np.float32)
        ),
        dtype=np.float64,
    ).copy()


def _track(tracker, track_id):
    for track in list(tracker.active_tracks) + list(tracker.lost_stracks):
        if int(track.id) == track_id:
            return track
    raise AssertionError(f"track {track_id} not found")


# ----------------------------------------------------------------------
# the classification rule itself
# ----------------------------------------------------------------------
def test_visible_ratio_counts_per_keypoint_scores():
    result = compute_detection_visibility(_scores(7), 0.30)
    assert result.available
    assert (result.visible_count, result.total_count) == (7, 17)
    assert result.visible_ratio == pytest.approx(7 / 17)


@pytest.mark.parametrize(
    "scores,note",
    [
        (None, "NO_KEYPOINT_SCORES"),
        (np.empty(0, dtype=np.float32), "EMPTY_KEYPOINT_SCORES"),
        (np.full(NUM_KEYPOINTS, np.nan, dtype=np.float32), "NON_FINITE_KEYPOINT_SCORES"),
    ],
)
def test_missing_keypoint_scores_are_unavailable_not_zero(scores, note):
    result = compute_detection_visibility(scores, 0.30)
    assert not result.available
    assert result.visible_ratio is None
    assert result.note == note
    # ... and an unavailable signal must never vote OCCLUDED on its own.
    assert classify_occlusion(result, PARAMS, 60.0, None, 0).state == NORMAL


def test_partial_keypoint_scores_shrink_the_denominator_and_are_flagged():
    scores = _scores(6).astype(np.float64)
    scores[15:] = np.nan
    result = compute_detection_visibility(scores, 0.30)
    assert result.available
    assert (result.visible_count, result.total_count) == (6, 15)
    assert result.note == "PARTIAL_KEYPOINT_SCORES"


def test_binary_rule_uses_visible_ratio_and_nothing_else():
    good, bad = _visibility(14), _visibility(7)

    assert classify_occlusion(good, PARAMS, 120.0, 120.0, 15).state == NORMAL

    occluded = classify_occlusion(bad, PARAMS, 120.0, 120.0, 15)
    assert occluded.state == OCCLUDED
    assert occluded.reasons == [REASON_VISIBLE_RATIO]

    # a collapsed box with a good skeleton is NOT occluded any more
    narrow = classify_occlusion(good, PARAMS, 50.0, 120.0, 15)
    assert narrow.state == NORMAL
    assert narrow.reasons == []
    # ... but the ratio is still reported, for the log only
    assert narrow.width_ratio == pytest.approx(50.0 / 120.0)
    assert narrow.width_ratio_available


@pytest.mark.parametrize("width", [5.0, 30.0, 120.0, 400.0])
def test_bbox_width_has_zero_effect_on_the_verdict(width):
    """Same skeleton, wildly different widths -> identical classification."""
    for n_visible, expected in ((16, NORMAL), (6, OCCLUDED)):
        result = classify_occlusion(_visibility(n_visible), PARAMS, width, 120.0, 15)
        assert result.state == expected
        assert result.reasons in ([], [REASON_VISIBLE_RATIO])
        assert "BBOX_WIDTH_RATIO_FAIL" not in result.reasons


def test_new_track_without_history_does_not_invent_a_reference_width():
    """Edge case: no reference width -> the width ratio is unavailable."""
    narrow = classify_occlusion(_visibility(14), PARAMS, 5.0, None, 0)
    assert narrow.state == NORMAL
    assert not narrow.width_ratio_available
    assert narrow.width_ratio is None


# ----------------------------------------------------------------------
# Test 1 / Test 2 -- the new rule at the tracker level
# ----------------------------------------------------------------------
def test_1_sideways_person_is_not_occluded():
    """16/17 joints visible, box collapsed to 0.45x -> NORMAL."""
    tracker = _build()
    for _ in range(15):
        _step(tracker, [_box(200, 300, w=120.0)], [_emb(1)], [_visibility(17)])
    reference = _track(tracker, 1).normal_reference_width()
    assert reference == pytest.approx(120.0)

    # turning sideways: same person, same skeleton, much narrower box
    _step(tracker, [_box(200, 300, w=54.0)], [_emb(1)], [_visibility(16)])

    track = _track(tracker, 1)
    assert track.visibility_state == NORMAL
    assert track.last_assessment.reasons == []
    assert track.last_assessment.width_ratio == pytest.approx(0.45)
    assert track.last_assessment.visibility.visible_ratio == pytest.approx(16 / 17)
    # and because it is NORMAL, the Kalman filter did take the measurement
    assert track.mean[2] < 120.0


def test_2_low_visible_ratio_is_occluded():
    """6/17 = 0.353 < 0.50 -> OCCLUDED, whatever the box is doing."""
    tracker = _build()
    for _ in range(15):
        _step(tracker, [_box(200, 300, w=120.0)], [_emb(1)], [_visibility(17)])

    _step(tracker, [_box(200, 300, w=120.0)], [_emb(1)], [_visibility(6)])

    track = _track(tracker, 1)
    assert track.visibility_state == OCCLUDED
    assert track.last_assessment.reasons == [REASON_VISIBLE_RATIO]
    assert track.last_assessment.visibility.visible_ratio == pytest.approx(6 / 17)
    # full-width box, so the old width rule would have said NORMAL here
    assert track.last_assessment.width_ratio == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Test A -- NORMAL behaviour is unchanged
# ----------------------------------------------------------------------
def test_a_normal_sequence_is_identical_to_stock_botsort():
    """Every detection fully visible => byte-for-byte stock BoT-SORT output."""
    frames = []
    for i in range(30):
        frames.append(
            (
                [_box(200, 300), _box(600 - 3 * i, 300)],
                [_emb(1), _emb(5)],
                [_visibility(17), _visibility(17)],
            )
        )

    BaseTrack.clear_count()
    stock_tracker = BotSort(**TRACKER_KWARGS)
    stock = [_stock_step(stock_tracker, b, e) for b, e, _ in frames]

    tracker = _build()
    aware = [_step(tracker, b, e, v) for b, e, v in frames]

    assert len(stock) == len(aware)
    for index, (a, b) in enumerate(zip(stock, aware)):
        assert a.shape == b.shape, f"frame {index} row count differs"
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index} differs")


def test_a_disabled_feature_is_never_constructed():
    """occlusion_aware_tracking=false must build a plain BotSort."""
    from skeleton_detection.inference.person_tracking import SkeletonTracker

    tracker = SkeletonTracker(with_reid=False, occlusion_aware_tracking=False)
    assert type(tracker.tracker) is BotSort
    assert not hasattr(tracker.tracker, "set_frame_visibility")


# ----------------------------------------------------------------------
# Test B -- ReID freeze
# ----------------------------------------------------------------------
def test_b_occluded_observation_does_not_change_the_appearance_feature():
    tracker = _build()
    feature_a, feature_b = _emb(1), _emb(4)

    for _ in range(8):  # establish the track and its NORMAL appearance
        _step(tracker, [_box(200, 300)], [feature_a], [_visibility(17)])
    track = _track(tracker, 1)
    smooth_before = track.smooth_feat.copy()
    curr_before = track.curr_feat.copy()
    history_before = len(track.features)

    # same place (so it still associates) but a badly occluded skeleton and a
    # completely different appearance vector
    _step(tracker, [_box(200, 300)], [feature_b], [_visibility(5)])

    track = _track(tracker, 1)
    assert track.visibility_state == OCCLUDED
    np.testing.assert_array_equal(track.smooth_feat, smooth_before)
    np.testing.assert_array_equal(track.curr_feat, curr_before)
    assert len(track.features) == history_before


def test_b_reid_would_have_changed_without_the_protection():
    """Control: stock BoT-SORT does absorb the contaminated feature."""
    BaseTrack.clear_count()
    tracker = BotSort(**TRACKER_KWARGS)
    for _ in range(8):
        _stock_step(tracker, [_box(200, 300)], [_emb(1)])
    before = tracker.active_tracks[0].smooth_feat.copy()
    _stock_step(tracker, [_box(200, 300)], [_emb(4)])
    assert not np.array_equal(tracker.active_tracks[0].smooth_feat, before)


# ----------------------------------------------------------------------
# Test C -- Kalman protection
# ----------------------------------------------------------------------
# A stationary person, 120 px wide, occluded from the left: the right edge
# stays at 260 while the visible part shrinks, so the box narrows AND its
# centre marches right -- exactly the failure mode this experiment targets.
FULL_BOX = _box(200, 300, w=120.0)          # [140, 260]
PARTIAL_BOXES = [
    _box(215, 300, w=90.0),                 # [170, 260]
    _box(230, 300, w=60.0),                 # [200, 260]
    _box(245, 300, w=30.0),                 # [230, 260]
]


def test_c_occluded_boxes_do_not_drive_the_kalman_velocity():
    """Stationary at x=200, then partial boxes at x=215, 230, 245."""
    tracker = _build()
    for _ in range(15):  # stationary, fully visible: velocity settles near 0
        _step(tracker, [FULL_BOX], [_emb(1)], [_visibility(17)])

    track = _track(tracker, 1)
    trusted_mean = track.mean.copy()
    assert abs(trusted_mean[4]) < 0.5, "precondition: the track was stationary"

    for box in PARTIAL_BOXES:
        _step(tracker, [box], [_emb(1)], [_visibility(6)])

    track = _track(tracker, 1)
    assert track.visibility_state == OCCLUDED
    assert track.state == TrackState.Tracked, "the track must stay alive"
    assert int(track.id) == 1
    # velocity was never overwritten by the partial boxes
    np.testing.assert_allclose(track.mean[4:8], trusted_mean[4:8], rtol=0, atol=1e-9)
    # prediction did continue -- the position advanced by 3 x the trusted vx
    assert track.mean[0] == pytest.approx(trusted_mean[0] + 3.0 * trusted_mean[4])
    assert track.mean[0] < 205.0, "the box must not have chased the partial dets"
    assert track.mean[2] > 115.0, "the width must not have collapsed either"


def test_c_control_stock_botsort_is_dragged_by_the_same_boxes():
    BaseTrack.clear_count()
    tracker = BotSort(**TRACKER_KWARGS)
    for _ in range(15):
        _stock_step(tracker, [FULL_BOX], [_emb(1)])
    before = tracker.active_tracks[0].mean.copy()
    for box in PARTIAL_BOXES:
        _stock_step(tracker, [box], [_emb(1)])
    after = tracker.active_tracks[0].mean
    assert after[4] > before[4] + 1.0, "stock BoT-SORT learns a spurious +vx"
    assert after[2] < before[2] * 0.8, "stock BoT-SORT also learns the narrow width"


def test_c_a_partial_box_outside_the_iou_gate_goes_lost_not_corrupted():
    """The documented cost of not chasing the partial box.

    Because the track no longer moves toward a drifting partial detection, a
    far enough partial box falls outside BoT-SORT's IoU gate and the track is
    marked LOST instead of being dragged. That is deliberate -- association is
    stock BoxMOT and was not touched -- and the track keeps its id, its clean
    velocity and its clean appearance for the whole track buffer.
    """
    tracker = _build()
    for _ in range(15):
        _step(tracker, [_box(200, 300, w=60.0)], [_emb(1)], [_visibility(17)])
    trusted = _track(tracker, 1).mean.copy()
    trusted_feature = _track(tracker, 1).smooth_feat.copy()

    # 15 px/frame with a half-width box: by the third one IoU is under the gate
    for cx in (215.0, 230.0, 245.0):
        _step(tracker, [_box(cx, 300, w=30.0)], [_emb(1)], [_visibility(6)])

    track = _track(tracker, 1)
    assert track.state == TrackState.Lost
    assert int(track.id) == 1, "still in the pool, still recoverable"
    assert track not in tracker.removed_stracks
    np.testing.assert_array_equal(track.smooth_feat, trusted_feature)
    assert track.mean[2] == pytest.approx(trusted[2], abs=1.0), "width uncorrupted"


# ----------------------------------------------------------------------
# Test D -- return to NORMAL
# ----------------------------------------------------------------------
def test_d_normal_updates_resume_after_the_occlusion():
    tracker = _build()
    for _ in range(15):
        _step(tracker, [_box(200, 300)], [_emb(1)], [_visibility(17)])

    _step(tracker, [_box(205, 300, w=25.0)], [_emb(4)], [_visibility(5)])
    _step(tracker, [_box(210, 300, w=25.0)], [_emb(4)], [_visibility(5)])
    assert _track(tracker, 1).visibility_state == OCCLUDED
    frozen_feature = _track(tracker, 1).smooth_feat.copy()
    frozen_mean = _track(tracker, 1).mean.copy()

    _step(tracker, [_box(212, 300)], [_emb(4)], [_visibility(17)])

    track = _track(tracker, 1)
    assert track.visibility_state == NORMAL
    assert not np.array_equal(track.smooth_feat, frozen_feature), "ReID resumed"
    assert not np.array_equal(track.mean[:4], frozen_mean[:4]), "Kalman resumed"
    assert track.mean[0] == pytest.approx(212.0, abs=6.0)
    assert int(track.id) == 1, "the id survived the occlusion"


# ----------------------------------------------------------------------
# Test E -- reference-width history protection
# ----------------------------------------------------------------------
def test_e_occluded_widths_never_enter_the_normal_width_history():
    tracker = _build()
    for _ in range(15):
        _step(tracker, [_box(200, 300, w=120.0)], [_emb(1)], [_visibility(17)])
    track = _track(tracker, 1)
    assert len(track.normal_widths) == PARAMS.normal_bbox_history_size
    assert set(track.normal_widths) == {120.0}
    assert track.normal_reference_width() == pytest.approx(120.0)

    for _ in range(5):
        _step(tracker, [_box(200, 300, w=30.0)], [_emb(1)], [_visibility(4)])

    track = _track(tracker, 1)
    assert track.visibility_state == OCCLUDED
    assert set(track.normal_widths) == {120.0}, "30.0 must never be recorded"
    assert track.normal_reference_width() == pytest.approx(120.0)

    # and a NORMAL frame does append again
    _step(tracker, [_box(200, 300, w=118.0)], [_emb(1)], [_visibility(17)])
    assert 118.0 in _track(tracker, 1).normal_widths


def test_e_width_alone_can_no_longer_mark_a_detection_occluded():
    """Full skeleton confidence, box collapsed to 37% -> still NORMAL.

    This is the regression the rule change exists for: under the old two-signal
    rule this frame was classified OCCLUDED.
    """
    tracker = _build()
    for _ in range(15):
        _step(tracker, [_box(200, 300, w=120.0)], [_emb(1)], [_visibility(17)])

    _step(tracker, [_box(200, 300, w=45.0)], [_emb(1)], [_visibility(17)])
    track = _track(tracker, 1)
    assert track.visibility_state == NORMAL
    assert track.last_assessment.reasons == []
    assert track.last_assessment.width_ratio == pytest.approx(0.375)


def test_e_history_is_per_track_and_does_not_jump_between_people():
    tracker = _build()
    for _ in range(15):
        _step(
            tracker,
            [_box(200, 300, w=120.0), _box(600, 300, w=50.0)],
            [_emb(1), _emb(5)],
            [_visibility(17), _visibility(17)],
        )
    wide, narrow = _track(tracker, 1), _track(tracker, 2)
    assert wide.normal_reference_width() == pytest.approx(120.0)
    assert narrow.normal_reference_width() == pytest.approx(50.0)
    # the narrow person is NOT occluded just because someone else is wider
    assert narrow.visibility_state == NORMAL


# ----------------------------------------------------------------------
# lifetime + re-activation
# ----------------------------------------------------------------------
def test_track_lifetime_bookkeeping_survives_an_occluded_match():
    tracker = _build()
    for _ in range(15):
        _step(tracker, [_box(200, 300, w=120.0)], [_emb(1)], [_visibility(17)])
    track = _track(tracker, 1)
    frame_before, length_before = track.frame_id, track.tracklet_len

    _step(tracker, [_box(200, 300, w=30.0)], [_emb(1)], [_visibility(4)])

    track = _track(tracker, 1)
    assert track.frame_id == frame_before + 1, "end_frame must advance"
    assert track.tracklet_len == length_before + 1
    assert track.state == TrackState.Tracked
    assert track.is_activated
    assert track in tracker.active_tracks


def test_reactivating_a_lost_track_on_an_occluded_detection_keeps_the_id():
    tracker = _build()
    for _ in range(15):
        _step(tracker, [_box(200, 300, w=120.0)], [_emb(1)], [_visibility(17)])
    track = _track(tracker, 1)
    trusted_velocity = track.mean[4:8].copy()
    trusted_feature = track.smooth_feat.copy()

    for _ in range(5):  # nothing detected: the track goes LOST
        _step(tracker, np.empty((0, 6), dtype=np.float32), np.empty((0, EMB_DIM)), [])
    assert _track(tracker, 1).state == TrackState.Lost

    _step(tracker, [_box(200, 300, w=30.0)], [_emb(4)], [_visibility(4)])

    track = _track(tracker, 1)
    assert int(track.id) == 1, "re_activate must not allocate a new id"
    assert track.state == TrackState.Tracked
    assert track.tracklet_len == 0, "re_activate resets tracklet_len as usual"
    np.testing.assert_array_equal(track.smooth_feat, trusted_feature)
    # multi_predict zeroes the velocity of a non-Tracked track, which is stock
    # BoxMOT behaviour and happens before this code sees the observation; what
    # matters is that the partial box did not write a NEW velocity.
    assert np.all(track.mean[4:8] == 0.0) or np.allclose(
        track.mean[4:8], trusted_velocity
    )


# ----------------------------------------------------------------------
# Test 3 -- the relaxed proximity gate
#
# These exercise the REAL BoxMOT code path: the cost matrix that stage 1 hands
# to the Hungarian solver is captured verbatim, so the assertions are about
# what BoT-SORT actually computed, not about a reimplementation of its mask.
# ----------------------------------------------------------------------
class _Stage1Recorder:
    """Pass-through proxies that snapshot stage 1's matrices.

    Each proxy returns exactly what the real helper returned, so the tracker's
    arithmetic is unchanged. The copies matter because BoxMOT masks emb_dists
    in place.
    """

    def __init__(self):
        self.iou = None
        self.emb_raw = None
        self.final = None

    def __enter__(self):
        import boxmot.trackers.botsort.botsort as botsort_module
        from boxmot.utils import matching

        self._module = botsort_module
        self._saved = (
            botsort_module.iou_distance,
            botsort_module.embedding_distance,
            botsort_module.linear_assignment,
        )

        def iou_distance(a, b, *args, **kwargs):
            result = matching.iou_distance(a, b, *args, **kwargs)
            if self.iou is None:
                self.iou = np.asarray(result, dtype=float).copy()
            return result

        def embedding_distance(t, d, *args, **kwargs):
            result = matching.embedding_distance(t, d, *args, **kwargs)
            if self.emb_raw is None:
                self.emb_raw = np.asarray(result, dtype=float).copy()
            return result

        def linear_assignment(cost, *args, **kwargs):
            if self.final is None:
                self.final = np.asarray(cost, dtype=float).copy()
            return matching.linear_assignment(cost, *args, **kwargs)

        botsort_module.iou_distance = iou_distance
        botsort_module.embedding_distance = embedding_distance
        botsort_module.linear_assignment = linear_assignment
        return self

    def __exit__(self, *exc):
        (
            self._module.iou_distance,
            self._module.embedding_distance,
            self._module.linear_assignment,
        ) = self._saved
        return False


def _emb_at_cosine_distance(base: np.ndarray, distance: float) -> np.ndarray:
    """A unit vector whose cosine distance from ``base`` is exactly ``distance``."""
    base = base / np.linalg.norm(base)
    orthogonal = np.zeros_like(base)
    orthogonal[int(np.argmin(np.abs(base)))] = 1.0
    orthogonal = orthogonal - np.dot(orthogonal, base) * base
    orthogonal /= np.linalg.norm(orthogonal)
    cosine = 1.0 - distance
    vector = cosine * base + np.sqrt(max(0.0, 1.0 - cosine**2)) * orthogonal
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _gate_probe(target_iou, reid_distance, width=100.0, settle=25):
    """Park a stationary track, then probe it with one offset detection.

    Returns ``(raw_iou, raw_reid, final_cost)`` as BoT-SORT computed them.
    Equal-size boxes offset by ``d`` have IoU ``(w - d) / (w + d)``.
    """
    tracker = _build()
    base = _emb(1)
    for _ in range(settle):  # let the Kalman prediction settle on the box
        _step(tracker, [_box(400, 300, w=width)], [base], [_visibility(17)])

    offset = width * (1.0 - target_iou) / (1.0 + target_iou)
    probe = _box(400 + offset, 300, w=width)
    feature = _emb_at_cosine_distance(base, reid_distance)

    with _Stage1Recorder() as rec:
        _step(tracker, [probe], [feature], [_visibility(17)])

    return (
        1.0 - float(rec.iou[0, 0]),
        float(rec.emb_raw[0, 0]),
        float(rec.final[0, 0]),
    )


@pytest.mark.parametrize("target_iou", [0.45, 0.35, 0.31, 0.30, 0.29, 0.20, 0.10])
def test_3_gate_masks_reid_exactly_when_iou_dist_exceeds_proximity_thresh(target_iou):
    """The masking rule, checked against what BoT-SORT actually computed.

    For a pair whose appearance passes, the stage-1 cost is the ReID distance
    when the pair is NOT proximity-masked and the IoU distance when it is, so
    the final cost reveals the mask. It must agree with
    ``iou_dist > proximity_thresh`` on every probe -- including the nominal
    0.30 knife edge, where BoxMOT's float32 IoU lands at iou_dist 0.7000000595
    and is therefore masked.
    """
    raw_iou, raw_reid, final = _gate_probe(target_iou=target_iou, reid_distance=0.10)
    iou_dist = 1.0 - raw_iou

    # the probe really produced the geometry and appearance it asked for
    assert raw_iou == pytest.approx(target_iou, abs=0.02)
    assert raw_reid == pytest.approx(0.10, abs=0.01)
    assert raw_reid <= 0.25, "precondition: the appearance gate passes"

    masked = iou_dist > 0.70
    expected = iou_dist if masked else raw_reid
    assert final == pytest.approx(expected, abs=1e-6)
    if target_iou >= 0.31:
        # e.g. IoU 0.35 + ReID 0.10: the appearance match reaches the solver
        assert not masked and final == pytest.approx(0.10, abs=0.01)
    if target_iou <= 0.29:
        assert masked and final > raw_reid


def test_4_skeleton_tracker_association_defaults():
    from skeleton_detection.inference.person_tracking import SkeletonTracker

    tracker = SkeletonTracker(with_reid=False)
    assert tracker.tracker.proximity_thresh == pytest.approx(0.70)
    # the other association thresholds keep the BoxMOT defaults
    assert tracker.tracker.appearance_thresh == pytest.approx(0.25)
    assert tracker.tracker.match_thresh == pytest.approx(0.80)
    assert tracker.tracker.track_high_thresh == pytest.approx(0.50)
    assert tracker.tracker.new_track_thresh == pytest.approx(0.60)
    assert tracker.max_time_lost == 165


# ----------------------------------------------------------------------
# the debug log
# ----------------------------------------------------------------------
class _Sink:
    def __init__(self):
        self.blocks = []

    def write_block(self, text):
        self.blocks.append(text)


def test_debug_log_records_both_transitions_and_no_embeddings():
    from skeleton_detection.inference.occlusion_tracking import OcclusionDebugLogger

    sink = _Sink()
    BaseTrack.clear_count()
    cls = make_occlusion_aware_botsort(BotSort, PARAMS, OcclusionDebugLogger(sink))
    tracker = cls(**TRACKER_KWARGS)

    for _ in range(15):
        _step(tracker, [_box(200, 300, w=120.0)], [_emb(1)], [_visibility(17)])
    _step(tracker, [_box(205, 300, w=30.0)], [_emb(1)], [_visibility(5)])
    _step(tracker, [_box(210, 300, w=30.0)], [_emb(1)], [_visibility(5)])
    _step(tracker, [_box(212, 300, w=120.0)], [_emb(1)], [_visibility(17)])

    text = "\n".join(sink.blocks)
    assert "Track 1: NORMAL -> OCCLUDED" in text
    assert "Track 1: OCCLUDED -> NORMAL" in text
    assert "Kalman measurement update: SKIPPED" in text
    assert "Kalman prediction: CONTINUED" in text
    assert "ReID feature update: FROZEN" in text
    assert "VISIBLE_RATIO_FAIL" in text
    assert "BBOX_WIDTH_RATIO_FAIL" not in text, "width is no longer a reason"
    assert "classification use: INFORMATIONAL ONLY" in text
    assert "bbox width ratio:" in text
    assert "velocity preserved from trusted state:" in text
    # steady NORMAL tracking must not flood the file
    assert sum(1 for b in sink.blocks if "visibility state: NORMAL" in b) == 1

