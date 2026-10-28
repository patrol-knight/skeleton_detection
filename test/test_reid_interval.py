"""Tests for ``SkeletonTracker(reid_interval=N)``.

ReID runs on processed frames 0, N, 2N, ...; the frames between use BoT-SORT's
IoU/Kalman matching only. A track born on a skipped frame has no appearance
feature until its next ReID frame, and the featureless guard on
``embedding_distance`` keeps that from crashing BoxMOT.

The ReID model is a fake whose feature is the mean colour of the crop, so
appearance matching is real but needs no checkpoint. Needs ``boxmot``, which
exists only inside the Docker image::

    python3 -m pytest test/test_reid_interval.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skeleton_detection.inference.rtmo_inference import PersonDetection  # noqa: E402

boxmot = pytest.importorskip(
    "boxmot", reason="boxmot only exists inside the Docker image"
)
import boxmot.trackers.botsort.botsort as botsort_module  # noqa: E402
from boxmot.trackers.botsort.botsort_track import STrack  # noqa: E402
from boxmot.utils.matching import embedding_distance as stock_distance  # noqa: E402

from skeleton_detection.inference.person_tracking import (  # noqa: E402
    NO_APPEARANCE_COST,
    SkeletonTracker,
    TrackerInitError,
    install_featureless_guard,
)

COLOURS = {"A": (200, 40, 40), "B": (40, 200, 40), "C": (40, 40, 200)}


class FakeReID:
    """Records which tracker frames ran ReID; feature = crop mean colour."""

    def __init__(self):
        self.tracker = None
        self.frames = []

    def get_features(self, xyxys, img):
        # Called inside update(), after BoxMOT's 1-based frame_count += 1.
        self.frames.append(self.tracker.frame_count - 1)
        feats = []
        for x1, y1, x2, y2 in np.asarray(xyxys, dtype=int):
            crop = img[max(y1, 0):y2, max(x1, 0):x2].reshape(-1, 3)
            feats.append(crop.mean(axis=0) + 1.0)
        return np.asarray(feats, dtype=np.float32)


def make_tracker(monkeypatch, reid_interval, occlusion_aware_tracking=False):
    reid = FakeReID()
    monkeypatch.setattr(SkeletonTracker, "_build_reid", lambda self: reid)
    tracker = SkeletonTracker(
        device="cpu",
        reid_interval=reid_interval,
        occlusion_aware_tracking=occlusion_aware_tracking,
    )
    reid.tracker = tracker.tracker
    return tracker, reid


def probe_sequence(frames=12):
    """A moves right; B is born on frame 1 (OFF for N=2) and misses frame 7;
    C appears on OFF frame 5 only."""
    for i in range(frames):
        people = {"A": (60 + 5 * i, 160)}
        if i >= 1 and i != 7:
            people["B"] = (450 - 5 * i, 160)
        if i == 5:
            people["C"] = (230, 400)
        yield people


def render(people):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = []
    for index, (name, (cx, cy)) in enumerate(people.items()):
        x1, y1, x2, y2 = cx - 45, cy - 145, cx + 45, cy + 145
        img[max(y1, 0):y2, max(x1, 0):x2] = COLOURS[name]
        detections.append(
            PersonDetection(
                bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
                score=0.9,
                keypoints_xy=np.full((17, 2), [cx, cy], dtype=np.float32),
                keypoint_scores=np.full(17, 0.8, dtype=np.float32),
                detection_index=index,
            )
        )
    return img, detections


def run(tracker, frames=12):
    """Per frame: {person name: track id or None}."""
    ids = []
    for people in probe_sequence(frames):
        img, detections = render(people)
        tracker.update(detections, img)
        ids.append({n: d.track_id for n, d in zip(people, detections)})
    return ids


def track_for(tracker, name_box):
    for track in tracker.tracker.active_tracks:
        if np.allclose(track.xyxy, name_box, atol=1.0):
            return track
    return None


# ----------------------------------------------------------------------
# cadence


def test_interval_one_runs_reid_every_frame(monkeypatch):
    tracker, reid = make_tracker(monkeypatch, reid_interval=1)
    run(tracker, frames=6)
    assert reid.frames == [0, 1, 2, 3, 4, 5]


def test_interval_two_runs_reid_only_on_even_frames(monkeypatch):
    tracker, reid = make_tracker(monkeypatch, reid_interval=2)
    run(tracker)
    # Births on OFF frames 1 and 5 trigger no ReID either.
    assert reid.frames == [0, 2, 4, 6, 8, 10]


def test_reid_interval_must_be_positive():
    with pytest.raises(TrackerInitError):
        SkeletonTracker(device="cpu", with_reid=False, reid_interval=0)


# ----------------------------------------------------------------------
# tracking behaviour


def stock_run(monkeypatch):
    """Today's behaviour: stock BoT-SORT with ReID every frame and the
    unwrapped BoxMOT embedding_distance."""
    tracker, _ = make_tracker(monkeypatch, reid_interval=1)
    monkeypatch.setattr(botsort_module, "embedding_distance", stock_distance)
    return run(tracker)


def test_interval_one_matches_stock_tracking(monkeypatch):
    expected = stock_run(monkeypatch)
    monkeypatch.undo()
    tracker, _ = make_tracker(monkeypatch, reid_interval=1)
    assert run(tracker) == expected


def test_interval_two_ids_match_interval_one_on_probe_sequence(monkeypatch):
    # Built and run one at a time: BoxMOT's track-id counter is class-global.
    ids1 = run(make_tracker(monkeypatch, reid_interval=1)[0])
    ids2 = run(make_tracker(monkeypatch, reid_interval=2)[0])
    assert ids2 == ids1
    # A keeps one id throughout, B keeps one id across its frame-7 gap.
    assert len({f["A"] for f in ids2}) == 1
    assert len({f["B"] for f in ids2[2:] if "B" in f}) == 1


@pytest.mark.parametrize("occlusion_aware_tracking", [False, True])
def test_track_born_on_off_frame_gets_feature_on_next_on_frame(
    monkeypatch, occlusion_aware_tracking
):
    tracker, reid = make_tracker(
        monkeypatch, reid_interval=2, occlusion_aware_tracking=occlusion_aware_tracking
    )
    sequence = list(probe_sequence(3))

    for i, people in enumerate(sequence[:2]):
        img, detections = render(people)
        tracker.update(detections, img)
    # Frame 1 was OFF: B was born there without a feature, A kept its own.
    assert reid.frames == [0]
    b = track_for(tracker, detections[1].bbox_xyxy)
    a = track_for(tracker, detections[0].bbox_xyxy)
    assert b is not None and b.smooth_feat is None
    assert a.smooth_feat is not None

    # Frame 2 is ON: no crash, B matched on IoU and now has an embedding.
    img, detections = render(sequence[2])
    tracker.update(detections, img)
    assert reid.frames == [0, 2]
    assert b.smooth_feat is not None
    assert detections[1].track_id == b.id


# ----------------------------------------------------------------------
# the guard itself


def make_stracks(feats):
    tracks = []
    for i, feat in enumerate(feats):
        det = np.array([10.0 * i, 0, 10.0 * i + 5, 20, 0.9, 0, i], dtype=np.float32)
        tracks.append(STrack(det, None if feat is None else np.asarray(feat, np.float32)))
    return tracks


def test_guard_neutral_rows_and_stock_rows():
    install_featureless_guard(botsort_module)
    guarded = botsort_module.embedding_distance
    tracks = make_stracks([[1, 0, 0], None, [0, 1, 1]])
    dets = make_stracks([[1, 1, 0], [0, 0, 1]])

    cost = guarded(tracks, dets)

    assert cost.shape == (3, 2)
    assert np.all(cost[1] == NO_APPEARANCE_COST)
    np.testing.assert_array_equal(cost[[0, 2]], stock_distance([tracks[0], tracks[2]], dets))
    assert np.isfinite(cost).all()
    # All rows featured -> exactly the stock result.
    featured = [tracks[0], tracks[2]]
    np.testing.assert_array_equal(guarded(featured, dets), stock_distance(featured, dets))


def test_guard_is_installed_once_regardless_of_order(monkeypatch):
    for interval in (2, 1, 2, 1):
        make_tracker(monkeypatch, reid_interval=interval)
    install_featureless_guard(botsort_module)

    guarded = botsort_module.embedding_distance
    assert guarded.featureless_guard_stock is stock_distance
    assert not hasattr(stock_distance, "featureless_guard_stock")

    # Creating trackers in a different order gives the same tracking.
    first, _ = make_tracker(monkeypatch, reid_interval=1)
    ids_first = run(first)
    make_tracker(monkeypatch, reid_interval=2)
    again, _ = make_tracker(monkeypatch, reid_interval=1)
    assert run(again) == ids_first
