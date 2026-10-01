"""Tests for the published ``person_id`` contract.

``person_id >= 0`` means the detection has a persistent BoT-SORT track id.
``person_id == -1`` means it has no persistent identity right now. The
frame-local detection index (0, 1, 2, ...) is NEVER published as an id, so
``person_id == 1`` can only ever mean "tracker id 1", never "second detection
in the frame". Untracked people are still published in full.

The PersonDetection tests are pure numpy. The tracker tests drive the real
SkeletonTracker + stock BoT-SORT (no ReID) and need ``boxmot``; the message
tests need ``patrolknight_msgs`` -- both exist only inside the Docker image
and are skipped otherwise::

    python3 -m pytest test/test_person_id.py -v

Coverage:

    A  detection matched to a confirmed track   -> person_id == tracker id
    B  new person whose track is not confirmed  -> -1, then the id once
       confirmed, never back-filled into the earlier frame
    C  detection no track is ever made for      -> -1 on every frame
    D  tracking disabled                        -> -1
    E  three untracked detections               -> [-1, -1, -1], not [0, 1, 2]
    F  real tracker ids 0 and 1 are preserved as-is
    G  untracked people are still published with bbox/joints/position
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skeleton_detection.inference.depth_estimation import CameraPoint  # noqa: E402
from skeleton_detection.inference.rtmo_inference import (  # noqa: E402
    UNTRACKED_PERSON_ID,
    PersonDetection,
)

IMG = np.zeros((480, 848, 3), dtype=np.uint8)


def make_detection(detection_index, cx=200.0, cy=300.0, score=0.9, track_id=None):
    w, h = 60.0, 160.0
    return PersonDetection(
        bbox_xyxy=np.array(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32
        ),
        score=score,
        keypoints_xy=np.full((17, 2), [cx, cy], dtype=np.float32),
        keypoint_scores=np.full(17, 0.8, dtype=np.float32),
        detection_index=detection_index,
        track_id=track_id,
    )


def make_tracker():
    pytest.importorskip("boxmot", reason="boxmot only exists inside the Docker image")
    from boxmot.trackers.botsort.basetrack import BaseTrack

    from skeleton_detection.inference.person_tracking import SkeletonTracker

    BaseTrack.clear_count()
    return SkeletonTracker(with_reid=False, device="cpu")


def ids(detections):
    return [d.person_id for d in detections]


# ----------------------------------------------------------------------
# PersonDetection.person_id itself
# ----------------------------------------------------------------------
def test_untracked_person_id_is_minus_one():
    assert UNTRACKED_PERSON_ID == -1
    assert make_detection(detection_index=3).person_id == -1


# ----------------------------------------------------------------------
# E - multiple untracked detections are all -1
# ----------------------------------------------------------------------
def test_e_multiple_untracked_detections_are_all_minus_one():
    detections = [make_detection(i, cx=100.0 + 200 * i) for i in range(3)]
    assert ids(detections) == [-1, -1, -1]


# ----------------------------------------------------------------------
# F - real tracker ids 0 and 1 are preserved, whatever the detection index
# ----------------------------------------------------------------------
def test_f_tracker_ids_zero_and_one_are_preserved():
    # Deliberately crossed: detection 0 carries track 1, detection 1 track 0.
    detections = [
        make_detection(0, track_id=1),
        make_detection(1, track_id=0),
        make_detection(2),
    ]
    assert ids(detections) == [1, 0, -1]


# ----------------------------------------------------------------------
# A/B - confirmed track -> tracker id; unconfirmed -> -1, no back-fill
# ----------------------------------------------------------------------
def test_a_b_confirmed_track_id_and_unconfirmed_minus_one():
    tracker = make_tracker()

    # Frame 1: BoT-SORT activates first-frame tracks immediately.
    frame1 = tracker.update([make_detection(0, cx=200.0)], IMG)
    (person_a,) = ids(frame1)
    assert person_a >= 0

    # Frame 2: a new person B appears far from A. Their track exists but is
    # not confirmed yet, so BoT-SORT does not emit it.
    frame2 = tracker.update(
        [make_detection(0, cx=200.0), make_detection(1, cx=600.0)], IMG
    )
    assert ids(frame2) == [person_a, -1]

    # Frame 3: B's track is confirmed and now carries its tracker id.
    frame3 = tracker.update(
        [make_detection(0, cx=200.0), make_detection(1, cx=600.0)], IMG
    )
    person_a_again, person_b = ids(frame3)
    assert person_a_again == person_a
    assert person_b >= 0 and person_b != person_a

    # Frame 4: B keeps the same id.
    frame4 = tracker.update(
        [make_detection(0, cx=200.0), make_detection(1, cx=600.0)], IMG
    )
    assert ids(frame4) == [person_a, person_b]

    # The earlier unconfirmed frame is never rewritten.
    assert frame2[1].person_id == -1


# ----------------------------------------------------------------------
# C - a detection the tracker never associates with a track stays -1
# ----------------------------------------------------------------------
def test_c_unmatched_detection_is_minus_one():
    tracker = make_tracker()
    tracker.update([make_detection(0, cx=200.0)], IMG)

    # score 0.55: above track_high_thresh (0.5) so it is associated if it can
    # be, below new_track_thresh (0.6) so an unmatched one never starts a
    # track. Placed far from A, it matches nothing on every frame.
    for _ in range(5):
        detections = tracker.update(
            [make_detection(0, cx=200.0), make_detection(1, cx=650.0, score=0.55)],
            IMG,
        )
        assert detections[0].person_id >= 0
        assert detections[1].person_id == -1
        # A "1" here would have been the old frame-local-index fallback.
        assert detections[1].person_id != detections[1].detection_index


# ----------------------------------------------------------------------
# D/E/G - through the message builder
# ----------------------------------------------------------------------
def test_d_tracking_disabled_publishes_minus_one_and_keeps_the_person():
    pytest.importorskip("patrolknight_msgs.msg")
    from std_msgs.msg import Header

    from skeleton_detection.output.message_builder import build_skeleton_frame

    # Tracking disabled: the node never calls a tracker, so no track_id is set.
    detection = make_detection(0, cx=200.0)
    detection.position = CameraPoint(0.5, -0.25, 3.0)
    frame = build_skeleton_frame(Header(), 7, [detection])

    (person,) = frame.persons
    assert person.person_id == -1
    # G - everything else is still published normally.
    assert list(person.bbox) == pytest.approx([170.0, 220.0, 60.0, 160.0])
    assert len(person.joints) == 51
    assert person.score == pytest.approx(0.9)
    assert person.position.z == pytest.approx(3.0)
    assert math.isfinite(person.depth)


def test_e_three_untracked_messages_are_not_zero_one_two():
    pytest.importorskip("patrolknight_msgs.msg")
    from std_msgs.msg import Header

    from skeleton_detection.output.message_builder import build_skeleton_frame

    detections = [make_detection(i, cx=100.0 + 200 * i) for i in range(3)]
    frame = build_skeleton_frame(Header(), 0, detections)

    assert len(frame.persons) == 3
    assert [p.person_id for p in frame.persons] == [-1, -1, -1]


def test_f_tracker_ids_zero_and_one_reach_the_message():
    pytest.importorskip("patrolknight_msgs.msg")
    from std_msgs.msg import Header

    from skeleton_detection.output.message_builder import build_skeleton_frame

    detections = [
        make_detection(0, track_id=1),
        make_detection(1, cx=500.0, track_id=0),
        make_detection(2, cx=700.0),
    ]
    frame = build_skeleton_frame(Header(), 0, detections)
    assert [p.person_id for p in frame.persons] == [1, 0, -1]
