"""Tracker unit tests.

Focus on the two failure modes CLAUDE.md §4.4 calls out:
  * IDs must be stable frame-to-frame for the same person.
  * IDs must NOT swap when two people cross close paths (Hungarian assignment
    picks the globally best match).
"""
import numpy as np

from tracking.person_tracker import PersonTracker


def _det(x, y, z):
    """Minimal detection stub: only root_position is required by the tracker."""
    class D: pass
    d = D()
    d.root_position = np.array([x, y, z], dtype=np.float64)
    d.person_id = None
    return d


def test_first_frame_assigns_new_ids():
    tr = PersonTracker(max_distance=1.0)
    dets = [_det(0, 0, 3), _det(1, 0, 3)]
    tr.update(dets)
    ids = [d.person_id for d in dets]
    assert None not in ids
    assert len(set(ids)) == 2


def test_id_stable_across_frames_when_person_barely_moves():
    tr = PersonTracker(max_distance=1.0)
    d0 = _det(0.0, 0.0, 3.0)
    tr.update([d0])
    pid0 = d0.person_id

    for step in range(1, 10):
        d = _det(0.01 * step, 0.0, 3.0)
        tr.update([d])
        assert d.person_id == pid0, f"ID changed at step {step}: {d.person_id} vs {pid0}"


def test_id_expires_after_max_missing_frames():
    tr = PersonTracker(max_distance=1.0, max_missing_frames=3)
    d0 = _det(0.0, 0.0, 3.0)
    tr.update([d0])
    pid0 = d0.person_id

    # 4 empty frames -> track retired.
    for _ in range(4):
        tr.update([])
    assert pid0 not in tr.active_ids

    d_new = _det(0.0, 0.0, 3.0)
    tr.update([d_new])
    assert d_new.person_id != pid0


def test_hungarian_avoids_id_swap_on_close_pass():
    """Two people move toward each other; assignment should pick the closer match."""
    tr = PersonTracker(max_distance=1.0)
    # Frame 0: A at x=-0.5, B at x=+0.5
    A = _det(-0.5, 0, 3.0)
    B = _det(+0.5, 0, 3.0)
    tr.update([A, B])
    pid_A, pid_B = A.person_id, B.person_id
    assert pid_A != pid_B

    # Frame 1: A moves to -0.2, B moves to +0.2. Order in detection list flipped
    # to make sure tracker doesn't just rely on ordering.
    A1 = _det(-0.2, 0, 3.0)
    B1 = _det(+0.2, 0, 3.0)
    tr.update([B1, A1])
    assert A1.person_id == pid_A
    assert B1.person_id == pid_B


def test_far_detection_gets_new_id_even_if_track_exists():
    tr = PersonTracker(max_distance=0.5)
    d0 = _det(0.0, 0.0, 3.0)
    tr.update([d0])
    pid0 = d0.person_id

    d_far = _det(5.0, 0.0, 3.0)  # too far to match under max_distance
    tr.update([d_far])
    assert d_far.person_id is not None
    assert d_far.person_id != pid0
