"""Frame-to-frame person_id tracker.

SAT-HMR is single-shot: detection order is not stable across frames, so we
must assign stable IDs ourselves (CLAUDE.md §4.4). Approach:

* For each frame, match previous tracks against new detections by 3D root
  distance (Hungarian assignment, ``scipy.optimize.linear_sum_assignment``).
* Reject matches whose distance exceeds ``max_distance``.
* Unmatched detections get a new ID.
* Tracks unmatched for more than ``max_missing_frames`` are retired.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class _Track:
    person_id: int
    position: np.ndarray  # (3,) most-recent root position, SMPL space
    last_seen_frame: int


class PersonTracker:
    def __init__(self, max_distance: float = 1.0, max_missing_frames: int = 15):
        self.max_distance = float(max_distance)
        self.max_missing_frames = int(max_missing_frames)
        self._tracks: dict[int, _Track] = {}
        self._next_id: int = 1
        self._frame_idx: int = -1

    def update(self, detections: Sequence) -> List:
        """Assign ``person_id`` on each detection (mutates in-place, returns the list).

        Each detection must expose a ``root_position`` attribute or key.
        """
        self._frame_idx += 1
        self._retire_stale()

        if not detections:
            return list(detections)

        det_positions = np.array(
            [np.asarray(_get_root(d), dtype=np.float64) for d in detections]
        )

        active_ids = list(self._tracks.keys())
        if active_ids:
            track_positions = np.array(
                [self._tracks[tid].position for tid in active_ids]
            )
            # cost[i, j] = distance from track i to detection j
            diff = track_positions[:, None, :] - det_positions[None, :, :]
            cost = np.linalg.norm(diff, axis=2)

            # Infeasible entries: cost > max_distance. Assign huge cost so Hungarian
            # avoids them; then filter out after.
            big = self.max_distance * 1000.0 + 1e6
            cost_h = np.where(cost > self.max_distance, big, cost)
            row_ind, col_ind = linear_sum_assignment(cost_h)

            assigned_dets = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= self.max_distance:
                    tid = active_ids[r]
                    _set_person_id(detections[c], tid)
                    self._tracks[tid].position = det_positions[c]
                    self._tracks[tid].last_seen_frame = self._frame_idx
                    assigned_dets.add(c)

            for j, det in enumerate(detections):
                if j not in assigned_dets:
                    self._spawn_track(det, det_positions[j])
        else:
            for j, det in enumerate(detections):
                self._spawn_track(det, det_positions[j])

        return list(detections)

    def _spawn_track(self, detection, position: np.ndarray) -> None:
        pid = self._next_id
        self._next_id += 1
        self._tracks[pid] = _Track(
            person_id=pid,
            position=position.copy(),
            last_seen_frame=self._frame_idx,
        )
        _set_person_id(detection, pid)

    def _retire_stale(self) -> None:
        stale = [
            tid
            for tid, t in self._tracks.items()
            if self._frame_idx - t.last_seen_frame > self.max_missing_frames
        ]
        for tid in stale:
            del self._tracks[tid]

    @property
    def active_ids(self) -> List[int]:
        return list(self._tracks.keys())


def _get_root(detection) -> np.ndarray:
    if hasattr(detection, "root_position"):
        return detection.root_position
    return detection["root_position"]


def _set_person_id(detection, pid: int) -> None:
    if hasattr(detection, "person_id"):
        detection.person_id = pid
    else:
        detection["person_id"] = pid
