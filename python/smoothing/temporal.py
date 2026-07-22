"""Per-person temporal smoothing for quaternions and root positions.

Simple EMA in position space and SLERP-based EMA in quaternion space, keyed
by ``person_id``. Sits between the tracker and the packet builder so what
Unreal receives is already denoised.

Rules:
  * Rotations stay as quaternions throughout — no Euler (CLAUDE.md §4.2).
  * Per-person state is retired after ``max_missing_frames`` so a stale
    filter doesn't warp a re-appearing person.
  * ``alpha = 0`` (or 1) short-circuits smoothing → useful for A/B perf tests
    with debug OFF (CLAUDE.md §8.2, §10).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from config.bone_mapping import NUM_SMPL_JOINTS


@dataclass
class _PersonState:
    quats: np.ndarray          # (24, 4) xyzw, previous smoothed value
    root: np.ndarray           # (3,) previous smoothed value
    last_seen_frame: int


def _slerp_pair(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation of two xyzw quaternions.

    Handles the double-cover: if ``q0 . q1 < 0``, flip ``q1`` sign so we take
    the shorter arc. Falls back to linear interpolation when the two are
    nearly identical (numerically stable).
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        # Near-identical — linear interp is stable and skips the sin()
        result = q0 + t * (q1 - q0)
        n = np.linalg.norm(result)
        return result / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


def _slerp_batch(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Row-wise SLERP for (N, 4) quaternion arrays."""
    if q0.shape != q1.shape:
        raise ValueError(f"Shape mismatch: {q0.shape} vs {q1.shape}")
    out = np.empty_like(q0, dtype=np.float64)
    for i in range(q0.shape[0]):
        out[i] = _slerp_pair(q0[i], q1[i], t)
    return out


class TemporalSmoother:
    """Per-person EMA + SLERP smoother.

    Args:
        alpha: [0, 1] weight of the *new* sample. 1.0 == no smoothing.
        max_missing_frames: retire per-person state after this many frames w/o updates.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        max_missing_frames: int = 15,
        root_alpha: Optional[float] = None,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if root_alpha is not None and not 0.0 <= root_alpha <= 1.0:
            raise ValueError(f"root_alpha must be in [0, 1], got {root_alpha}")
        self.alpha = float(alpha)
        # Separate root smoothing: monocular depth is noisier than pose (bbox
        # size shifts when limbs move), so root usually wants a smaller alpha
        # than pose. None = same as alpha (legacy behaviour).
        self.root_alpha = float(root_alpha) if root_alpha is not None else self.alpha
        self.max_missing_frames = int(max_missing_frames)
        self._states: Dict[int, _PersonState] = {}
        self._frame_idx: int = -1

    def step(
        self,
        person_id: int,
        quats_xyzw: np.ndarray,
        root: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the smoothed (quats, root) for one person.

        Call once per frame per person. Order across persons is irrelevant.
        """
        # Joint count is inferred from input (24 for SMPL, 55 for SMPL-X, etc.).
        q_new = np.asarray(quats_xyzw, dtype=np.float64).reshape(-1, 4)
        r_new = np.asarray(root, dtype=np.float64).reshape(3)

        state = self._states.get(person_id)
        if state is None:
            q_out = q_new.copy()
            r_out = r_new.copy()
        else:
            q_out = q_new.copy() if self.alpha >= 1.0 else _slerp_batch(state.quats, q_new, self.alpha)
            if self.root_alpha >= 1.0:
                r_out = r_new.copy()
            else:
                r_out = (1.0 - self.root_alpha) * state.root + self.root_alpha * r_new

        self._states[person_id] = _PersonState(
            quats=q_out.copy(),
            root=r_out.copy(),
            last_seen_frame=self._frame_idx,
        )
        return q_out, r_out

    def advance_frame(self) -> None:
        """Advance the internal frame counter and retire stale per-person states.

        Call once at the top of every frame, before ``step`` for each person.
        """
        self._frame_idx += 1
        stale = [
            pid
            for pid, s in self._states.items()
            if self._frame_idx - s.last_seen_frame > self.max_missing_frames
        ]
        for pid in stale:
            del self._states[pid]

    def reset(self) -> None:
        self._states.clear()
        self._frame_idx = -1

    @property
    def active_ids(self) -> list[int]:
        return list(self._states.keys())
