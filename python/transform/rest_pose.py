"""Rest-pose (T-pose) offset per joint.

Bridges SMPL rest pose vs UE mannequin rest pose (CLAUDE.md §4.5).
Real values must be measured empirically during integration; MVP defaults to identity.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from config.bone_mapping import NUM_SMPL_JOINTS
from transform.rotation import normalize_quat, quat_multiply_xyzw


IDENTITY_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def identity_offsets() -> np.ndarray:
    """Return (24, 4) array of identity quaternions."""
    q = np.zeros((NUM_SMPL_JOINTS, 4), dtype=np.float64)
    q[:, 3] = 1.0
    return q


def load_offsets(path: str) -> np.ndarray:
    """Load per-joint offset quaternions from a .npy file.

    Returns identity offsets if path is empty or file is missing.
    Shape must be (24, 4), xyzw ordered, unit quaternions.
    """
    if not path:
        return identity_offsets()
    p = Path(path)
    if not p.is_file():
        return identity_offsets()
    arr = np.load(str(p))
    if arr.shape != (NUM_SMPL_JOINTS, 4):
        raise ValueError(
            f"Offset file {p} has shape {arr.shape}, expected ({NUM_SMPL_JOINTS}, 4)."
        )
    return normalize_quat(arr.astype(np.float64))


def apply_rest_pose_offset(joint_quats_xyzw: np.ndarray, offsets_xyzw: np.ndarray) -> np.ndarray:
    """Apply per-joint offset: ``q_out[i] = offsets[i] * joint_quats[i]``.

    Multiplicative order: offset first (parent-frame realignment), then the incoming
    SMPL-derived local rotation.
    """
    q = np.asarray(joint_quats_xyzw, dtype=np.float64)
    o = np.asarray(offsets_xyzw, dtype=np.float64)
    if q.shape != o.shape:
        raise ValueError(f"Shape mismatch: joints={q.shape} offsets={o.shape}")
    return normalize_quat(quat_multiply_xyzw(o, q))
