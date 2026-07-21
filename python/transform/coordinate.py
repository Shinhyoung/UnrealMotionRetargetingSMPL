"""Coordinate-system swizzling: SMPL (right-hand, Y-up) → Unreal (left-hand, Z-up, X-forward).

Single source of truth for axis mapping. Do NOT reimplement this anywhere else
(CLAUDE.md §4.3).

Axis mapping used here (person facing camera along SMPL -Z):

    UE +X (forward) = -SMPL Z
    UE +Y (right)   = +SMPL X
    UE +Z (up)      = +SMPL Y

That yields a 3x3 change-of-basis matrix M with det(M) = -1
(chirality flip: right-hand → left-hand). Applying ``M R M^T`` to a rotation
matrix re-expresses the same physical rotation in the UE basis (both bases
still give proper rotations because det(M R M^T) = det(R) = +1).

If the integration reveals a mirror flip (a common symptom is left/right
swapped bones), tweak the sign of one row here and add a unit test — do NOT
sprinkle sign flips across the pipeline.
"""
from __future__ import annotations

import numpy as np

from transform.rotation import matrix_to_quat_xyzw, quat_xyzw_to_matrix

# Columns are SMPL basis vectors expressed in UE frame:
#   [smpl_x_in_ue | smpl_y_in_ue | smpl_z_in_ue]
# Equivalently: v_ue = SMPL_TO_UE_BASIS @ v_smpl
SMPL_TO_UE_BASIS = np.array(
    [
        [0.0, 0.0, -1.0],   # ue_x = -smpl_z
        [1.0, 0.0, 0.0],    # ue_y =  smpl_x
        [0.0, 1.0, 0.0],    # ue_z =  smpl_y
    ],
    dtype=np.float64,
)


def smpl_pos_to_ue_pos(pos: np.ndarray, scale: float = 100.0) -> np.ndarray:
    """Convert a 3D position from SMPL space to UE space.

    Args:
        pos: (..., 3) position in SMPL space (meters).
        scale: multiplicative unit scale. Default 100 (meters → centimeters).

    Returns:
        (..., 3) position in UE space, same shape.
    """
    p = np.asarray(pos, dtype=np.float64)
    return (p @ SMPL_TO_UE_BASIS.T) * scale


def smpl_quat_to_ue_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    """Re-express an xyzw quaternion from SMPL basis to UE basis.

    Works for any (..., 4) shape. Handles rotation matrix conjugation
    ``R' = M R M^T`` and returns a unit quaternion.
    """
    q = np.asarray(quat_xyzw, dtype=np.float64)
    single = q.ndim == 1
    if single:
        q = q.reshape(1, 4)
    R = quat_xyzw_to_matrix(q)                          # (N, 3, 3)
    R_ue = SMPL_TO_UE_BASIS @ R @ SMPL_TO_UE_BASIS.T    # broadcast over N
    q_ue = matrix_to_quat_xyzw(R_ue)
    return q_ue[0] if single else q_ue


def smpl_quat_to_ue_quat_chirality_preserving(quat_xyzw: np.ndarray) -> np.ndarray:
    """Chirality-preserving variant of :func:`smpl_quat_to_ue_quat`.

    Rationale
    ---------
    ``smpl_quat_to_ue_quat`` follows the strict axial-vector rule: for a
    reflection ``M`` (det = -1), the rotation axis of the conjugated rotation
    is ``-M n`` rather than ``M n``. Physically this **flips the sign of the
    rotation angle**: a SMPL "raise right arm forward" (+θ about person's
    right axis) becomes "raise right arm backward" (-θ) in UE.

    For retargeting to a mirrored target skeleton the axis should be treated
    as a polar vector (``+M n``), which negates the vector part of the
    conjugated quaternion:

        q_preserved = (+M · q_xyz_conjugated, q_w) = (-q_xyz_conjugated_matrix_form, q_w)

    In practice this is the quaternion whose vector part is negated relative
    to the matrix-conjugation result. Use only for joint-local rotations
    where the target skeleton (UE mannequin) is a mirror of the source (SMPL);
    positions still use :func:`smpl_pos_to_ue_pos` (mirrored world).
    """
    q_conj = smpl_quat_to_ue_quat(quat_xyzw)
    q_conj = np.asarray(q_conj, dtype=np.float64).copy()
    q_conj[..., :3] *= -1.0
    return q_conj


# ---- Camera (OpenCV, Y-down) → SMPL Y-up bridge -----------------------------
#
# Most HMR-family models (SAT-HMR included) emit their `global_orient` and
# `transl` in **OpenCV camera coordinates** (X-right, Y-DOWN, Z-forward). Our
# `transform/coordinate.py` treats the pipeline's "SMPL" side as **Y-up**, so
# we need a one-shot pre-transform before the SMPL→UE basis change.
#
# The mapping is a reflection F = diag(1, -1, 1) (chirality flip). Under it:
#   * position: (x, y, z) → (x, -y, z)
#   * rotation matrix: R' = F R F  (still a proper rotation, det=+1)
#   * axis-angle (a_x, a_y, a_z): → (-a_x, a_y, -a_z)
#   * quaternion (x, y, z, w):    → (-x, y, -z, w)


def opencv_cam_yflip_axis_angle(axis_angle: np.ndarray) -> np.ndarray:
    """Reflect axis-angle rotations from OpenCV camera (Y-down) to SMPL Y-up."""
    aa = np.asarray(axis_angle, dtype=np.float64).copy()
    aa[..., 0] *= -1
    aa[..., 2] *= -1
    return aa


def opencv_cam_yflip_position(pos: np.ndarray) -> np.ndarray:
    """Reflect a 3D position from OpenCV camera (Y-down) to SMPL Y-up."""
    p = np.asarray(pos, dtype=np.float64).copy()
    p[..., 1] *= -1
    return p
