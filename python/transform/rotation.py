"""Rotation conversions. Quaternion order is **xyzw** everywhere (scipy default,
matches UE ``FQuat`` component order). Never convert to Euler (CLAUDE.md §4.2).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def axis_angle_to_quat_xyzw(axis_angles: np.ndarray) -> np.ndarray:
    """Convert axis-angle vectors to unit quaternions in xyzw order.

    Args:
        axis_angles: (..., 3) axis-angle. Magnitude = angle in radians.

    Returns:
        (..., 4) quaternions, xyzw, unit norm.
    """
    aa = np.asarray(axis_angles, dtype=np.float64)
    single = aa.ndim == 1
    if single:
        aa = aa.reshape(1, 3)
    q = Rotation.from_rotvec(aa).as_quat()
    return q[0] if single else q


def quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Quaternion (xyzw) → 3x3 rotation matrix."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    single = q.ndim == 1
    if single:
        q = q.reshape(1, 4)
    m = Rotation.from_quat(q).as_matrix()
    return m[0] if single else m


def matrix_to_quat_xyzw(rotmat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → quaternion (xyzw)."""
    m = np.asarray(rotmat, dtype=np.float64)
    single = m.ndim == 2
    if single:
        m = m.reshape(1, 3, 3)
    q = Rotation.from_matrix(m).as_quat()
    return q[0] if single else q


def quat_multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b for xyzw-ordered quaternions.

    Result quaternion applies ``b`` first then ``a`` (same as matrix multiply).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return np.stack([x, y, z, w], axis=-1)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize xyzw quaternion(s) to unit length."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return q / n
