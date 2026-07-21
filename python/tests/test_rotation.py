import numpy as np
import pytest

from transform.rotation import (
    axis_angle_to_quat_xyzw,
    matrix_to_quat_xyzw,
    normalize_quat,
    quat_multiply_xyzw,
    quat_xyzw_to_matrix,
)


def test_identity_axis_angle_is_identity_quat():
    q = axis_angle_to_quat_xyzw(np.zeros(3))
    assert np.allclose(q, [0.0, 0.0, 0.0, 1.0])


def test_pi_rotation_about_x():
    q = axis_angle_to_quat_xyzw(np.array([np.pi, 0.0, 0.0]))
    # Rotating +Y by 180° about X gives -Y.
    m = quat_xyzw_to_matrix(q)
    y = m @ np.array([0.0, 1.0, 0.0])
    assert np.allclose(y, [0.0, -1.0, 0.0], atol=1e-12)


def test_batch_axis_angle_conversion_shape():
    aa = np.zeros((24, 3))
    aa[3] = [np.pi / 2, 0, 0]
    q = axis_angle_to_quat_xyzw(aa)
    assert q.shape == (24, 4)
    # Norm ~1 everywhere.
    assert np.allclose(np.linalg.norm(q, axis=-1), 1.0)


def test_quat_matrix_roundtrip():
    rng = np.random.default_rng(0)
    aa = rng.normal(0, 1, (16, 3))
    q = axis_angle_to_quat_xyzw(aa)
    m = quat_xyzw_to_matrix(q)
    q2 = matrix_to_quat_xyzw(m)
    # Quaternion double-cover: q and -q are the same rotation.
    diff = np.minimum(np.linalg.norm(q - q2, axis=-1), np.linalg.norm(q + q2, axis=-1))
    assert np.all(diff < 1e-9)


def test_quat_multiply_identity():
    q = axis_angle_to_quat_xyzw(np.array([0.3, -0.7, 1.1]))
    ident = np.array([0.0, 0.0, 0.0, 1.0])
    out = quat_multiply_xyzw(q, ident)
    assert np.allclose(out, q)
    out2 = quat_multiply_xyzw(ident, q)
    assert np.allclose(out2, q)


def test_quat_multiply_matches_matrix_composition():
    q1 = axis_angle_to_quat_xyzw(np.array([0.5, 0.0, 0.0]))
    q2 = axis_angle_to_quat_xyzw(np.array([0.0, 0.7, 0.0]))
    q_prod = quat_multiply_xyzw(q1, q2)
    m_prod = quat_xyzw_to_matrix(q_prod)
    m_expected = quat_xyzw_to_matrix(q1) @ quat_xyzw_to_matrix(q2)
    assert np.allclose(m_prod, m_expected, atol=1e-12)


def test_normalize_zero_quat_safe():
    q = np.array([0.0, 0.0, 0.0, 0.0])
    out = normalize_quat(q)
    assert np.all(np.isfinite(out))
