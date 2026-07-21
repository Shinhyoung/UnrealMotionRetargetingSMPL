import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from config.bone_mapping import NUM_SMPL_JOINTS
from transform.bone_correction import (
    apply_bone_correction,
    identity_correction,
    load_correction,
)
from transform.rotation import normalize_quat, quat_multiply_xyzw


def test_identity_correction_shape_and_values():
    parent, own = identity_correction()
    assert parent.shape == (NUM_SMPL_JOINTS, 4)
    assert own.shape == (NUM_SMPL_JOINTS, 4)
    assert np.allclose(parent[:, 3], 1.0) and np.allclose(parent[:, :3], 0.0)
    assert np.allclose(own[:, 3], 1.0) and np.allclose(own[:, :3], 0.0)


def test_apply_identity_correction_is_noop():
    rng = np.random.default_rng(0)
    aa = rng.normal(0.0, 0.3, size=(NUM_SMPL_JOINTS, 3))
    q = Rotation.from_rotvec(aa).as_quat()
    parent, own = identity_correction()
    out = apply_bone_correction(q, parent, own)
    assert np.allclose(out, q, atol=1e-9)


def test_apply_correction_matches_manual_multiplication():
    """Q_send[i] = parent_inv[i] · q[i] · own[i] must match direct multiply."""
    rng = np.random.default_rng(1)
    q = normalize_quat(rng.normal(size=(NUM_SMPL_JOINTS, 4)))
    parent = normalize_quat(rng.normal(size=(NUM_SMPL_JOINTS, 4)))
    own = normalize_quat(rng.normal(size=(NUM_SMPL_JOINTS, 4)))

    out = apply_bone_correction(q, parent, own)
    expected = normalize_quat(quat_multiply_xyzw(quat_multiply_xyzw(parent, q), own))
    assert np.allclose(out, expected, atol=1e-10)


def test_rest_pose_yields_ue_local_rest():
    """At R_smpl = I (identity conjugated), Q_send == parent_inv · own == L_ref.

    Sanity: with correction loaded, identity input should preserve UE
    mannequin's rest local rotation (the mannequin holds T-pose).
    """
    # Fake W_ref: parent has some rotation A, child has A*B (so L_ref = B).
    A = normalize_quat(Rotation.from_rotvec([0.5, 0.2, 0.1]).as_quat())
    B = normalize_quat(Rotation.from_rotvec([0.1, -0.3, 0.4]).as_quat())
    AB = normalize_quat(quat_multiply_xyzw(A, B))

    parent = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (NUM_SMPL_JOINTS, 1))
    own = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (NUM_SMPL_JOINTS, 1))
    parent[5] = np.array([-A[0], -A[1], -A[2], A[3]])   # inverse(A)
    own[5] = AB

    q_identity = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (NUM_SMPL_JOINTS, 1))
    out = apply_bone_correction(q_identity, parent, own)
    # Expected local rest for joint 5 = A^-1 · AB = B.
    assert np.allclose(out[5], B, atol=1e-9)
    # Other joints (identity parent/own) stay identity.
    assert np.allclose(out[0], [0.0, 0.0, 0.0, 1.0], atol=1e-9)


def test_load_correction_empty_path_returns_identity():
    parent, own = load_correction("")
    assert np.allclose(parent[:, 3], 1.0)
    assert np.allclose(own[:, 3], 1.0)


def test_load_correction_missing_file_returns_identity():
    parent, own = load_correction("nowhere.npz")
    assert np.allclose(parent[:, 3], 1.0)
    assert np.allclose(own[:, 3], 1.0)


def test_load_correction_bad_shape(tmp_path):
    bad = tmp_path / "bad.npz"
    np.savez(bad, parent_world_inv=np.zeros((10, 4)), own_world=np.zeros((10, 4)))
    with pytest.raises(ValueError):
        load_correction(str(bad))


def test_load_correction_missing_keys(tmp_path):
    bad = tmp_path / "bad.npz"
    np.savez(bad, foo=np.zeros((NUM_SMPL_JOINTS, 4)))
    with pytest.raises(ValueError):
        load_correction(str(bad))


def test_load_correction_round_trip(tmp_path):
    rng = np.random.default_rng(2)
    parent_in = normalize_quat(rng.normal(size=(NUM_SMPL_JOINTS, 4)))
    own_in = normalize_quat(rng.normal(size=(NUM_SMPL_JOINTS, 4)))
    path = tmp_path / "corr.npz"
    np.savez(path, parent_world_inv=parent_in, own_world=own_in)
    parent, own = load_correction(str(path))
    assert np.allclose(parent, parent_in, atol=1e-12)
    assert np.allclose(own, own_in, atol=1e-12)
