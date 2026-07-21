import numpy as np
import pytest

from config.bone_mapping import NUM_SMPL_JOINTS
from transform.rest_pose import (
    apply_rest_pose_offset,
    identity_offsets,
    load_offsets,
)


def test_identity_offsets_shape_and_values():
    ident = identity_offsets()
    assert ident.shape == (NUM_SMPL_JOINTS, 4)
    assert np.allclose(ident[:, 3], 1.0)
    assert np.allclose(ident[:, :3], 0.0)


def test_apply_identity_offset_is_noop():
    q = np.tile(np.array([0.1, 0.2, 0.3, np.sqrt(1 - 0.14)]), (NUM_SMPL_JOINTS, 1))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    out = apply_rest_pose_offset(q, identity_offsets())
    assert np.allclose(out, q, atol=1e-9)


def test_load_offsets_empty_path_returns_identity():
    out = load_offsets("")
    assert out.shape == (NUM_SMPL_JOINTS, 4)
    assert np.allclose(out[:, 3], 1.0)


def test_load_offsets_missing_file_returns_identity():
    out = load_offsets("does_not_exist.npy")
    assert np.allclose(out[:, 3], 1.0)


def test_load_offsets_bad_shape(tmp_path):
    bad = tmp_path / "bad.npy"
    np.save(bad, np.zeros((10, 4)))
    with pytest.raises(ValueError):
        load_offsets(str(bad))


def test_shape_mismatch_raises():
    q = identity_offsets()
    bad = np.zeros((5, 4))
    with pytest.raises(ValueError):
        apply_rest_pose_offset(q, bad)
