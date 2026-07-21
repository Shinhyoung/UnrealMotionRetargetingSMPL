import numpy as np
import pytest

from config.bone_mapping import NUM_SMPL_JOINTS
from smoothing.temporal import TemporalSmoother, _slerp_pair
from transform.rotation import axis_angle_to_quat_xyzw


def _identity_quats():
    q = np.zeros((NUM_SMPL_JOINTS, 4))
    q[:, 3] = 1.0
    return q


def test_slerp_endpoints():
    q0 = axis_angle_to_quat_xyzw(np.array([0.0, 0.0, 0.0]))
    q1 = axis_angle_to_quat_xyzw(np.array([np.pi / 2, 0.0, 0.0]))
    assert np.allclose(_slerp_pair(q0, q1, 0.0), q0)
    # SLERP double-cover: result can be ±q1. Compare up to sign.
    r = _slerp_pair(q0, q1, 1.0)
    diff = min(np.linalg.norm(r - q1), np.linalg.norm(r + q1))
    assert diff < 1e-9


def test_slerp_double_cover_takes_shorter_arc():
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = np.array([0.0, 0.0, 0.0, -1.0])  # same rotation as q0
    r = _slerp_pair(q0, q1, 0.5)
    # Should stay near identity, not flip to some intermediate.
    assert abs(abs(r[3]) - 1.0) < 1e-9


def test_alpha_one_is_passthrough():
    sm = TemporalSmoother(alpha=1.0)
    q_new = _identity_quats()
    root_new = np.array([0.5, 0.0, 3.0])
    sm.advance_frame()
    q_out, r_out = sm.step(1, q_new, root_new)
    assert np.allclose(q_out, q_new)
    assert np.allclose(r_out, root_new)


def test_position_ema_moves_toward_new_sample():
    sm = TemporalSmoother(alpha=0.5)
    sm.advance_frame()
    sm.step(1, _identity_quats(), np.array([0.0, 0.0, 0.0]))
    sm.advance_frame()
    q_out, r_out = sm.step(1, _identity_quats(), np.array([1.0, 0.0, 0.0]))
    # EMA with alpha=0.5 → halfway
    assert np.allclose(r_out, [0.5, 0.0, 0.0])


def test_new_person_bypasses_history():
    sm = TemporalSmoother(alpha=0.5)
    sm.advance_frame()
    sm.step(1, _identity_quats(), np.array([10.0, 0.0, 0.0]))
    # Second person has no history — should be returned as-is
    q_out, r_out = sm.step(2, _identity_quats(), np.array([0.0, 5.0, 0.0]))
    assert np.allclose(r_out, [0.0, 5.0, 0.0])


def test_stale_state_retired():
    sm = TemporalSmoother(alpha=0.5, max_missing_frames=2)
    sm.advance_frame()
    sm.step(1, _identity_quats(), np.array([0.0, 0.0, 0.0]))
    # Advance 3 frames without seeing pid=1
    for _ in range(3):
        sm.advance_frame()
    assert 1 not in sm.active_ids


def test_alpha_out_of_range_raises():
    with pytest.raises(ValueError):
        TemporalSmoother(alpha=-0.1)
    with pytest.raises(ValueError):
        TemporalSmoother(alpha=1.5)


def test_smoothed_quats_stay_unit_norm():
    sm = TemporalSmoother(alpha=0.3)
    rng = np.random.default_rng(0)
    prev = _identity_quats()
    for i in range(10):
        sm.advance_frame()
        aa = rng.normal(0, 0.5, (NUM_SMPL_JOINTS, 3))
        q = axis_angle_to_quat_xyzw(aa)
        q_out, _ = sm.step(1, q, np.zeros(3))
        norms = np.linalg.norm(q_out, axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-6)
        prev = q_out
