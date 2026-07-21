import numpy as np

from transform.coordinate import (
    SMPL_TO_UE_BASIS,
    opencv_cam_yflip_axis_angle,
    opencv_cam_yflip_position,
    smpl_pos_to_ue_pos,
    smpl_quat_to_ue_quat,
)
from transform.rotation import axis_angle_to_quat_xyzw, quat_xyzw_to_matrix


def test_axis_mapping_from_basis_matrix():
    # SMPL +X (right) → UE +Y (right)
    assert np.allclose(SMPL_TO_UE_BASIS @ [1, 0, 0], [0, 1, 0])
    # SMPL +Y (up) → UE +Z (up)
    assert np.allclose(SMPL_TO_UE_BASIS @ [0, 1, 0], [0, 0, 1])
    # SMPL +Z (back) → UE -X (backward)
    assert np.allclose(SMPL_TO_UE_BASIS @ [0, 0, 1], [-1, 0, 0])


def test_basis_matrix_is_chirality_flip():
    # right-hand SMPL → left-hand UE requires det = -1
    assert np.isclose(np.linalg.det(SMPL_TO_UE_BASIS), -1.0)


def test_smpl_pos_to_ue_pos_scale_and_axis():
    # 1.5 m up in SMPL → 150 cm along UE +Z with default scale=100
    ue = smpl_pos_to_ue_pos(np.array([0.0, 1.5, 0.0]))
    assert np.allclose(ue, [0.0, 0.0, 150.0])


def test_smpl_pos_batch_shape_preserved():
    p = np.zeros((5, 3))
    p[:, 1] = 1.0
    out = smpl_pos_to_ue_pos(p, scale=1.0)
    assert out.shape == (5, 3)
    assert np.allclose(out[:, 2], 1.0)


def test_identity_quat_preserved_through_basis_change():
    q_id = np.array([0.0, 0.0, 0.0, 1.0])
    q_out = smpl_quat_to_ue_quat(q_id)
    assert np.allclose(q_out, [0.0, 0.0, 0.0, 1.0])


def test_yaw_rotation_physical_meaning():
    """Rotate SMPL person 90° CCW about SMPL +Y (up, right-hand rule).

    Their right hand (SMPL +X = UE +Y) should end up pointing forward
    (UE +X).
    """
    q_smpl = axis_angle_to_quat_xyzw(np.array([0.0, np.pi / 2, 0.0]))
    q_ue = smpl_quat_to_ue_quat(q_smpl)
    R_ue = quat_xyzw_to_matrix(q_ue)
    # UE 'right' vector = [0, 1, 0]. After the rotation, expected: UE forward = [1, 0, 0].
    rotated_right = R_ue @ np.array([0.0, 1.0, 0.0])
    assert np.allclose(rotated_right, [1.0, 0.0, 0.0], atol=1e-9)


def test_batch_quat_conversion_shape():
    aa = np.zeros((24, 3))
    aa[0] = [np.pi, 0, 0]
    q = axis_angle_to_quat_xyzw(aa)
    q_ue = smpl_quat_to_ue_quat(q)
    assert q_ue.shape == (24, 4)
    norms = np.linalg.norm(q_ue, axis=-1)
    assert np.allclose(norms, 1.0)


def test_opencv_yflip_position_flips_only_y():
    p = np.array([1.0, 2.0, 3.0])
    out = opencv_cam_yflip_position(p)
    assert np.allclose(out, [1.0, -2.0, 3.0])
    # Batched
    p_batch = np.arange(12, dtype=float).reshape(4, 3)
    out_batch = opencv_cam_yflip_position(p_batch)
    assert out_batch.shape == p_batch.shape
    assert np.allclose(out_batch[:, 1], -p_batch[:, 1])


def test_opencv_yflip_axis_angle_matches_matrix_reflection():
    """(-a_x, a_y, -a_z) must equal F R F for F = diag(1,-1,1)."""
    rng = np.random.default_rng(0)
    F = np.diag([1.0, -1.0, 1.0])
    for _ in range(5):
        aa = rng.normal(0, 1.0, 3)
        aa_flipped = opencv_cam_yflip_axis_angle(aa)
        R = quat_xyzw_to_matrix(axis_angle_to_quat_xyzw(aa))
        R_expected = F @ R @ F
        R_actual = quat_xyzw_to_matrix(axis_angle_to_quat_xyzw(aa_flipped))
        assert np.allclose(R_actual, R_expected, atol=1e-10)


def test_opencv_yflip_axis_angle_batched():
    aa = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, -0.25]])
    out = opencv_cam_yflip_axis_angle(aa)
    assert out.shape == (2, 3)
    assert np.allclose(out, [[-1.0, 2.0, -3.0], [1.0, 0.5, 0.25]])
