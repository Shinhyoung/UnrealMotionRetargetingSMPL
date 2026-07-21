"""End-to-end plumbing test — Mock detector → tracker → transform → packet.

Skips the real camera / UDP socket; just proves the modules compose.
"""
import numpy as np

from config.bone_mapping import NUM_SMPL_JOINTS
from detectors.sat_hmr_wrapper import MockSATHMRDetector
from network.packet import PER_PERSON_SIZE, build_packet, parse_packet
from tracking.person_tracker import PersonTracker
from transform.coordinate import smpl_pos_to_ue_pos, smpl_quat_to_ue_quat
from transform.rest_pose import apply_rest_pose_offset, identity_offsets
from transform.rotation import axis_angle_to_quat_xyzw


def test_mock_pipeline_produces_valid_packet():
    detector = MockSATHMRDetector(n_persons=3, seed=1)
    tracker = PersonTracker(max_distance=1.0)
    offsets = identity_offsets()

    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    last_ids: list[int] = []
    for frame_id in range(5):
        detections = detector.infer(dummy_frame)
        tracker.update(detections)

        payload = []
        for det in detections:
            aa = det.full_axis_angles()
            q_smpl = axis_angle_to_quat_xyzw(aa)
            q_ue = smpl_quat_to_ue_quat(q_smpl)
            q_ue = apply_rest_pose_offset(q_ue, offsets)
            root_ue = smpl_pos_to_ue_pos(det.root_position)
            payload.append((det.person_id, root_ue, q_ue))

        data = build_packet(frame_id, payload)
        assert len(data) == 5 + 3 * PER_PERSON_SIZE

        parsed_frame, parsed_persons = parse_packet(data)
        assert parsed_frame == frame_id
        assert len(parsed_persons) == 3

        # Track stability: IDs must be identical set from frame 1 onward.
        ids = sorted(p[0] for p in parsed_persons)
        if last_ids:
            assert ids == last_ids, "person_id set changed across frames — tracker unstable"
        last_ids = ids

        # Quats unit norm.
        for _, _, quats in parsed_persons:
            assert quats.shape == (NUM_SMPL_JOINTS, 4)
            norms = np.linalg.norm(quats, axis=-1)
            assert np.allclose(norms, 1.0, atol=1e-4)
