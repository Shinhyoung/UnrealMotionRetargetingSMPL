"""Rest-pose (T-pose) calibration.

CLAUDE.md §4.5. SMPL canonical rest pose vs UE mannequin rest pose 사이의
관절별 축 차이를 사용자가 카메라 앞에서 T-pose 로 서 있는 동안 자동 측정.

사용:
    # 방법 A correction 없이 (레거시)
    python tools/calibrate_rest_pose.py --output rest_pose_offset.npy

    # 방법 A correction 위에 얹기 (권장) — 잔여 twist 만 측정
    python tools/calibrate_rest_pose.py \\
        --bone-correction ../bone_correction.npz \\
        --output ../rest_pose_offset.npy

결과 파일 (24, 4) xyzw quaternion 을 저장한다. 다음 파이프라인 실행 시
``--rest-pose-offset`` 로 지정하면 자동 로드된다.

수학:
  * 사용자 T-pose 상태의 관절별 회전 예측을 파이프라인과 **동일한 변환** 을
    통과시킨다: ``smpl_quat_to_ue_quat`` 후 (옵션) ``apply_bone_correction``.
  * 결과를 ``q_T[i]`` 라 하면, UE 가 T-pose 를 유지하려면 그 뒤 offset 이
    ``offset[i] * q_T[i] = identity`` 여야 함 → ``offset[i] = conj(q_T[i])``.
  * ``apply_rest_pose_offset`` 는 이미 ``offset * q`` 로 곱하므로 그대로 저장.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# make /python importable when this file is run standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture.realsense import RealSenseCapture
from capture.threaded import ThreadedCapture
from config.bone_mapping import NUM_SMPL_JOINTS
from detectors.sat_hmr_wrapper import SATHMRDetector
from tracking.person_tracker import PersonTracker
from transform.bone_correction import apply_bone_correction, load_correction
from transform.coordinate import smpl_quat_to_ue_quat
from transform.rotation import axis_angle_to_quat_xyzw, normalize_quat


def _average_quats_signed(quat_list: list[np.ndarray]) -> np.ndarray:
    """Sum-and-normalize quaternion mean with sign alignment.

    Quaternions have double-cover ambiguity — q and -q are the same rotation.
    Naive averaging can cancel them out. We flip each sample to the same
    hemisphere as the first before averaging. This is a first-order
    approximation but is accurate when all samples cluster near one rotation
    (which is the case for a still T-pose).
    """
    stacked = np.stack(quat_list, axis=0)  # (N, 24, 4)
    reference = stacked[0]                 # (24, 4)
    dots = np.einsum("nji,ji->nj", stacked, reference)   # (N, 24)
    signs = np.where(dots < 0.0, -1.0, 1.0)              # (N, 24)
    stacked = stacked * signs[:, :, None]
    mean = stacked.mean(axis=0)
    return normalize_quat(mean)


def _inverse_unit_quat(q: np.ndarray) -> np.ndarray:
    """For unit quaternions, inverse == conjugate: (x,y,z,w) → (-x,-y,-z,w)."""
    out = q.copy()
    out[..., :3] *= -1.0
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sat-hmr-root", default="c:/0.shinhyoung/Project/SAT-HMR")
    p.add_argument("--sat-hmr-ckpt", default="")
    p.add_argument("--input-size", type=int, default=672)
    p.add_argument("--conf-thresh", type=float, default=0.3)
    p.add_argument("--frames", type=int, default=60,
                   help="Number of frames to average.")
    p.add_argument("--countdown", type=int, default=5,
                   help="Seconds of countdown before capture starts.")
    p.add_argument("--output", default="rest_pose_offset.npy",
                   help="Path to save the (24, 4) xyzw offset quaternions.")
    p.add_argument("--rs-width", type=int, default=1280)
    p.add_argument("--rs-height", type=int, default=720)
    p.add_argument("--rs-fps", type=int, default=30)
    p.add_argument("--bone-correction", default="",
                   help="Path to a .npz from tools/compute_bone_correction.py. "
                        "When set, calibration measures the residual T-pose error "
                        "*after* the correction — so runtime pipeline must supply "
                        "the same --bone-correction alongside the produced offset.")
    args = p.parse_args()

    print("=" * 60)
    print("Rest-Pose Calibration")
    print("=" * 60)

    bone_correction = None
    if args.bone_correction:
        parent_inv, own = load_correction(args.bone_correction)
        bone_correction = (parent_inv, own)
        print(f"[bone_correction] loaded {args.bone_correction} "
              f"— calibrating residual T-pose error on top of it.")

    print(f"Loading SAT-HMR from {args.sat_hmr_root} ...")
    detector = SATHMRDetector(
        sat_hmr_root=args.sat_hmr_root,
        checkpoint_path=(args.sat_hmr_ckpt or None),
        device="cuda",
        conf_thresh=args.conf_thresh,
        input_size_override=(args.input_size if args.input_size > 0 else None),
    )
    tracker = PersonTracker(max_distance=1.0, max_missing_frames=15)

    print("Opening RealSense capture ...")
    cap = ThreadedCapture(
        RealSenseCapture(width=args.rs_width, height=args.rs_height, fps=args.rs_fps)
    )

    # Warm the camera up.
    for _ in range(15):
        cap.read()

    print()
    print("카메라 앞에서 팔을 벌린 T-포즈 (십자 자세) 로 서 주세요.")
    print("팔은 어깨 높이, 몸통과 90° 로. 자세가 안정되어야 정확합니다.")
    print()
    for i in range(args.countdown, 0, -1):
        print(f"  {i} 초 후 캡처 시작 ...")
        time.sleep(1.0)
    print(">>> 캡처 시작. 움직이지 마세요.")

    # Collect quaternions.
    collected: list[np.ndarray] = []
    misses = 0
    while len(collected) < args.frames:
        ok, frame = cap.read()
        if not ok:
            misses += 1
            if misses > 30:
                print("Frame read failed repeatedly — aborting.")
                cap.release()
                return 1
            continue
        misses = 0

        detections = detector.infer(frame)
        tracker.update(detections)

        # Take the closest / most confident person. Since we assume the user is
        # alone in front of the camera, first detection is fine.
        target = None
        for det in detections:
            if det.person_id is not None:
                target = det
                break
        if target is None:
            continue

        aa = target.full_axis_angles()                # (24, 3)
        quats_smpl = axis_angle_to_quat_xyzw(aa)      # (24, 4)
        quats_ue = smpl_quat_to_ue_quat(quats_smpl)   # (24, 4) after SMPL→UE swizzle
        if bone_correction is not None:
            parent_inv, own = bone_correction
            quats_ue = apply_bone_correction(quats_ue, parent_inv, own)
        collected.append(quats_ue)

        if len(collected) % 10 == 0:
            print(f"  captured {len(collected)}/{args.frames}")

    cap.release()

    if len(collected) < args.frames:
        print(f"Only captured {len(collected)} frames. Aborting.")
        return 1

    print(">>> 캡처 완료. 평균 pose 계산 ...")
    q_T = _average_quats_signed(collected)   # (24, 4) unit quaternion, user's T-pose
    offset = _inverse_unit_quat(q_T)         # (24, 4)

    # Sanity: offset * q_T should be near identity.
    from transform.rotation import quat_multiply_xyzw
    product = normalize_quat(quat_multiply_xyzw(offset, q_T))
    residual = np.linalg.norm(product - np.array([0.0, 0.0, 0.0, 1.0]), axis=-1).max()
    print(f"  residual (max ||offset*q_T - identity||): {residual:.4e}")

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), offset)
    print(f">>> Saved offset to: {out_path}")
    print()
    print("이제 파이프라인 실행 시 이 파일을 지정하세요:")
    if bone_correction is not None:
        print(f"  python main.py --source realsense --bone-correction {args.bone_correction} \\")
        print(f"                 --rest-pose-offset {out_path} --skip-pelvis")
    else:
        print(f"  python main.py --source realsense --rest-pose-offset {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
