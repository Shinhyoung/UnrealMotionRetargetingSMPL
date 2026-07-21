"""Pelvis-only rest calibration (SMPL global_orient).

Why
---
SAT-HMR's ``global_orient`` (joint 0) does not evaluate to identity for a
person standing upright — it includes the fixed rotation that maps SMPL
canonical (Y-up) to the OpenCV camera world (Y-DOWN, camera-facing). This
π-ish rotation slips through the SMPL→UE pipeline and flips the UE mannequin
upside down when the pelvis slot is wired live.

The fix is to subtract this baseline SMPL rest prediction from every future
``global_orient``: ``R_smpl_effective = R_smpl · inv(R_smpl_rest)``. In the
T-pose the effective rotation collapses to identity, so method A's correction
yields the UE pelvis rest local rotation (mannequin stays upright). Any body
turn shows up as a delta on top of rest → mannequin rotates correctly.

This tool captures the baseline ``R_smpl_rest`` from N frames of a T-pose.
It is intentionally pelvis-only — arm/leg calibration is not needed once
``--bone-correction`` + ``--preserve-rot-chirality`` are in effect.

Usage::

    python tools/calibrate_pelvis_rest.py --output ../pelvis_rest.npy

Then::

    python main.py --source realsense --bone-correction ../bone_correction.npz \\
                   --preserve-rot-chirality --pelvis-rest ../pelvis_rest.npy
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# make /python importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture.realsense import RealSenseCapture
from capture.threaded import ThreadedCapture
from detectors.sat_hmr_wrapper import SATHMRDetector
from tracking.person_tracker import PersonTracker
from transform.rotation import axis_angle_to_quat_xyzw, normalize_quat


def _sign_aligned_mean(quats: np.ndarray) -> np.ndarray:
    """Sign-align quaternions then mean-and-normalize.

    Same trick as in ``calibrate_rest_pose.py``: quaternions have double-cover,
    naive average can cancel. Flip each sample to the same hemisphere as the
    first before averaging. Fine for near-static poses.
    """
    reference = quats[0]
    dots = np.einsum("nj,j->n", quats, reference)
    signs = np.where(dots < 0.0, -1.0, 1.0)
    aligned = quats * signs[:, None]
    mean = aligned.mean(axis=0)
    return normalize_quat(mean)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sat-hmr-root", default="c:/0.shinhyoung/Project/SAT-HMR")
    p.add_argument("--sat-hmr-ckpt", default="")
    p.add_argument("--input-size", type=int, default=672)
    p.add_argument("--conf-thresh", type=float, default=0.3)
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--countdown", type=int, default=5)
    p.add_argument("--output", default="pelvis_rest.npy",
                   help="Path to save the (4,) xyzw quaternion.")
    p.add_argument("--rs-width", type=int, default=1280)
    p.add_argument("--rs-height", type=int, default=720)
    p.add_argument("--rs-fps", type=int, default=30)
    args = p.parse_args()

    print("=" * 60)
    print("Pelvis Rest Calibration (SMPL global_orient baseline)")
    print("=" * 60)

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
    for _ in range(15):
        cap.read()  # warmup

    print()
    print("카메라 앞에서 자연스럽게 서 주세요 (T-pose 필수 아님).")
    print("몸통 방향이 정면(카메라)을 향한 upright 상태이면 됩니다.")
    print()
    for i in range(args.countdown, 0, -1):
        print(f"  {i} 초 후 캡처 시작 ...")
        time.sleep(1.0)
    print(">>> 캡처 시작. 몸통을 움직이지 마세요.")

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

        target = None
        for det in detections:
            if det.person_id is not None:
                target = det
                break
        if target is None:
            continue

        # global_orient is (3,) axis-angle in SMPL Y-up basis
        # (the SAT-HMR wrapper already applied opencv_cam_yflip).
        aa = np.asarray(target.global_orient, dtype=np.float64).reshape(3)
        q = axis_angle_to_quat_xyzw(aa)          # (4,) xyzw
        collected.append(q)

        if len(collected) % 10 == 0:
            print(f"  captured {len(collected)}/{args.frames}")

    cap.release()

    if len(collected) < args.frames:
        print(f"Only captured {len(collected)} frames. Aborting.")
        return 1

    stacked = np.stack(collected, axis=0)         # (N, 4)
    q_rest = _sign_aligned_mean(stacked)          # (4,) xyzw
    print(">>> 캡처 완료.")
    print(f"  pelvis rest quat (SMPL basis, xyzw) = {q_rest.round(4).tolist()}")

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), q_rest)
    print(f">>> Saved to: {out_path}")
    print()
    print("이제 파이프라인 실행 시 이 파일을 지정하세요:")
    print(f"  python main.py --source realsense \\")
    print(f"      --bone-correction ../bone_correction.npz --preserve-rot-chirality \\")
    print(f"      --pelvis-rest {out_path}")
    print(f"  ( --skip-pelvis 는 넘기지 마세요 — pelvis 를 살려야 몸통 회전이 반영됩니다 )")
    return 0


if __name__ == "__main__":
    sys.exit(main())
