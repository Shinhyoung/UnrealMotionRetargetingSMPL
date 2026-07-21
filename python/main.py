"""Realtime multi-person retargeting entrypoint.

Pipeline (CLAUDE.md §1 dataflow):

    webcam → SAT-HMR (or Mock) → tracker → transform (SMPL→UE + rest-pose)
           → UDP send   (realtime path)
           → debug overlay  (optional, isolated)

Usage examples:

    # Mock detector, no window, prove UDP is going out:
    python main.py --mock --source 0

    # Real SAT-HMR (once weights are wired), with debug overlay:
    python main.py --debug

Coordinate transform lives in ``transform/coordinate.py`` (CLAUDE.md §4.3).
Debug rendering is isolated behind ``--debug`` and its failures are swallowed
(CLAUDE.md §8).
"""
from __future__ import annotations

import argparse
import time
from typing import List, Optional

import cv2
import numpy as np

from config.bone_mapping import NUM_SMPL_JOINTS
from config.settings import Settings
from detectors.sat_hmr_wrapper import MockSATHMRDetector, PoseDetection
from network.packet import build_packet
from network.udp_sender import UDPSender
from smoothing.temporal import TemporalSmoother
from tracking.person_tracker import PersonTracker
from transform.bone_correction import apply_bone_correction, load_correction
from transform.coordinate import (
    smpl_pos_to_ue_pos,
    smpl_quat_to_ue_quat,
    smpl_quat_to_ue_quat_chirality_preserving,
)
from transform.rest_pose import apply_rest_pose_offset, load_offsets
from transform.rotation import axis_angle_to_quat_xyzw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="realsense",
                   help="'realsense' (default, Intel D455 via pyrealsense2), "
                        "camera index (e.g. '0'), video file path, "
                        "or 'synthetic' for blank frames (headless smoke tests).")
    p.add_argument("--rs-width", type=int, default=1280)
    p.add_argument("--rs-height", type=int, default=720)
    p.add_argument("--rs-fps", type=int, default=30)
    p.add_argument("--rs-serial", default="", help="RealSense device serial (optional).")
    p.add_argument("--config", default="",
                   help="Optional JSON config path; defaults if omitted.")
    p.add_argument("--mock", action="store_true",
                   help="Use MockSATHMRDetector instead of real SAT-HMR.")
    p.add_argument("--n-mock-persons", type=int, default=2,
                   help="Number of synthetic people when --mock is on.")
    p.add_argument("--sat-hmr-root", default="c:/0.shinhyoung/Project/SAT-HMR",
                   help="Path to the SAT-HMR repository clone (real detector).")
    p.add_argument("--sat-hmr-ckpt", default="",
                   help="Path to sat_644.pth (defaults to configs/run/demo.yaml value).")
    p.add_argument("--conf-thresh", type=float, default=0.3,
                   help="SAT-HMR detection confidence threshold.")
    p.add_argument("--input-size", type=int, default=672,
                   help="Override SAT-HMR input_size (must be a multiple of 56 when SAT "
                        "is on: e.g. 672=12x56, 616=11x56). Default 672 yields ~3-4x speed "
                        "vs the config's 1288 (23x56). Set to 0 to use the config value.")
    p.add_argument("--no-threaded-capture", action="store_true",
                   help="Disable background capture thread. When ON (default), stale "
                        "camera frames are dropped so inference always sees the newest.")
    p.add_argument("--debug", action="store_true",
                   help="Enable 2D debug overlay. Do NOT use for perf measurement.")
    p.add_argument("--show-mesh", action="store_true",
                   help="With --debug, overlay SAT-HMR SMPL mesh on the source frame "
                        "(uses SAT-HMR's own vis_meshes_img; requires pyrender).")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after N frames (0 = run until source ends).")
    p.add_argument("--smooth-alpha", type=float, default=0.5,
                   help="Temporal smoothing weight of the *new* sample for "
                        "per-joint rotations (SLERP EMA). 1.0 disables smoothing.")
    p.add_argument("--smooth-root-alpha", type=float, default=-1.0,
                   help="Separate smoothing weight for root position (EMA). "
                        "-1 = same as --smooth-alpha (legacy). Lower values (e.g. "
                        "0.1) heavily damp monocular depth jitter without slowing "
                        "pose response.")
    p.add_argument("--rest-pose-offset", default="",
                   help="Path to a (24, 4) xyzw .npy from tools/calibrate_rest_pose.py. "
                        "Overrides settings.rest_pose.offset_path when non-empty. "
                        "Applied AFTER --bone-correction when both are supplied "
                        "(방법 A 의 잔여 twist 를 empirical 하게 마무리).")
    p.add_argument("--bone-correction", default="",
                   help="Path to a .npz from tools/compute_bone_correction.py "
                        "(per-joint rest-orientation correction, 방법 A). "
                        "Applies Q_send[i] = W_ref[parent]^-1 · (M R_smpl M^-1) · W_ref[i]. "
                        "Combine with --rest-pose-offset to clean up residual twist.")
    p.add_argument("--tracker-max-distance", type=float, default=0.0,
                   help="Override PersonTracker.max_distance (meters). 0 = use settings default.")
    p.add_argument("--tracker-max-missing-frames", type=int, default=0,
                   help="Override PersonTracker.max_missing_frames. 0 = use settings default. "
                        "Larger values stop ID inflation when a single person briefly leaves the frame.")
    p.add_argument("--swap-lr", action="store_true",
                   help="Swap SMPL left/right joint pairs before send. Use when the "
                        "mannequin's limbs come out mirrored (X-cross pose).")
    p.add_argument("--offset-order", choices=["pre", "post"], default="pre",
                   help="Apply rest-pose offset as pre-multiply (offset*q, default) or "
                        "post-multiply (q*offset). Some skeletons prefer post-multiply.")
    p.add_argument("--skip-pelvis", action="store_true",
                   help="Zero out the pelvis (index 0) rotation before send so the "
                        "UE actor transform decides world orientation instead. "
                        "Use when the mannequin ends up lying down / rotated.")
    p.add_argument("--force-identity-pose", action="store_true",
                   help="DIAGNOSTIC: override SAT-HMR output — send identity SMPL "
                        "rotations for every joint. With --bone-correction, the "
                        "mannequin must hold a clean T-pose (Q_send[i] collapses "
                        "to UE local rest). If it does not, the correction / "
                        "AnimNode is at fault; if it does, the SMPL prediction "
                        "itself is misbehaving for the affected joints.")
    p.add_argument("--preserve-rot-chirality", action="store_true",
                   help="Use chirality-preserving SMPL→UE quaternion mapping "
                        "(axis = +M n) instead of the strict matrix conjugation "
                        "(axis = -M n). Fixes the arms-move-opposite issue that "
                        "shows up because M is a reflection (SMPL right-hand → "
                        "UE left-hand). Legs look fine without this because their "
                        "SMPL body_pose stays near identity at rest.")
    p.add_argument("--pelvis-rest", default="",
                   help="Path to a (4,) xyzw .npy from tools/calibrate_pelvis_rest.py. "
                        "Subtracts SAT-HMR's baseline global_orient (measured at "
                        "upright standing) so the UE mannequin does not flip "
                        "upside-down when the pelvis slot is wired live. Applied "
                        "as R_smpl[0] · inv(R_smpl_rest) BEFORE basis change.")
    p.add_argument("--raw-smpl", action="store_true",
                   help="Send raw SMPL joint rotations without method-A hacks. "
                       "Assumes UE target is an SMPL-native skeleton (imported via "
                       "SMPL_Skeleton.fbx), so UE-side IK Retargeter will handle "
                       "the SMPL→Mannequin mapping. Disables bone-correction, "
                       "chirality-preserving, pelvis-rest, swap-lr, skip-pelvis. "
                       "Basis change (SMPL Y-up → UE Z-up) still applied.")
    p.add_argument("--use-depth", action="store_true",
                   help="Enable RealSense depth stream and per-frame joint depth "
                       "verification. Diagnostic-only for now (prints per-limb "
                       "measured-vs-predicted depth delta every N frames). "
                       "Requires --source realsense.")
    p.add_argument("--depth-log-every", type=int, default=30,
                   help="Log depth verification summary every N processed frames.")
    p.add_argument("--depth-root-lock", action="store_true",
                   help="Replace SAT-HMR's monocular root translation with a "
                        "position back-projected from RealSense depth at the "
                        "pelvis pixel. Cures forward/backward drift caused by "
                        "bbox-size depth ambiguity. Enables depth capture.")
    p.add_argument("--single-front-person", action="store_true",
                   help="Keep only the front-most detection (min camera-Z) each frame. "
                        "Simplifies demo to one person; future multi-person work will "
                        "remove this filter.")
    p.add_argument("--smpl-native", action="store_true",
                   help="Send raw SMPL Y-up quaternions with NO basis change. "
                        "Used by UE's SMPLProceduralActor (Option F) which runs "
                        "SMPL LBS directly on the CPU and needs original SMPL "
                        "pose parameters. Implies --raw-smpl semantics.")
    return p.parse_args()


class _SyntheticCapture:
    """cv2.VideoCapture stand-in: yields blank frames. Used with --mock when no
    camera/video source is available (e.g. CI, headless smoke tests)."""

    def __init__(self, width: int = 1280, height: int = 720):
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)

    def read(self):
        return True, self._frame

    def release(self):
        pass


def open_capture(source: str, args=None):
    if source == "synthetic":
        return _SyntheticCapture()
    if source == "realsense":
        from capture.realsense import RealSenseCapture
        return RealSenseCapture(
            width=(args.rs_width if args else 1280),
            height=(args.rs_height if args else 720),
            fps=(args.rs_fps if args else 30),
            serial=(args.rs_serial if args and args.rs_serial else None),
            enable_depth=(bool(args and (args.use_depth or args.depth_root_lock))),
        )
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open capture: {source!r}")
    return cap


def build_detector(args, settings: Settings, keep_debug_output: bool = False):
    if args.mock:
        return MockSATHMRDetector(n_persons=args.n_mock_persons)
    from detectors.sat_hmr_wrapper import SATHMRDetector  # deferred import
    return SATHMRDetector(
        sat_hmr_root=args.sat_hmr_root,
        checkpoint_path=(args.sat_hmr_ckpt or None),
        device=settings.model.device,
        conf_thresh=args.conf_thresh,
        keep_debug_output=keep_debug_output,
        input_size_override=(args.input_size if args.input_size > 0 else None),
    )


# SMPL left/right joint pairs (index_left, index_right).
_LR_PAIRS = (
    (1, 2), (4, 5), (7, 8), (10, 11), (13, 14),
    (16, 17), (18, 19), (20, 21), (22, 23),
)


def _swap_lr(quats: np.ndarray) -> np.ndarray:
    """L/R mirror with separate handling for global orient vs body pose.

    Body pose (idx 1-23): swap L/R indices AND reflect across the SMPL YZ
    plane (F=diag(-1,1,1)) → quat (x, y, z, w) → (x, -y, -z, w). This flips
    both Y and Z rotations, giving a proper mirror-image body pose.

    Global orient (idx 0): reflect across the SMPL XZ plane (F=diag(1,-1,1))
    → quat (x, y, z, w) → (-x, y, -z, w). This preserves Y-axis rotation (yaw)
    while flipping Z-axis rotation (roll/tilt). Needed because the whole-body
    yaw was already correct without any mirror; only tilt needed inverting.
    """
    out = quats.copy()
    for l, r in _LR_PAIRS:
        out[[l, r]] = out[[r, l]]
    # Body pose: YZ-plane mirror
    out[1:, 1] *= -1
    out[1:, 2] *= -1
    # Global orient: XZ-plane mirror (preserves yaw, flips tilt/pitch)
    out[0, 0] *= -1
    out[0, 2] *= -1
    return out


def detection_to_ue_payload(
    det: PoseDetection,
    rest_offsets: np.ndarray,
    ue_scale: float,
    swap_lr: bool = False,
    skip_pelvis: bool = False,
    offset_order: str = "pre",
    bone_correction: Optional[tuple] = None,
    preserve_rot_chirality: bool = False,
    pelvis_rest_inv: Optional[np.ndarray] = None,
    raw_smpl: bool = False,
    smpl_native: bool = False,
) -> tuple:
    """Convert one PoseDetection into the (person_id, root_ue, quats_ue) tuple.

    All heavy per-joint math is here so the main loop stays readable.

    If ``bone_correction`` is provided it must be ``(parent_world_inv, own_world)``
    of shape ``(24, 4)`` each — 방법 A. When set it supersedes ``rest_offsets``:
    the per-joint rest-orientation basis change subsumes what the offset-only
    approach was trying to approximate.
    """
    from transform.rotation import normalize_quat, quat_multiply_xyzw

    aa = det.full_axis_angles()                       # (24, 3)
    quats_smpl = axis_angle_to_quat_xyzw(aa)          # (24, 4)

    if smpl_native:
        # SMPL native mode: send RAW SMPL Y-up axis-angle-quats and
        # UNBASIS-CHANGED root translation (SMPL Y-up meters). UE-side
        # SMPLProceduralActor consumes these to run SMPL LBS directly.
        # The wire format stays 24 quats + 1 root but semantics differ —
        # UE actor knows it because BP toggles smpl_native mode.
        if pelvis_rest_inv is not None:
            quats_smpl[0] = normalize_quat(quat_multiply_xyzw(quats_smpl[0], pelvis_rest_inv))
        # Swap L/R joint rotations so the mannequin's on-screen limbs match
        # the user's mirrored self-view (viewer facing the mannequin's front).
        # Global orient (idx 0) is NOT swapped — that would flip yaw direction
        # which is already correct via the chirality-preserving actor basis.
        if swap_lr:
            quats_smpl = _swap_lr(quats_smpl)
        # Root translation stays in SMPL Y-up meters (no basis change).
        root_out = np.asarray(det.root_position, dtype=np.float64).copy()
        return (int(det.person_id), root_out, quats_smpl.astype(np.float64))

    if raw_smpl:
        # Bypass method-A per-joint correction; downstream (IK Retargeter or
        # C++ chain-FK) does the retargeting. Basis change still applies.
        if pelvis_rest_inv is not None:
            quats_smpl[0] = normalize_quat(quat_multiply_xyzw(quats_smpl[0], pelvis_rest_inv))
        if skip_pelvis:
            quats_smpl[0] = np.array([0.0, 0.0, 0.0, 1.0])
        # For SMPL-skeleton-in-UE with matching bone directions, the swing
        # retargeter reads bone directions from FK. Matrix conjugation's
        # axial vector flip (axis = -M n) reverses rotation direction; the
        # chirality-preserving variant (axis = +M n) matches the physical
        # rotation. This is the default here; the flag controls the opposite.
        if preserve_rot_chirality:
            quats_ue = smpl_quat_to_ue_quat(quats_smpl)   # legacy: axis = -M n
        else:
            quats_ue = smpl_quat_to_ue_quat_chirality_preserving(quats_smpl)
        root_ue = smpl_pos_to_ue_pos(det.root_position, scale=ue_scale)
        return (int(det.person_id), root_ue, quats_ue)

    # Legacy code path (bone_correction etc.) below.
    # (unchanged)

    # Pelvis-only calibration: subtract SAT-HMR's upright baseline from
    # global_orient BEFORE basis change, so the effective rotation is identity
    # at rest (and correction returns UE pelvis rest local).
    if pelvis_rest_inv is not None:
        quats_smpl[0] = normalize_quat(quat_multiply_xyzw(quats_smpl[0], pelvis_rest_inv))

    if preserve_rot_chirality:
        quats_ue = smpl_quat_to_ue_quat_chirality_preserving(quats_smpl)  # axis = +M n
    else:
        quats_ue = smpl_quat_to_ue_quat(quats_smpl)   # (24, 4) — M R M^-1, axis = -M n
    if swap_lr:
        quats_ue = _swap_lr(quats_ue)

    if bone_correction is not None:
        parent_inv, own = bone_correction
        quats_ue = apply_bone_correction(quats_ue, parent_inv, own)
    if offset_order == "post":
        quats_ue = normalize_quat(quat_multiply_xyzw(quats_ue, rest_offsets))
    else:
        quats_ue = apply_rest_pose_offset(quats_ue, rest_offsets)

    if skip_pelvis:
        # "Skip" = pretend SMPL predicted identity for pelvis, then re-run the
        # correction on that identity so the UE bone still gets its rest local
        # (which is NOT identity when the UE bone is `pelvis`, only when it's
        # the virtual `root`). Without this, forcing raw identity on the pelvis
        # slot when it maps to UE `pelvis` collapses the rest orientation and
        # the mannequin falls over.
        if bone_correction is not None:
            parent_inv, own = bone_correction
            quats_ue[0] = normalize_quat(quat_multiply_xyzw(parent_inv[0], own[0]))
        else:
            quats_ue[0] = np.array([0.0, 0.0, 0.0, 1.0])
    root_ue = smpl_pos_to_ue_pos(det.root_position, scale=ue_scale)
    return (int(det.person_id), root_ue, quats_ue)


def main() -> int:
    args = parse_args()
    settings = Settings.load(args.config or None)
    debug_on = args.debug or settings.debug.overlay
    if args.show_mesh:
        settings.debug.show_mesh = True
        debug_on = True

    # keep_debug_output also required for depth verify (needs mesh verts + intrinsics)
    keep_debug = (debug_on or args.use_depth or args.depth_root_lock) and not args.mock
    detector = build_detector(args, settings, keep_debug_output=keep_debug)
    tracker = PersonTracker(
        max_distance=(args.tracker_max_distance if args.tracker_max_distance > 0
                      else settings.tracking.max_distance),
        max_missing_frames=(args.tracker_max_missing_frames if args.tracker_max_missing_frames > 0
                            else settings.tracking.max_missing_frames),
    )
    bone_correction: Optional[tuple] = None
    if args.bone_correction:
        parent_inv, own = load_correction(args.bone_correction)
        bone_correction = (parent_inv, own)
        print(f"[bone_correction] loaded correction from {args.bone_correction}")

    pelvis_rest_inv: Optional[np.ndarray] = None
    if args.pelvis_rest:
        from transform.rotation import normalize_quat as _norm
        q_rest = np.asarray(np.load(args.pelvis_rest), dtype=np.float64).reshape(4)
        # inverse of unit quaternion = conjugate = (-x, -y, -z, w)
        q_inv = q_rest.copy()
        q_inv[:3] *= -1.0
        pelvis_rest_inv = _norm(q_inv)
        print(f"[pelvis_rest] loaded baseline from {args.pelvis_rest} "
              f"(applies R_smpl[0] · inv(rest) before basis change)")

    offset_path = args.rest_pose_offset or settings.rest_pose.offset_path
    rest_offsets = load_offsets(offset_path)
    if offset_path:
        print(f"[rest_pose] loaded offsets from {offset_path}"
              + ("  (applied on top of bone_correction)" if bone_correction else ""))
    smoother = TemporalSmoother(
        alpha=args.smooth_alpha,
        max_missing_frames=settings.tracking.max_missing_frames,
        root_alpha=(args.smooth_root_alpha if args.smooth_root_alpha >= 0 else None),
    )
    sender = UDPSender(settings.network.host, settings.network.port)

    overlay = None
    if debug_on:
        try:
            from debug.overlay import DebugOverlay
            overlay = DebugOverlay(
                settings.debug,
                sat_hmr_root=(args.sat_hmr_root if not args.mock else None),
                smpl_faces=(getattr(detector, "smpl_faces", None) if not args.mock else None),
            )
        except Exception as e:
            print(f"[warn] debug overlay init failed: {e}. Continuing without debug.")
            overlay = None

    cap = open_capture(args.source, args)
    if not args.no_threaded_capture:
        from capture.threaded import ThreadedCapture
        depth_on = args.use_depth or args.depth_root_lock
        cap = ThreadedCapture(cap, with_depth=depth_on)
        print("[capture] threaded capture ON (drops stale frames)"
              + (" + depth" if depth_on else ""))

    depth_verifier = None
    depth_root_lock = None
    if args.use_depth or args.depth_root_lock:
        try:
            from depth.verify import DepthVerifier, format_limb_summary
            _dv = DepthVerifier()
            if args.use_depth:
                depth_verifier = _dv
                print("[depth] verifier ready — per-limb depth deltas will log every "
                      f"{args.depth_log_every} frames")
            if args.depth_root_lock:
                depth_root_lock = _dv
                print("[depth] root-lock ON — pelvis position taken from RealSense depth")
        except Exception as e:
            print(f"[warn] depth verifier init failed: {e}. Continuing without depth.")
            depth_verifier = None
            depth_root_lock = None

    frame_id = 0
    n_sent = 0
    t_start = time.time()
    min_frame_interval = 1.0 / settings.network.max_fps if settings.network.max_fps > 0 else 0.0
    last_send_ts = 0.0
    # [total_attempts, successful_corrections, last_measured_z_m, last_sat_hmr_z_m]
    _depth_lock_stats = [0, 0, 0.0, 0.0]

    try:
        while True:
            if args.max_frames and frame_id >= args.max_frames:
                break

            depth_frame = None
            if (args.use_depth or args.depth_root_lock) and hasattr(cap, "read_with_depth"):
                ok, frame, depth_frame = cap.read_with_depth()
            else:
                ok, frame = cap.read()
            if not ok:
                break

            # --- Inference (SAT-HMR or Mock) ---
            detections: List[PoseDetection] = detector.infer(frame)

            # Optional: reduce to the single front-most person (min camera Z).
            # Root position is in SMPL Y-up meters; camera-forward = +Z, so
            # the closest person has the smallest Z.
            if args.single_front_person and detections:
                detections = [min(detections, key=lambda d: float(d.root_position[2]))]

            # --- Depth root lock: replace SAT-HMR monocular root with real depth ---
            if depth_root_lock is not None and depth_frame is not None and depth_frame.size:
                for det in detections:
                    if det.debug_verts is None or det.debug_intrinsics is None:
                        continue
                    root_old = det.root_position.copy()
                    root_new = depth_root_lock.corrected_root_smpl(
                        mesh_verts_cam=det.debug_verts,
                        intrinsics=det.debug_intrinsics,
                        depth_mm=depth_frame,
                        resize_rate=det.debug_resize_rate or 1.0,
                    )
                    _depth_lock_stats[0] += 1
                    if root_new is not None:
                        det.root_position = root_new
                        _depth_lock_stats[1] += 1
                        _depth_lock_stats[2] = float(root_new[2])   # last measured Z
                        _depth_lock_stats[3] = float(root_old[2])   # last SAT-HMR Z
                if (frame_id % 30) == 0 and _depth_lock_stats[0] > 0:
                    tot, ok, z_meas, z_pred = _depth_lock_stats
                    print(f"[depth-lock] {ok}/{tot} success "
                          f"(last Z: measured={z_meas:.2f}m sat-hmr={z_pred:.2f}m)")

            # --- Depth verification (diagnostic) ---
            if depth_verifier is not None and depth_frame is not None and depth_frame.size:
                if (frame_id % max(1, args.depth_log_every)) == 0:
                    for det in detections:
                        if det.debug_verts is None or det.debug_intrinsics is None:
                            continue
                        report = depth_verifier.analyze(
                            mesh_verts_cam=det.debug_verts,
                            intrinsics=det.debug_intrinsics,
                            depth_mm=depth_frame,
                            resize_rate=det.debug_resize_rate or 1.0,
                        )
                        print(f"[depth] frame={frame_id} pid={det.person_id} "
                              f"{format_limb_summary(report)}  "
                              f"(+ = joint FARTHER than SMPL predicts)")

            # DIAGNOSTIC: force every joint to identity so we can verify the
            # correction/AnimNode path in isolation (should render T-pose).
            if args.force_identity_pose:
                for _d in detections:
                    _d.global_orient = np.zeros(3, dtype=np.float64)
                    _d.body_pose = np.zeros((NUM_SMPL_JOINTS - 1, 3), dtype=np.float64)

            # --- Tracking (assigns person_id) ---
            tracker.update(detections)

            # --- Realtime path: no meshes, no Euler ---
            smoother.advance_frame()
            persons_payload = []
            for det in detections:
                if det.person_id is None:
                    continue
                pid, root_ue, quats_ue = detection_to_ue_payload(
                    det, rest_offsets, settings.transform.smpl_to_ue_scale,
                    swap_lr=args.swap_lr,
                    skip_pelvis=args.skip_pelvis,
                    offset_order=args.offset_order,
                    bone_correction=bone_correction,
                    preserve_rot_chirality=args.preserve_rot_chirality,
                    pelvis_rest_inv=pelvis_rest_inv,
                    raw_smpl=args.raw_smpl,
                    smpl_native=args.smpl_native,
                )
                quats_ue, root_ue = smoother.step(pid, quats_ue, root_ue)
                persons_payload.append((pid, root_ue, quats_ue))

            # Optional pacing
            if min_frame_interval > 0.0:
                dt = time.time() - last_send_ts
                if dt < min_frame_interval:
                    time.sleep(min_frame_interval - dt)

            packet = build_packet(frame_id, persons_payload)
            if sender.send(packet):
                n_sent += 1
            last_send_ts = time.time()

            # --- Debug path (isolated, never blocks realtime) ---
            if overlay is not None:
                try:
                    overlay.render(frame, detections, frame_id)
                except Exception as e:
                    print(f"[warn] debug render failed: {e}")
                if overlay.should_exit:
                    print("[info] q/ESC pressed — exiting.")
                    break

            frame_id += 1
    finally:
        cap.release()
        if overlay is not None:
            try:
                overlay.close()
            except Exception:
                pass
        sender.close()

        elapsed = time.time() - t_start
        fps = frame_id / elapsed if elapsed > 0 else 0.0
        print(
            f"[info] frames={frame_id}  sent={n_sent}  "
            f"elapsed={elapsed:.1f}s  fps={fps:.1f}"
            + ("   (debug ON — perf not reliable)" if debug_on else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
