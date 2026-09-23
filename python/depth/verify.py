"""Depth-based front/back verification for SMPL predictions.

Uses RealSense depth stream to detect and correct the classic monocular
front/back ambiguity in SAT-HMR's SMPL body_pose output.

Algorithm
---------
1. SAT-HMR predicts SMPL pose. We already have (per debug output):
   - mesh vertices in OpenCV camera frame (Y-DOWN, Z-forward, meters)
   - pinhole intrinsics for the resized input frame
   - resize_rate mapping input pixels → original frame pixels
2. Compute 24 joint 3D positions via SMPL's J_regressor (constant matrix,
   loaded once from SMPL_NEUTRAL.pkl).
3. Project joints to 2D pixel coordinates in the original color frame.
4. Sample the depth image at those pixels — this is the ACTUAL depth of
   the user's body at each joint.
5. Compare measured depth (RealSense) vs predicted depth (SMPL Z coord).
   Large positive delta at a joint = joint is behind body but SMPL says
   it's in front (or vice versa).
6. Emit per-joint diagnostic and (optionally) apply a flip correction.

Depth image conventions
-----------------------
RealSense D455 depth is uint16, millimeters, aligned to color stream (we
call ``rs.align(color)`` in ``capture/realsense.py``). Values of 0 = invalid
(no return). Depth measures distance along the optical axis (Z).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


DEFAULT_SMPL_PKL = "c:/0.shinhyoung/Project/SAT-HMR/weights/smpl_data/smpl/SMPL_NEUTRAL.pkl"


@dataclass
class JointDepthReport:
    """Per-joint depth comparison for one detection."""

    joint_names: list
    predicted_mm: np.ndarray   # (24,) SMPL's Z (in mm)
    measured_mm: np.ndarray    # (24,) RealSense depth (in mm; 0 = invalid)
    delta_mm: np.ndarray       # measured - predicted (positive = joint is farther than SMPL says)
    pixel_xy: np.ndarray       # (24, 2) 2D pixel positions used to sample

    def valid_mask(self) -> np.ndarray:
        """True where measured depth is valid (non-zero)."""
        return self.measured_mm > 0

    def summarize_limbs(self) -> dict:
        """Per-limb depth deltas RELATIVE TO PELVIS.

        Monocular HMR (SAT-HMR) has systematic absolute-depth bias, but the
        relative depth between joints is much more reliable. We subtract each
        joint's delta from the pelvis' delta so the bias cancels, leaving the
        actual front/back mismatch we care about.

        For each limb (mean over its joints):
          value = (measured_joint - measured_pelvis) - (predicted_joint - predicted_pelvis)
                = delta_joint - delta_pelvis

        Interpretation (in mm):
          +N ⇒ limb is N mm FARTHER from camera than SMPL says (relative to
               torso). E.g., arm is behind body but SMPL predicted it in front.
          -N ⇒ limb is N mm CLOSER than SMPL says.
          ~0 ⇒ SMPL depth prediction correct for this limb.
        """
        groups = {
            "l_arm": (16, 18, 20),   # L_shoulder, L_elbow, L_wrist
            "r_arm": (17, 19, 21),
            "l_leg": (4, 7, 10),
            "r_leg": (5, 8, 11),
            "spine": (3, 6, 9),
            "head":  (12, 15),
        }
        mask = self.valid_mask()
        pelvis_ok = mask[0]
        delta_pelvis = self.delta_mm[0] if pelvis_ok else 0.0
        out = {}
        for name, idxs in groups.items():
            vals = [self.delta_mm[i] - delta_pelvis for i in idxs if mask[i] and pelvis_ok]
            out[name] = float(np.mean(vals)) if vals else None
        return out


class DepthVerifier:
    """Compute per-joint depth mismatch for SMPL predictions."""

    JOINT_NAMES = (
        "pelvis", "L_hip", "R_hip", "spine1",
        "L_knee", "R_knee", "spine2",
        "L_ankle", "R_ankle", "spine3",
        "L_foot", "R_foot", "neck",
        "L_collar", "R_collar", "head",
        "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
        "L_wrist", "R_wrist", "L_hand", "R_hand",
    )

    def __init__(self, smpl_pkl_path: str = DEFAULT_SMPL_PKL):
        p = Path(smpl_pkl_path)
        if not p.is_file():
            raise FileNotFoundError(f"SMPL pkl not found: {p}")

        with p.open("rb") as f:
            data = pickle.load(f, encoding="latin1")

        # J_regressor: sparse (24, 6890) matrix that computes joint positions
        # from mesh vertices. SMPL uses first 24 rows for body joints.
        jr = data["J_regressor"]
        try:
            jr = jr.toarray()   # sparse → dense
        except AttributeError:
            jr = np.asarray(jr)
        self.J_regressor = jr[:24].astype(np.float64)   # (24, 6890)
        # Per-person temporal state for A/B/C stability (see corrected_root_smpl).
        # Keyed by person_id; each entry holds smoothed pelvis pixel + last Z.
        self._state: dict = {}
        print(f"[depth_verify] loaded J_regressor {self.J_regressor.shape} from {p.name}")

    def analyze(
        self,
        mesh_verts_cam: np.ndarray,       # (V, 3) SMPL mesh in OpenCV camera frame, meters
        intrinsics: np.ndarray,           # (3, 3) intrinsics (from SAT-HMR, in input_size frame)
        depth_mm: np.ndarray,             # (H, W) uint16 depth in mm, aligned to color frame
        resize_rate: float,               # SAT-HMR input_size / max(orig_w, orig_h)
        sample_radius_px: int = 4,        # neighborhood radius when sampling depth (for noise)
    ) -> JointDepthReport:
        """Compute predicted vs measured depth for all 24 joints.

        Returns per-joint deltas useful for front/back diagnosis.
        """
        # 1) joint 3D positions in camera frame (still meters, Y-DOWN OpenCV)
        joints_cam = self.J_regressor @ mesh_verts_cam        # (24, 3)

        # 2) project to 2D pixels in the SAT-HMR input_size frame
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        xs = joints_cam[:, 0]
        ys = joints_cam[:, 1]
        zs = joints_cam[:, 2]
        # Avoid divide-by-zero for joints behind camera
        zs_safe = np.where(np.abs(zs) < 1e-6, 1e-6, zs)
        u_input = fx * xs / zs_safe + cx
        v_input = fy * ys / zs_safe + cy

        # 3) map back to ORIGINAL frame coords (SAT-HMR resized frame → orig)
        u_orig = u_input / max(resize_rate, 1e-8)
        v_orig = v_input / max(resize_rate, 1e-8)

        # 4) sample depth at each pixel (with small neighborhood median for noise)
        h, w = depth_mm.shape
        measured_mm = np.zeros(24, dtype=np.float64)
        for j in range(24):
            cx_px = int(round(u_orig[j]))
            cy_px = int(round(v_orig[j]))
            if not (0 <= cx_px < w and 0 <= cy_px < h):
                measured_mm[j] = 0.0
                continue
            r = sample_radius_px
            x0 = max(0, cx_px - r)
            x1 = min(w, cx_px + r + 1)
            y0 = max(0, cy_px - r)
            y1 = min(h, cy_px + r + 1)
            patch = depth_mm[y0:y1, x0:x1]
            valid = patch[patch > 0]
            measured_mm[j] = float(np.median(valid)) if valid.size else 0.0

        # 5) predicted depths (mm)
        predicted_mm = zs * 1000.0

        return JointDepthReport(
            joint_names=list(self.JOINT_NAMES),
            predicted_mm=predicted_mm,
            measured_mm=measured_mm,
            delta_mm=(measured_mm - predicted_mm),
            pixel_xy=np.stack([u_orig, v_orig], axis=-1),
        )


    def corrected_root_smpl(
        self,
        mesh_verts_cam: np.ndarray,       # (V, 3) OpenCV Y-down camera frame, meters
        intrinsics: np.ndarray,           # (3, 3) intrinsics in SAT-HMR input_size frame
        depth_mm: np.ndarray,             # (H, W) uint16 depth (mm) aligned to color
        resize_rate: float,
        sample_radius_px: int = 10,       # A: larger patch (21×21) → resilient to bbox jitter
        max_delta_m: float = 1.5,
        person_id: int = 0,
        pixel_smooth_alpha: float = 0.4,  # B: EMA on projected pixel (0=off, 1=no smoothing)
        max_z_jump_m: float = 0.3,        # C: sanity cap on frame-to-frame Z change
    ) -> Optional[np.ndarray]:
        """Back-project pelvis 2D pixel + measured depth → SMPL Y-up meters.

        Stability improvements:
          A) Wider sampling patch + 25th percentile (foreground bias) — robust
             to bbox jitter that projects pelvis onto the edge of the person.
          B) Temporal EMA on the projected pelvis pixel (per person_id).
          C) Reject Z jumps > max_z_jump_m vs previous accepted Z (outlier).
        """
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        # Pelvis 3D in OpenCV cam frame (from mesh).
        pelvis_cam = self.J_regressor[0] @ mesh_verts_cam    # (3,)
        z_pred = float(pelvis_cam[2])
        if z_pred < 1e-3:
            return None

        # Project pelvis to input-frame pixel, then to original-frame pixel.
        u_input = fx * pelvis_cam[0] / z_pred + cx
        v_input = fy * pelvis_cam[1] / z_pred + cy
        rr = max(resize_rate, 1e-8)
        u_orig_raw = u_input / rr
        v_orig_raw = v_input / rr

        # B: temporal EMA on pixel coords.
        state = self._state.setdefault(int(person_id), {})
        if "px" in state:
            u_orig = pixel_smooth_alpha * u_orig_raw + (1 - pixel_smooth_alpha) * state["px"]
            v_orig = pixel_smooth_alpha * v_orig_raw + (1 - pixel_smooth_alpha) * state["py"]
        else:
            u_orig, v_orig = u_orig_raw, v_orig_raw
        state["px"], state["py"] = u_orig, v_orig

        h, w = depth_mm.shape
        cx_px = int(round(u_orig))
        cy_px = int(round(v_orig))
        if not (0 <= cx_px < w and 0 <= cy_px < h):
            return None

        # A: wider patch + 25th percentile.
        r = sample_radius_px
        x0 = max(0, cx_px - r); x1 = min(w, cx_px + r + 1)
        y0 = max(0, cy_px - r); y1 = min(h, cy_px + r + 1)
        patch = depth_mm[y0:y1, x0:x1]
        valid = patch[patch > 0]
        if valid.size < 20:
            return None
        z_meas_m = float(np.percentile(valid, 25)) / 1000.0
        if not np.isfinite(z_meas_m) or z_meas_m <= 0.2 or z_meas_m > 8.0:
            return None
        if abs(z_meas_m - z_pred) > max_delta_m:
            return None

        # C: outlier rejection vs previous Z (fresh person → accept first sample).
        last_z = state.get("z")
        if last_z is not None and abs(z_meas_m - last_z) > max_z_jump_m:
            return None
        state["z"] = z_meas_m

        # Back-project through smoothed input-frame pixel.
        u_input_s = u_orig * rr
        v_input_s = v_orig * rr
        x_cam = (u_input_s - cx) * z_meas_m / fx
        y_cam = (v_input_s - cy) * z_meas_m / fy
        return np.array([x_cam, -y_cam, z_meas_m], dtype=np.float64)


    def corrected_root_from_bbox(
        self,
        bbox: np.ndarray,                 # (4,) x1,y1,x2,y2 in original frame pixels
        depth_mm: np.ndarray,             # (H, W) uint16 aligned depth
        current_root_smpl: np.ndarray,    # (3,) current SMPL Y-up meters
        percentile: float = 25.0,
    ) -> Optional[np.ndarray]:
        """Depth correction from a person bbox — foreground-biased median.

        Samples the middle 60% width × 30-70% height of the bbox (roughly the
        torso), keeps valid depths, returns the ``percentile``-th (default 25th)
        for foreground bias — this suppresses stray background pixels (walls
        behind the person) that a naive center-median would include.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        if w < 20 or h < 40:
            return None
        # Torso region: middle-width × upper-mid height (pelvis area).
        tx1 = int(x1 + 0.20 * w)
        tx2 = int(x1 + 0.80 * w)
        ty1 = int(y1 + 0.30 * h)
        ty2 = int(y1 + 0.70 * h)

        H, W = depth_mm.shape
        tx1 = max(0, tx1); tx2 = min(W, tx2)
        ty1 = max(0, ty1); ty2 = min(H, ty2)
        if tx2 <= tx1 or ty2 <= ty1:
            return None

        patch = depth_mm[ty1:ty2, tx1:tx2]
        valid = patch[patch > 0]
        if valid.size < 20:
            return None
        # Lower-quartile: closer than median → biases toward the person
        # (foreground) over occasional background pixels within the bbox.
        z_m = float(np.percentile(valid, percentile)) / 1000.0
        if z_m <= 0.2 or z_m > 8.0:
            return None
        new_root = current_root_smpl.copy().astype(np.float64)
        new_root[2] = z_m
        return new_root


def format_limb_summary(report: JointDepthReport) -> str:
    """One-line diagnostic string suitable for periodic logging."""
    summary = report.summarize_limbs()
    parts = []
    for name in ("l_arm", "r_arm", "l_leg", "r_leg", "spine", "head"):
        v = summary.get(name)
        parts.append(f"{name}={v:+.0f}mm" if v is not None else f"{name}=--")
    return " ".join(parts)
