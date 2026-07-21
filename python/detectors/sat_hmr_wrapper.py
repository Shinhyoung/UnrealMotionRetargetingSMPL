"""SAT-HMR inference wrapper.

The real detector requires the SAT-HMR repo + checkpoint. This module defines:

* ``PoseDetection`` — the per-person output payload used across the pipeline.
* ``SATHMRDetector`` — real-model interface (raises until the SAT-HMR clone
  is wired up).
* ``MockSATHMRDetector`` — synthetic multi-person output so the entire
  pipeline (tracking → transform → network → debug) can be exercised without
  weights. Use ``python main.py --mock`` while environment/weights are pending.

Per CLAUDE.md §4.1, only ``pose`` / ``global_orient`` / ``root_position`` are
required for realtime transmission. ``beta`` and ``bbox`` are DEBUG-ONLY.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from config.bone_mapping import NUM_SMPL_JOINTS


@dataclass
class PoseDetection:
    """Per-person single-shot output from SAT-HMR (or Mock).

    Convention (matches SMPL-X):
      * ``global_orient``: (3,) axis-angle for the pelvis / root (joint 0).
      * ``body_pose``:     (23, 3) axis-angle for joints 1..23, local to parent.
      * ``root_position``: (3,) camera-space translation, meters, SMPL basis.
      * ``beta``:          (10,) shape params. DEBUG ONLY (mesh generation).
      * ``bbox``:          (4,) x0 y0 x1 y1 in pixels. Debug/tracking helper.
      * ``person_id``:     assigned by :class:`PersonTracker`, not the model.
    """
    global_orient: np.ndarray
    body_pose: np.ndarray
    root_position: np.ndarray
    beta: Optional[np.ndarray] = None
    bbox: Optional[np.ndarray] = None
    person_id: Optional[int] = None
    # DEBUG-ONLY: SMPL vertices in the original camera frame and camera intrinsics.
    # Populated only when the detector was built with ``keep_debug_output=True``.
    # Never transmit these (CLAUDE.md §4.1).
    debug_verts: Optional[np.ndarray] = None      # (V, 3) in OpenCV camera (Y-down) meters
    debug_intrinsics: Optional[np.ndarray] = None  # (3, 3) pinhole intrinsics for input_size frame
    debug_resize_rate: Optional[float] = None      # frame-size -> input_size scale

    def full_axis_angles(self) -> np.ndarray:
        """Return (24, 3) axis-angle: [global_orient, body_pose]."""
        go = np.asarray(self.global_orient, dtype=np.float64).reshape(3)
        bp = np.asarray(self.body_pose, dtype=np.float64).reshape(NUM_SMPL_JOINTS - 1, 3)
        return np.vstack([go[None, :], bp])


class SATHMRDetector:
    """Real SAT-HMR inference wrapper.

    See ``docs/sat_hmr_integration.md`` for the layout requirements.

    Output dict from ``models/sat_model.py::Model.forward`` (verified against
    the 2026-07-13 clone):

        pred_poses      (B, Q, 24, 3) axis-angle in **OpenCV camera coord**
        pred_betas      (B, Q, 10)
        pred_transl     (B, Q, 3)     translation, OpenCV camera (Y-DOWN)
        pred_boxes      (B, Q, 4)     normalized cxcywh in ``input_size`` frame
        pred_confs      (B, Q)        query confidence
        pred_verts      (B, Q, V, 3)  debug-only
        pred_intrinsics (B, 3, 3)     debug/2D projection

    We convert the OpenCV Y-down output to SMPL Y-up (via
    :func:`transform.coordinate.opencv_cam_yflip_*`) before wrapping into
    :class:`PoseDetection` — the rest of the pipeline stays Y-up.
    """

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        sat_hmr_root: str,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        conf_thresh: float = 0.3,
        cfg_run: str = "demo",
        cfg_model: str = "default",
        opencv_camera_input: bool = True,
        amp_dtype: str = "bfloat16",  # {"none","float16","bfloat16"} — Blackwell (sm_120) needs half for xformers
        keep_debug_output: bool = False,  # populate PoseDetection.debug_* (verts, intrinsics) — CLAUDE.md §8
        input_size_override: Optional[int] = None,  # override cfg's input_size (e.g. 644 for ~2-3x speedup)
    ):
        self.sat_hmr_root = _abspath(sat_hmr_root)
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.conf_thresh = float(conf_thresh)
        self.cfg_run = cfg_run
        self.cfg_model = cfg_model
        self.opencv_camera_input = bool(opencv_camera_input)
        self.amp_dtype_str = amp_dtype
        self.keep_debug_output = bool(keep_debug_output)
        self.input_size_override = input_size_override

        self._torch = None
        self._amp_dtype = None
        self._nested_tensor_from_tensor_list = None
        self._model = None
        self._input_size: int = 0
        self._patch_size: int = 0
        self._smpl_faces: Optional[np.ndarray] = None

        self._load_model()

    @property
    def smpl_faces(self) -> Optional[np.ndarray]:
        """(F, 3) int face indices for the SMPL mesh, or None if model not loaded."""
        return self._smpl_faces

    @property
    def input_size(self) -> int:
        return self._input_size

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        import os
        import sys
        import contextlib

        import yaml
        import torch
        self._torch = torch

        if not os.path.isdir(self.sat_hmr_root):
            raise FileNotFoundError(f"SAT-HMR repo not found at: {self.sat_hmr_root}")

        # Configs use CWD-relative paths (./weights/...). Import + build must
        # happen with the SAT-HMR root as cwd, and the root must be on sys.path.
        if self.sat_hmr_root not in sys.path:
            sys.path.insert(0, self.sat_hmr_root)

        cfg = self._load_cfgs()

        # Blackwell (sm_120) has no cutlass-based xformers kernel yet, and the
        # Windows xformers wheel doesn't ship fa2/fa3. Fall back to PyTorch's
        # native SDPA. Safe for batch=1 inference where BlockDiagonalMask is
        # degenerate (only one block).
        self._patch_xformers_for_blackwell()

        with contextlib.chdir(self.sat_hmr_root):
            from models import build_sat_model  # deferred so imports find configs/
            from utils.misc import nested_tensor_from_tensor_list  # for infer path
            self._nested_tensor_from_tensor_list = nested_tensor_from_tensor_list

            model, _ = build_sat_model(cfg, set_criterion=False)

            ckpt_path = self.checkpoint_path or cfg.pretrain_path
            if not os.path.isabs(ckpt_path):
                ckpt_path = os.path.join(self.sat_hmr_root, ckpt_path)
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(
                    f"SAT-HMR checkpoint not found: {ckpt_path}"
                )
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            # Resize checkpoint's 2D positional embeddings if we downshifted input_size.
            self._resize_pos_embeds(model, state_dict)
            # ``cam_intrinsics`` is a registered buffer baked at the training
            # input_size. When we downshift input_size, keep our fresh init
            # (already correct for the new size) — the checkpoint value would
            # overwrite (fx, cx, cy) to the training scale and drop the mesh in
            # the wrong quadrant of the valid area.
            if self.input_size_override is not None and "cam_intrinsics" in state_dict:
                print(f"[sat_hmr] dropping checkpoint's cam_intrinsics "
                      f"(training-scale) so our input_size={self.input_size_override} "
                      f"init survives")
                del state_dict["cam_intrinsics"]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[sat_hmr] missing keys: {len(missing)} (showing first 5): {missing[:5]}")
            if unexpected:
                print(f"[sat_hmr] unexpected keys: {len(unexpected)} (showing first 5): {unexpected[:5]}")

            model.to(self.device).eval()
            self._model = model
            self._input_size = int(cfg.input_size)
            # SAT-HMR uses patch=56 when SAT is enabled, else 14.
            self._patch_size = 56 if bool(cfg.sat_cfg.get("use_sat", False)) else 14

            # Cache the SMPL face topology for debug mesh rendering (no state_dict entry).
            self._smpl_faces = np.asarray(model.human_model.faces, dtype=np.int32)

        # Autocast dtype (Blackwell needs fp16/bf16 for xformers memory_efficient_attention).
        self._amp_dtype = {
            "none": None,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(self.amp_dtype_str)
        if self.amp_dtype_str not in ("none", "float16", "bfloat16"):
            raise ValueError(f"amp_dtype must be one of none/float16/bfloat16, got {self.amp_dtype_str!r}")

        print(f"[sat_hmr] loaded {ckpt_path} on {self.device} "
              f"(input_size={self._input_size}, patch={self._patch_size}, "
              f"conf_thresh={self.conf_thresh}, opencv_camera_input={self.opencv_camera_input}, "
              f"amp={self.amp_dtype_str})")

    def _load_cfgs(self):
        import os
        import argparse
        import yaml

        with open(os.path.join(self.sat_hmr_root, "configs", "run", f"{self.cfg_run}.yaml")) as f:
            run_cfg = yaml.safe_load(f) or {}
        with open(os.path.join(self.sat_hmr_root, "configs", "models", f"{self.cfg_model}.yaml")) as f:
            model_cfg = yaml.safe_load(f) or {}

        merged = {}
        merged.update(run_cfg)
        merged.update(model_cfg)
        # build_sat_model / encoder builder inspect args.mode to decide whether to
        # load DINOv2 pretrain weights. In 'infer' mode they skip that.
        merged.setdefault("mode", "infer")
        merged.setdefault("pretrain", True)

        # Optional input-size override. Must be a multiple of encoder_patch_size × 4
        # (= 56 when use_sat=True with patch=14). SAT-HMR asserts this at model init.
        # Valid smaller sizes: 616, 672, 728, ... The DINOv2 position embed
        # interpolates cleanly to the new size.
        if self.input_size_override is not None:
            new_size = int(self.input_size_override)
            if new_size % 56 != 0:
                raise ValueError(
                    f"input_size_override={new_size} is not a multiple of 56 "
                    "(required when SAT is on). Try 616, 672, 728, ..."
                )
            merged["input_size"] = new_size
        return argparse.Namespace(**merged)

    def _resize_pos_embeds(self, model, state_dict) -> None:
        """Bicubic-resize checkpoint pos-embed tensors whose spatial dims don't match.

        Two shapes appear in SAT-HMR/DINOv2 checkpoints:

        1) ``encoder_pos_embeds`` — shape (H, W, C). SAT-HMR's own 2D pos map,
           scales linearly with ``input_size``.
        2) ``encoder.pos_embed`` — shape (1, 1 + N, C) from DINOv2 (cls-token +
           flattened HxW patches). N = (input_size / patch_size)^2. Cls-token
           row is preserved verbatim; the remaining rows are reshaped to
           (Hc, Wc, C), interpolated, and reflattened.

        Anything else with a shape mismatch is left untouched (surfaces later
        as ``missing_keys`` from ``load_state_dict(strict=False)``).
        """
        import torch
        import torch.nn.functional as F

        model_state = model.state_dict()
        for k, v_ckpt in list(state_dict.items()):
            if k not in model_state:
                continue
            v_model = model_state[k]
            if v_ckpt.shape == v_model.shape:
                continue

            # Case 1: (H, W, C).
            if v_ckpt.dim() == 3 and v_model.dim() == 3 \
                    and v_ckpt.shape[-1] == v_model.shape[-1]:
                H_new, W_new, C = v_model.shape
                pe = v_ckpt.float().permute(2, 0, 1).unsqueeze(0)   # (1, C, H_old, W_old)
                pe_resized = F.interpolate(pe, size=(H_new, W_new),
                                           mode="bicubic", align_corners=False)
                state_dict[k] = pe_resized.squeeze(0).permute(1, 2, 0).to(v_ckpt.dtype).contiguous()
                print(f"[sat_hmr] resized {k}: {tuple(v_ckpt.shape)} -> "
                      f"{tuple(state_dict[k].shape)}")
                continue

            # Case 2: DINOv2 style (1, 1+N, C).
            if v_ckpt.dim() == 3 and v_model.dim() == 3 \
                    and v_ckpt.shape[0] == 1 and v_model.shape[0] == 1 \
                    and v_ckpt.shape[-1] == v_model.shape[-1]:
                N_old = v_ckpt.shape[1] - 1
                N_new = v_model.shape[1] - 1
                Hc_old = int(round(N_old ** 0.5))
                Hc_new = int(round(N_new ** 0.5))
                if Hc_old * Hc_old != N_old or Hc_new * Hc_new != N_new:
                    continue  # non-square, bail
                cls = v_ckpt[:, :1, :]                      # (1, 1, C)
                patch = v_ckpt[:, 1:, :]                    # (1, N_old, C)
                C = patch.shape[-1]
                patch2d = (patch.float()
                           .reshape(1, Hc_old, Hc_old, C)
                           .permute(0, 3, 1, 2))            # (1, C, Hc_old, Hc_old)
                patch_r = F.interpolate(patch2d, size=(Hc_new, Hc_new),
                                        mode="bicubic", align_corners=False)
                patch_r = patch_r.permute(0, 2, 3, 1).reshape(1, N_new, C).to(v_ckpt.dtype)
                state_dict[k] = torch.cat([cls, patch_r], dim=1).contiguous()
                print(f"[sat_hmr] resized {k}: N {N_old} -> {N_new} "
                      f"(cls+{Hc_new}x{Hc_new})")
                continue

    def _patch_xformers_for_blackwell(self) -> None:
        """Replace xformers.ops.memory_efficient_attention with a native SDPA fallback.

        Rationale: xformers pre-built wheels (as of 0.0.35 on Windows) only include
        the cutlass CPU/GPU kernel which caps at compute capability 9.0. On
        Blackwell (sm_120) the dispatcher raises NotImplementedError. PyTorch's
        native ``scaled_dot_product_attention`` works on sm_120 and matches the
        math for the batch=1 inference path (BlockDiagonalMask degenerates to
        full attention when there is a single block).
        """
        import torch.nn.functional as F
        try:
            import xformers.ops as _xops  # noqa: F401
        except ImportError:
            return  # nothing to patch

        def _me_attention_sdpa(q, k, v, attn_bias=None, p=0.0, scale=None, **kw):
            # xformers layout: (B, S, H, D). SDPA expects (B, H, S, D).
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            attn_mask = None  # BlockDiagonalMask with a single block == no mask
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                attn_mask=attn_mask,
                dropout_p=float(p),
                is_causal=False,
                scale=scale,
            )
            return out.transpose(1, 2).contiguous()

        _xops.memory_efficient_attention = _me_attention_sdpa
        # Some call sites import directly:
        import sys as _sys
        for name, mod in list(_sys.modules.items()):
            if name.startswith("xformers.") and hasattr(mod, "memory_efficient_attention"):
                mod.memory_efficient_attention = _me_attention_sdpa
        print("[sat_hmr] xformers.memory_efficient_attention → native SDPA (Blackwell workaround)")

    # ------------------------------------------------------------------
    # Preprocessing (mirrors SAT-HMR datasets/base.py infer path)
    # ------------------------------------------------------------------

    def _preprocess(self, frame_bgr: np.ndarray):
        import math
        import cv2

        torch = self._torch
        h_orig, w_orig = frame_bgr.shape[:2]

        # Resize longest side to input_size while preserving aspect.
        if w_orig >= h_orig:
            resize_rate = self._input_size / w_orig
            new_w = self._input_size
            new_h = int(round(resize_rate * h_orig))
        else:
            resize_rate = self._input_size / h_orig
            new_h = self._input_size
            new_w = int(round(resize_rate * w_orig))
        img = cv2.resize(frame_bgr, (new_w, new_h))

        # Pad to multiple of patch_size (rows/cols independently).
        pad_h = math.ceil(new_h / self._patch_size) * self._patch_size
        pad_w = math.ceil(new_w / self._patch_size) * self._patch_size
        padded = np.zeros((pad_h, pad_w, 3), dtype=img.dtype)
        padded[:new_h, :new_w] = img

        # BGR → RGB, HWC → CHW, [0,1] float, ImageNet normalize.
        rgb = padded[:, :, ::-1].astype(np.float32) / 255.0
        rgb = (rgb - np.asarray(self._IMAGENET_MEAN, dtype=np.float32)) \
              / np.asarray(self._IMAGENET_STD, dtype=np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        tensor = tensor.to(self.device, non_blocking=True)

        # The model consumes a list of tensors of possibly different sizes.
        samples = [tensor]
        # img_size stores the VALID (unpadded) region as (H, W) — matches upstream convention.
        img_size = torch.tensor([new_h, new_w], device=self.device)
        targets = [{"img_size": img_size}]
        return samples, targets, resize_rate

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, frame_bgr: np.ndarray) -> List[PoseDetection]:
        torch = self._torch
        samples, targets, resize_rate = self._preprocess(frame_bgr)

        with torch.no_grad():
            if self._amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
                    out = self._model(samples, targets)
            else:
                out = self._model(samples, targets)

        # pred_confs comes as (1, Q, 1) — squeeze the trailing dim.
        confs = out["pred_confs"][0].detach().float().cpu().numpy().reshape(-1)  # (Q,)
        keep = np.where(confs > self.conf_thresh)[0]
        if keep.size == 0:
            return []

        # pred_poses is flattened as (Q, 72) — reshape to (N, 24, 3).
        poses = (
            out["pred_poses"][0][keep].detach().float().cpu().numpy()
            .reshape(-1, NUM_SMPL_JOINTS, 3)
        )                                                                       # (N, 24, 3)
        transl = out["pred_transl"][0][keep].detach().float().cpu().numpy()     # (N, 3)
        betas = out["pred_betas"][0][keep].detach().float().cpu().numpy()       # (N, 10)
        boxes = out["pred_boxes"][0][keep].detach().float().cpu().numpy()       # (N, 4) cxcywh

        # cxcywh (normalized to input_size) → xyxy in ORIGINAL image pixel coords.
        boxes_xyxy_input = _cxcywh_to_xyxy(boxes) * self._input_size
        boxes_orig = boxes_xyxy_input / max(resize_rate, 1e-8)

        # DEBUG-ONLY tap (verts + intrinsics) — kept in original OpenCV camera frame
        # so we can re-project through SAT-HMR's own vis_meshes_img() utility.
        debug_verts_cam = None
        debug_intrinsics = None
        if self.keep_debug_output:
            # pred_verts shape (1, Q, V, 3); slice per kept query.
            verts_full = out["pred_verts"][0][keep].detach().float().cpu().numpy()  # (N, V, 3)
            debug_verts_cam = verts_full
            debug_intrinsics = out["pred_intrinsics"][0].reshape(3, 3).detach().float().cpu().numpy()

        # OpenCV camera (Y-down) → SMPL Y-up for the downstream pipeline.
        if self.opencv_camera_input:
            from transform.coordinate import (
                opencv_cam_yflip_axis_angle,
                opencv_cam_yflip_position,
            )
            # Only global_orient (idx 0) is a WORLD rotation. body_pose is joint-local
            # and lives in the SMPL kinematic tree — no world basis change needed.
            poses = poses.copy()
            poses[:, 0, :] = opencv_cam_yflip_axis_angle(poses[:, 0, :])
            transl = opencv_cam_yflip_position(transl)

        detections: List[PoseDetection] = []
        for i in range(poses.shape[0]):
            det = PoseDetection(
                global_orient=poses[i, 0].astype(np.float64),
                body_pose=poses[i, 1:24].astype(np.float64),
                root_position=transl[i].astype(np.float64),
                beta=betas[i].astype(np.float64),
                bbox=boxes_orig[i].astype(np.float64),
            )
            if self.keep_debug_output:
                det.debug_verts = debug_verts_cam[i]
                det.debug_intrinsics = debug_intrinsics
                det.debug_resize_rate = float(resize_rate)
            detections.append(det)
        return detections


def _cxcywh_to_xyxy(cxcywh: np.ndarray) -> np.ndarray:
    cx, cy, w, h = cxcywh[..., 0], cxcywh[..., 1], cxcywh[..., 2], cxcywh[..., 3]
    return np.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=-1)


def _abspath(path: str) -> str:
    import os
    return os.path.abspath(os.path.expanduser(path))


class MockSATHMRDetector:
    """Synthetic detector: smoothly moving people with small pose noise.

    Deterministic given ``seed``. Meant for pipeline plumbing tests, not
    perceptual verification.
    """

    def __init__(self, n_persons: int = 2, seed: int = 42, image_size=(1280, 720)):
        self.n_persons = int(n_persons)
        self.rng = np.random.default_rng(seed)
        self.image_size = image_size  # (w, h) for synthetic bbox
        self._frame = 0

    def infer(self, frame_bgr: np.ndarray) -> List[PoseDetection]:
        self._frame += 1
        t = self._frame * 0.05
        w, h = self.image_size

        detections: List[PoseDetection] = []
        for i in range(self.n_persons):
            # Root motion — smooth sinusoid so the tracker has a target to lock onto.
            phase = i * 2.0
            x = 0.5 * np.sin(t + phase)
            y = 0.05 * np.sin(t * 0.3 + phase)
            z = 3.0 + 0.4 * i + 0.1 * np.cos(t * 0.5 + phase)
            root = np.array([x, y, z], dtype=np.float64)

            # Small per-joint jitter; global orient roughly facing camera.
            body_pose = self.rng.normal(0.0, 0.03, size=(NUM_SMPL_JOINTS - 1, 3))
            global_orient = np.array([np.pi, 0.0, 0.0], dtype=np.float64)

            # Debug fields.
            beta = np.zeros(10, dtype=np.float64)
            cx = int(w * (0.3 + 0.15 * i))
            cy = int(h * 0.5)
            bbox = np.array(
                [cx - 80, cy - 200, cx + 80, cy + 200], dtype=np.float64
            )

            detections.append(
                PoseDetection(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    root_position=root,
                    beta=beta,
                    bbox=bbox,
                )
            )
        return detections
