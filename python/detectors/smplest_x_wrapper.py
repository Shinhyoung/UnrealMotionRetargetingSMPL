"""SMPLest-X detector wrapper — real-time SMPL-X pose estimation.

Repo: https://github.com/MotrixLab/SMPLest-X

Prerequisites (user setup, one-time):
    1. Clone SMPLest-X and its dependencies:
           git clone https://github.com/MotrixLab/SMPLest-X.git <SMPLEST_X_ROOT>
       Follow their install guide (conda env, pip install requirements).
    2. Download SMPL-X model files (smplx.is.tue.mpg.de) and place under
       <SMPLEST_X_ROOT>/human_models/human_model_files/smplx/.
    3. Download pretrained weights (e.g. SMPLest-X Huge from Hugging Face)
       to <SMPLEST_X_ROOT>/pretrained_models/smplest_x_h/smplest_x_h.pth.tar.
    4. YOLOv8x weights auto-download on first use via ultralytics.

Runtime:
    --detector smplest-x
    --smplest-x-root <SMPLEST_X_ROOT>
    --smplest-x-ckpt smplest_x_h            (ckpt directory name)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from detectors.sat_hmr_wrapper import PoseDetection


class SMPLestXDetector:
    """Real-time SMPL-X estimator. Output shape:
      - global_orient (3,)
      - body_pose (23, 3)      # SMPL layout — first 21 filled from SMPL-X body,
                                #                last 2 (L_hand/R_hand) left zero
      - left_hand_pose (15, 3)  # MANO order: index, middle, pinky, ring, thumb
      - right_hand_pose (15, 3)
      - root_position (3,)
    """

    def __init__(
        self,
        smplest_x_root: str,
        ckpt_name: str = "smplest_x_h",
        device: str = "cuda",
        conf_thresh: float = 0.5,
        input_size: int = 512,
        keep_debug_output: bool = False,
    ):
        self.smplest_x_root = str(Path(smplest_x_root).resolve())
        self.ckpt_name = ckpt_name
        self.device_name = device
        self.conf_thresh = float(conf_thresh)
        self.input_size = int(input_size)
        self.keep_debug_output = keep_debug_output
        self._load_model()

    def _load_model(self):
        """Load SMPLest-X. SMPLest-X repo is on sys.path with __init__.py files
        so 'from main.config import Config' etc. resolve to the SMPLest-X package
        (must be BEFORE any 'import main' in our own project)."""
        import os
        root = Path(self.smplest_x_root)
        root_str = str(root)

        prev_cwd = os.getcwd()
        os.chdir(root)   # SMPLest-X uses relative paths for model files
        # Remove any cached 'main' module from prior imports so it doesn't shadow
        # SMPLest-X's main/ package.
        for k in [k for k in sys.modules if k == "main" or k.startswith("main.")]:
            del sys.modules[k]

        try:
            if root_str not in sys.path:
                sys.path.insert(0, root_str)

            import torch
            from ultralytics import YOLO
            from main.config import Config
            from main.base import Tester
            from human_models.human_models import SMPLX
            self._torch = torch

            cfg_path = root / "pretrained_models" / self.ckpt_name / "config_base.py"
            if not cfg_path.is_file():
                raise FileNotFoundError(f"SMPLest-X config not found: {cfg_path}")
            cfg = Config.load_config(str(cfg_path))
            ckpt_path = root / "pretrained_models" / self.ckpt_name / f"{self.ckpt_name}.pth.tar"
            log_dir = root / "outputs" / f"realtime_{self.ckpt_name}" / "log"
            log_dir.mkdir(parents=True, exist_ok=True)
            cfg.update_config({
                "model": {"pretrained_model_path": str(ckpt_path)},
                "log": {
                    "exp_name": f"realtime_{self.ckpt_name}",
                    "log_dir": str(log_dir),
                },
            })
            cfg.prepare_log()

            SMPLX(cfg.model.human_model_path)   # init singleton

            self._detector = YOLO("yolov8x.pt")

            self._tester = Tester(cfg)
            self._tester._make_model()
            self._tester.model.eval()

            self._focal = cfg.model.focal
            self._princpt = cfg.model.princpt

            print(f"[smplest_x] loaded '{self.ckpt_name}' from {root} on {self.device_name}")
        finally:
            os.chdir(prev_cwd)

    def _detect_people(self, frame_bgr: np.ndarray):
        """Run YOLOv8x → list of (x1, y1, x2, y2, conf) bboxes for persons only."""
        # YOLO expects RGB
        results = self._detector(frame_bgr[:, :, ::-1], verbose=False)
        boxes = []
        for r in results:
            for b in r.boxes:
                if int(b.cls) != 0:      # class 0 = person in COCO
                    continue
                conf = float(b.conf)
                if conf < self.conf_thresh:
                    continue
                x1, y1, x2, y2 = b.xyxy.cpu().numpy().flatten().tolist()
                boxes.append((x1, y1, x2, y2, conf))
        return boxes

    def _run_smplx_on_crop(self, frame_bgr: np.ndarray, bbox):
        """Run SMPLest-X on one person crop. Returns dict of numpy arrays."""
        torch = self._torch
        from utils.data_utils import process_bbox, generate_patch_image

        img_h, img_w = frame_bgr.shape[:2]
        x1, y1, x2, y2, _ = bbox
        bb = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)   # xywh
        input_shape = self._tester.cfg.model.input_body_shape
        bb = process_bbox(bb, img_w, img_h, input_shape)
        if bb is None:
            return None

        img_patch, img2bb_trans, bb2img_trans = generate_patch_image(
            frame_bgr[:, :, ::-1], bb, 1.0, 0.0, False, input_shape
        )
        img_patch = img_patch.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_patch.transpose(2, 0, 1))[None].to(self.device_name)

        inputs = {"img": img_tensor}
        targets = {}
        meta_info = {}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self._tester.model(inputs, targets, meta_info, "test")

        def _to_np(t):
            return t.detach().float().cpu().numpy()[0]

        return {
            "root_pose": _to_np(out["smplx_root_pose"]).astype(np.float64).reshape(3),
            "body_pose": _to_np(out["smplx_body_pose"]).astype(np.float64).reshape(21, 3),
            "lhand_pose": _to_np(out["smplx_lhand_pose"]).astype(np.float64).reshape(15, 3),
            "rhand_pose": _to_np(out["smplx_rhand_pose"]).astype(np.float64).reshape(15, 3),
            "cam_trans": _to_np(out["cam_trans"]).astype(np.float64).reshape(3),
            "shape": _to_np(out["smplx_shape"]).astype(np.float64).reshape(-1),
            "bbox_orig": np.array([x1, y1, x2, y2], dtype=np.float64),
        }

    def infer(self, frame_bgr: np.ndarray) -> List[PoseDetection]:
        detections: List[PoseDetection] = []
        for bbox in self._detect_people(frame_bgr):
            out = self._run_smplx_on_crop(frame_bgr, bbox)
            if out is None:
                continue
            # SMPL body_pose = 23 joints; SMPL-X provides 21. Last 2 (SMPL L_hand,
            # R_hand joints) are superseded by MANO fingers → leave as identity.
            body_pose = np.zeros((23, 3), dtype=np.float64)
            body_pose[:21] = out["body_pose"]

            # SMPLest-X outputs pose in camera frame (OpenCV Y-down). Apply the
            # same yflip as sat_hmr_wrapper for consistency downstream.
            from transform.coordinate import (
                opencv_cam_yflip_axis_angle,
                opencv_cam_yflip_position,
            )
            go = opencv_cam_yflip_axis_angle(out["root_pose"].reshape(1, 3))[0]
            transl = opencv_cam_yflip_position(out["cam_trans"])

            detections.append(PoseDetection(
                global_orient=go,
                body_pose=body_pose,
                root_position=transl,
                beta=out["shape"][:10] if out["shape"].size >= 10 else out["shape"],
                bbox=out["bbox_orig"],
                left_hand_pose=out["lhand_pose"],
                right_hand_pose=out["rhand_pose"],
            ))
        return detections
