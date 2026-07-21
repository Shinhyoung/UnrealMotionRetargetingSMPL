"""SMPL forward (LBS) — DEBUG ONLY.

Isolated here so the production pipeline never depends on ``smplx`` /
``trimesh`` / ``pyrender`` (CLAUDE.md §8.2). If SAT-HMR ships its own
visualization utility, prefer that instead of re-implementing.

Import lazily to keep the failure surface small when the model files are
missing.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class SMPLForward:
    """Thin wrapper around ``smplx.SMPL`` for debug mesh generation.

    Raises RuntimeError with a clear message if smplx or the model file is
    unavailable — the caller (debug overlay) should catch and disable mesh
    rendering rather than crash the pipeline.
    """

    def __init__(self, model_path: str, gender: str = "neutral", device: str = "cpu"):
        try:
            import smplx  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "smplx not installed. `pip install smplx` for debug mesh rendering."
            ) from e
        try:
            import torch  # type: ignore
        except ImportError as e:
            raise RuntimeError("torch required for smplx forward.") from e

        self._torch = torch
        self._smplx = smplx
        self.device = device
        try:
            self.model = smplx.SMPL(
                model_path=model_path,
                gender=gender,
                batch_size=1,
            ).to(device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SMPL model from {model_path!r}. "
                "Download SMPL model files (EULA required) and set the correct path."
            ) from e

    def forward(
        self,
        global_orient: np.ndarray,
        body_pose: np.ndarray,
        betas: Optional[np.ndarray] = None,
        transl: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return (6890, 3) vertices in SMPL space (meters)."""
        torch = self._torch
        go = torch.as_tensor(global_orient, dtype=torch.float32, device=self.device).reshape(1, 3)
        bp = torch.as_tensor(body_pose, dtype=torch.float32, device=self.device).reshape(1, 69)
        b = betas if betas is not None else np.zeros(10, dtype=np.float32)
        b_t = torch.as_tensor(b, dtype=torch.float32, device=self.device).reshape(1, 10)
        t_t = None
        if transl is not None:
            t_t = torch.as_tensor(transl, dtype=torch.float32, device=self.device).reshape(1, 3)
        with torch.no_grad():
            out = self.model(global_orient=go, body_pose=bp, betas=b_t, transl=t_t)
        return out.vertices[0].detach().cpu().numpy()
