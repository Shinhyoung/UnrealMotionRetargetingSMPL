"""Runtime settings for the retargeting pipeline.

Defaults are baked in for MVP. Override via a JSON file or via code by
constructing sub-dataclasses. Config loading is intentionally simple —
we don't need YAML here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ModelSettings:
    checkpoint_path: str = ""
    device: str = "cuda"
    input_resolution: int = 512


@dataclass
class TrackingSettings:
    # Max frame-to-frame root-position jump (meters, SMPL space) allowed for the same person.
    max_distance: float = 2.0
    # Frames a track survives without a detection before its ID is retired.
    # Raised from 15 → 300 (~10s at 30 FPS) to stop ID inflation for a single-user
    # setup where the user briefly leaves frame or SAT-HMR blinks.
    max_missing_frames: int = 300


@dataclass
class TransformSettings:
    # Meters (SMPL) → centimeters (UE default). Set to 1.0 if UE project uses meters.
    smpl_to_ue_scale: float = 100.0


@dataclass
class RestPoseSettings:
    # Path to a .npy of shape (24, 4) — per-joint offset quaternions (xyzw) applied
    # AFTER coordinate swizzling. Empty string = identity offsets.
    offset_path: str = ""


@dataclass
class NetworkSettings:
    host: str = "127.0.0.1"
    port: int = 9527
    # Optional: cap outbound rate; 0 = as fast as inference allows.
    max_fps: float = 0.0


@dataclass
class DebugSettings:
    # Master switch. Default OFF per CLAUDE.md §8: perf must be measured with debug OFF.
    overlay: bool = False
    show_hud: bool = True
    show_bbox: bool = True
    show_skeleton: bool = False    # requires SMPL forward joints; enable when smplx wired
    show_mesh: bool = False        # requires smplx + SMPL model files
    window_name: str = "SAT-HMR Debug (do not use for perf test)"


@dataclass
class Settings:
    model: ModelSettings = field(default_factory=ModelSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    transform: TransformSettings = field(default_factory=TransformSettings)
    rest_pose: RestPoseSettings = field(default_factory=RestPoseSettings)
    network: NetworkSettings = field(default_factory=NetworkSettings)
    debug: DebugSettings = field(default_factory=DebugSettings)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Settings":
        if not path:
            return cls()
        p = Path(path)
        if not p.is_file():
            return cls()
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "Settings":
        s = cls()
        for k, v in data.items():
            if hasattr(s, k) and isinstance(v, dict):
                sub = getattr(s, k)
                for kk, vv in v.items():
                    if hasattr(sub, kk):
                        setattr(sub, kk, vv)
        return s

    def dump(self, path: str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
