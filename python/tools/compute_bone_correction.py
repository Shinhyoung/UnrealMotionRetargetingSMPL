"""Build per-SMPL-joint correction quaternions from a UE ref-pose JSON.

Input
-----
``ue_ref_pose.json`` produced by ``tools/ue_export_ref_pose.py`` (run inside
UE Editor). Structure::

    {
      "source_mesh": "/Game/...",
      "bones": {
        "pelvis":     {"parent": "root",     "world_translation": [...], "world_rotation_xyzw": [...]},
        "thigh_l":    {"parent": "pelvis",   "world_translation": [...], "world_rotation_xyzw": [...]},
        ...
      }
    }

Output
------
An ``.npz`` (default: ``bone_correction.npz``) with two ``(24, 4)`` xyzw
arrays — see ``transform/bone_correction.py`` for the format and math.

Usage
-----
    python tools/compute_bone_correction.py \\
        --input  ../ue_ref_pose.json \\
        --output ../bone_correction.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# make /python importable when run standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.bone_mapping import (
    NUM_SMPL_JOINTS,
    SMPL_PARENTS,
    SMPL_TO_UE_BONE,
)
from transform.rotation import normalize_quat


IDENTITY_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def _inverse_unit_quat(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).copy()
    out[..., :3] *= -1.0
    return out


def _bone_world_quat(bones: dict, name: str) -> np.ndarray:
    """Look up a bone's world_rotation_xyzw. Return identity if missing."""
    if not name or name not in bones:
        return IDENTITY_QUAT_XYZW.copy()
    q = np.asarray(bones[name]["world_rotation_xyzw"], dtype=np.float64)
    return normalize_quat(q)


def build_correction(ue_ref: dict, verbose: bool = True):
    bones = ue_ref["bones"]

    parent_inv = np.zeros((NUM_SMPL_JOINTS, 4), dtype=np.float64)
    own = np.zeros((NUM_SMPL_JOINTS, 4), dtype=np.float64)
    for j in range(NUM_SMPL_JOINTS):
        ue_name = SMPL_TO_UE_BONE.get(j, "")
        smpl_parent = SMPL_PARENTS[j]

        # Prefer the parent listed in the JSON (actual UE bone hierarchy).
        # If the JSON has empty/missing parent (some UE Python API paths don't
        # expose parent lookup), fall back to SMPL kinematic parent's UE bone.
        # The fallback is safe for the mannequin because UE parent chain aligns
        # with SMPL joint semantics; intermediate UE bones like spine_04/05
        # keep identity local at rest so skipping them is inert.
        ue_parent_name = ""
        if ue_name and ue_name in bones:
            ue_parent_name = bones[ue_name].get("parent", "") or ""
        if not ue_parent_name and smpl_parent >= 0:
            ue_parent_name = SMPL_TO_UE_BONE.get(smpl_parent, "")

        own_q = _bone_world_quat(bones, ue_name)
        parent_q = _bone_world_quat(bones, ue_parent_name)
        parent_inv[j] = _inverse_unit_quat(parent_q)
        own[j] = own_q

        if verbose:
            print(
                f"  [{j:2d}] SMPL {SMPL_TO_UE_BONE.get(j, '<none>'):>12s}"
                f"  ue_parent={ue_parent_name:>12s}"
                f"  own={own[j].round(3).tolist()}"
            )
    return parent_inv, own


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "-i", required=True,
                   help="Path to ue_ref_pose.json (from tools/ue_export_ref_pose.py).")
    p.add_argument("--output", "-o", default="bone_correction.npz",
                   help="Output .npz path (default: bone_correction.npz in CWD).")
    p.add_argument("--quiet", action="store_true", help="Suppress per-joint dump.")
    args = p.parse_args()

    in_path = Path(args.input).resolve()
    if not in_path.is_file():
        print(f"[error] input not found: {in_path}", file=sys.stderr)
        return 1

    with in_path.open("r", encoding="utf-8") as f:
        ue_ref = json.load(f)
    if "bones" not in ue_ref:
        print(f"[error] {in_path} has no 'bones' key.", file=sys.stderr)
        return 1

    print(f"[info] loaded {len(ue_ref['bones'])} UE bones from {in_path}")
    parent_inv, own = build_correction(ue_ref, verbose=not args.quiet)

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), parent_world_inv=parent_inv, own_world=own)
    print(f"[info] wrote correction to {out_path}")
    print()
    print("이제 파이프라인 실행 시 이 파일을 지정하세요:")
    print(f"  python main.py --source realsense --bone-correction {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
