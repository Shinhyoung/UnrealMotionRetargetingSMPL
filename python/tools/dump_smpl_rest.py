"""Dump SMPL neutral model's rest-pose skeleton + T-pose mesh.

Purpose
-------
방향 2 (UE IK Retargeter) 를 위해 UE 에서 SMPL skeleton 을 자산으로 만들려면
SMPL β=0 rest pose 상태에서:
- 각 관절의 world 좌표 (24개)
- 관절 계층 (parent index)
- T-pose 시 mesh vertices + faces

이 정보를 뽑아 JSON + OBJ 로 저장한다. 이후 별도 도구 (Blender addon,
SMPL-to-FBX 스크립트, 또는 UE Python API) 로 FBX/glTF 를 생성하거나 직접
사용한다.

Usage
-----
    python tools/dump_smpl_rest.py \\
        --smpl-pkl <path-to-SMPL_NEUTRAL.pkl> \\
        --output-json ../smpl_rest.json \\
        --output-obj  ../smpl_tpose.obj

기본값은 SAT-HMR 저장소의 SMPL_NEUTRAL.pkl 을 찾음.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.bone_mapping import SMPL_JOINT_NAMES, SMPL_PARENTS


DEFAULT_SMPL_PKL = "c:/0.shinhyoung/Project/SAT-HMR/weights/smpl_data/smpl/SMPL_NEUTRAL.pkl"


def load_smpl(smpl_pkl_path: str):
    """Load SMPL via smplx. Returns the loaded model instance."""
    import torch
    import smplx

    # smplx.create looks for <model_path>/smpl/SMPL_NEUTRAL.pkl.
    # Our pkl lives at .../weights/smpl_data/smpl/SMPL_NEUTRAL.pkl,
    # so model_path must be .../weights/smpl_data (parent of "smpl" folder).
    smpl_parent = str(Path(smpl_pkl_path).parent.parent)
    model = smplx.create(
        model_path=smpl_parent,
        model_type="smpl",
        gender="neutral",
        num_betas=10,
    )
    return model, torch


def dump_rest(smpl_pkl: str, output_json: str, output_obj: str,
              output_weights: str = "") -> None:
    print(f"[dump_smpl_rest] loading SMPL from {smpl_pkl}")
    model, torch = load_smpl(smpl_pkl)

    # β=0, pose=0 → canonical T-pose
    betas = torch.zeros(1, 10)
    body_pose = torch.zeros(1, 23 * 3)          # 23 joints (excluding pelvis)
    global_orient = torch.zeros(1, 3)
    transl = torch.zeros(1, 3)

    with torch.no_grad():
        out = model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
        )

    # joints: (1, N, 3). SMPL has 24 body joints + extra joints for hands/face
    # in SMPL-X. We only want the first 24.
    joints = out.joints[0, :24].cpu().numpy()   # (24, 3), meters, Y-up
    vertices = out.vertices[0].cpu().numpy()    # (V, 3)
    faces = model.faces.astype(np.int32)        # (F, 3)

    print(f"[dump_smpl_rest] joints shape={joints.shape}, verts={vertices.shape}, faces={faces.shape}")

    # ---- JSON: skeleton hierarchy + joint positions ----
    bones = []
    for i, name in enumerate(SMPL_JOINT_NAMES):
        parent = int(SMPL_PARENTS[i])
        pos = joints[i].tolist()
        bones.append({
            "index": i,
            "name": name,
            "parent_index": parent,
            "parent_name": SMPL_JOINT_NAMES[parent] if parent >= 0 else "",
            "position_world": pos,      # (x, y, z) meters, SMPL Y-up
            "position_local": (         # relative to parent
                (joints[i] - joints[parent]).tolist() if parent >= 0 else pos
            ),
        })

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "convention": "SMPL Y-up, right-handed, meters. Positions are world (root at origin).",
            "num_joints": 24,
            "bones": bones,
        }, f, indent=2)
    print(f"[dump_smpl_rest] wrote skeleton JSON to {output_json}")

    # ---- OBJ: T-pose mesh (for reference / mesh-based FBX generation) ----
    Path(output_obj).parent.mkdir(parents=True, exist_ok=True)
    with open(output_obj, "w", encoding="utf-8") as f:
        f.write("# SMPL β=0 T-pose mesh (SMPL Y-up, meters)\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            # OBJ is 1-indexed
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    print(f"[dump_smpl_rest] wrote T-pose mesh OBJ to {output_obj}")

    # ---- NPZ: SMPL LBS weights (used by Blender FBX builder) ----
    if output_weights:
        # Extract weights from the underlying SMPL layer. smplx keeps the raw
        # .pkl arrays on the model instance; `lbs_weights` is the (V, 24) tensor.
        try:
            w = model.lbs_weights.detach().cpu().numpy()
        except AttributeError:
            # Some smplx versions expose it differently
            w = model.weights.detach().cpu().numpy() \
                if hasattr(model, "weights") else None
        if w is None:
            print("[warn] could not extract LBS weights — skipping NPZ")
        else:
            if w.ndim != 2 or w.shape[0] != vertices.shape[0]:
                raise ValueError(f"unexpected weights shape {w.shape}")
            Path(output_weights).parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(output_weights), weights=w.astype(np.float32))
            print(f"[dump_smpl_rest] wrote weights NPZ {w.shape} to {output_weights}")

    # ---- Print summary ----
    print()
    print("Joint hierarchy (index: name → parent):")
    for b in bones:
        parent_str = b["parent_name"] if b["parent_name"] else "(root)"
        pos_str = "[{:+.3f}, {:+.3f}, {:+.3f}]".format(*b["position_world"])
        print(f"  [{b['index']:2d}] {b['name']:<12s} → {parent_str:<10s}  pos={pos_str}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smpl-pkl", default=DEFAULT_SMPL_PKL)
    p.add_argument("--output-json", default="smpl_rest.json")
    p.add_argument("--output-obj", default="smpl_tpose.obj")
    p.add_argument("--output-weights", default="smpl_weights.npz",
                   help="Path to save SMPL LBS weights (6890,24) as .npz. "
                        "Consumed by make_smpl_fbx_blender.py --weights.")
    args = p.parse_args()

    smpl_pkl = Path(args.smpl_pkl).resolve()
    if not smpl_pkl.is_file():
        print(f"[error] SMPL pkl not found: {smpl_pkl}", file=sys.stderr)
        print("        --smpl-pkl 로 경로 지정", file=sys.stderr)
        return 1

    dump_rest(str(smpl_pkl), args.output_json, args.output_obj, args.output_weights)
    return 0


if __name__ == "__main__":
    sys.exit(main())
