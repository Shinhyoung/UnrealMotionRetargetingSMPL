"""Dump SMPL model as a compact binary blob for C++ consumption.

Emits a little-endian binary file with the following layout:

  Header (16 bytes):
    magic:      4 bytes  "SMPB"
    version:    uint32   = 1
    num_joints: uint32   (=24)
    num_verts:  uint32   (=6890)

  num_faces:   uint32                  (=13776; kept separate for alignment)
  padding:     uint32                  (reserved)

  parents:      int32 * num_joints
  rest_joints:  float32 * num_joints * 3  (SMPL Y-up meters)
  rest_verts:   float32 * num_verts * 3   (SMPL Y-up meters)
  faces:        int32 * num_faces * 3
  weights:      float32 * num_verts * num_joints  (LBS skinning weights)

The blob is loaded once at C++ init via a plain binary read; format
doesn't require any parser dependency in UE.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.bone_mapping import SMPL_PARENTS


DEFAULT_SMPL_PKL = "c:/0.shinhyoung/Project/SAT-HMR/weights/smpl_data/smpl/SMPL_NEUTRAL.pkl"


def _load_smpl(smpl_pkl_path: str):
    import torch
    import smplx
    smpl_parent = str(Path(smpl_pkl_path).parent.parent)
    model = smplx.create(
        model_path=smpl_parent,
        model_type="smpl",
        gender="neutral",
        num_betas=10,
    )
    return model, torch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smpl-pkl", default=DEFAULT_SMPL_PKL)
    p.add_argument("--output", "-o", default="smpl_model.bin",
                   help="Output binary blob path.")
    args = p.parse_args()

    smpl_pkl = Path(args.smpl_pkl).resolve()
    if not smpl_pkl.is_file():
        print(f"[error] SMPL pkl not found: {smpl_pkl}", file=sys.stderr)
        return 1

    print(f"[dump_smpl_blob] loading SMPL from {smpl_pkl}")
    model, torch = _load_smpl(str(smpl_pkl))

    with torch.no_grad():
        # β=0, pose=0 → canonical T-pose
        out = model(
            betas=torch.zeros(1, 10),
            body_pose=torch.zeros(1, 69),
            global_orient=torch.zeros(1, 3),
            transl=torch.zeros(1, 3),
        )

    rest_joints = out.joints[0, :24].cpu().numpy().astype(np.float32)   # (24, 3)
    rest_verts = out.vertices[0].cpu().numpy().astype(np.float32)       # (6890, 3)
    faces = model.faces.astype(np.int32)                                # (13776, 3)

    # LBS weights (V, J)
    try:
        weights = model.lbs_weights.detach().cpu().numpy().astype(np.float32)
    except AttributeError:
        weights = model.weights.detach().cpu().numpy().astype(np.float32)

    num_joints = 24
    num_verts = rest_verts.shape[0]
    num_faces = faces.shape[0]

    if rest_joints.shape != (num_joints, 3):
        raise ValueError(f"bad rest_joints shape: {rest_joints.shape}")
    if weights.shape != (num_verts, num_joints):
        raise ValueError(f"bad weights shape: {weights.shape}, expected ({num_verts}, {num_joints})")

    parents = np.asarray(SMPL_PARENTS, dtype=np.int32)   # (24,)
    if parents.shape != (num_joints,):
        raise ValueError(f"bad parents shape: {parents.shape}")

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        # Header (little-endian)
        f.write(b"SMPB")                                       # magic
        f.write(struct.pack("<I", 1))                          # version
        f.write(struct.pack("<I", num_joints))
        f.write(struct.pack("<I", num_verts))
        f.write(struct.pack("<I", num_faces))
        f.write(struct.pack("<I", 0))                          # padding

        f.write(parents.tobytes(order="C"))                    # int32 * 24
        f.write(rest_joints.tobytes(order="C"))                # float32 * 24 * 3
        f.write(rest_verts.tobytes(order="C"))                 # float32 * 6890 * 3
        f.write(faces.tobytes(order="C"))                      # int32 * 13776 * 3
        f.write(weights.tobytes(order="C"))                    # float32 * 6890 * 24

    size = out_path.stat().st_size
    print(f"[dump_smpl_blob] wrote {out_path}")
    print(f"  {num_joints} joints, {num_verts} verts, {num_faces} faces, "
          f"weights nonzeros = {int(np.count_nonzero(weights > 1e-4))}")
    print(f"  file size: {size:,} bytes ({size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
