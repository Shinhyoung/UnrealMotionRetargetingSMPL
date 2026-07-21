"""Sanity-check the SMPL binary blob produced by dump_smpl_blob.py.

Loads the blob, reruns SMPL forward with the same pose in Python, and
prints max/median differences. Useful for verifying the C++ side later
(any C++ LBS bug reveals itself as large per-vertex disagreement).

Run:
    python tools/verify_smpl_blob.py --blob ../smpl_model.bin
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_blob(path: str):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"SMPB", "bad magic"
    p = 4
    version = struct.unpack_from("<I", data, p)[0]; p += 4
    assert version == 1
    n_j = struct.unpack_from("<I", data, p)[0]; p += 4
    n_v = struct.unpack_from("<I", data, p)[0]; p += 4
    n_f = struct.unpack_from("<I", data, p)[0]; p += 4
    _pad = struct.unpack_from("<I", data, p)[0]; p += 4

    parents = np.frombuffer(data, dtype=np.int32, count=n_j, offset=p); p += 4 * n_j
    rest_joints = np.frombuffer(data, dtype=np.float32, count=n_j * 3, offset=p).reshape(n_j, 3); p += 12 * n_j
    rest_verts = np.frombuffer(data, dtype=np.float32, count=n_v * 3, offset=p).reshape(n_v, 3); p += 12 * n_v
    faces = np.frombuffer(data, dtype=np.int32, count=n_f * 3, offset=p).reshape(n_f, 3); p += 12 * n_f
    weights = np.frombuffer(data, dtype=np.float32, count=n_v * n_j, offset=p).reshape(n_v, n_j); p += 4 * n_v * n_j
    assert p == len(data), f"leftover {len(data) - p} bytes"
    return dict(parents=parents, rest_joints=rest_joints, rest_verts=rest_verts,
                faces=faces, weights=weights, n_j=n_j, n_v=n_v, n_f=n_f)


def python_lbs(blob, body_pose_aa, global_orient_aa, root_trans):
    """Reimplement SMPL LBS in numpy for comparison. Assumes β=0 (rest verts)."""
    from scipy.spatial.transform import Rotation as R
    n_j = blob["n_j"]
    n_v = blob["n_v"]
    parents = blob["parents"]
    rest_joints = blob["rest_joints"]
    rest_verts = blob["rest_verts"]
    weights = blob["weights"]

    # Joint world transforms (4x4)
    world = np.zeros((n_j, 4, 4), dtype=np.float64)
    world[0] = np.eye(4)
    world[0, :3, :3] = R.from_rotvec(global_orient_aa).as_matrix()
    world[0, :3, 3] = rest_joints[0] + root_trans

    for i in range(1, n_j):
        parent = parents[i]
        rest_offset = rest_joints[i] - rest_joints[parent]
        local = np.eye(4)
        local[:3, :3] = R.from_rotvec(body_pose_aa[i - 1]).as_matrix()
        local[:3, 3] = rest_offset
        world[i] = world[parent] @ local

    # Rest world inverse (translation only at rest)
    rest_world_inv = np.tile(np.eye(4), (n_j, 1, 1))
    rest_world_inv[:, :3, 3] = -rest_joints

    # Skinning matrix
    skin = np.einsum("jab,jbc->jac", world, rest_world_inv)   # (n_j, 4, 4)

    # LBS
    verts_h = np.concatenate([rest_verts, np.ones((n_v, 1), dtype=np.float32)], axis=-1)  # (n_v, 4)
    # per-vertex blend: (n_v, 4) = sum_j weights[v,j] * skin[j] @ verts_h[v]
    # Compute per-vertex transformed by each bone, then blend
    transformed = np.einsum("jab,vb->vja", skin, verts_h)   # (n_v, n_j, 4)
    blended = np.einsum("vj,vja->va", weights, transformed)  # (n_v, 4)
    return blended[:, :3]


def smplx_reference(body_pose_aa, global_orient_aa, root_trans):
    """Ground truth: run smplx forward and get vertices."""
    import torch
    import smplx
    from pathlib import Path
    default_smpl_pkl = "c:/0.shinhyoung/Project/SAT-HMR/weights/smpl_data/smpl/SMPL_NEUTRAL.pkl"
    smpl_parent = str(Path(default_smpl_pkl).parent.parent)
    model = smplx.create(smpl_parent, model_type="smpl", gender="neutral", num_betas=10)

    with torch.no_grad():
        out = model(
            betas=torch.zeros(1, 10),
            body_pose=torch.tensor(body_pose_aa.reshape(1, -1), dtype=torch.float32),
            global_orient=torch.tensor(global_orient_aa.reshape(1, 3), dtype=torch.float32),
            transl=torch.tensor(root_trans.reshape(1, 3), dtype=torch.float32),
        )
    return out.vertices[0].cpu().numpy()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blob", default="smpl_model.bin")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"[verify] loading blob {args.blob}")
    blob = load_blob(args.blob)

    rng = np.random.default_rng(args.seed)
    body_pose = rng.normal(0.0, 0.3, size=(23, 3)).astype(np.float32)
    global_orient = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    root_trans = np.array([0.0, 0.0, 3.0], dtype=np.float32)

    print("[verify] running our python-LBS (using loaded blob) ...")
    ours = python_lbs(blob, body_pose, global_orient, root_trans)
    print("[verify] running smplx reference forward ...")
    ref = smplx_reference(body_pose, global_orient, root_trans)

    diff = np.linalg.norm(ours - ref, axis=-1)
    print(f"[verify] per-vertex L2 diff: max={diff.max()*1000:.3f}mm  "
          f"median={np.median(diff)*1000:.3f}mm  mean={diff.mean()*1000:.3f}mm")
    print(f"[verify] {(diff < 1e-3).sum()} / {len(diff)} vertices within 1 mm")
    if diff.max() < 1e-2:
        print("[verify] OK — LBS math agrees with smplx to <10mm")
        return 0
    else:
        print("[verify] MISMATCH — investigate LBS implementation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
