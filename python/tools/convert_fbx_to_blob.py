"""Wrapper that launches Blender headless to convert a T-pose FBX → SMPL blob.

Usage:
    python tools/convert_fbx_to_blob.py \
        --fbx <mixamo_or_custom_tpose>.fbx \
        --smpl-pkl <path>/SMPL_NEUTRAL.pkl \
        --out <repo>/smpl_model_custom.bin \
        [--blender <path/to/blender.exe>]

Blender is auto-detected on Windows/Mac/Linux common install paths.
Override with --blender if it's installed somewhere unusual.

The SMPL armature is built programmatically inside Blender from the PKL —
no separate SMPL_Skeleton.fbx is needed.
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


NUM_JOINTS = 24


def preprocess_smpl_pkl(pkl_path: str, out_npz_path: str) -> None:
    """Extract rest_joints + parents from SMPL PKL, save as npz.

    Runs in system Python (has scipy for J_regressor sparse matrix).
    Blender's bundled Python doesn't have scipy, so it consumes this npz.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    v_template = np.asarray(data["v_template"], dtype=np.float64)
    j_reg = data["J_regressor"]
    try:
        j_reg = j_reg.toarray()
    except AttributeError:
        j_reg = np.asarray(j_reg)
    rest_joints = (j_reg[:NUM_JOINTS] @ v_template).astype(np.float32)
    parents = np.asarray(data["kintree_table"][0][:NUM_JOINTS], dtype=np.int32).copy()
    parents[0] = -1
    np.savez(out_npz_path, rest_joints=rest_joints, parents=parents)


def find_blender() -> str | None:
    # 1) On PATH
    exe = shutil.which("blender")
    if exe:
        return exe
    # 2) Common Windows install paths
    if sys.platform == "win32":
        candidates = []
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        blender_root = Path(pf) / "Blender Foundation"
        if blender_root.is_dir():
            for sub in sorted(blender_root.iterdir(), reverse=True):
                cand = sub / "blender.exe"
                if cand.is_file():
                    candidates.append(str(cand))
        return candidates[0] if candidates else None
    # 3) Common macOS install path
    if sys.platform == "darwin":
        cand = "/Applications/Blender.app/Contents/MacOS/Blender"
        return cand if Path(cand).is_file() else None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbx", required=True, help="T-pose FBX (Mixamo or custom).")
    ap.add_argument("--smpl-pkl", required=True, help="SMPL_NEUTRAL.pkl.")
    ap.add_argument("--out", required=True, help="Output blob path (e.g. smpl_model_custom.bin).")
    ap.add_argument("--blender", default=None, help="Path to blender executable (auto-detect if omitted).")
    args = ap.parse_args()

    blender = args.blender or find_blender()
    if not blender:
        print("[error] Could not find Blender. Pass --blender <path/to/blender.exe>",
              file=sys.stderr)
        return 1
    print(f"[wrapper] using Blender at: {blender}")

    for p, name in [(args.fbx, "--fbx"), (args.smpl_pkl, "--smpl-pkl")]:
        if not Path(p).is_file():
            print(f"[error] {name} not found: {p}", file=sys.stderr)
            return 1

    script = Path(__file__).parent / "blender_fbx_to_blob.py"
    if not script.is_file():
        print(f"[error] Blender script missing: {script}", file=sys.stderr)
        return 1

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-extract SMPL rest data (needs scipy — Blender's Python doesn't have it).
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        smpl_rest_npz = tmp.name
    try:
        print(f"[wrapper] pre-extracting SMPL rest data → {smpl_rest_npz}")
        preprocess_smpl_pkl(args.smpl_pkl, smpl_rest_npz)

        cmd = [
            blender, "--background",
            "--python", str(script),
            "--",
            str(Path(args.fbx).resolve()),
            smpl_rest_npz,
            str(out_path),
        ]
        print(f"[wrapper] launching: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[error] Blender exited with code {result.returncode}", file=sys.stderr)
            return result.returncode
    finally:
        try:
            os.unlink(smpl_rest_npz)
        except OSError:
            pass

    if not out_path.is_file():
        print(f"[error] Expected output not produced: {out_path}", file=sys.stderr)
        return 1
    print(f"[wrapper] blob ready: {out_path} ({out_path.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
