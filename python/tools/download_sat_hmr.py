"""Download the SAT-HMR checkpoint (sat_644.pth) into the expected location.

Requires ``huggingface_hub``. The checkpoint (~875 MB) lands at:
    ${SAT_HMR_ROOT}/weights/sat_hmr/sat_644.pth
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sat-hmr-root", default="c:/0.shinhyoung/Project/SAT-HMR",
                   help="Root of your local SAT-HMR clone.")
    p.add_argument("--variant", default="sat_644.pth",
                   choices=("sat_644.pth", "sat_644_3dpw.pth", "sat_644_agora.pth"),
                   help="Checkpoint variant on ChiSu001/SAT-HMR HuggingFace repo.")
    args = p.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub not installed. `pip install huggingface_hub`.")
        return 1

    target_dir = os.path.join(args.sat_hmr_root, "weights", "sat_hmr")
    os.makedirs(target_dir, exist_ok=True)
    path = hf_hub_download(
        repo_id="ChiSu001/SAT-HMR",
        filename=f"weights/sat_hmr/{args.variant}",
        local_dir=args.sat_hmr_root,
    )
    print(f"downloaded → {path}")
    print(f"size: {os.path.getsize(path):,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
