"""Environment verification — RTX 5070 (Blackwell sm_120) + CUDA 12.8 + PyTorch nightly.

Per CLAUDE.md §2: if this doesn't pass, do NOT touch model code. Fix the
environment first. Run with:

    python verify_env.py
"""
from __future__ import annotations

import sys


def _print_header(title: str) -> None:
    print("=" * 64)
    print(title)
    print("=" * 64)


def main() -> int:
    _print_header("Environment verification (CLAUDE.md §2)")
    ok = True

    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("  ! Python 3.10+ required.")
        ok = False

    try:
        import torch  # type: ignore
    except ImportError:
        print("PyTorch: NOT INSTALLED")
        print("  Install nightly:")
        print("  pip install --pre torch torchvision --index-url "
              "https://download.pytorch.org/whl/nightly/cu128")
        return 1

    print(f"PyTorch: {torch.__version__}")
    is_nightly = ("dev" in torch.__version__) or ("nightly" in torch.__version__)
    # If the CUDA runtime is already 12.8+, the wheel supports sm_120 regardless of
    # nightly vs stable tag — the "must be nightly" advice only applies to older builds.
    cuda_supports_sm120 = False
    if torch.version.cuda:
        try:
            cuda_supports_sm120 = tuple(int(p) for p in torch.version.cuda.split(".")) >= (12, 8)
        except ValueError:
            pass
    if not is_nightly and not cuda_supports_sm120:
        print("  ! WARNING: Not a nightly build and CUDA < 12.8. "
              "Blackwell (sm_120) needs nightly + CUDA 12.8.")

    if not torch.cuda.is_available():
        print("CUDA: NOT AVAILABLE — check driver + CUDA 12.8 install.")
        return 1

    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")

    saw_blackwell = False
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        cc = torch.cuda.get_device_capability(i)
        tag = f"sm_{cc[0]*10 + cc[1]}"
        line = f"  [{i}] {name}   compute={cc[0]}.{cc[1]} ({tag})"
        if cc == (12, 0):
            saw_blackwell = True
            line += "   -- Blackwell OK"
        elif cc[0] >= 12:
            line += "   -- next-gen (>=Blackwell)"
        elif cc[0] < 8:
            line += "   -- older GPU (sm_120 detection N/A)"
        print(line)

    # Small compute sanity test
    try:
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x.T
        torch.cuda.synchronize()
        print(f"GPU matmul test: OK  (shape={tuple(y.shape)})")
    except Exception as e:
        print(f"GPU matmul test: FAILED — {e}")
        ok = False

    print("-" * 64)
    if not saw_blackwell:
        print("Note: no sm_120 GPU detected. This is only fatal if targeting RTX 5070.")
    print("Result:", "PASS" if ok else "FAIL")
    _print_header("done")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
