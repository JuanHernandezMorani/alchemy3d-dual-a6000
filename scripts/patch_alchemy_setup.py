#!/usr/bin/env python3
"""Allow compilation of CUDA extensions in GPU-less Docker build environments.

Alchemy3D upstream setup.sh infers build toolchain from nvidia-smi/rocminfo.
The GitHub Actions Docker builder has nvcc but no physical GPU. We set a
build-only override, without modifying runtime device checks.
"""
from pathlib import Path

p = Path("/workspace/repos/Alchemy3D/setup.sh")
src = p.read_text(encoding="utf-8")
old = 'if command -v nvidia-smi > /dev/null; then'
new = 'if [ "${ALCHEMY_BUILD_PLATFORM:-}" = "cuda" ] || command -v nvidia-smi > /dev/null; then'
if src.count(old) == 1:
    p.write_text(src.replace(old, new, 1), encoding="utf-8")
    print("[CUDA BUILD] Alchemy3D setup.sh: use nvcc build override (no GPU required).", flush=True)
elif src.count(new) == 1:
    print("[CUDA BUILD] Alchemy3D setup.sh already patched.", flush=True)
else:
    raise RuntimeError("Unexpected Alchemy3D setup.sh: cannot safely patch the GPU build detection")
