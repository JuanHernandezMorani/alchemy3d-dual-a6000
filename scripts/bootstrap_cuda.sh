#!/usr/bin/env bash
set -euo pipefail
ROOT="${LAB_ROOT:-/workspace}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" MAX_JOBS="${MAX_JOBS:-2}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTHONPATH="$ROOT/repos/Alchemy3D:$HERE:${PYTHONPATH:-}"
nvidia-smi
python - <<'PY'
import torch
print('torch:',torch.__version__,'CUDA:',torch.cuda.is_available())
assert torch.cuda.is_available(), 'No hay GPU CUDA visible. Revisar plantilla y --gpus all.'
print('GPU:',torch.cuda.get_device_name(0),'capability:',torch.cuda.get_device_capability(0))
PY
bash "$HERE/scripts/clone_sources.sh"
mkdir -p "$ROOT/logs"
if [[ -f "$ROOT/.bootstrap_cuda_ok" && "${FORCE_BOOTSTRAP:-0}" != "1" ]]; then
  echo "[OK] Ya se instalaron extensiones. Para repetir: FORCE_BOOTSTRAP=1"; exit 0
fi
# Official setup script, without --new-env: preinstalled image has torch cu124.
# This stage may compile several CUDA extensions and needs working nvcc.
cd "$ROOT/repos/Alchemy3D"
if [[ "${SKIP_EXTENSIONS:-0}" != "1" ]]; then
  bash -c '. ./setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --flexgemm --blender' 2>&1 | tee "$ROOT/logs/bootstrap_cuda.log"
else
  echo "[AVISO] SKIP_EXTENSIONS=1: modo solo diagnóstico, no habilita inferencia completa."
  exit 0
fi
echo "[O-VOXEL] Instalando el subárbol fijado en TRELLIS.2 commit (no última main)..."
python -m pip install "$ROOT/repos/TRELLIS.2/o-voxel" --no-build-isolation 2>&1 | tee "$ROOT/logs/o_voxel_build.log"
cd "$ROOT"
python - <<'PY'
import torch, o_voxel, flash_attn, flex_gemm
print('CUDA',torch.cuda.get_device_name(0))
print('o_voxel, flash_attn, flex_gemm: OK')
PY
export BLENDER_PATH="$ROOT/repos/Alchemy3D/blender-4.3.2-linux-x64/blender"
if [[ ! -x "$BLENDER_PATH" ]]; then echo 'No se encontró Blender instalado por setup.sh'; exit 2; fi
printf 'BLENDER_PATH=%s\n' "$BLENDER_PATH" > "$ROOT/.lab_runtime.env"
date -u +%FT%TZ > "$ROOT/.bootstrap_cuda_ok"
echo "[OK] CUDA preparado, fuentes fijadas y extensiones compiladas."
