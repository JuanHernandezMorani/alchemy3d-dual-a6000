#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LAB_ROOT="${LAB_ROOT:-/workspace}"
export PYTHONPATH="$LAB_ROOT/repos/Alchemy3D:$HERE:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$LAB_ROOT/.hf}"
export HF_HUB_CACHE="$HF_HOME/hub"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
if [[ -f "$LAB_ROOT/.lab_runtime.env" ]]; then set -a; source "$LAB_ROOT/.lab_runtime.env"; set +a; fi
cmd="${1:-help}"; shift || true
case "$cmd" in
  bootstrap) bash "$HERE/scripts/bootstrap_cuda.sh" "$@" ;;
  download) python "$HERE/scripts/download_models.py" "$@" ;;
  overlay) python "$HERE/scripts/create_overlay.py" "$@" ;;
  merge) python "$HERE/scripts/merge_texture.py" "$@" ;;
  preflight) python "$HERE/scripts/preflight.py" "$@" ;;
  test) python "$HERE/scripts/run_case_cuda.py" "$@" ;;
  verify) python "$HERE/scripts/verify_results.py" "$@" ;;
  server) python "$HERE/scripts/serve_lab.py" "$@" ;;
  *) printf '%s\n' 'Uso: lab.sh {bootstrap|download|overlay|merge|preflight|test|verify|server} [argumentos]' ;;
esac
