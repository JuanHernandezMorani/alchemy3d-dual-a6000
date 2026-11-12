#!/usr/bin/env bash
set -Eeuo pipefail

# A single Empirio custom Docker image. Services are isolated by GPU index.
export LAB_ROOT="${LAB_ROOT:-/workspace}"
export LAB_GPU="${LAB_GPU:-0}"
export QWEN_GPU="${QWEN_GPU:-1}"
export LAB_PORT="${LAB_PORT:-8765}"
export QWEN_PORT="${QWEN_PORT:-8081}"
mkdir -p "$LAB_ROOT/logs" "$LAB_ROOT/results" "$LAB_ROOT/qwen"

if [[ -z "${LAB_TOKEN:-}" || ${#LAB_TOKEN} -lt 24 ]]; then
  echo 'ERROR: configure LAB_TOKEN (minimum 24 characters) in Empirio environment. Never expose the API without authentication.' >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null; then
  echo 'ERROR: NVIDIA GPU runtime not available (nvidia-smi missing)' >&2
  exit 3
fi
count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if (( count < 2 )); then
  echo "ERROR: expected 2 NVIDIA GPUs, found $count. Select the dual A6000 offer." >&2
  exit 4
fi
if [[ "$LAB_GPU" == "$QWEN_GPU" ]]; then
  echo 'ERROR: LAB_GPU and QWEN_GPU must differ.' >&2
  exit 4
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

qwen_pid=''
if [[ -n "${QWEN_MODEL:-}" ]]; then
  if [[ ! -f "$QWEN_MODEL" ]]; then
    echo "ERROR: QWEN_MODEL file not found: $QWEN_MODEL" >&2
    exit 5
  fi
  if ! command -v llama-server >/dev/null; then
    echo 'ERROR: llama-server is missing from image.' >&2
    exit 5
  fi
  echo "Starting Qwen on GPU $QWEN_GPU (loopback only)."
  CUDA_VISIBLE_DEVICES="$QWEN_GPU" llama-server \
    --model "$QWEN_MODEL" \
    --host 127.0.0.1 --port "$QWEN_PORT" \
    --ctx-size "${QWEN_CTX:-32768}" \
    --n-gpu-layers 999 --split-mode none \
    > "$LAB_ROOT/logs/qwen.log" 2>&1 &
  qwen_pid=$!
else
  echo 'QWEN_MODEL not set: lab API starts; download GGUF, set model path and restart for dual mode.'
fi

stop(){
  if [[ -n "$qwen_pid" ]]; then kill "$qwen_pid" 2>/dev/null || true; wait "$qwen_pid" 2>/dev/null || true; fi
}
trap stop EXIT TERM INT

# 0.0.0.0 needed for Empirio's authenticated port forwarding; LAB_TOKEN is mandatory.
# API subprocess inherits GPU0 restriction. llama-server remains bound to loopback.
echo "Starting authenticated Alchemy3D API on GPU $LAB_GPU, port $LAB_PORT."
CUDA_VISIBLE_DEVICES="$LAB_GPU" bash /opt/lab/scripts/lab.sh server \
    --host 0.0.0.0 --port "$LAB_PORT"
