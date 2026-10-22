#!/usr/bin/env bash
# Source this file from bash: source scripts/env.sh
YSDA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export YSDA_ROOT
if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda
  elif command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
    export CUDA_HOME
  else
    echo 'CUDA Toolkit not found. Set CUDA_HOME to a CUDA 13.3+ installation.' >&2
    return 1
  fi
fi
export PATH="$YSDA_ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$YSDA_ROOT/.cache/uv}"
export TRITON_CACHE_DIR="$YSDA_ROOT/.cache/triton"
export CUDA_CACHE_PATH="$YSDA_ROOT/.cache/cuda"
export CUTE_DSL_CACHE_DIR="$YSDA_ROOT/.cache/cute"
export PYTHONPATH="$YSDA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# CUDA_VISIBLE_DEVICES is deliberately selected by the caller.
