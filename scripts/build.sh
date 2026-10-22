#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
source scripts/env.sh
# Query the selected CUDA device, not the first entry returned by nvidia-smi.
YSDA_ARCH="${YSDA_CUDA_ARCH:-$(python -c 'import torch; a,b=torch.cuda.get_device_capability(); print(f"{a}{b}")')}"
if [[ ! "$YSDA_ARCH" =~ ^[0-9]+$ ]]; then
  echo 'YSDA_CUDA_ARCH must be a numeric SM architecture, e.g. 86, 90, 100, 120.' >&2
  exit 1
fi
cmake -S cuda -B "build/sm_$YSDA_ARCH" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$YSDA_ARCH" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "build/sm_$YSDA_ARCH" --parallel "${YSDA_BUILD_JOBS:-2}"
