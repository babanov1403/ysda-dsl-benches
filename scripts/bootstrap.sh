#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'This lock targets Linux x86_64 + NVIDIA GPUs. See docs/portability.md.' >&2
  exit 1
fi
source scripts/env.sh
python3 - <<'PY'
import re, subprocess
s = subprocess.check_output(['nvcc', '--version'], text=True)
m = re.search(r'release (\d+)\.(\d+)', s)
if not m or tuple(map(int, m.groups())) < (13, 3):
    raise SystemExit('CUDA Toolkit >=13.3 required by this cu13 profile; set CUDA_HOME.')
PY
if [[ ! -x .tools/venv/bin/uv ]]; then
  python3 -m venv .tools/venv
  .tools/venv/bin/python -m pip install --disable-pip-version-check uv==0.13.5
fi
.tools/venv/bin/uv sync --locked --python 3.12
bash scripts/build.sh
echo 'Ready. source scripts/env.sh; python -m bench.doctor; python -m pytest'
