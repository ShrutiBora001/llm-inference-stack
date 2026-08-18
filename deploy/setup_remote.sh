#!/usr/bin/env bash
# Idempotent environment setup on the box. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x .venv/bin/python ] && .venv/bin/python -c "import lis" 2>/dev/null; then
  echo "  env already good"; exit 0
fi

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

# Inherit the image's CUDA-matched torch. Installing our own would mismatch the
# driver, and NGC's system interpreter is externally managed (PEP 668) so a
# plain install fails outright.
if python3 -c "import torch" 2>/dev/null; then
  echo "  system torch $(python3 -c 'import torch;print(torch.__version__)') -- inheriting"
  uv venv --system-site-packages .venv
else
  echo "  no system torch -- installing cu124 wheels"
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124
fi

[ -d vendor/dattn ] && uv pip install --python .venv/bin/python -e vendor/dattn
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -c "import torch,lis;print(f'  ready: torch {torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}')"
