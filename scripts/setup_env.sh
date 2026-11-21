#!/usr/bin/env bash
# Rebuild the project virtualenv. Idempotent.
# Pitfall: a conda env may be active; always target .venv explicitly with --python.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d third_party/videoseal ]; then
  git clone --depth 1 https://github.com/facebookresearch/videoseal third_party/videoseal
fi
uv sync
# videoseal declares decord, which has no macOS arm64 wheel: install it without deps
# (its other runtime deps are already listed in pyproject.toml).
uv pip install --python .venv/bin/python --no-deps -e third_party/videoseal
.venv/bin/python -c "import torch, videoseal; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"
