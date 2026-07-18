#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
# The host's CUDA 13 driver is backward-compatible with this tested wheel.
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

python - <<'PY'
import torch
import pyspiel
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; check the image and NVIDIA driver")
print("gpu", torch.cuda.get_device_name(0))
PY

echo "Setup complete. Run: bash scripts/cloud_benchmark.sh"
