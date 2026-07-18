#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after installing requirements.txt.
mkdir -p results/cloud_benchmark
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log="results/cloud_benchmark/${stamp}.log"

{
  echo "=== host ==="
  date -u
  uname -a
  lscpu
  nvidia-smi
  python - <<'PY'
import os
import torch
print("python_cpu_count", os.cpu_count())
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

  echo "=== real Agent B benchmark ==="
  python -m experiments.benchmark_training \
    --agent B \
    --rounds 2 \
    --steps-per-phase 16 \
    --deals-per-step 512 \
    --batch-size 256 \
    --num-epochs 4 \
    --prewarm-deals 512 \
    --prewarm-epochs 1 \
    --calibration-deals 512
} 2>&1 | tee "$log"

echo "Benchmark saved to $log"
