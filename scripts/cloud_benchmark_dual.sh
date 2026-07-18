#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/cloud_benchmark
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log0="results/cloud_benchmark/${stamp}_gpu0.log"
log1="results/cloud_benchmark/${stamp}_gpu1.log"

args=(
  -m experiments.benchmark_training
  --agent B
  --rounds 2
  --steps-per-phase 16
  --deals-per-step 512
  --batch-size 256
  --num-epochs 4
  --prewarm-deals 512
  --prewarm-epochs 1
  --calibration-deals 512
)

CUDA_VISIBLE_DEVICES=0 python "${args[@]}" >"$log0" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 python "${args[@]}" >"$log1" 2>&1 &
pid1=$!

status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?

echo "GPU 0 log: $log0"
echo "GPU 1 log: $log1"
echo "=== GPU 0 result ==="
grep -A 80 BENCHMARK_RESULT "$log0" || true
echo "=== GPU 1 result ==="
grep -A 80 BENCHMARK_RESULT "$log1" || true
exit "$status"
