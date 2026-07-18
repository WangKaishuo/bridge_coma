#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/quad

started=$(date +%s)
pids=()
num_processes="${NUM_PROCESSES:-4}"
steps_per_phase="${STEPS_PER_PHASE:-8}"
num_epochs="${NUM_EPOCHS:-2}"
ppo_batch_size="${PPO_BATCH_SIZE:-256}"
deals_per_process=$((2 * steps_per_phase * 512))
total_deals=$((num_processes * deals_per_process))
for ((slot=0; slot<num_processes; slot++)); do
  gpu=$((slot % 2))
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -m experiments.benchmark_training \
    --agent B \
    --seed "$((42 + slot))" \
    --rounds 1 \
    --steps-per-phase "$steps_per_phase" \
    --deals-per-step 512 \
    --collector-workers 1 \
    --fast-observation-encoding \
    --batch-size "$ppo_batch_size" \
    --num-epochs "$num_epochs" \
    --prewarm-deals 128 \
    --prewarm-epochs 1 \
    --calibration-deals 128 \
    > "results/quad/slot_${slot}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
elapsed=$(( $(date +%s) - started ))

echo "QUAD_WALL_SECONDS=$elapsed"
echo "QUAD_TOTAL_DEALS=$total_deals"
awk -v elapsed="$elapsed" -v total="$total_deals" \
  'BEGIN { printf "QUAD_AGGREGATE_DEALS_PER_SECOND=%.3f\n", total / elapsed }'
for ((slot=0; slot<num_processes; slot++)); do
  echo "===== SLOT $slot ====="
  grep -E '"training_seconds"|"deals_per_second"|"gpu_util_mean_pct"|"process_cpu_mean' \
    "results/quad/slot_${slot}.log" || true
done
exit "$status"
