#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/root/bridge-coma}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
OUT="$ROOT/results/main_lr1e-5_seed42/A/curve_vs_sl_20000"
SL="$ROOT/results/sl_base.pt"
DATA="$ROOT/data/pgx_eval_500k_memmap"
SEED=20260714

if [[ -e "$OUT/RUN_COMPLETE" || -e "$OUT/RUN_FAILED" ]]; then
    echo "refusing to overwrite completed curve evaluation: $OUT" >&2
    exit 1
fi
mkdir -p "$OUT"
printf 'started_utc=%s\nseed=%s\ndeals_per_checkpoint=20000\nrounds=70,80,90,100\n' \
    "$(date -u +%FT%TZ)" "$SEED" > "$OUT/manifest.txt"

pids=()
rounds=(70 80 90 100)
gpus=(0 1 0 1)
for index in "${!rounds[@]}"; do
    round="${rounds[$index]}"
    gpu="${gpus[$index]}"
    checkpoint="$ROOT/results/main_lr1e-5_seed42/A/snapshots/agent_a_round${round}.resume.pt"
    env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
        "$ROOT/experiments/evaluate_single_vs_sl.py" \
        --agent "$checkpoint" --sl-checkpoint "$SL" --data "$DATA" \
        --deals 20000 --seed "$SEED" \
        --output "$OUT/a${round}_vs_sl_20000.json" \
        > "$OUT/a${round}_vs_sl.log" 2>&1 &
    pids+=("$!")
done

printf 'a70=%s\na80=%s\na90=%s\na100=%s\n' \
    "${pids[0]}" "${pids[1]}" "${pids[2]}" "${pids[3]}" > "$OUT/pids.txt"

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
printf 'finished_utc=%s\nstatus=%s\n' "$(date -u +%FT%TZ)" "$status" >> "$OUT/manifest.txt"
if (( status == 0 )); then
    touch "$OUT/RUN_COMPLETE"
else
    touch "$OUT/RUN_FAILED"
fi
exit "$status"
