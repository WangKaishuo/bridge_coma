#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/root/bridge-coma}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
OUT="$ROOT/results/main_lr1e-5_seed42/round60_eval_20000"
A="$ROOT/results/main_lr1e-5_seed42/A/snapshots/agent_a_round60.resume.pt"
B="$ROOT/results/main_lr1e-5_seed42/B/agent_b_seed42.resume.pt"
SL="$ROOT/results/sl_base.pt"
DATA="$ROOT/data/pgx_eval_500k_memmap"
SEED=20260714

mkdir -p "$OUT"
rm -f "$OUT/RUN_COMPLETE" "$OUT/RUN_FAILED"
printf 'started_utc=%s\nseed=%s\ndeals_per_match=20000\n' \
    "$(date -u +%FT%TZ)" "$SEED" > "$OUT/manifest.txt"

env CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$ROOT/experiments/evaluate_single_vs_sl.py" \
    --agent "$A" --sl-checkpoint "$SL" --data "$DATA" \
    --deals 20000 --seed "$SEED" --output "$OUT/a60_vs_sl_20000.json" \
    > "$OUT/a60_vs_sl.log" 2>&1 &
pid_a=$!

env CUDA_VISIBLE_DEVICES=1 "$PYTHON" -u "$ROOT/experiments/evaluate_single_vs_sl.py" \
    --agent "$B" --sl-checkpoint "$SL" --data "$DATA" \
    --deals 20000 --seed "$SEED" --output "$OUT/b60_vs_sl_20000.json" \
    > "$OUT/b60_vs_sl.log" 2>&1 &
pid_b=$!

env CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$ROOT/experiments/evaluate_resume_ab.py" \
    --agent-a "$A" --agent-b "$B" --data "$DATA" \
    --deals 20000 --seed "$SEED" --output "$OUT/b60_vs_a60_20000.json" \
    > "$OUT/b60_vs_a60.log" 2>&1 &
pid_ab=$!

env CUDA_VISIBLE_DEVICES=1 "$PYTHON" -u "$ROOT/experiments/inspect_unrestricted_match.py" \
    --agent-a "$A" --agent-b "$B" --data "$DATA" \
    --deals 100 --seed "$SEED" --output "$OUT/B60_vs_A60_100_auctions.txt" \
    > "$OUT/trace_100.log" 2>&1 &
pid_trace=$!

printf 'a_vs_sl=%s\nb_vs_sl=%s\nb_vs_a=%s\ntrace=%s\n' \
    "$pid_a" "$pid_b" "$pid_ab" "$pid_trace" > "$OUT/pids.txt"

status=0
wait "$pid_a" || status=1
wait "$pid_b" || status=1
wait "$pid_ab" || status=1
wait "$pid_trace" || status=1

printf 'finished_utc=%s\nstatus=%s\n' "$(date -u +%FT%TZ)" "$status" >> "$OUT/manifest.txt"
if (( status == 0 )); then
    touch "$OUT/RUN_COMPLETE"
else
    touch "$OUT/RUN_FAILED"
fi
exit "$status"
