#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/bridge-coma}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
FORMAL="$ROOT/results/main_unrestricted_formal_memory_safe_seed42"
GATE="$FORMAL/round30_gate"
POLL_SECONDS="${POLL_SECONDS:-30}"
mkdir -p "$GATE"

pause_and_snapshot() {
    local label="$1" lower pid resume snapshot log
    lower="$(printf '%s' "$label" | tr '[:upper:]' '[:lower:]')"
    pid="$(cat "$FORMAL/$label/pid")"
    resume="$FORMAL/$label/agent_${lower}_seed42.resume.pt"
    snapshot="$GATE/agent_${lower}_round30.resume.pt"
    log="$FORMAL/$label/run.log"
    while ! grep -q '\[Checkpoint\] round 30 ->' "$log"; do
        kill -0 "$pid" 2>/dev/null || {
            echo "$label stopped before round 30" >&2
            exit 1
        }
        sleep "$POLL_SECONDS"
    done
    kill -STOP "$pid"
    cp --reflink=auto "$resume" "$snapshot"
    echo "$pid" > "$GATE/$label.paused.pid"
    echo "[$(date -u +%FT%TZ)] paused $label and saved $snapshot"
}

pause_and_snapshot A & child_a=$!
pause_and_snapshot B & child_b=$!
wait "$child_a"
wait "$child_b"

pid_a="$(cat "$GATE/A.paused.pid")"
pid_b="$(cat "$GATE/B.paused.pid")"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$ROOT/experiments/evaluate_resume_ab.py" \
    --agent-a "$GATE/agent_a_round30.resume.pt" \
    --agent-b "$GATE/agent_b_round30.resume.pt" \
    --data "$ROOT/data/pgx_eval_500k_memmap" \
    --deals 5000 --seed 20260714 \
    --output "$GATE/result.json" \
    > "$GATE/evaluation.log" 2>&1

decision="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "$GATE/result.json")"
echo "$decision" > "$GATE/decision.txt"
if [[ "$decision" == "WIN" ]]; then
    kill -CONT "$pid_a" "$pid_b"
    touch "$GATE/RESUMED_TO_60"
    echo "[$(date -u +%FT%TZ)] B clearly wins; resumed A/B to round 60"
else
    touch "$GATE/HELD_FOR_REVIEW"
    echo "[$(date -u +%FT%TZ)] decision=$decision; A/B remain paused"
fi
