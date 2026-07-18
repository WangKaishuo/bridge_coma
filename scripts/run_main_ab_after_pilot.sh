#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/bridge-coma}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
PILOT="$ROOT/results/main_unrestricted_pilot_seed42"
FORMAL="$ROOT/results/main_unrestricted_formal_seed42"
SUPERVISOR_LOG="$ROOT/results/main_pipeline_supervisor.log"
POLL_SECONDS="${POLL_SECONDS:-300}"

exec >>"$SUPERVISOR_LOG" 2>&1
echo "[$(date -u +%FT%TZ)] supervisor started pid=$$"

failed_log() {
    grep -Eq 'Traceback|CUDA out of memory|(^|[^A-Za-z])NaN([^A-Za-z]|$)' "$1"
}

process_alive() {
    local pid_file="$1"
    [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

pilot_complete() {
    local label="$1" lower
    lower="$(printf '%s' "$label" | tr '[:upper:]' '[:lower:]')"
    [[ -f "$PILOT/$label/agent_${lower}_seed42.pt" ]] \
        && grep -q 'Training complete' "$PILOT/$label/run.log"
}

while true; do
    all_complete=1
    for label in A B C; do
        log="$PILOT/$label/run.log"
        if [[ -f "$log" ]] && failed_log "$log"; then
            echo "[$(date -u +%FT%TZ)] pilot $label failed"
            touch "$ROOT/results/MAIN_PIPELINE_BLOCKED"
            exit 1
        fi
        if ! pilot_complete "$label"; then
            all_complete=0
            if ! process_alive "$PILOT/$label/pid"; then
                echo "[$(date -u +%FT%TZ)] pilot $label stopped before completion"
                touch "$ROOT/results/MAIN_PIPELINE_BLOCKED"
                exit 1
            fi
        fi
    done
    [[ "$all_complete" -eq 1 ]] && break
    sleep "$POLL_SECONDS"
done

echo "[$(date -u +%FT%TZ)] pilot complete; launching formal A/B"

launch_agent() {
    local label="$1" gpu="$2" lower out
    lower="$(printf '%s' "$label" | tr '[:upper:]' '[:lower:]')"
    out="$FORMAL/$label"
    mkdir -p "$out"

    if [[ -f "$out/agent_${lower}_seed42.pt" ]]; then
        echo "[$(date -u +%FT%TZ)] formal $label already complete"
        return
    fi
    if process_alive "$out/pid"; then
        echo "[$(date -u +%FT%TZ)] formal $label already running"
        return
    fi
    if [[ -s "$out/run.log" ]]; then
        echo "[$(date -u +%FT%TZ)] refusing to overwrite incomplete $out/run.log"
        touch "$ROOT/results/MAIN_PIPELINE_BLOCKED"
        exit 1
    fi

    nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
        "$ROOT/experiments/main_experiment.py" \
        --data "$ROOT/data/pgx_train_10m" \
        --eval-data "$ROOT/data/pgx_eval_500k" \
        --sl-checkpoint "$ROOT/results/sl_base.pt" \
        --belief-checkpoint "$ROOT/results/sl_base_bca.pt" \
        --output-dir "$out" \
        --train-agents "$label" --seed 42 --eval-seed 20260714 \
        --rounds 60 --steps-per-phase 256 --deals-per-step 512 \
        --batch-size 512 --num-epochs 4 --eval-deals 5000 \
        --beta 0.05 --info-weight 0.05 --actor-belief-coef 0.1 \
        --learning-rate 3e-6 --entropy-coef 0.01 \
        --checkpoint-interval 1 \
        >"$out/run.log" 2>&1 < /dev/null &
    echo "$!" > "$out/pid"
    echo "[$(date -u +%FT%TZ)] launched formal $label gpu=$gpu pid=$!"
}

launch_agent A 0
launch_agent B 1

while true; do
    all_complete=1
    for label in A B; do
        lower="$(printf '%s' "$label" | tr '[:upper:]' '[:lower:]')"
        out="$FORMAL/$label"
        if [[ -f "$out/run.log" ]] && failed_log "$out/run.log"; then
            echo "[$(date -u +%FT%TZ)] formal $label failed"
            touch "$ROOT/results/MAIN_PIPELINE_BLOCKED"
            exit 1
        fi
        if [[ ! -f "$out/agent_${lower}_seed42.pt" ]]; then
            all_complete=0
            if ! process_alive "$out/pid"; then
                echo "[$(date -u +%FT%TZ)] formal $label stopped before completion"
                touch "$ROOT/results/MAIN_PIPELINE_BLOCKED"
                exit 1
            fi
        fi
    done
    [[ "$all_complete" -eq 1 ]] && break
    sleep "$POLL_SECONDS"
done

touch "$ROOT/results/MAIN_AB_COMPLETE"
echo "[$(date -u +%FT%TZ)] formal A/B complete; awaiting C parameters"
