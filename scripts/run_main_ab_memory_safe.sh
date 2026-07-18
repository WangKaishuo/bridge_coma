#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/bridge-coma}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
FORMAL="$ROOT/results/main_unrestricted_formal_memory_safe_seed42"
SUPERVISOR_LOG="$ROOT/results/main_memory_safe_supervisor.log"
POLL_SECONDS="${POLL_SECONDS:-300}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"
MEMORY_GUARD_KB="${MEMORY_GUARD_KB:-16777216}"
RESOURCE_LOG="$FORMAL/resources.tsv"

exec >>"$SUPERVISOR_LOG" 2>&1
echo "[$(date -u +%FT%TZ)] memory-safe A/B supervisor started pid=$$"

for data_file in \
    "$ROOT/data/pgx_train_10m_memmap/dds.npy" \
    "$ROOT/data/pgx_eval_500k_memmap/dds.npy"; do
    [[ -s "$data_file" ]] || { echo "missing $data_file"; exit 1; }
done

free_gb="$(df -Pk "$ROOT" | awk 'NR==2 {print int($4/1024/1024)}')"
if (( free_gb < MIN_FREE_GB )); then
    echo "insufficient free disk: ${free_gb}GB < ${MIN_FREE_GB}GB"
    exit 1
fi

failed_log() {
    grep -Eq 'Traceback|CUDA out of memory|OutOfMemoryError|Killed|(^|[^A-Za-z])NaN([^A-Za-z]|$)' "$1"
}

process_alive() {
    local pid_file="$1"
    [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

launch_agent() {
    local label="$1" gpu="$2" lower out
    lower="$(printf '%s' "$label" | tr '[:upper:]' '[:lower:]')"
    out="$FORMAL/$label"
    mkdir -p "$out"
    [[ ! -e "$out/run.log" ]] || { echo "refusing to overwrite $out"; exit 1; }

    nohup env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        MALLOC_ARENA_MAX=2 \
        PYTORCH_ALLOC_CONF=expandable_segments:True \
        "$PYTHON" -u "$ROOT/experiments/main_experiment.py" \
        --data "$ROOT/data/pgx_train_10m_memmap" \
        --eval-data "$ROOT/data/pgx_eval_500k_memmap" \
        --sl-checkpoint "$ROOT/results/sl_base.pt" \
        --belief-checkpoint "$ROOT/results/sl_base_bca.pt" \
        --output-dir "$out" \
        --train-agents "$label" --seed 42 --eval-seed 20260714 \
        --rounds 60 --steps-per-phase 256 --deals-per-step 512 \
        --rollout-chunk-deals 8192 \
        --batch-size 512 --num-epochs 4 --eval-deals 5000 \
        --fsp-pool-size 10 --fsp-add-interval 1 \
        --beta 0.05 --info-weight 0.05 --actor-belief-coef 0.1 \
        --learning-rate 3e-6 --entropy-coef 0.01 \
        --checkpoint-interval 1 \
        >"$out/run.log" 2>&1 < /dev/null &
    echo "$!" > "$out/pid"
    echo "[$(date -u +%FT%TZ)] launched $label gpu=$gpu pid=$!"
}

launch_agent A 0
launch_agent B 1
printf 'utc\tpid_a_rss_kb\tpid_b_rss_kb\tmem_available_kb\tgpu\n' > "$RESOURCE_LOG"

while true; do
    all_complete=1
    for label in A B; do
        lower="$(printf '%s' "$label" | tr '[:upper:]' '[:lower:]')"
        out="$FORMAL/$label"
        if failed_log "$out/run.log"; then
            echo "[$(date -u +%FT%TZ)] $label failed"
            touch "$ROOT/results/MAIN_AB_MEMORY_SAFE_BLOCKED"
            exit 1
        fi
        if [[ ! -f "$out/agent_${lower}_seed42.pt" ]]; then
            all_complete=0
            if ! process_alive "$out/pid"; then
                echo "[$(date -u +%FT%TZ)] $label stopped before completion"
                touch "$ROOT/results/MAIN_AB_MEMORY_SAFE_BLOCKED"
                exit 1
            fi
        fi
    done
    pid_a="$(cat "$FORMAL/A/pid")"
    pid_b="$(cat "$FORMAL/B/pid")"
    rss_a="$(ps -o rss= -p "$pid_a" 2>/dev/null | tr -d ' ' || true)"
    rss_b="$(ps -o rss= -p "$pid_b" 2>/dev/null | tr -d ' ' || true)"
    available="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    gpu="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';')"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$(date -u +%FT%TZ)" "${rss_a:-0}" "${rss_b:-0}" "$available" "$gpu" \
        >> "$RESOURCE_LOG"
    if (( available < MEMORY_GUARD_KB )); then
        echo "[$(date -u +%FT%TZ)] memory guard triggered: ${available}KB"
        touch "$ROOT/results/MAIN_AB_MEMORY_SAFE_BLOCKED"
        kill -TERM "$pid_a" "$pid_b" 2>/dev/null || true
        exit 1
    fi
    [[ "$all_complete" -eq 1 ]] && break
    sleep "$POLL_SECONDS"
done

touch "$ROOT/results/MAIN_AB_MEMORY_SAFE_COMPLETE"
echo "[$(date -u +%FT%TZ)] formal A/B complete; awaiting C parameters"
