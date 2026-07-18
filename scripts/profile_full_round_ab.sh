#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/bridge-coma}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
OUT="$ROOT/results/profile_full_round_memory_safe_seed42"
RESOURCE_LOG="$OUT/resources.tsv"
MEMORY_GUARD_KB="${MEMORY_GUARD_KB:-16777216}"

mkdir -p "$OUT"
[[ ! -e "$OUT/A/run.log" && ! -e "$OUT/B/run.log" ]] || {
    echo "refusing to overwrite existing profile output" >&2
    exit 1
}

launch() {
    local label="$1" gpu="$2" out
    out="$OUT/$label"
    mkdir -p "$out"
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
        --rounds 1 --steps-per-phase 256 --deals-per-step 512 \
        --rollout-chunk-deals 8192 \
        --batch-size 512 --num-epochs 4 --eval-deals 5000 \
        --info-calibration-deals 2048 \
        --fsp-pool-size 10 --fsp-add-interval 1 \
        --beta 0.05 --info-weight 0.05 --actor-belief-coef 0.1 \
        --learning-rate 3e-6 --entropy-coef 0.01 \
        --checkpoint-interval 1 \
        >"$out/run.log" 2>&1 < /dev/null &
    echo "$!" > "$out/pid"
}

launch A 0
launch B 1
pid_a="$(cat "$OUT/A/pid")"
pid_b="$(cat "$OUT/B/pid")"
printf 'utc\tpid_a_rss_kb\tpid_b_rss_kb\tmem_available_kb\tgpu\n' > "$RESOURCE_LOG"

while kill -0 "$pid_a" 2>/dev/null || kill -0 "$pid_b" 2>/dev/null; do
    rss_a="$(ps -o rss= -p "$pid_a" 2>/dev/null | tr -d ' ' || true)"
    rss_b="$(ps -o rss= -p "$pid_b" 2>/dev/null | tr -d ' ' || true)"
    available="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    gpu="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';')"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$(date -u +%FT%TZ)" "${rss_a:-0}" "${rss_b:-0}" "$available" "$gpu" \
        >> "$RESOURCE_LOG"
    if (( available < MEMORY_GUARD_KB )); then
        echo "memory guard triggered at ${available}KB" > "$OUT/MEMORY_GUARD_TRIGGERED"
        kill -TERM "$pid_a" "$pid_b" 2>/dev/null || true
        break
    fi
    sleep 15
done

status_a=0; status_b=0
wait "$pid_a" || status_a=$?
wait "$pid_b" || status_b=$?
printf 'A=%s B=%s\n' "$status_a" "$status_b" > "$OUT/exit_status.txt"
if (( status_a != 0 || status_b != 0 )); then
    exit 1
fi
