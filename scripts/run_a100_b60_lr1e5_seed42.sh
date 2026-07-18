#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/bridge-coma}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SOURCE_A="$ROOT/results/a_lr_sweep_seed42/lr1e-5/agent_a_seed42.resume.pt"
OUT="$ROOT/results/main_lr1e-5_seed42"
A_OUT="$OUT/A"
B_OUT="$OUT/B"
LOG="$OUT/supervisor.log"
RESOURCE_LOG="$OUT/resources.tsv"
MEMORY_GUARD_KB="${MEMORY_GUARD_KB:-16777216}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"
SNAPSHOT_MIN_FREE_GB="${SNAPSHOT_MIN_FREE_GB:-25}"

mkdir -p "$A_OUT/snapshots" "$B_OUT/snapshots"
exec >>"$LOG" 2>&1
echo "[$(date -u +%FT%TZ)] A100/B60 lr=1e-5 supervisor started pid=$$"

for required in \
    "$SOURCE_A" \
    "$ROOT/data/pgx_train_10m_memmap/dds.npy" \
    "$ROOT/data/pgx_eval_500k_memmap/dds.npy" \
    "$ROOT/results/sl_base.pt" \
    "$ROOT/results/sl_base_bca.pt"; do
    [[ -s "$required" ]] || { echo "missing $required"; exit 1; }
done

free_gb="$(df -Pk "$ROOT" | awk 'NR==2 {print int($4/1024/1024)}')"
(( free_gb >= MIN_FREE_GB )) || {
    echo "insufficient free disk: ${free_gb}GB < ${MIN_FREE_GB}GB"
    exit 1
}
[[ ! -e "$A_OUT/run.log" && ! -e "$B_OUT/run.log" ]] || {
    echo "refusing to overwrite existing A/B run logs"
    exit 1
}

# Preserve the completed round-30 sweep artifact.  The continuation overwrites
# only this reflinked copy as it advances through rounds 31..100.
cp --reflink=auto "$SOURCE_A" "$A_OUT/agent_a_seed42.resume.pt"
cp --reflink=auto "$SOURCE_A" "$A_OUT/snapshots/agent_a_round30.resume.pt"

nohup env \
    CUDA_VISIBLE_DEVICES=0 \
    MALLOC_ARENA_MAX=2 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" -u "$ROOT/experiments/main_experiment.py" \
    --data "$ROOT/data/pgx_train_10m_memmap" \
    --eval-data "$ROOT/data/pgx_eval_500k_memmap" \
    --sl-checkpoint "$ROOT/results/sl_base.pt" \
    --belief-checkpoint "$ROOT/results/sl_base_bca.pt" \
    --output-dir "$A_OUT" --train-agents A --seed 42 --eval-seed 20260714 \
    --resume "$A_OUT/agent_a_seed42.resume.pt" \
    --rounds 100 --steps-per-phase 256 --deals-per-step 512 \
    --rollout-chunk-deals 8192 --batch-size 512 --num-epochs 4 \
    --eval-deals 5000 --info-calibration-deals 2048 \
    --fsp-pool-size 10 --fsp-add-interval 1 \
    --beta 0.05 --info-weight 0.05 --actor-belief-coef 0.1 \
    --learning-rate 1e-5 --entropy-coef 0.01 --checkpoint-interval 1 \
    >"$A_OUT/run.log" 2>&1 < /dev/null &
pid_a=$!
echo "$pid_a" > "$A_OUT/pid"

nohup env \
    CUDA_VISIBLE_DEVICES=1 \
    MALLOC_ARENA_MAX=2 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" -u "$ROOT/experiments/main_experiment.py" \
    --data "$ROOT/data/pgx_train_10m_memmap" \
    --eval-data "$ROOT/data/pgx_eval_500k_memmap" \
    --sl-checkpoint "$ROOT/results/sl_base.pt" \
    --belief-checkpoint "$ROOT/results/sl_base_bca.pt" \
    --output-dir "$B_OUT" --train-agents B --seed 42 --eval-seed 20260714 \
    --rounds 60 --steps-per-phase 256 --deals-per-step 512 \
    --rollout-chunk-deals 8192 --batch-size 512 --num-epochs 4 \
    --eval-deals 5000 --info-calibration-deals 2048 \
    --fsp-pool-size 10 --fsp-add-interval 1 \
    --beta 0.05 --info-weight 0.05 --actor-belief-coef 0.1 \
    --learning-rate 1e-5 --entropy-coef 0.01 --checkpoint-interval 1 \
    >"$B_OUT/run.log" 2>&1 < /dev/null &
pid_b=$!
echo "$pid_b" > "$B_OUT/pid"

echo "[$(date -u +%FT%TZ)] launched A resume round30->100 gpu=0 pid=$pid_a"
echo "[$(date -u +%FT%TZ)] launched B fresh 0->60 gpu=1 pid=$pid_b"
printf 'utc\tpid_a_rss_kb\tpid_b_rss_kb\tmem_available_kb\tfree_disk_gb\tgpu\n' \
    > "$RESOURCE_LOG"

failed_log() {
    grep -Eq 'Traceback|CUDA out of memory|OutOfMemoryError|Killed|(^|[^A-Za-z])NaN([^A-Za-z]|$)' "$1"
}

snapshot_and_evaluate_a() {
    local round="$1"
    local marker="$A_OUT/snapshots/ROUND${round}_EVAL_COMPLETE"
    [[ ! -e "$marker" ]] || return 0

    local stopped=0
    if kill -0 "$pid_a" 2>/dev/null; then
        kill -STOP "$pid_a"
        stopped=1
    fi
    local free_now
    free_now="$(df -Pk "$ROOT" | awk 'NR==2 {print int($4/1024/1024)}')"
    if (( free_now < SNAPSHOT_MIN_FREE_GB )); then
        echo "[$(date -u +%FT%TZ)] snapshot disk guard: ${free_now}GB"
        touch "$OUT/RUN_BLOCKED"
        exit 1
    fi
    cp --reflink=auto \
        "$A_OUT/agent_a_seed42.resume.pt" \
        "$A_OUT/snapshots/agent_a_round${round}.resume.pt"
    echo "[$(date -u +%FT%TZ)] saved A round $round; evaluating 5000 vs SL"
    if ! CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
        "$ROOT/experiments/evaluate_single_vs_sl.py" \
        --agent "$A_OUT/snapshots/agent_a_round${round}.resume.pt" \
        --sl-checkpoint "$ROOT/results/sl_base.pt" \
        --data "$ROOT/data/pgx_eval_500k_memmap" \
        --deals 5000 --seed 20260714 \
        --output "$A_OUT/snapshots/round${round}_vs_sl_5000.json" \
        > "$A_OUT/snapshots/round${round}_evaluation.log" 2>&1; then
        echo "[$(date -u +%FT%TZ)] A round $round evaluation failed"
        touch "$OUT/RUN_BLOCKED"
        exit 1
    fi
    touch "$marker"
    if (( stopped == 1 )) && kill -0 "$pid_a" 2>/dev/null; then
        kill -CONT "$pid_a"
    fi
    echo "[$(date -u +%FT%TZ)] A round $round evaluation complete"
}

snapshot_b30=0
resource_tick=0
while true; do
    if failed_log "$A_OUT/run.log" || failed_log "$B_OUT/run.log"; then
        echo "[$(date -u +%FT%TZ)] failure signature detected"
        touch "$OUT/RUN_BLOCKED"
        kill -TERM "$pid_a" "$pid_b" 2>/dev/null || true
        exit 1
    fi

    complete_a=0
    complete_b=0
    [[ -f "$A_OUT/agent_a_seed42.pt" ]] && complete_a=1
    [[ -f "$B_OUT/agent_b_seed42.pt" ]] && complete_b=1
    if (( complete_a == 0 )) && ! kill -0 "$pid_a" 2>/dev/null; then
        echo "[$(date -u +%FT%TZ)] A stopped before round 100"
        touch "$OUT/RUN_BLOCKED"
        kill -TERM "$pid_b" 2>/dev/null || true
        exit 1
    fi
    if (( complete_b == 0 )) && ! kill -0 "$pid_b" 2>/dev/null; then
        echo "[$(date -u +%FT%TZ)] B stopped before round 60"
        touch "$OUT/RUN_BLOCKED"
        kill -TERM "$pid_a" 2>/dev/null || true
        exit 1
    fi

    available="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    free_now="$(df -Pk "$ROOT" | awk 'NR==2 {print int($4/1024/1024)}')"
    if (( available < MEMORY_GUARD_KB || free_now < SNAPSHOT_MIN_FREE_GB )); then
        echo "[$(date -u +%FT%TZ)] resource guard: mem=${available}KB disk=${free_now}GB"
        touch "$OUT/RUN_BLOCKED"
        kill -TERM "$pid_a" "$pid_b" 2>/dev/null || true
        exit 1
    fi

    if (( resource_tick % 12 == 0 )); then
        rss_a="$(ps -o rss= -p "$pid_a" 2>/dev/null | tr -d ' ' || true)"
        rss_b="$(ps -o rss= -p "$pid_b" 2>/dev/null | tr -d ' ' || true)"
        gpu="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';')"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date -u +%FT%TZ)" "${rss_a:-0}" "${rss_b:-0}" \
            "$available" "$free_now" "$gpu" >> "$RESOURCE_LOG"
    fi
    resource_tick=$((resource_tick + 1))

    for round in 40 50 60 70 80 90 100; do
        marker="$A_OUT/snapshots/ROUND${round}_EVAL_COMPLETE"
        if [[ ! -e "$marker" ]] \
                && grep -q "\[Checkpoint\] round $round ->" "$A_OUT/run.log"; then
            snapshot_and_evaluate_a "$round"
        fi
    done

    if (( snapshot_b30 == 0 )) \
            && grep -q '\[Checkpoint\] round 30 ->' "$B_OUT/run.log"; then
        kill -STOP "$pid_b"
        cp --reflink=auto "$B_OUT/agent_b_seed42.resume.pt" \
            "$B_OUT/snapshots/agent_b_round30.resume.pt"
        kill -CONT "$pid_b"
        touch "$B_OUT/snapshots/ROUND30_SAVED"
        snapshot_b30=1
        echo "[$(date -u +%FT%TZ)] saved B round 30 snapshot"
    fi

    if (( complete_a == 1 && complete_b == 1 )) \
            && [[ -f "$A_OUT/snapshots/ROUND100_EVAL_COMPLETE" ]]; then
        break
    fi
    sleep 5
done

status_a=0
status_b=0
wait "$pid_a" || status_a=$?
wait "$pid_b" || status_b=$?
if (( status_a != 0 || status_b != 0 )); then
    echo "[$(date -u +%FT%TZ)] process exit failure A=$status_a B=$status_b"
    touch "$OUT/RUN_BLOCKED"
    exit 1
fi

echo "[$(date -u +%FT%TZ)] training complete; evaluating B60 vs SL and B60 vs A60"
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -u "$ROOT/experiments/evaluate_single_vs_sl.py" \
    --agent "$B_OUT/agent_b_seed42.resume.pt" \
    --sl-checkpoint "$ROOT/results/sl_base.pt" \
    --data "$ROOT/data/pgx_eval_500k_memmap" \
    --deals 5000 --seed 20260714 \
    --output "$B_OUT/b60_vs_sl_5000.json" \
    > "$B_OUT/b60_vs_sl_evaluation.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$ROOT/experiments/evaluate_resume_ab.py" \
    --agent-a "$A_OUT/snapshots/agent_a_round60.resume.pt" \
    --agent-b "$B_OUT/agent_b_seed42.resume.pt" \
    --data "$ROOT/data/pgx_eval_500k_memmap" \
    --deals 5000 --seed 20260714 \
    --output "$OUT/b60_vs_a60_5000.json" \
    > "$OUT/b60_vs_a60_evaluation.log" 2>&1

touch "$OUT/RUN_COMPLETE"
echo "[$(date -u +%FT%TZ)] A100/B60 run and requested evaluations complete"
