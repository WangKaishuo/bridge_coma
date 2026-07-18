#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/root/bridge-coma}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
OUT="$ROOT/results/main_lr1e-5_seed42/C_beta_sweep_seed42"
C1="$OUT/beta1"
C2="$OUT/beta2"
A60="$ROOT/results/main_lr1e-5_seed42/A/snapshots/agent_a_round60.resume.pt"
B60="$ROOT/results/main_lr1e-5_seed42/B/agent_b_seed42.resume.pt"
SL="$ROOT/results/sl_base.pt"
JUDGE="$ROOT/results/sl_base_bca.pt"
TRAIN_DATA="$ROOT/data/pgx_train_10m_memmap"
EVAL_DATA="$ROOT/data/pgx_eval_500k_memmap"
RESOURCE_LOG="$OUT/resources.tsv"
MEMORY_GUARD_KB="${MEMORY_GUARD_KB:-16777216}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"

mkdir -p "$C1" "$C2"
if [[ -e "$OUT/RUN_COMPLETE" || -e "$OUT/RUN_BLOCKED" \
      || -e "$C1/run.log" || -e "$C2/run.log" ]]; then
    echo "refusing to overwrite an existing C beta run" >&2
    exit 1
fi
for required in \
    "$TRAIN_DATA/dds.npy" "$EVAL_DATA/dds.npy" "$SL" "$JUDGE" "$A60" "$B60"; do
    [[ -s "$required" ]] || { echo "missing required artifact: $required" >&2; exit 1; }
done
free_gb="$(df -Pk "$ROOT" | awk 'NR==2 {print int($4/1024/1024)}')"
(( free_gb >= MIN_FREE_GB )) || {
    echo "insufficient free disk: ${free_gb}GB < ${MIN_FREE_GB}GB" >&2
    exit 1
}

printf 'started_utc=%s\nseed=42\neval_seed=20260714\nvariants=beta1,beta2\n' \
    "$(date -u +%FT%TZ)" > "$OUT/manifest.txt"
printf 'utc\tpid_beta1\trss_beta1_kb\tpid_beta2\trss_beta2_kb\tmem_available_kb\tfree_disk_gb\tgpu0_used_mib\tgpu1_used_mib\n' \
    > "$RESOURCE_LOG"

launch_c() {
    local gpu="$1"
    local beta="$2"
    local output="$3"
    env CUDA_VISIBLE_DEVICES="$gpu" \
        MALLOC_ARENA_MAX=2 \
        PYTORCH_ALLOC_CONF=expandable_segments:True \
        "$PYTHON" -u "$ROOT/experiments/main_experiment.py" \
        --data "$TRAIN_DATA" --eval-data "$EVAL_DATA" \
        --sl-checkpoint "$SL" --belief-checkpoint "$JUDGE" \
        --output-dir "$output" --train-agents C --seed 42 --eval-seed 20260714 \
        --rounds 60 --steps-per-phase 256 --deals-per-step 512 \
        --rollout-chunk-deals 8192 --batch-size 512 --num-epochs 4 \
        --eval-deals 5000 --info-calibration-deals 2048 \
        --fsp-pool-size 10 --fsp-add-interval 1 \
        --beta "$beta" --info-weight 0.05 --actor-belief-coef 0.1 \
        --learning-rate 1e-5 --entropy-coef 0.01 --checkpoint-interval 1 \
        > "$output/run.log" 2>&1 < /dev/null &
    launched_pid="$!"
}

launched_pid=""
launch_c 0 1.0 "$C1"
pid1="$launched_pid"
launch_c 1 2.0 "$C2"
pid2="$launched_pid"
printf '%s\n' "$pid1" > "$C1/pid"
printf '%s\n' "$pid2" > "$C2/pid"

guard_failed=0
while kill -0 "$pid1" 2>/dev/null || kill -0 "$pid2" 2>/dev/null; do
    rss1="$(awk '/VmRSS/ {print $2}' "/proc/$pid1/status" 2>/dev/null || echo 0)"
    rss2="$(awk '/VmRSS/ {print $2}' "/proc/$pid2/status" 2>/dev/null || echo 0)"
    mem_available="$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
    free_gb="$(df -Pk "$ROOT" | awk 'NR==2 {print int($4/1024/1024)}')"
    gpu0="$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits -i 0 2>/dev/null | awk '{s+=$1} END {print s+0}')"
    gpu1="$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits -i 1 2>/dev/null | awk '{s+=$1} END {print s+0}')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -u +%FT%TZ)" "$pid1" "$rss1" "$pid2" "$rss2" \
        "$mem_available" "$free_gb" "$gpu0" "$gpu1" >> "$RESOURCE_LOG"
    if (( mem_available < MEMORY_GUARD_KB || free_gb < MIN_FREE_GB )); then
        guard_failed=1
        kill "$pid1" "$pid2" 2>/dev/null || true
        break
    fi
    sleep 300
done

status1=0
status2=0
wait "$pid1" || status1=$?
wait "$pid2" || status2=$?
if (( guard_failed != 0 || status1 != 0 || status2 != 0 )); then
    printf 'training_failed_utc=%s\nstatus_beta1=%s\nstatus_beta2=%s\n' \
        "$(date -u +%FT%TZ)" "$status1" "$status2" >> "$OUT/manifest.txt"
    touch "$OUT/RUN_BLOCKED"
    exit 1
fi

C1_MODEL="$C1/agent_c_seed42.resume.pt"
C2_MODEL="$C2/agent_c_seed42.resume.pt"
[[ -s "$C1_MODEL" && -s "$C2_MODEL" ]] || {
    echo "missing final C resume checkpoint" >> "$OUT/manifest.txt"
    touch "$OUT/RUN_BLOCKED"
    exit 1
}
if grep -Eqi 'Traceback|CUDA out of memory|(^|[^A-Za-z])NaN([^A-Za-z]|$)|killed process' \
        "$C1/run.log" "$C2/run.log"; then
    echo "failure signature found in training log" >> "$OUT/manifest.txt"
    touch "$OUT/RUN_BLOCKED"
    exit 1
fi

run_eval_set() {
    local gpu="$1"
    local model="$2"
    local output="$3"
    mkdir -p "$output/evaluation"
    env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
        "$ROOT/experiments/evaluate_single_vs_sl.py" \
        --agent "$model" --sl-checkpoint "$SL" --data "$EVAL_DATA" \
        --deals 20000 --seed 20260714 --output "$output/evaluation/c_vs_sl_20000.json" \
        > "$output/evaluation/c_vs_sl.log" 2>&1 &
    local p_sl=$!
    env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
        "$ROOT/experiments/evaluate_resume_ab.py" \
        --agent-a "$A60" --agent-b "$model" --data "$EVAL_DATA" \
        --deals 20000 --seed 20260714 --output "$output/evaluation/c_vs_a60_20000.json" \
        > "$output/evaluation/c_vs_a60.log" 2>&1 &
    local p_a=$!
    env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
        "$ROOT/experiments/evaluate_resume_ab.py" \
        --agent-a "$B60" --agent-b "$model" --data "$EVAL_DATA" \
        --deals 20000 --seed 20260714 --output "$output/evaluation/c_vs_b60_20000.json" \
        > "$output/evaluation/c_vs_b60.log" 2>&1 &
    local p_b=$!
    env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
        "$ROOT/experiments/analyze_judge_information.py" \
        --agent-a "$B60" --agent-b "$model" --judge-checkpoint "$JUDGE" \
        --data "$EVAL_DATA" --deals 5000 --seed 20260714 \
        --output "$output/evaluation/c_vs_b60_judge_5000.json" \
        > "$output/evaluation/c_vs_b60_judge.log" 2>&1 &
    local p_j=$!
    local status=0
    wait "$p_sl" || status=1
    wait "$p_a" || status=1
    wait "$p_b" || status=1
    wait "$p_j" || status=1
    return "$status"
}

run_eval_set 0 "$C1_MODEL" "$C1" &
eval1=$!
run_eval_set 1 "$C2_MODEL" "$C2" &
eval2=$!
eval_status1=0
eval_status2=0
wait "$eval1" || eval_status1=$?
wait "$eval2" || eval_status2=$?
if (( eval_status1 != 0 || eval_status2 != 0 )); then
    printf 'evaluation_failed_utc=%s\nstatus_beta1=%s\nstatus_beta2=%s\n' \
        "$(date -u +%FT%TZ)" "$eval_status1" "$eval_status2" >> "$OUT/manifest.txt"
    touch "$OUT/RUN_BLOCKED"
    exit 1
fi

printf 'finished_utc=%s\nstatus=0\n' "$(date -u +%FT%TZ)" >> "$OUT/manifest.txt"
touch "$OUT/RUN_COMPLETE"
