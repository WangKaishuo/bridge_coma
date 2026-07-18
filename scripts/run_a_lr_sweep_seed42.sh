#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/bridge-coma}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
OUT="$ROOT/results/a_lr_sweep_seed42"
SUPERVISOR_LOG="$OUT/supervisor.log"
RESOURCE_LOG="$OUT/resources.tsv"
MEMORY_GUARD_KB="${MEMORY_GUARD_KB:-16777216}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"

mkdir -p "$OUT"
exec >>"$SUPERVISOR_LOG" 2>&1
echo "[$(date -u +%FT%TZ)] A learning-rate sweep supervisor started pid=$$"

for required in \
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

failed_log() {
    grep -Eq 'Traceback|CUDA out of memory|OutOfMemoryError|Killed|(^|[^A-Za-z])NaN([^A-Za-z]|$)' "$1"
}

launch_branch() {
    local name="$1" lr="$2" gpu="$3"
    local branch="$OUT/$name"
    mkdir -p "$branch/snapshots"
    [[ ! -e "$branch/run.log" ]] || {
        echo "refusing to overwrite $branch/run.log"
        exit 1
    }
    nohup env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        MALLOC_ARENA_MAX=2 \
        PYTORCH_ALLOC_CONF=expandable_segments:True \
        "$PYTHON" -u "$ROOT/experiments/main_experiment.py" \
        --data "$ROOT/data/pgx_train_10m_memmap" \
        --eval-data "$ROOT/data/pgx_eval_500k_memmap" \
        --sl-checkpoint "$ROOT/results/sl_base.pt" \
        --belief-checkpoint "$ROOT/results/sl_base_bca.pt" \
        --output-dir "$branch" \
        --train-agents A --seed 42 --eval-seed 20260714 \
        --rounds 30 --steps-per-phase 256 --deals-per-step 512 \
        --rollout-chunk-deals 8192 \
        --batch-size 512 --num-epochs 4 --eval-deals 5000 \
        --info-calibration-deals 2048 \
        --fsp-pool-size 10 --fsp-add-interval 1 \
        --beta 0.05 --info-weight 0.05 --actor-belief-coef 0.1 \
        --learning-rate "$lr" --entropy-coef 0.01 \
        --checkpoint-interval 1 \
        >"$branch/run.log" 2>&1 < /dev/null &
    echo "$!" > "$branch/pid"
    printf '%s\n' "$lr" > "$branch/learning_rate.txt"
    echo "[$(date -u +%FT%TZ)] launched $name lr=$lr gpu=$gpu pid=$!"
}

launch_branch lr1e-5 1e-5 0
launch_branch lr3e-5 3e-5 1
printf 'utc\tlr1_rss_kb\tlr3_rss_kb\tmem_available_kb\tgpu\n' > "$RESOURCE_LOG"

pid_lr1="$(cat "$OUT/lr1e-5/pid")"
pid_lr3="$(cat "$OUT/lr3e-5/pid")"
gate10_complete=0
snapshot20_lr1=0
snapshot20_lr3=0
resource_tick=0

health_check() {
    local log="$1"
    "$PYTHON" - "$log" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
health = re.findall(
    r"auction_health: len=([0-9.]+).*?competitive=([0-9.]+)% "
    r"all_pass=([0-9.]+)% dbl=([0-9.]+)% rdbl=([0-9.]+)%",
    text,
)
entropy = re.findall(r"ent=([-+0-9.]+)", text)
if not health or not entropy:
    raise SystemExit("missing health metrics")
length, competitive, all_pass, double, redouble = map(float, health[-1])
ent = sum(map(float, entropy[-4:])) / min(4, len(entropy))
ok = (
    6.0 <= length <= 20.0
    and 0.0 <= competitive <= 100.0
    and all_pass <= 5.0
    and double <= 15.0
    and redouble <= 2.0
    and ent >= 0.02
)
print(
    f"len={length:.2f} competitive={competitive:.1f}% "
    f"all_pass={all_pass:.2f}% dbl={double:.2f}% "
    f"rdbl={redouble:.2f}% ent={ent:.4f} ok={ok}"
)
raise SystemExit(0 if ok else 1)
PY
}

pause_and_snapshot() {
    local name="$1" pid="$2" round="$3"
    local branch="$OUT/$name"
    kill -STOP "$pid"
    cp --reflink=auto \
        "$branch/agent_a_seed42.resume.pt" \
        "$branch/snapshots/agent_a_round${round}.resume.pt"
    touch "$branch/snapshots/ROUND${round}_PAUSED"
    echo "[$(date -u +%FT%TZ)] paused $name and saved round $round"
}

while true; do
    for item in "lr1e-5:$pid_lr1" "lr3e-5:$pid_lr3"; do
        name="${item%%:*}"
        pid="${item##*:}"
        branch="$OUT/$name"
        if failed_log "$branch/run.log"; then
            echo "[$(date -u +%FT%TZ)] $name failure signature detected"
            touch "$OUT/SWEEP_BLOCKED"
            kill -TERM "$pid_lr1" "$pid_lr3" 2>/dev/null || true
            exit 1
        fi
        if ! kill -0 "$pid" 2>/dev/null \
                && [[ ! -f "$branch/agent_a_seed42.pt" ]]; then
            echo "[$(date -u +%FT%TZ)] $name stopped before completion"
            touch "$OUT/SWEEP_BLOCKED"
            kill -TERM "$pid_lr1" "$pid_lr3" 2>/dev/null || true
            exit 1
        fi
    done

    available="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    if (( available < MEMORY_GUARD_KB )); then
        echo "[$(date -u +%FT%TZ)] memory guard triggered: ${available}KB"
        touch "$OUT/SWEEP_BLOCKED"
        kill -TERM "$pid_lr1" "$pid_lr3" 2>/dev/null || true
        exit 1
    fi

    if (( resource_tick % 12 == 0 )); then
        rss_lr1="$(ps -o rss= -p "$pid_lr1" 2>/dev/null | tr -d ' ' || true)"
        rss_lr3="$(ps -o rss= -p "$pid_lr3" 2>/dev/null | tr -d ' ' || true)"
        gpu="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';')"
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$(date -u +%FT%TZ)" "${rss_lr1:-0}" "${rss_lr3:-0}" \
            "$available" "$gpu" >> "$RESOURCE_LOG"
    fi
    resource_tick=$((resource_tick + 1))

    if (( gate10_complete == 0 )); then
        for item in "lr1e-5:$pid_lr1" "lr3e-5:$pid_lr3"; do
            name="${item%%:*}"
            pid="${item##*:}"
            branch="$OUT/$name"
            if [[ ! -f "$branch/snapshots/ROUND10_PAUSED" ]] \
                    && grep -q '\[Checkpoint\] round 10 ->' "$branch/run.log"; then
                pause_and_snapshot "$name" "$pid" 10
            fi
        done
        if [[ -f "$OUT/lr1e-5/snapshots/ROUND10_PAUSED" \
              && -f "$OUT/lr3e-5/snapshots/ROUND10_PAUSED" ]]; then
            echo "[$(date -u +%FT%TZ)] running round-10 health gate"
            if ! health_check "$OUT/lr1e-5/run.log" \
                    > "$OUT/lr1e-5/snapshots/health_round10.txt" \
                    || ! health_check "$OUT/lr3e-5/run.log" \
                    > "$OUT/lr3e-5/snapshots/health_round10.txt"; then
                echo "[$(date -u +%FT%TZ)] round-10 health gate failed; branches remain paused"
                touch "$OUT/SWEEP_BLOCKED"
                exit 1
            fi
            if ! CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
                "$ROOT/experiments/evaluate_agents_vs_sl.py" \
                --agent-a "$OUT/lr1e-5/snapshots/agent_a_round10.resume.pt" \
                --agent-b "$OUT/lr3e-5/snapshots/agent_a_round10.resume.pt" \
                --sl-checkpoint "$ROOT/results/sl_base.pt" \
                --data "$ROOT/data/pgx_eval_500k_memmap" \
                --deals 1000 --seed 20260714 \
                --output "$OUT/round10_vs_sl_1000.json" \
                > "$OUT/round10_evaluation.log" 2>&1; then
                echo "[$(date -u +%FT%TZ)] round-10 evaluation failed; branches remain paused"
                touch "$OUT/SWEEP_BLOCKED"
                exit 1
            fi
            kill -CONT "$pid_lr1" "$pid_lr3"
            touch "$OUT/ROUND10_GATE_PASSED_AND_RESUMED"
            gate10_complete=1
            echo "[$(date -u +%FT%TZ)] round-10 gate passed; resumed both branches"
        fi
    else
        if (( snapshot20_lr1 == 0 )) \
                && grep -q '\[Checkpoint\] round 20 ->' "$OUT/lr1e-5/run.log"; then
            pause_and_snapshot lr1e-5 "$pid_lr1" 20
            kill -CONT "$pid_lr1"
            snapshot20_lr1=1
        fi
        if (( snapshot20_lr3 == 0 )) \
                && grep -q '\[Checkpoint\] round 20 ->' "$OUT/lr3e-5/run.log"; then
            pause_and_snapshot lr3e-5 "$pid_lr3" 20
            kill -CONT "$pid_lr3"
            snapshot20_lr3=1
        fi
    fi

    complete_lr1=0
    complete_lr3=0
    [[ -f "$OUT/lr1e-5/agent_a_seed42.pt" ]] && complete_lr1=1
    [[ -f "$OUT/lr3e-5/agent_a_seed42.pt" ]] && complete_lr3=1
    if (( complete_lr1 == 1 && complete_lr3 == 1 )); then
        break
    fi
    sleep 5
done

status_lr1=0
status_lr3=0
wait "$pid_lr1" || status_lr1=$?
wait "$pid_lr3" || status_lr3=$?
if (( status_lr1 != 0 || status_lr3 != 0 )); then
    echo "[$(date -u +%FT%TZ)] branch exit failure: lr1=$status_lr1 lr3=$status_lr3"
    touch "$OUT/SWEEP_BLOCKED"
    exit 1
fi

echo "[$(date -u +%FT%TZ)] both 30-round branches complete; starting final evaluation"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$ROOT/experiments/evaluate_agents_vs_sl.py" \
    --agent-a "$OUT/lr1e-5/agent_a_seed42.resume.pt" \
    --agent-b "$OUT/lr3e-5/agent_a_seed42.resume.pt" \
    --sl-checkpoint "$ROOT/results/sl_base.pt" \
    --data "$ROOT/data/pgx_eval_500k_memmap" \
    --deals 5000 --seed 20260714 \
    --output "$OUT/round30_vs_sl_5000.json" \
    > "$OUT/round30_vs_sl_evaluation.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$ROOT/experiments/evaluate_resume_ab.py" \
    --agent-a "$OUT/lr1e-5/agent_a_seed42.resume.pt" \
    --agent-b "$OUT/lr3e-5/agent_a_seed42.resume.pt" \
    --data "$ROOT/data/pgx_eval_500k_memmap" \
    --deals 5000 --seed 20260714 \
    --output "$OUT/lr3e-5_vs_lr1e-5_round30.json" \
    > "$OUT/lr_direct_evaluation.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$ROOT/experiments/evaluate_agents_vs_sl.py" \
    --agent-a "$ROOT/results/main_unrestricted_formal_memory_safe_seed42/round30_gate/agent_a_round30.resume.pt" \
    --agent-b "$ROOT/results/main_unrestricted_formal_memory_safe_seed42/round30_gate/agent_b_round30.resume.pt" \
    --sl-checkpoint "$ROOT/results/sl_base.pt" \
    --data "$ROOT/data/pgx_eval_500k_memmap" \
    --deals 5000 --seed 20260714 \
    --output "$OUT/original_lr3e-6_round30_vs_sl_5000.json" \
    > "$OUT/original_round30_evaluation.log" 2>&1

touch "$OUT/SWEEP_COMPLETE"
echo "[$(date -u +%FT%TZ)] A learning-rate sweep and evaluations complete"
