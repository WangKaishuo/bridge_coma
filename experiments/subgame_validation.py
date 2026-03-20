"""
Subgame Validation — Competitive Path (Phase 1 排雷)
=====================================================

用法:
    python experiments/subgame_validation.py \\
        --type competitive \\
        --data path/to/1h1s_100k.npz \\
        --seed 42 \\
        --beta 0.05 \\
        --rounds 10 \\
        --quick

排雷判断标准（README §Phase 1）:
    ✅  ir > 0（info gain 有效）
    ✅  entropy 不坍塌（> 0.5）
    ✅  KL anchor 可控（< 0.3）
    ✅  value_loss 收敛（无爆炸）

主实验（Phase 3）入口见 experiments/train.py。
"""

import argparse
import os
import random
import sys
from pathlib import Path

# 将 bridge_coma 根目录加入 sys.path，确保包结构可被找到
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from subgames.competitive_env import CompetitiveSubgameEnv, make_agent_policy
from subgames.subgame_trainer import SubgameTrainer, SubgameConfig
from utils.running_stats import RunningStats


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_competitive(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[Competitive Subgame] seed={args.seed}  device={device}")
    print(f"  beta={args.beta}  rounds={args.rounds}  quick={args.quick}")

    # ── 环境 ────────────────────────────────────────────────────────────────
    print("\n[1] Initializing environment...")
    env = CompetitiveSubgameEnv(data_path=args.data)

    # ── Agent A（控制组：纯 MAPPO，beta=0）──────────────────────────────────
    print("\n[2] Building Agent A (MAPPO, β=0)...")
    cfg_a = SubgameConfig(
        num_rounds      = args.rounds,
        steps_per_phase = 50 if args.quick else 200,
        deals_per_step  = 32,
        lr              = 1e-6,
        batch_size      = 256,
        use_info_bonus  = False,
        beta            = 0.0,
        fsp_pool_size   = 10,
        fsp_add_interval= 2,
        kl_lambda_start = 0.5,
        kl_lambda_end   = 0.1,
        bc_warmup_samples= 1000 if args.quick else 5000,
        bc_warmup_epochs = 5   if args.quick else 20,
        device          = device,
    )
    reward_stats_a = RunningStats()
    trainer_a = SubgameTrainer(env, cfg_a, reward_stats=reward_stats_a)

    # ── Agent B（实验组：MAPPO + r_info，beta=args.beta）────────────────────
    print(f"\n[3] Building Agent B (MAPPO + r_info, β={args.beta})...")
    cfg_b = SubgameConfig(
        num_rounds      = args.rounds,
        steps_per_phase = 50 if args.quick else 200,
        deals_per_step  = 32,
        lr              = 1e-6,
        batch_size      = 256,
        use_info_bonus  = True,
        beta            = args.beta,
        fsp_pool_size   = 10,
        fsp_add_interval= 2,
        kl_lambda_start = 0.5,
        kl_lambda_end   = 0.1,
        bc_warmup_samples= 1000 if args.quick else 5000,
        bc_warmup_epochs = 5   if args.quick else 20,
        device          = device,
    )
    reward_stats_b = RunningStats()
    trainer_b = SubgameTrainer(env, cfg_b, reward_stats=reward_stats_b)

    # ── Stage 1: BC 预热（rule-based，~5k 局）────────────────────────────────
    print("\n[Stage 1] BC Warmup (rule-based)...")
    trainer_a.run_bc_warmup()
    trainer_b.run_bc_warmup()

    # BC 结束后，将当前 actor 设为 KL anchor
    trainer_a.set_bc_anchor(trainer_a.agent)
    trainer_b.set_bc_anchor(trainer_b.agent)
    print("  [KL Anchor] BC anchor set for both agents.")

    # ── Stage 2: RL 微调 ────────────────────────────────────────────────────
    print("\n[Stage 2] RL Fine-tuning...")
    print("  ── Agent A ──")
    log_a = trainer_a.run(num_rounds=args.rounds)

    print("\n  ── Agent B ──")
    log_b = trainer_b.run(num_rounds=args.rounds)

    # ── Stage 3: 评估 ───────────────────────────────────────────────────────
    print("\n[Stage 3] Evaluation...")

    print("\n  → DDS Oracle Evaluation (Agent A):")
    result_a = trainer_a.evaluate_oracle(num_deals=200 if args.quick else 1000)

    print("\n  → DDS Oracle Evaluation (Agent B):")
    result_b = trainer_b.evaluate_oracle(num_deals=200 if args.quick else 1000)

    # 打印对比
    _print_comparison(result_a, result_b)

    # Belief Net 评估（Agent B only）
    if cfg_b.use_info_bonus:
        print("\n  → Belief Network Evaluation (Agent B):")
        trainer_b.evaluate_belief(num_deals=50 if args.quick else 100)

    # ── 排雷诊断 ────────────────────────────────────────────────────────────
    print("\n[Diagnostics]")
    _print_diagnostics(log_a, log_b, cfg_b)

    # ── 保存 checkpoint ─────────────────────────────────────────────────────
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        trainer_a.agent.save(os.path.join(args.save_dir, f'agent_a_seed{args.seed}.pt'))
        trainer_b.agent.save(os.path.join(args.save_dir, f'agent_b_seed{args.seed}.pt'))
        print(f"\n  Checkpoints saved → {args.save_dir}")


def _print_comparison(result_a: dict, result_b: dict):
    print("\n  ── IMP Regret vs DDS Oracle ──")
    print(f"  Agent A: {result_a['mean_regret']:+.3f} ± {result_a['std_regret']:.3f} IMP"
          f"  95% CI [{result_a['ci_lo']:+.3f}, {result_a['ci_hi']:+.3f}]")
    print(f"  Agent B: {result_b['mean_regret']:+.3f} ± {result_b['std_regret']:.3f} IMP"
          f"  95% CI [{result_b['ci_lo']:+.3f}, {result_b['ci_hi']:+.3f}]")
    delta = result_b['mean_regret'] - result_a['mean_regret']
    verdict = "✅ B better" if delta > 0 else "❌ A better / tie"
    print(f"  Δ (B − A): {delta:+.3f} IMP  → {verdict}")


def _print_diagnostics(log_a: list, log_b: list, cfg_b: SubgameConfig):
    """排雷判断：检查训练健康性指标."""

    def _last_metric(log: list, player_key: int, metric: str) -> float:
        for entry in reversed(log):
            # 兼容新格式 ns_metrics/ew_metrics 和旧格式 s_metrics/n_metrics
            for key in ('ns_metrics', 'ew_metrics', 's_metrics', 'n_metrics'):
                m = entry.get(key, {}).get(player_key, {})
                if metric in m:
                    return m[metric]
        return float('nan')

    from env import SOUTH, EAST
    ent_a   = _last_metric(log_a, SOUTH, 'entropy')
    ent_b   = _last_metric(log_b, SOUTH, 'entropy')
    vl_a    = _last_metric(log_a, SOUTH, 'value_loss')
    vl_b    = _last_metric(log_b, SOUTH, 'value_loss')
    kl_b    = _last_metric(log_b, SOUTH, 'kl_loss')
    klam_b  = _last_metric(log_b, SOUTH, 'kl_lambda')
    # EW 指标
    ent_a_e = _last_metric(log_a, EAST, 'entropy')
    vl_a_e  = _last_metric(log_a, EAST, 'value_loss')
    ent_b_e = _last_metric(log_b, EAST, 'entropy')
    vl_b_e  = _last_metric(log_b, EAST, 'value_loss')

    print(f"  Agent A: NS entropy={ent_a:.3f} vl={vl_a:.4f} │ EW entropy={ent_a_e:.3f} vl={vl_a_e:.4f}")
    print(f"  Agent B: NS entropy={ent_b:.3f} vl={vl_b:.4f} kl={kl_b:.5f}(λ={klam_b:.3f}) │ EW entropy={ent_b_e:.3f} vl={vl_b_e:.4f}")

    ok = True
    if ent_a < 0.5 or ent_b < 0.5:
        print("  ⚠️  Entropy collapse detected! Consider higher entropy_coef.")
        ok = False
    if vl_a > 10 or vl_b > 10:
        print("  ⚠️  Value loss too high! Consider more critic warmup.")
        ok = False
    if kl_b > 0.3:
        print("  ⚠️  KL too high! Consider higher kl_lambda_start.")
        ok = False

    if cfg_b.use_info_bonus:
        # ir 健康性: 从日志中找 info_ratio
        ir_vals = []
        for entry in log_b:
            if 'info_ratio' in entry:
                ir_vals.append(entry['info_ratio'])
        if ir_vals:
            mean_ir = np.mean(ir_vals)
            print(f"  Agent B: mean ir={mean_ir:.4f}")
            if mean_ir <= 0:
                print("  ⚠️  ir ≤ 0! Check Belief Net and r_info wiring.")
                ok = False

    if ok:
        print("  ✅ All diagnostics passed.")
    else:
        print("  ❌ Fix above issues before running Phase 3.")


def parse_args():
    parser = argparse.ArgumentParser(description='Bridge-COMA Subgame Validation')
    parser.add_argument('--type', default='competitive',
                        choices=['competitive'],
                        help='Subgame type (competitive only for now)')
    parser.add_argument('--data', '--competitive_data', default='data/competitive_100k.npz',
                        help='Path to DDS data (npz)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--beta', type=float, default=0.05,
                        help='Agent B r_info coefficient')
    parser.add_argument('--rounds', type=int, default=10,
                        help='Number of IBR rounds')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer deals/deals for debugging')
    parser.add_argument('--save_dir', default='results/competitive',
                        help='Directory to save checkpoints')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.type == 'competitive':
        run_competitive(args)
    else:
        raise ValueError(f"Unknown subgame type: {args.type}")
