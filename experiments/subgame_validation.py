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
        steps_per_phase = 50 if args.quick else 100,   # P56: 200→100 (deals_per_step×4)
        deals_per_step  = 32 if args.quick else 128,   # P56: 32→128, 4× larger batches
        lr              = 3e-6,
        batch_size      = 256,
        use_info_bonus  = False,
        beta            = 0.0,
        fsp_pool_size   = 10,
        fsp_add_interval= 2,
        kl_lambda_start = 0.1,
        kl_lambda_end   = 0.01,
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
        steps_per_phase = 50 if args.quick else 100,   # P56: same as A
        deals_per_step  = 32 if args.quick else 128,   # P56: same as A
        lr              = 3e-6,
        batch_size      = 256,
        use_info_bonus  = True,
        beta            = args.beta,
        fsp_pool_size   = 10,
        fsp_add_interval= 2,
        kl_lambda_start = 0.1,
        kl_lambda_end   = 0.01,
        bc_warmup_samples= 1000 if args.quick else 5000,
        bc_warmup_epochs = 5   if args.quick else 20,
        device          = device,
    )
    reward_stats_b = RunningStats()
    trainer_b = SubgameTrainer(env, cfg_b, reward_stats=reward_stats_b)

    # ── Stage 1: 初始化（SL checkpoint 优先，否则 rule-based BC）──────────────
    from env import NORTH as _N, EAST as _E, SOUTH as _S, WEST as _W
    sl_path = getattr(args, 'sl_checkpoint', None)
    if sl_path and os.path.exists(sl_path):
        print(f"\n[Stage 1] Loading SL checkpoint: {sl_path}")
        ckpt = torch.load(sl_path, map_location=device)
        player_key_map = [(_N, 'actor_n'), (_E, 'actor_e'),
                          (_S, 'actor_s'), (_W, 'actor_w')]
        for trainer in (trainer_a, trainer_b):
            for player, key in player_key_map:
                if key in ckpt:
                    trainer.agent.get_actor(player).load_state_dict(
                        {k: v.to(device) for k, v in ckpt[key].items()})
        print("  [SL Init] Weights loaded for N/E/S/W actors (both agents).")
    else:
        print("\n[Stage 1] BC Warmup (rule-based, SL checkpoint not found)...")
        trainer_a.run_bc_warmup()
        trainer_b.run_bc_warmup()

    # Stage 1 结束：将当前 actor 设为 KL anchor
    trainer_a.set_bc_anchor(trainer_a.agent)
    trainer_b.set_bc_anchor(trainer_b.agent)
    print("  [KL Anchor] BC anchor set for both agents.")

    # ── Stage 1.5: Belief Net 独立预训练（Agent B only）─────────────────────
    belief_pretrain_rounds = getattr(args, 'belief_pretrain_rounds', 5)
    if belief_pretrain_rounds > 0 and trainer_b.belief_net is not None:
        # num_rounds × deals_per_round = 总数据量
        # 训练到 early stopping，不再按轮数固定 epoch 数
        deals = 200 if args.quick else 2000
        print(f"\n[Stage 1.5] Belief Net Pretrain "
              f"(total {belief_pretrain_rounds * deals} deals, early stopping)...")
        trainer_b.pretrain_belief(
            num_rounds=belief_pretrain_rounds,
            deals_per_round=deals,
            epochs_per_round=5,
            max_epochs=getattr(args, 'belief_pretrain_max_epochs', 300),
        )

    # ── Stage 2: RL 微调 ────────────────────────────────────────────────────
    print("\n[Stage 2] RL Fine-tuning...")
    print("  ── Agent A ──")
    log_a = trainer_a.run(num_rounds=args.rounds)

    print("\n  ── Agent B ──")
    log_b = trainer_b.run(num_rounds=args.rounds)

    # ── Stage 3: 评估 ───────────────────────────────────────────────────────
    print("\n[Stage 3] Evaluation...")
    eval_deals = 200 if args.quick else 1000

    print("\n  → DDS Oracle Evaluation (Agent A):")
    result_a = trainer_a.evaluate_oracle(num_deals=eval_deals)

    print("\n  → DDS Oracle Evaluation (Agent B):")
    result_b = trainer_b.evaluate_oracle(num_deals=eval_deals)

    _print_oracle_comparison(result_a, result_b)

    # ── Head-to-head: A vs B 直接对战 ──────────────────────────────────────
    print("\n  → Head-to-Head: Agent A vs Agent B")
    h2h_deals = 200 if args.quick else 500
    h2h = trainer_a.evaluate_head_to_head(
        trainer_b,
        num_deals=h2h_deals,
        label_self="A",
        label_other="B",
    )

    # Belief Net 评估（Agent B only）
    if cfg_b.use_info_bonus:
        print("\n  → Belief Network Evaluation (Agent B):")
        trainer_b.evaluate_belief(num_deals=50 if args.quick else 200)

    # ── 排雷诊断 ────────────────────────────────────────────────────────────
    print("\n[Diagnostics]")
    _print_diagnostics(log_a, log_b, cfg_b)

    # ── 最终摘要 ────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  FINAL SUMMARY")
    print("═" * 60)
    print(f"  Oracle regret  A: {result_a['mean_regret']:+.3f} ± {result_a['std_regret']:.3f} IMP")
    print(f"  Oracle regret  B: {result_b['mean_regret']:+.3f} ± {result_b['std_regret']:.3f} IMP")
    print(f"  Δ oracle (B−A):   {result_b['mean_regret'] - result_a['mean_regret']:+.3f} IMP")
    print(f"  Head-to-head A IMP: {h2h['mean_imp']:+.3f} ± {h2h['std_imp']:.3f}  "
          f"win_rate={h2h['win_rate']:.1%}  p={h2h['p_value']:.3f} "
          f"{'✅ sig' if h2h['significant'] else '(ns)'}")
    conclusion = (
        "✅ B significantly better than A"
        if (h2h['mean_imp'] < 0 and h2h['significant'])
        else "❌ No significant difference" if not h2h['significant']
        else "⚠️  A outperforms B"
    )
    print(f"  Conclusion: {conclusion}")
    print("═" * 60)

    # ── 保存 checkpoint ─────────────────────────────────────────────────────
    if args.save_dir:
        import json
        os.makedirs(args.save_dir, exist_ok=True)
        trainer_a.agent.save(os.path.join(args.save_dir, f'agent_a_seed{args.seed}.pt'))
        trainer_b.agent.save(os.path.join(args.save_dir, f'agent_b_seed{args.seed}.pt'))
        report = {
            'seed': args.seed, 'beta': args.beta, 'rounds': args.rounds,
            'oracle_a': result_a, 'oracle_b': result_b,
            'head_to_head': h2h,
        }
        report_path = os.path.join(args.save_dir, f'report_seed{args.seed}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n  Checkpoints + report saved → {args.save_dir}")


def _print_oracle_comparison(result_a: dict, result_b: dict):
    print("\n  ── DDS Oracle Regret Comparison ──")
    print(f"  Agent A: {result_a['mean_regret']:+.3f} ± {result_a['std_regret']:.3f} IMP"
          f"  95% CI [{result_a['ci_lo']:+.3f}, {result_a['ci_hi']:+.3f}]")
    print(f"  Agent B: {result_b['mean_regret']:+.3f} ± {result_b['std_regret']:.3f} IMP"
          f"  95% CI [{result_b['ci_lo']:+.3f}, {result_b['ci_hi']:+.3f}]")
    delta = result_b['mean_regret'] - result_a['mean_regret']
    verdict = "B closer to oracle" if delta > 0 else "A closer to oracle / tie"
    print(f"  Δ oracle (B−A): {delta:+.3f} IMP  → {verdict}")


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
    parser.add_argument('--rounds', type=int, default=30,
                        help='Number of IBR rounds (default: 30)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer deals for debugging')
    parser.add_argument('--belief_pretrain_rounds', type=int, default=5,
                        help='Rounds of data collection for Belief Net pretrain (default: 5, total_deals=rounds×2000)')
    parser.add_argument('--belief_pretrain_max_epochs', type=int, default=300,
                        help='Max training epochs for Belief Net pretrain (default: 300)')
    parser.add_argument('--sl_checkpoint', default='results/sl_base.pt',
                        help='SL pretrained checkpoint (4-actor format from sl_pretrain.py)')
    parser.add_argument('--save_dir', default='results/competitive',
                        help='Directory to save checkpoints')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.type == 'competitive':
        run_competitive(args)
    else:
        raise ValueError(f"Unknown subgame type: {args.type}")
