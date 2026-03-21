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
    print(f"  beta={args.beta}  info_weight={args.info_weight}  rounds={args.rounds}  quick={args.quick}")

    # ── 环境 ────────────────────────────────────────────────────────────────
    print("\n[1] Initializing environment...")
    env = CompetitiveSubgameEnv(data_path=args.data)

    # ── Agent A（控制组：纯 MAPPO，beta=0）──────────────────────────────────
    print("\n[2] Building Agent A (MAPPO, β=0)...")
    cfg_a = SubgameConfig(
        num_rounds       = args.rounds,
        steps_per_phase  = 10  if args.quick else 64,
        deals_per_step   = 32  if args.quick else 512,
        lr               = 3e-6,
        batch_size       = 256,
        use_info_bonus   = False,
        beta             = 0.0,
        fsp_pool_size    = 10,   # P90: restored (P89's 3 was too homogeneous)
        fsp_add_interval = 2,    # P90: restored
        kl_lambda_start  = 0.5,   # P88
        kl_lambda_end    = 0.1,   # P90: floor 0.1 (was 0.0, caused entropy collapse)
        kl_anneal_frac   = 0.3,   # P90: fast anneal in first 30%, then hold at 0.1
        bc_warmup_samples= 1000 if args.quick else 5000,
        bc_warmup_epochs = 5    if args.quick else 20,
        device           = device,
    )
    reward_stats_a = RunningStats()
    trainer_a = SubgameTrainer(env, cfg_a, reward_stats=reward_stats_a)

    # ── Agent B（实验组：MAPPO + r_info，beta=args.beta）────────────────────
    print(f"\n[3] Building Agent B (MAPPO + r_info, β={args.beta}, w={args.info_weight})...")
    cfg_b = SubgameConfig(
        # 与A完全对称，唯一区别是use_info_bonus、beta、info_reward_weight
        num_rounds       = args.rounds,
        steps_per_phase  = 10  if args.quick else 64,
        deals_per_step   = 32  if args.quick else 512,
        lr               = 3e-6,
        batch_size       = 256,
        use_info_bonus   = True,
        beta             = args.beta,
        info_reward_weight = args.info_weight,
        fsp_pool_size    = 10,   # P90: restored
        fsp_add_interval = 2,    # P90: restored
        kl_lambda_start  = 0.5,   # P88
        kl_lambda_end    = 0.1,   # P90: floor 0.1
        kl_anneal_frac   = 0.3,   # P90: fast anneal in first 30%
        bc_warmup_samples= 1000 if args.quick else 5000,
        bc_warmup_epochs = 5    if args.quick else 20,
        device           = device,
    )
    reward_stats_b = RunningStats()
    trainer_b = SubgameTrainer(env, cfg_b, reward_stats=reward_stats_b)

    # ── Stage 1: 初始化（SL checkpoint 优先，否则 rule-based BC）──────────────
    from env import NORTH as _N, EAST as _E, SOUTH as _S, WEST as _W
    sl_path = getattr(args, 'sl_checkpoint', None)

    # 确定哪些trainer需要SL初始化
    trainers_to_init = [trainer_b] if args.load_agent_a else [trainer_a, trainer_b]

    if sl_path and os.path.exists(sl_path):
        print(f"\n[Stage 1] Loading SL checkpoint: {sl_path}")
        ckpt = torch.load(sl_path, map_location=device)
        player_key_map = [(_N, 'actor_n'), (_E, 'actor_e'),
                          (_S, 'actor_s'), (_W, 'actor_w')]
        for trainer in trainers_to_init:
            for player, key in player_key_map:
                if key in ckpt:
                    trainer.agent.get_actor(player).load_state_dict(
                        {k: v.to(device) for k, v in ckpt[key].items()})
        init_names = "B only" if args.load_agent_a else "both agents"
        print(f"  [SL Init] Weights loaded for N/E/S/W actors ({init_names}).")
    else:
        print("\n[Stage 1] BC Warmup (rule-based, SL checkpoint not found)...")
        for trainer in trainers_to_init:
            trainer.run_bc_warmup()

    # Stage 1 结束：将当前 actor 设为 KL anchor
    for trainer in trainers_to_init:
        trainer.set_bc_anchor(trainer.agent)
    anchor_names = "Agent B" if args.load_agent_a else "both agents"
    print(f"  [KL Anchor] BC anchor set for {anchor_names}.")

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

    if args.load_agent_a:
        # 直接加载预训练好的Agent A checkpoint，跳过训练
        print(f"  ── Agent A (loading from {args.load_agent_a}) ──")
        trainer_a.agent.load(args.load_agent_a)
        log_a = []
        print(f"  [Agent A] Loaded from checkpoint. Training skipped.")
    else:
        print("  ── Agent A ──")
        log_a = trainer_a.run(num_rounds=args.rounds)

    print("\n  ── Agent B ──")
    log_b = trainer_b.run(num_rounds=args.rounds)

    # ── Stage 3: 评估 ───────────────────────────────────────────────────────
    print("\n[Stage 3] Evaluation...")
    h2h_deals = 200 if args.quick else 1000

    # ── Build SL baseline agent for evaluation ─────────────────────────────
    from algorithms.mappo import MAPPOAgent, MAPPOConfig
    sl_agent_eval = MAPPOAgent(MAPPOConfig(device=device))
    if sl_path and os.path.exists(sl_path):
        ckpt_sl = torch.load(sl_path, map_location=device)
        for player, key in [(_N, 'actor_n'), (_E, 'actor_e'),
                            (_S, 'actor_s'), (_W, 'actor_w')]:
            if key in ckpt_sl:
                sl_agent_eval.get_actor(player).load_state_dict(
                    {k: v.to(device) for k, v in ckpt_sl[key].items()})
    # Wrap SL agent in a temporary trainer for H2H API
    sl_trainer = SubgameTrainer(env, cfg_a, reward_stats=RunningStats())
    sl_trainer.agent = sl_agent_eval

    # ── A vs SL ────────────────────────────────────────────────────────────
    print("\n  → Agent A vs SL baseline")
    h2h_a_sl = trainer_a.evaluate_head_to_head(
        sl_trainer, num_deals=h2h_deals, label_self="A", label_other="SL")

    # ── B vs SL ────────────────────────────────────────────────────────────
    print("\n  → Agent B vs SL baseline")
    h2h_b_sl = trainer_b.evaluate_head_to_head(
        sl_trainer, num_deals=h2h_deals, label_self="B", label_other="SL")

    # ── A vs B ─────────────────────────────────────────────────────────────
    print("\n  → Agent A vs Agent B")
    h2h = trainer_a.evaluate_head_to_head(
        trainer_b, num_deals=h2h_deals, label_self="A", label_other="B")

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
    print(f"  A vs SL:  {h2h_a_sl['mean_imp']:+.3f} ± {h2h_a_sl['std_imp']:.3f} IMP  "
          f"p={h2h_a_sl['p_value']:.3f} {'✅' if h2h_a_sl['significant'] else '(ns)'}")
    print(f"  B vs SL:  {h2h_b_sl['mean_imp']:+.3f} ± {h2h_b_sl['std_imp']:.3f} IMP  "
          f"p={h2h_b_sl['p_value']:.3f} {'✅' if h2h_b_sl['significant'] else '(ns)'}")
    print(f"  A vs B:   {h2h['mean_imp']:+.3f} ± {h2h['std_imp']:.3f} IMP  "
          f"p={h2h['p_value']:.3f} {'✅' if h2h['significant'] else '(ns)'}")
    # Conclusion
    if h2h['mean_imp'] < 0 and h2h['significant']:
        conclusion = "✅ B significantly better than A"
    elif h2h['mean_imp'] > 0 and h2h['significant']:
        conclusion = "⚠️  A outperforms B"
    else:
        conclusion = "❌ No significant difference"
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
            'a_vs_sl': h2h_a_sl,
            'b_vs_sl': h2h_b_sl,
            'a_vs_b': h2h,
        }
        report_path = os.path.join(args.save_dir, f'report_seed{args.seed}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n  Checkpoints + report saved → {args.save_dir}")


def _print_diagnostics(log_a: list, log_b: list, cfg_b: SubgameConfig):
    """排雷判断：检查训练健康性指标."""

    def _last_metric(log: list, player_key: int, metric: str) -> float:
        for entry in reversed(log):
            for key in ('ns_metrics', 'ew_metrics', 's_metrics', 'n_metrics'):
                m = entry.get(key, {}).get(player_key, {})
                if metric in m:
                    return m[metric]
        return float('nan')

    from env import SOUTH, EAST

    # Agent A（可能从checkpoint加载，无训练日志）
    if log_a:
        ent_a   = _last_metric(log_a, SOUTH, 'entropy')
        vl_a    = _last_metric(log_a, SOUTH, 'value_loss')
        ent_a_e = _last_metric(log_a, EAST, 'entropy')
        vl_a_e  = _last_metric(log_a, EAST, 'value_loss')
        print(f"  Agent A: NS entropy={ent_a:.3f} vl={vl_a:.4f} │ EW entropy={ent_a_e:.3f} vl={vl_a_e:.4f}")
    else:
        print(f"  Agent A: (loaded from checkpoint, no training log)")

    ent_b   = _last_metric(log_b, SOUTH, 'entropy')
    vl_b    = _last_metric(log_b, SOUTH, 'value_loss')
    kl_b    = _last_metric(log_b, SOUTH, 'kl_loss')
    klam_b  = _last_metric(log_b, SOUTH, 'kl_lambda')
    ent_b_e = _last_metric(log_b, EAST, 'entropy')
    vl_b_e  = _last_metric(log_b, EAST, 'value_loss')
    print(f"  Agent B: NS entropy={ent_b:.3f} vl={vl_b:.4f} kl={kl_b:.5f}(λ={klam_b:.3f}) │ EW entropy={ent_b_e:.3f} vl={vl_b_e:.4f}")

    ok = True
    if ent_b < 0.5:
        print("  ⚠️  Entropy collapse detected! Consider higher entropy_coef.")
        ok = False
    if vl_b > 10:
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
                        help='Internal β: r_info = I(partner) - β·I(opponent)')
    parser.add_argument('--info_weight', type=float, default=0.2,
                        help='r_info as fraction of IMP variance (P87b: 0.02→0.2, step_ir≈1.4)')
    parser.add_argument('--rounds', type=int, default=30,
                        help='Number of IBR rounds (P90: 20→30)')
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
    parser.add_argument('--load_agent_a', default=None,
                        help='Path to pre-trained Agent A checkpoint (.pt). '
                        'If set, skip A training and load directly.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.type == 'competitive':
        run_competitive(args)
    else:
        raise ValueError(f"Unknown subgame type: {args.type}")
