#!/usr/bin/env python3
"""
Subgame Validation (Phase 2)
==============================

三阶段 Stayman 实验:
  Stage 1: N 规则策略, 训练 S_base (公共基线)
  Stage 1.5: Belief Network 预训练 (用 Stage 1 rollout 数据)
  Stage 2: 加载 S_base, 解冻 N, 分支 A (MAPPO) vs B (MAPPO+r_info)
  Stage 3: 评估 + 定性分析 (N 策略偏移) + 训练诊断

Usage:
    cd bridge-coma/
    python -m utils.generate_subgame_data --type both --num_workers 4
    python experiments/subgame_validation.py \\
        --stayman_data data/stayman_50k.npz \\
        --competitive_data data/competitive_100k.npz \\
        --device cuda
"""

import argparse
import json
import sys
import time
import copy
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter
from typing import Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from subgames.stayman_env import StaymanSubgameEnv
from subgames.competitive_env import (
    CompetitiveSubgameEnv, cross_evaluate, make_agent_policy,
)
from subgames.subgame_trainer import SubgameTrainer, SubgameConfig
from subgames.action_mask import count_suit_length, count_hcp
from subgames.stayman_env import create_bc_dataset_for_stayman
from algorithms.behavioral_cloning import (
    BCDataset, behavioral_cloning_warmup,
    create_bc_dataset_for_competitive, evaluate_pass_rate,
)
from env import BridgeBiddingEnv, NORTH, SOUTH, string_to_bid, bid_to_string


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Phase2Config:
    """Phase 2 全局配置."""
    stayman_data: str = "data/stayman_50k.npz"
    competitive_data: str = "data/competitive_100k.npz"
    device: str = "cpu"
    output_dir: str = "results/"

    # Stage 1: BC warmup + optional RL fine-tune
    stage1_steps: int = 0             # 0 = 纯 BC (推荐), >0 = BC 后 RL 微调
    stage1_deals_per_step: int = 32
    stage1_accumulate: int = 4

    # Stage 1.5: Belief pre-training
    belief_pretrain_deals: int = 2000
    belief_pretrain_epochs: int = 20
    belief_pretrain_target_acc: float = 0.80

    # Stage 2: Alternating fine-tune (single_step, batch-mean baseline)
    # S-N-S-N... 交替训练 + 联合微调收尾
    stage2_alt_rounds: int = 4         # 交替轮数 (每轮 = 1×S + 1×N)
    stage2_alt_steps: int = 200        # 每个半轮的步数
    stage2_joint_steps: int = 400      # 最终联合微调步数
    stage2_deals_per_step: int = 32
    stage2_accumulate: int = 8         # 256 deals/update
    stage2_lr: float = 3e-5            # 交替阶段用
    stage2_lr_joint: float = 1e-5      # 联合微调用
    stage2_entropy_start: float = 0.05 # 防 3-action 坍缩
    stage2_entropy_end: float = 0.02
    stage2_entropy_anneal: float = 0.8 # 延后退火

    # Competitive
    competitive_steps: int = 5000
    competitive_deals_per_step: int = 32
    competitive_eval_deals: int = 500

    # BC (Stayman)
    stayman_bc_samples: int = 10000
    stayman_bc_epochs: int = 15

    # BC (Competitive)
    bc_num_samples: int = 30000
    bc_epochs: int = 10

    # Eval
    eval_deals: int = 200

    # Diagnostics
    diag_deals: int = 500

    # Go/No-Go
    go_info_ratio: float = 1.0
    go_belief_acc: float = 0.7


# ============================================================================
# Stage 1: BC warmup + RL fine-tune (N+S jointly)
# ============================================================================

def run_stage1(config: Phase2Config) -> dict:
    """
    Stage 1: 联合训练 N+S base agent.

    流程:
      1. BC warmup: 用 north/south_stayman_rule 生成 N+S 联合监督数据
      2. RL fine-tune: N+S 联合 PPO 微调 (north_rule=False)

    BC 保证 N 和 S 都从合理策略出发 (避免随机初始化导致的乱叫),
    RL 允许 N+S 在规则覆盖不到的边界情况上联合优化.
    """
    print("=" * 60)
    print("Stage 1: Train N+S base (BC warmup + RL fine-tune)")
    print("=" * 60)

    env = StaymanSubgameEnv(config.stayman_data, north_rule=False)

    sub_config = SubgameConfig(
        num_steps=config.stage1_steps,
        deals_per_step=config.stage1_deals_per_step,
        accumulate_steps=config.stage1_accumulate,
        use_info_bonus=False,
        lr=1e-4,
        device=config.device,
        active_players=[NORTH, SOUTH],  # Both learn
    )
    trainer = SubgameTrainer(env, sub_config)

    # --- BC Warmup (N+S) ---
    print("\n--- BC Warmup (N+S rules) ---")
    bc_data = create_bc_dataset_for_stayman(
        config.stayman_data,
        num_samples=config.stayman_bc_samples,
        players='both',
    )
    bc_dataset = BCDataset(bc_data)
    bc_stats = behavioral_cloning_warmup(
        trainer.agent,
        bc_dataset,
        epochs=config.stayman_bc_epochs,
        lr=1e-3,
        batch_size=256,
    )
    print(f"  BC result: loss={bc_stats['final_loss']:.4f}, "
          f"acc={bc_stats['final_acc']:.3f}")

    # BC 后诊断
    print("\n--- Post-BC Diagnostics ---")
    post_bc_diag = _run_diagnostics(env, trainer, config.diag_deals)

    # --- RL Fine-tune (optional) ---
    log = []
    train_time = 0.0
    if config.stage1_steps > 0:
        print("\n--- RL Fine-tune ---")
        t0 = time.time()
        log = trainer.train()
        train_time = time.time() - t0
    else:
        print("\n--- Skipping RL fine-tune (stage1_steps=0, pure BC) ---")

    # Evaluate
    eval_results = _evaluate_stayman_full(env, trainer, config.eval_deals)

    results = {
        'mean_imp': eval_results['mean_imp'],
        'std_imp': eval_results['std_imp'],
        'train_time_sec': train_time,
        'bc_stats': bc_stats,
        'final_log': log[-1] if log else {},
    }

    print(f"\nStage 1 result: IMP={results['mean_imp']:+.2f}±{results['std_imp']:.2f} "
          f"(time={train_time:.0f}s)")

    # Diagnostics
    print("\n--- Stage 1 Diagnostics ---")
    diag = _run_diagnostics(env, trainer, config.diag_deals)
    results['diagnostics'] = diag

    return {'results': results, 'trainer': trainer, 'log': log}


# ============================================================================
# Stage 1.5: Belief Network Pre-training
# ============================================================================

def run_belief_pretrain(config: Phase2Config, base_trainer: SubgameTrainer) -> dict:
    """
    Stage 1.5: 用 Stage 1 的 rollout 数据预训练 Belief Network.

    科学依据: Belief net (supervised) 与 policy (RL) 是不同优化目标.
    混合训练时未收敛的 belief 会通过 r_info 向 policy 注入噪声梯度.
    分阶段预训练是 CTDE 文献中的标准做法 (cf. LICA, Ding et al. 2020).
    """
    print("\n" + "=" * 60)
    print("Stage 1.5: Belief Network Pre-training")
    print("=" * 60)

    env = StaymanSubgameEnv(config.stayman_data, north_rule=False)

    # 收集大量 rollout 数据 (用 base agent 的策略)
    print(f"  Collecting {config.belief_pretrain_deals} rollout episodes...")
    episodes = base_trainer.collect_episodes(config.belief_pretrain_deals)

    # 创建一个临时 trainer 来训练 belief (带 info bonus 配置)
    sub_config = SubgameConfig(
        num_steps=0,  # 不做 RL 训练
        use_info_bonus=True,
        beta=0.0,
        belief_lr=1e-3,
        device=config.device,
        active_players=[SOUTH],
    )
    belief_trainer = SubgameTrainer(env, sub_config)

    # 多轮训练 belief network
    best_acc = 0.0
    final_epoch = 0
    for epoch in range(1, config.belief_pretrain_epochs + 1):
        loss = belief_trainer.train_belief_step(episodes)
        acc = belief_trainer.evaluate_belief_accuracy(num_deals=100)
        final_epoch = epoch

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [Epoch {epoch}/{config.belief_pretrain_epochs}] "
                  f"belief_loss={loss:.4f}, belief_acc={acc:.3f}")

        best_acc = max(best_acc, acc)
        if acc >= config.belief_pretrain_target_acc:
            print(f"  ✓ Target accuracy {config.belief_pretrain_target_acc:.2f} "
                  f"reached at epoch {epoch}")
            break

    results = {
        'final_acc': float(best_acc),
        'epochs_trained': final_epoch,
        'target_reached': best_acc >= config.belief_pretrain_target_acc,
    }

    print(f"\nBelief pre-train: acc={best_acc:.3f} "
          f"({'✓ reached' if results['target_reached'] else '✗ not reached'} "
          f"target={config.belief_pretrain_target_acc:.2f})")

    return {
        'results': results,
        'belief_state': copy.deepcopy(belief_trainer.belief_net.state_dict()),
    }


# ============================================================================
# Stage 2: Alternating fine-tuning (A vs B)
# ============================================================================

def _make_sub_config(config: Phase2Config, num_steps: int, lr: float,
                     active_players: list, use_info: bool = False,
                     beta: float = 0.0, belief_warmup: int = 0,
                     entropy_start: float = None,
                     entropy_end: float = None) -> SubgameConfig:
    """构建单阶段 SubgameConfig 的 helper."""
    return SubgameConfig(
        num_steps=num_steps,
        deals_per_step=config.stage2_deals_per_step,
        accumulate_steps=config.stage2_accumulate,
        use_info_bonus=use_info,
        beta=beta,
        lr=lr,
        device=config.device,
        active_players=active_players,
        single_step=True,
        belief_warmup_steps=belief_warmup,
        entropy_coef_start=entropy_start or config.stage2_entropy_start,
        entropy_coef_end=entropy_end or config.stage2_entropy_end,
        entropy_anneal_frac=config.stage2_entropy_anneal,
        eval_interval=100,
        log_interval=20,
    )


def _run_one_phase(config: Phase2Config, env, sub_config: SubgameConfig,
                   prev_state: dict, belief_state: dict = None,
                   phase_label: str = "") -> Tuple:
    """
    运行一个训练阶段, 返回 (trainer, log, eval_results).

    加载 prev_state 权重, 可选加载 belief_state.
    """
    trainer = SubgameTrainer(env, sub_config)
    trainer.agent.model.load_state_dict(prev_state)

    if sub_config.use_info_bonus and belief_state and trainer.belief_net is not None:
        trainer.belief_net.load_state_dict(belief_state)

    log = trainer.train()

    # 评估 (用 north_rule=False env 以测试 N+S agent 策略)
    eval_env = StaymanSubgameEnv(config.stayman_data, north_rule=False)
    eval_results = _evaluate_stayman_full(eval_env, trainer, config.eval_deals)

    if phase_label:
        print(f"    {phase_label}: IMP={eval_results['mean_imp']:+.2f}"
              f"±{eval_results['std_imp']:.2f}")

    return trainer, log, eval_results


def run_stage2(config: Phase2Config, base_trainer: SubgameTrainer,
               belief_pretrain: dict = None) -> dict:
    """
    Stage 2: Alternating fine-tuning (Iterated Best Response).

    交替训练解决 N↔S 同时变化的 credit assignment 问题:

      Round 1: S learns (N=BC frozen) → N learns (S=round1 frozen)
      Round 2: S learns (N=round1)    → N learns (S=round2)
      ...
      Round K: S learns → N learns
      Final:   N+S joint fine-tune (low lr)

    每个半轮, 非 active player 由 agent 以 deterministic 模式执行
    (SubgameTrainer.collect_episodes 中 non-active player 路径).
    第一轮 S 训练时 N 用 north_rule=True (规则策略),
    后续轮次全部用 north_rule=False (agent 策略).

    A (control) 和 B (r_info) 各自独立跑完整流程.
    """
    print("\n" + "=" * 60)
    print("Stage 2: Alternating Fine-tuning")
    print(f"  {config.stage2_alt_rounds} rounds × {config.stage2_alt_steps} steps"
          f" + {config.stage2_joint_steps} joint")
    print("  A=MAPPO (control) vs B=MAPPO+r_info")
    print("=" * 60)

    base_state = copy.deepcopy(base_trainer.agent.model.state_dict())
    belief_state = belief_pretrain['belief_state'] if belief_pretrain else None

    results = {}

    for name, use_info, beta in [
        ("A_control", False, 0.0),
        ("B_partner_only", True, 0.0),
    ]:
        print(f"\n{'─'*60}")
        print(f"  Agent: {name}")
        print(f"{'─'*60}")

        current_state = copy.deepcopy(base_state)
        current_belief = copy.deepcopy(belief_state) if belief_state else None
        round_imps = []

        # ================================================================
        # 交替轮次: S → N → S → N → ...
        # ================================================================
        for rnd in range(1, config.stage2_alt_rounds + 1):
            print(f"\n  ── Round {rnd}/{config.stage2_alt_rounds} ──")

            # --- S phase: 训 S, 冻结 N ---
            if rnd == 1:
                # 第一轮: N 用规则策略 (最稳定的起点)
                env_s = StaymanSubgameEnv(config.stayman_data, north_rule=True)
            else:
                # 后续轮: N 用 agent 策略 (上轮训好的)
                env_s = StaymanSubgameEnv(config.stayman_data, north_rule=False)

            cfg_s = _make_sub_config(
                config, config.stage2_alt_steps, config.stage2_lr,
                active_players=[SOUTH],
                use_info=False,  # S 阶段不用 info bonus
            )

            trainer_s, _, eval_s = _run_one_phase(
                config, env_s, cfg_s, current_state,
                phase_label=f"R{rnd} S-phase",
            )
            current_state = copy.deepcopy(trainer_s.agent.model.state_dict())

            # --- N phase: 训 N, 冻结 S ---
            env_n = StaymanSubgameEnv(config.stayman_data, north_rule=False)

            cfg_n = _make_sub_config(
                config, config.stage2_alt_steps, config.stage2_lr,
                active_players=[NORTH],
                use_info=use_info,  # B 在 N-phase 用 info bonus
                beta=beta,
                belief_warmup=30 if (use_info and rnd == 1) else 0,
            )

            trainer_n, log_n, eval_n = _run_one_phase(
                config, env_n, cfg_n, current_state,
                belief_state=current_belief,
                phase_label=f"R{rnd} N-phase",
            )
            current_state = copy.deepcopy(trainer_n.agent.model.state_dict())

            # 更新 belief state (如果有)
            if use_info and trainer_n.belief_net is not None:
                current_belief = copy.deepcopy(trainer_n.belief_net.state_dict())

            round_imps.append({
                'round': rnd,
                'S_phase': eval_s['mean_imp'],
                'N_phase': eval_n['mean_imp'],
            })

        # ================================================================
        # 联合微调收尾
        # ================================================================
        print(f"\n  ── Joint fine-tune ({config.stage2_joint_steps} steps) ──")

        env_j = StaymanSubgameEnv(config.stayman_data, north_rule=False)

        cfg_j = _make_sub_config(
            config, config.stage2_joint_steps, config.stage2_lr_joint,
            active_players=[NORTH, SOUTH],
            use_info=use_info,
            beta=beta,
            # 联合阶段: 低 entropy, 不退火
            entropy_start=config.stage2_entropy_end,
            entropy_end=config.stage2_entropy_end,
        )

        trainer_j, log_j, eval_j = _run_one_phase(
            config, env_j, cfg_j, current_state,
            belief_state=current_belief,
            phase_label="Joint",
        )

        # ================================================================
        # 结果汇总
        # ================================================================
        belief_acc = trainer_j.evaluate_belief_accuracy(
            num_deals=config.eval_deals
        ) if use_info else 0.0

        info_metrics = {}
        if use_info and log_n:
            last = log_n[-1]
            info_metrics = {
                'info_ratio': last.get('info_ratio', 0),
                'partner_gain': last.get('partner_gain', 0),
                'opponent_leak': last.get('opponent_leak', 0),
            }

        results[name] = {
            'mean_imp': eval_j['mean_imp'],
            'std_imp': eval_j['std_imp'],
            'belief_accuracy': belief_acc,
            'round_imps': round_imps,
            'final_log': log_j[-1] if log_j else {},
            **info_metrics,
        }

        print(f"\n  {name} progression:")
        print(f"    BC base:  IMP ≈ -4.1")
        for ri in round_imps:
            print(f"    R{ri['round']}: S→{ri['S_phase']:+.2f}  N→{ri['N_phase']:+.2f}")
        print(f"    Joint:    IMP = {eval_j['mean_imp']:+.2f}")

        # Diagnostics
        print(f"\n--- {name} Final Diagnostics ---")
        diag = _run_diagnostics(env_j, trainer_j, config.diag_deals)
        results[name]['diagnostics'] = diag

        results[name]['_trainer'] = trainer_j

    return results


# ============================================================================
# Stage 3: Evaluation & Analysis
# ============================================================================

def run_stage3(config: Phase2Config, stage1_results: dict,
               stage2_results: dict) -> dict:
    """
    Stage 3: 核心指标评估 + 定性分析.

    定量: IMP 对比
    定性: N 的策略偏移 (Agent B 的 N 是否偏离标准规则?)
    """
    print("\n" + "=" * 60)
    print("Stage 3: Evaluation & Analysis")
    print("=" * 60)

    analysis = {}

    # 定量对比
    s_base_imp = stage1_results['results']['mean_imp']
    a_imp = stage2_results['A_control']['mean_imp']
    b_imp = stage2_results['B_partner_only']['mean_imp']

    analysis['quantitative'] = {
        's_base_imp': s_base_imp,
        'a_control_imp': a_imp,
        'b_partner_only_imp': b_imp,
        'a_vs_sbase': a_imp - s_base_imp,
        'b_vs_sbase': b_imp - s_base_imp,
        'b_vs_a': b_imp - a_imp,
    }

    print(f"\n  S_base (N=rule):  IMP = {s_base_imp:+.2f}")
    print(f"  A (MAPPO):        IMP = {a_imp:+.2f} (Δ vs S_base: {a_imp - s_base_imp:+.2f})")
    print(f"  B (MAPPO+r_info): IMP = {b_imp:+.2f} (Δ vs S_base: {b_imp - s_base_imp:+.2f})")
    print(f"  B vs A:           Δ = {b_imp - a_imp:+.2f}")

    # 定性分析: N 的策略偏移
    print("\n--- N's Policy Shift Analysis ---")
    for name in ['A_control', 'B_partner_only']:
        trainer = stage2_results[name].get('_trainer')
        if trainer is None:
            continue

        env = StaymanSubgameEnv(config.stayman_data, north_rule=False)
        policy = _analyze_north_policy(env, trainer, num_deals=500)
        analysis[f'{name}_north_policy'] = policy

        print(f"\n  {name} — N's bidding distribution:")
        for situation, dist in policy.items():
            print(f"    {situation}:")
            for bid, pct in sorted(dist.items(), key=lambda x: -x[1]):
                bar = "█" * int(pct * 40)
                print(f"      {bid:4s}: {pct:5.1%} {bar}")

    # Go/No-Go
    go = _stayman_go_no_go(stage1_results, stage2_results, config)
    analysis['go_no_go'] = go

    return analysis


def _analyze_north_policy(env, trainer, num_deals: int = 500) -> dict:
    """
    分析 N 的策略分布.

    对每种手牌类型 (有4H / 有4S / 都没有), 统计 N 叫什么.
    """
    agent = trainer.agent
    counts = {
        'has_4H': {},   # N 有 4+H 时的叫品分布
        'has_4S': {},   # N 有 4+S (无 4H) 时
        'no_4M': {},    # 都没有时
    }

    for _ in range(num_deals):
        hands, dd_table = env.generate_deal()
        obs = env.env.reset(hands, dealer=NORTH, vulnerability=(False, False))

        # 执行固定前缀
        for bid_str_prefix in env.FIXED_PREFIX:
            bid = string_to_bid(bid_str_prefix)
            obs, _, done, _ = env.env.step(bid)

        if done:
            continue

        # N's turn: get agent's action
        obs['legal_actions'] = env._get_stayman_mask()
        obs_t = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(agent.device)
                 for k, v in obs.items()}
        with torch.no_grad():
            all_h = torch.tensor(hands, dtype=torch.float32).unsqueeze(0).to(agent.device)
            action, _, _, _ = agent.model.get_action_and_value(obs_t, all_h, deterministic=True)
            action = action.item()

        bid_str = bid_to_string(action)

        # Categorize N's hand
        n_hand = hands[NORTH]
        h = count_suit_length(n_hand, 2)  # hearts
        s = count_suit_length(n_hand, 3)  # spades

        if h >= 4:
            key = 'has_4H'
        elif s >= 4:
            key = 'has_4S'
        else:
            key = 'no_4M'

        counts[key][bid_str] = counts[key].get(bid_str, 0) + 1

    # Convert to percentages
    policy = {}
    for situation, bids in counts.items():
        total = sum(bids.values())
        if total > 0:
            policy[situation] = {bid: n / total for bid, n in bids.items()}
        else:
            policy[situation] = {}

    return policy


def _stayman_go_no_go(stage1_results, stage2_results, config) -> dict:
    """Go/No-Go 判定."""
    b_result = stage2_results.get('B_partner_only', {})
    a_result = stage2_results.get('A_control', {})
    s_base = stage1_results['results']

    b_imp = b_result.get('mean_imp', -999)
    a_imp = a_result.get('mean_imp', -999)
    s_base_imp = s_base.get('mean_imp', -999)
    info_ratio = b_result.get('info_ratio', 0)
    belief_acc = b_result.get('belief_accuracy', 0)

    checks = {
        'sbase_converged': s_base_imp > -5.0,
        'b_better_than_a': b_imp > a_imp,
        'b_better_than_sbase': b_imp > s_base_imp,
        'info_ratio': info_ratio,
        'belief_acc': belief_acc,
        'imp_values': {
            's_base': s_base_imp,
            'a_control': a_imp,
            'b_partner_only': b_imp,
        },
    }

    if checks['sbase_converged'] and checks['b_better_than_a']:
        checks['decision'] = 'GO'
        print(f"\n✅ Stayman Go/No-Go: GO "
              f"(B={b_imp:+.2f} > A={a_imp:+.2f}, S_base={s_base_imp:+.2f})")
    elif checks['sbase_converged']:
        checks['decision'] = 'MARGINAL'
        print(f"\n⚠️  Stayman: S_base converged but B not better than A "
              f"(B={b_imp:+.2f}, A={a_imp:+.2f})")
    else:
        checks['decision'] = 'NO_GO'
        print(f"\n❌ Stayman: S_base didn't converge (IMP={s_base_imp:+.2f})")

    return checks


# ============================================================================
# Diagnostics: Contract distribution + Fit detection + Reward stats
# ============================================================================

def _classify_contract(contract) -> str:
    """将定约分类为诊断类别."""
    if contract is None:
        return 'passed_out'
    level = contract.level
    suit = contract.suit
    if level >= 7:
        return 'grand_slam'
    elif level >= 6:
        return 'small_slam'
    elif level >= 4 and suit in (2, 3):  # 4H, 4S (and 5H/5S)
        return '4M'
    elif level >= 5 and suit in (0, 1):  # 5C, 5D
        return '5m'
    elif suit == 4 and level == 3:       # 3NT
        return '3NT'
    elif (suit == 4 and level >= 4) or (suit <= 1 and level >= 3 and level < 5):
        # 4NT+, 3C/3D/4C/4D — 其他成局级别
        return 'other_game'
    else:
        return 'part_score'


def _has_major_fit(hands, declarer_side_players=(NORTH, SOUTH)) -> dict:
    """检测 N-S 是否有 4-4 高花配合."""
    p1, p2 = declarer_side_players
    h1 = count_suit_length(hands[p1], 2)  # hearts
    h2 = count_suit_length(hands[p2], 2)
    s1 = count_suit_length(hands[p1], 3)  # spades
    s2 = count_suit_length(hands[p2], 3)
    return {
        'heart_fit': h1 >= 4 and h2 >= 4,
        'spade_fit': s1 >= 4 and s2 >= 4,
        'any_fit': (h1 >= 4 and h2 >= 4) or (s1 >= 4 and s2 >= 4),
    }


def _run_diagnostics(env, trainer, num_deals: int) -> dict:
    """
    全面训练诊断:
      1. Contract Distribution (定约等级分布)
      2. N-S Fit Detection (4-4 高花配合检测率)
      3. Reward / IMP Distribution (奖励分布统计)
    """
    agent = trainer.agent

    contract_counts = Counter()
    imp_values = []
    reward_values = []

    # Fit detection tracking
    fit_total = 0           # 有 4-4 fit 的牌副数
    fit_bid_4M = 0          # 有 fit 且叫到 4M 的次数
    fit_bid_3NT = 0         # 有 fit 但叫到 3NT 的次数
    fit_bid_other = 0       # 有 fit 但叫到其他的次数

    for _ in range(num_deals):
        hands, dd_table = env.generate_deal()
        obs = env.reset(hands, dd_table)
        done = False

        while not done:
            obs_t = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(agent.device)
                     for k, v in obs.items()}
            with torch.no_grad():
                all_hands = env._current_hands
                all_h_t = torch.tensor(all_hands, dtype=torch.float32).unsqueeze(0).to(agent.device)
                action, _, _, _ = agent.model.get_action_and_value(
                    obs_t, all_h_t, deterministic=True
                )
                action = action.item()

            obs, reward, done, info = env.step(action)

        # Record IMP and reward
        imp_val = info.get('imp', 0)
        imp_values.append(imp_val)
        reward_values.append(reward)

        # Classify contract
        contract = env.env.state.final_contract
        category = _classify_contract(contract)
        contract_counts[category] += 1

        # Fit detection
        fit_info = _has_major_fit(hands)
        if fit_info['any_fit']:
            fit_total += 1
            if contract is not None:
                if contract.suit in (2, 3) and contract.level >= 4:
                    fit_bid_4M += 1
                elif contract.suit == 4 and contract.level == 3:
                    fit_bid_3NT += 1
                else:
                    fit_bid_other += 1
            else:
                fit_bid_other += 1

    # === Print diagnostics ===
    total = sum(contract_counts.values())
    print(f"\n  Contract Distribution ({total} deals):")
    display_order = ['passed_out', 'part_score', '3NT', '4M', '5m',
                     'other_game', 'small_slam', 'grand_slam']
    for cat in display_order:
        n = contract_counts.get(cat, 0)
        pct = n / max(1, total)
        bar = "█" * int(pct * 40)
        print(f"    {cat:14s}: {n:4d} ({pct:5.1%}) {bar}")

    print(f"\n  N-S Fit Detection:")
    if fit_total > 0:
        print(f"    Deals with 4-4 major fit: {fit_total}/{total} ({fit_total/total:.1%})")
        print(f"    Correctly bid 4M when fit:  {fit_bid_4M}/{fit_total} ({fit_bid_4M/fit_total:.1%})")
        print(f"    Bid 3NT instead when fit:   {fit_bid_3NT}/{fit_total} ({fit_bid_3NT/fit_total:.1%})")
        print(f"    Other when fit:             {fit_bid_other}/{fit_total} ({fit_bid_other/fit_total:.1%})")
    else:
        print(f"    No 4-4 major fit found in {total} deals")

    imp_arr = np.array(imp_values)
    rw_arr = np.array(reward_values)
    print(f"\n  Reward Distribution:")
    print(f"    reward: mean={rw_arr.mean():+.3f}, std={rw_arr.std():.3f}, "
          f"min={rw_arr.min():+.3f}, max={rw_arr.max():+.3f}")
    print(f"    IMP:    mean={imp_arr.mean():+.2f}, median={np.median(imp_arr):+.1f}, "
          f"[p5={np.percentile(imp_arr,5):+.0f}, p95={np.percentile(imp_arr,95):+.0f}]")

    # Build result dict
    diag = {
        'contract_distribution': {k: contract_counts.get(k, 0) / max(1, total)
                                  for k in display_order},
        'fit_detection': {
            'total_with_fit': fit_total,
            'correctly_bid_4M': fit_bid_4M,
            'bid_3NT_instead': fit_bid_3NT,
            'bid_other': fit_bid_other,
            'fit_4M_rate': fit_bid_4M / max(1, fit_total),
            'fit_3NT_rate': fit_bid_3NT / max(1, fit_total),
        },
        'reward_stats': {
            'mean': float(rw_arr.mean()),
            'std': float(rw_arr.std()),
            'min': float(rw_arr.min()),
            'max': float(rw_arr.max()),
        },
        'imp_stats': {
            'mean': float(imp_arr.mean()),
            'std': float(imp_arr.std()),
            'median': float(np.median(imp_arr)),
            'p5': float(np.percentile(imp_arr, 5)),
            'p95': float(np.percentile(imp_arr, 95)),
        },
    }
    return diag


# ============================================================================
# Evaluation helper
# ============================================================================

def _evaluate_stayman_full(env, trainer, num_deals: int) -> dict:
    """在 Stayman 子博弈中评估, 返回完整统计."""
    agent = trainer.agent
    imps = []

    for _ in range(num_deals):
        hands, dd_table = env.generate_deal()
        obs = env.reset(hands, dd_table)
        done = False

        while not done:
            obs_t = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(agent.device)
                     for k, v in obs.items()}
            with torch.no_grad():
                all_hands = env._current_hands
                all_h_t = torch.tensor(all_hands, dtype=torch.float32).unsqueeze(0).to(agent.device)
                action, _, _, _ = agent.model.get_action_and_value(
                    obs_t, all_h_t, deterministic=True
                )
                action = action.item()

            obs, reward, done, info = env.step(action)

        imps.append(info.get('imp', 0))

    imp_arr = np.array(imps)
    return {
        'mean_imp': float(imp_arr.mean()),
        'std_imp': float(imp_arr.std()),
        'imps': imps,
    }


# ============================================================================
# Main pipeline
# ============================================================================

def run_phase2(config: Phase2Config) -> dict:
    """完整 Phase 2 流程."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'config': asdict(config),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Stage 1: S_base
    stage1 = run_stage1(config)
    report['stage1'] = stage1['results']

    # Save S_base checkpoint
    _save_checkpoint(stage1['trainer'], output_dir / "s_base.pt", "S_base")

    # Stage 1.5: Belief pre-training
    belief_pretrain = run_belief_pretrain(config, stage1['trainer'])
    report['belief_pretrain'] = belief_pretrain['results']

    # Stage 2: A vs B (with pre-trained belief for B)
    stage2 = run_stage2(config, stage1['trainer'], belief_pretrain)
    report['stage2'] = {
        name: {k: v for k, v in data.items() if k != '_trainer'}
        for name, data in stage2.items()
    }

    # Save Stage 2 checkpoints
    for name in ['A_control', 'B_partner_only']:
        trainer = stage2[name].get('_trainer')
        if trainer:
            _save_checkpoint(trainer, output_dir / f"{name}.pt", name)

    # Stage 3: Analysis
    stage3 = run_stage3(config, stage1, stage2)
    report['stage3'] = {
        k: v for k, v in stage3.items()
        if not k.startswith('_')
    }

    # Save report
    report_path = output_dir / "phase2_report.json"
    _save_json(report, report_path)
    print(f"\nReport saved to {report_path}")

    return report


def _save_checkpoint(trainer, path: Path, name: str):
    """保存模型权重 (policy + belief)."""
    ckpt = {
        'model': trainer.agent.model.state_dict(),
    }
    if trainer.belief_net is not None:
        ckpt['belief_net'] = trainer.belief_net.state_dict()
    torch.save(ckpt, path)
    print(f"  💾 Saved {name} checkpoint → {path}")


def _save_json(data: dict, path: Path):
    def default(o):
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.tolist()
        return str(o)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=default)


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Phase 2: Subgame Validation")
    p.add_argument('--stayman_data', default='data/stayman_50k.npz')
    p.add_argument('--competitive_data', default='data/competitive_100k.npz')
    p.add_argument('--device', default=None)
    p.add_argument('--output_dir', default='results/')
    p.add_argument('--stage1_steps', type=int, default=0)
    p.add_argument('--alt_rounds', type=int, default=4)
    p.add_argument('--alt_steps', type=int, default=200)
    p.add_argument('--joint_steps', type=int, default=400)
    p.add_argument('--eval_deals', type=int, default=200)
    p.add_argument('--diag_deals', type=int, default=500)
    # Quick mode for testing
    p.add_argument('--quick', action='store_true',
                   help='Quick test run (fewer steps)')
    args = p.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    config = Phase2Config(
        stayman_data=args.stayman_data,
        competitive_data=args.competitive_data,
        device=device,
        output_dir=args.output_dir,
        stage1_steps=args.stage1_steps,
        stage2_alt_rounds=args.alt_rounds,
        stage2_alt_steps=args.alt_steps,
        stage2_joint_steps=args.joint_steps,
        eval_deals=args.eval_deals,
        diag_deals=args.diag_deals,
    )

    if args.quick:
        config.stage1_steps = 0
        config.stage2_alt_rounds = 2
        config.stage2_alt_steps = 50
        config.stage2_joint_steps = 50
        config.stage2_accumulate = 4
        config.eval_deals = 50
        config.diag_deals = 100
        config.stayman_bc_samples = 1000
        config.stayman_bc_epochs = 3
        config.belief_pretrain_deals = 200
        config.belief_pretrain_epochs = 5
        print("⚡ Quick mode: reduced steps for testing")

    run_phase2(config)


if __name__ == "__main__":
    main()
