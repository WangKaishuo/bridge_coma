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

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from subgames.stayman_env import StaymanSubgameEnv
from subgames.competitive_env import (
    CompetitiveSubgameEnv, cross_evaluate, make_agent_policy,
)
from subgames.subgame_trainer import SubgameTrainer, SubgameConfig
from subgames.action_mask import count_suit_length, count_hcp
from algorithms.behavioral_cloning import (
    create_bc_dataset_for_competitive, behavioral_cloning_warmup, evaluate_pass_rate,
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

    # Stage 1: Train S_base (N=rule, only S learns)
    stage1_steps: int = 5000
    stage1_deals_per_step: int = 32
    stage1_accumulate: int = 4

    # Stage 1.5: Belief pre-training
    belief_pretrain_deals: int = 2000
    belief_pretrain_epochs: int = 20
    belief_pretrain_target_acc: float = 0.80

    # Stage 2: Joint fine-tune (N+S both learn)
    stage2_steps: int = 3000
    stage2_deals_per_step: int = 32
    stage2_accumulate: int = 4

    # Competitive
    competitive_steps: int = 5000
    competitive_deals_per_step: int = 32
    competitive_eval_deals: int = 500

    # BC
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
# Stage 1: Train S_base with N=rule
# ============================================================================

def run_stage1(config: Phase2Config) -> dict:
    """
    Stage 1: N 硬编码规则, 只训练 S.

    环境完全 stationary (N 的行为固定),
    S 面对的是一个标准 contextual bandit.
    """
    print("=" * 60)
    print("Stage 1: Train S_base (N=rule, S learns)")
    print("=" * 60)

    env = StaymanSubgameEnv(config.stayman_data, north_rule=True)

    sub_config = SubgameConfig(
        num_steps=config.stage1_steps,
        deals_per_step=config.stage1_deals_per_step,
        accumulate_steps=config.stage1_accumulate,
        use_info_bonus=False,
        lr=1e-4,
        device=config.device,
        active_players=[SOUTH],  # Only S learns
    )
    trainer = SubgameTrainer(env, sub_config)

    t0 = time.time()
    log = trainer.train()
    train_time = time.time() - t0

    # Evaluate
    eval_results = _evaluate_stayman_full(env, trainer, config.eval_deals)

    results = {
        'mean_imp': eval_results['mean_imp'],
        'std_imp': eval_results['std_imp'],
        'train_time_sec': train_time,
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

def run_belief_pretrain(config: Phase2Config, s_base_trainer: SubgameTrainer) -> dict:
    """
    Stage 1.5: 用 Stage 1 的 rollout 数据预训练 Belief Network.

    科学依据: Belief net (supervised) 与 policy (RL) 是不同优化目标.
    混合训练时未收敛的 belief 会通过 r_info 向 policy 注入噪声梯度.
    分阶段预训练是 CTDE 文献中的标准做法 (cf. LICA, Ding et al. 2020).
    """
    print("\n" + "=" * 60)
    print("Stage 1.5: Belief Network Pre-training")
    print("=" * 60)

    env = StaymanSubgameEnv(config.stayman_data, north_rule=True)

    # 收集大量 rollout 数据 (用 S_base 的策略)
    print(f"  Collecting {config.belief_pretrain_deals} rollout episodes...")
    episodes = s_base_trainer.collect_episodes(config.belief_pretrain_deals)

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
# Stage 2: Divergent fine-tuning (A vs B)
# ============================================================================

def run_stage2(config: Phase2Config, s_base_trainer: SubgameTrainer,
               belief_pretrain: dict = None) -> dict:
    """
    Stage 2: 加载 S_base, 解冻 N, 分支微调.

    Agent A: MAPPO (control)
    Agent B: MAPPO + r_info (β=0), belief net 从预训练权重初始化
    """
    print("\n" + "=" * 60)
    print("Stage 2: Divergent Fine-tuning (A=MAPPO vs B=MAPPO+r_info)")
    print("=" * 60)

    # 保存 S_base 权重
    s_base_state = copy.deepcopy(s_base_trainer.agent.model.state_dict())
    belief_state = belief_pretrain['belief_state'] if belief_pretrain else None

    results = {}

    for name, use_info, beta in [
        ("A_control", False, 0.0),
        ("B_partner_only", True, 0.0),
    ]:
        print(f"\n--- Stage 2: {name} ---")

        # 创建新环境: north_rule=False (N 由 agent 决策)
        env = StaymanSubgameEnv(config.stayman_data, north_rule=False)

        sub_config = SubgameConfig(
            num_steps=config.stage2_steps,
            deals_per_step=config.stage2_deals_per_step,
            accumulate_steps=config.stage2_accumulate,
            use_info_bonus=use_info,
            beta=beta,
            lr=5e-5,  # 微调用更小 lr
            device=config.device,
            active_players=[NORTH, SOUTH],  # Both learn
            # Belief warmup 大幅降低: 预训练已完成, 只做微调
            belief_warmup_steps=100 if use_info else 0,
        )
        trainer = SubgameTrainer(env, sub_config)

        # 加载 S_base 权重 (公平起跑线)
        trainer.agent.model.load_state_dict(s_base_state)
        print(f"  Loaded S_base weights")

        # 加载预训练的 belief net 权重
        if use_info and belief_state is not None and trainer.belief_net is not None:
            trainer.belief_net.load_state_dict(belief_state)
            print(f"  Loaded pre-trained belief network")

        t0 = time.time()
        log = trainer.train()
        train_time = time.time() - t0

        # Evaluate
        eval_results = _evaluate_stayman_full(env, trainer, config.eval_deals)
        belief_acc = trainer.evaluate_belief_accuracy(
            num_deals=config.eval_deals
        ) if use_info else 0.0

        # 提取 info metrics
        info_metrics = {}
        if use_info and log:
            last = log[-1]
            info_metrics = {
                'info_ratio': last.get('info_ratio', 0),
                'partner_gain': last.get('partner_gain', 0),
                'opponent_leak': last.get('opponent_leak', 0),
            }

        results[name] = {
            'mean_imp': eval_results['mean_imp'],
            'std_imp': eval_results['std_imp'],
            'belief_accuracy': belief_acc,
            'train_time_sec': train_time,
            'final_log': log[-1] if log else {},
            **info_metrics,
        }

        print(f"  {name}: IMP={results[name]['mean_imp']:+.2f}±{results[name]['std_imp']:.2f}, "
              f"belief_acc={belief_acc:.3f}")

        # Diagnostics
        print(f"\n--- {name} Diagnostics ---")
        diag = _run_diagnostics(env, trainer, config.diag_deals)
        results[name]['diagnostics'] = diag

        # 保存 trainer 用于 Stage 3 分析
        results[name]['_trainer'] = trainer

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

    # Stage 1.5: Belief pre-training
    belief_pretrain = run_belief_pretrain(config, stage1['trainer'])
    report['belief_pretrain'] = belief_pretrain['results']

    # Stage 2: A vs B (with pre-trained belief for B)
    stage2 = run_stage2(config, stage1['trainer'], belief_pretrain)
    report['stage2'] = {
        name: {k: v for k, v in data.items() if k != '_trainer'}
        for name, data in stage2.items()
    }

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
    p.add_argument('--stage1_steps', type=int, default=5000)
    p.add_argument('--stage2_steps', type=int, default=3000)
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
        stage2_steps=args.stage2_steps,
        eval_deals=args.eval_deals,
        diag_deals=args.diag_deals,
    )

    if args.quick:
        config.stage1_steps = 500
        config.stage2_steps = 300
        config.eval_deals = 50
        config.diag_deals = 100
        config.belief_pretrain_deals = 200
        config.belief_pretrain_epochs = 5
        print("⚡ Quick mode: reduced steps for testing")

    run_phase2(config)


if __name__ == "__main__":
    main()
