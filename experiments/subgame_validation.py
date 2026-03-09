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

    # Stage 1: BC warmup + Critic rollout warmup (双轨预热)
    stage1_steps: int = 0             # 0 = 纯 BC+Critic 预热, >0 = 之后接 RL 微调
    stage1_deals_per_step: int = 32
    stage1_accumulate: int = 4
    # 双轨预热: BC 和 Critic 交替训练
    critic_warmup_rounds: int = 10    # 原 5; 给 Critic 更多收敛机会
    critic_warmup_deals: int = 512
    critic_warmup_log_interval: int = 2

    # Stage 1.5: Belief pre-training
    belief_pretrain_deals: int = 10000   # 原 2000, 增加 5x 训练数据
    belief_pretrain_epochs: int = 50     # 原 20, 充分收敛
    belief_pretrain_target_acc: float = 0.40  # Top-13 命中率目标 (随机基线=0.25, 有意义≥0.35)

    # Stage 2: Alternating fine-tune (single_step, batch-mean baseline)
    # S-N-S-N... 交替训练 + 联合微调收尾
    stage2_alt_rounds: int = 4         # 交替轮数 (每轮 = 1×S + 1×N)
    stage2_alt_steps: int = 200        # 每个半轮的步数
    stage2_joint_steps: int = 400      # 最终联合微调步数
    stage2_deals_per_step: int = 32
    stage2_accumulate: int = 8         # 256 deals/update
    stage2_lr: float = 3e-5            # 交替阶段用
    stage2_lr_joint: float = 1e-5      # 联合微调用
    stage2_entropy_start: float = 0.10 # 每轮重置到此值 (防 3-action 坍缩)
    stage2_entropy_end: float = 0.05   # 每轮退火终点
    stage2_entropy_anneal: float = 0.8 # 延后退火

    # KL Anchor: Stage 2 中限制 RL 策略偏离 BC 的程度
    # 前两轮强锚定, 后两轮逐渐放松
    stage2_kl_lambda_start: float = 0.5  # 初始 KL 系数
    stage2_kl_lambda_end: float = 0.1    # 退火终点 (S-phase 用)
    stage2_kl_anneal_frac: float = 1.0   # 覆盖整个训练过程退火

    # N-phase 专用 KL anchor (更强, 防止 N 退化为 100% 2D)
    # 行动三: kl_lambda_end 提至 0.5, 无论 S 是否倾听, 强制 N 诚实报高花
    stage2_n_kl_lambda_start: float = 0.5  # N-phase KL 起始值 (与全局相同)
    stage2_n_kl_lambda_end: float = 0.5    # N-phase KL 终点值 (不退火, 始终保持强锚定)

    # 课程学习前缀 (行动二): 在交替训练前, 先冻结 N 用规则策略, 只训 S N_warmup_rounds 轮
    # 目的: 让 S 在"完美发信机"环境下建立"倾听信心", 再解冻 N
    stage2_n_warmup_rounds: int = 1       # 0 = 关闭课程学习前缀; 1 = 保守折中

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
    eval_deals: int = 1000   # 原 200; 标准误差从 ±0.28 降至 ±0.14 IMP

    # Diagnostics
    diag_deals: int = 2000   # 原 500; 合同分布统计更稳定 (500 deals 方差约 ±2%)

    # Go/No-Go
    go_info_ratio: float = 1.0
    go_belief_acc: float = 0.7


# ============================================================================
# Stage 1: BC warmup + RL fine-tune (N+S jointly)
# ============================================================================

def run_stage1(config: Phase2Config) -> dict:
    """
    Stage 1: 联合训练 N+S base agent — 双轨预热.

    流程:
      Phase A: BC warmup (静态数据, Cross-Entropy, 教 Actor "该叫什么")
      Phase B: Critic Rollout warmup × N 轮 (动态数据, MSE, 教 Critic
               "在当前策略下, 这手牌能拿多少分")

    为什么必须双轨:
      纯 BC 后 Critic 从随机初始化起步. Stage 2 进入时 V(s) 完全无意义.
      Advantage = reward - V(s) 充满噪声. S 的梯度是"哪个动作看起来比
      随机 Critic 更好", 而非"哪个动作真的更好". 结果必然坍缩到 4M.

      Critic rollout target = 当前 Actor 在环境里实际拿到的 final_reward.
      这是 V(s) 的无偏估计. 经过预热的 Critic 能精准区分
      "8HCP 无配合手牌期望 -5 IMP" 和 "8HCP 有配合期望 +0 IMP",
      Stage 2 的 Advantage 才能剥离牌运方差, S 才能真正听懂 N 的叫牌.
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
        active_players=[NORTH, SOUTH],
        single_step=False,   # 双轨预热后 Critic 有效, 启用标准 GAE
    )
    trainer = SubgameTrainer(env, sub_config)

    # ---------------------------------------------------------------
    # Phase A: BC Warmup (静态数据, 教 Actor)
    # ---------------------------------------------------------------
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
    _run_diagnostics(env, trainer, config.diag_deals)

    # ---------------------------------------------------------------
    # Phase B: Critic Rollout Warmup (动态数据, 教 Critic)
    # ---------------------------------------------------------------
    print(f"\n--- Critic Rollout Warmup ({config.critic_warmup_rounds} rounds"
          f" × {config.critic_warmup_deals} deals) ---")
    critic_losses = []
    for rnd in range(1, config.critic_warmup_rounds + 1):
        v_loss = trainer.critic_warmup_step(num_deals=config.critic_warmup_deals)
        critic_losses.append(v_loss)
        if rnd % config.critic_warmup_log_interval == 0 or rnd == 1:
            print(f"  [Round {rnd}/{config.critic_warmup_rounds}] "
                  f"critic_value_loss={v_loss:.4f}")

    print(f"  Critic warmup done: "
          f"loss {critic_losses[0]:.4f} → {critic_losses[-1]:.4f}")

    # ---------------------------------------------------------------
    # Optional RL fine-tune
    # ---------------------------------------------------------------
    log = []
    train_time = 0.0
    if config.stage1_steps > 0:
        print("\n--- RL Fine-tune ---")
        t0 = time.time()
        log = trainer.train()
        train_time = time.time() - t0
    else:
        print("\n--- Skipping RL fine-tune (stage1_steps=0, pure BC+Critic) ---")

    # Evaluate
    eval_results = _evaluate_stayman_full(env, trainer, config.eval_deals)

    results = {
        'mean_imp': eval_results['mean_imp'],
        'std_imp': eval_results['std_imp'],
        'train_time_sec': train_time,
        'bc_stats': bc_stats,
        'critic_warmup_losses': critic_losses,
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
                  f"belief_loss={loss:.4f}, top13_hit={acc:.3f}"
                  f" (random_baseline=0.25)")

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

    print(f"\nBelief pre-train: top13_hit={best_acc:.3f} "
          f"({'✓ reached' if results['target_reached'] else '✗ not reached'} "
          f"target={config.belief_pretrain_target_acc:.2f}, random_baseline=0.25)")

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
                     entropy_end: float = None,
                     kl_lambda_start: float = None,
                     kl_lambda_end: float = None) -> SubgameConfig:
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
        single_step=False,   # Critic 已预热, 启用标准 GAE (不再用 batch-mean baseline)
        belief_warmup_steps=belief_warmup,
        entropy_coef_start=entropy_start if entropy_start is not None else config.stage2_entropy_start,
        entropy_coef_end=entropy_end if entropy_end is not None else config.stage2_entropy_end,
        entropy_anneal_frac=config.stage2_entropy_anneal,
        # KL anchor: 每个半轮独立退火 (每轮重新从 kl_lambda_start 开始)
        kl_lambda_start=kl_lambda_start if kl_lambda_start is not None else config.stage2_kl_lambda_start,
        kl_lambda_end=kl_lambda_end if kl_lambda_end is not None else config.stage2_kl_lambda_end,
        kl_anneal_frac=config.stage2_kl_anneal_frac,
        eval_interval=100,
        log_interval=20,
    )


def _run_one_phase(config: Phase2Config, env, sub_config: SubgameConfig,
                   prev_state: dict, belief_state: dict = None,
                   bc_state: dict = None,
                   phase_label: str = "") -> Tuple:
    """
    运行一个训练阶段, 返回 (trainer, log, eval_results).

    加载 prev_state 权重, 可选加载 belief_state.
    若提供 bc_state, 注入 KL anchor (防 RL 摧毁 BC 策略).
    """
    trainer = SubgameTrainer(env, sub_config)
    trainer.agent.model.load_state_dict(prev_state)

    # 注入 KL anchor: BC 快照 = Stage 1 结束时的参数
    # 每个半轮重新注入同一个 bc_state, 确保 anchor 始终指向 BC 分布
    if bc_state is not None and sub_config.kl_lambda_start > 0.0:
        trainer.set_bc_anchor(bc_state)

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
    # BC 快照: Stage 1 结束时的参数, 用于 KL anchor
    bc_state = copy.deepcopy(base_state)

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
        # 行动二: 课程学习前缀 — 冻结 N (规则策略), 只训 S
        # 目的: 让 S 在"完美信号源"下学会倾听, 建立"叫 4M 是安全的"信心
        # 条件: N_warmup_rounds > 0 时启用; 完成后再进入正式交替训练
        # ================================================================
        if config.stage2_n_warmup_rounds > 0:
            print(f"\n  ── Curriculum Warmup: {config.stage2_n_warmup_rounds} rounds"
                  f" with N=rule (S only) ──")
            for warmup_rnd in range(1, config.stage2_n_warmup_rounds + 1):
                env_sw = StaymanSubgameEnv(config.stayman_data, north_rule=True)
                cfg_sw = _make_sub_config(
                    config, config.stage2_alt_steps, config.stage2_lr,
                    active_players=[SOUTH],
                    use_info=False,
                    entropy_start=config.stage2_entropy_start,
                    entropy_end=config.stage2_entropy_end,
                )
                trainer_sw, _, eval_sw = _run_one_phase(
                    config, env_sw, cfg_sw, current_state,
                    bc_state=bc_state,
                    phase_label=f"CurriculumWarmup R{warmup_rnd} S-phase",
                )
                current_state = copy.deepcopy(trainer_sw.agent.model.state_dict())
                round_imps.append({
                    'round': f'CW{warmup_rnd}',
                    'S_phase': eval_sw['mean_imp'],
                    'N_phase': None,
                })

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
                # entropy 每轮重置到 start (不跨轮退火)
                entropy_start=config.stage2_entropy_start,
                entropy_end=config.stage2_entropy_end,
            )

            trainer_s, _, eval_s = _run_one_phase(
                config, env_s, cfg_s, current_state,
                bc_state=bc_state,
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
                # entropy 每轮重置到 start
                entropy_start=config.stage2_entropy_start,
                entropy_end=config.stage2_entropy_end,
                # 行动三: N-phase 使用更强的 KL anchor, 不退到 0.1
                # 原理: N 退化成 100% 2D 时 KL 梯度为 0 (已到 BC 分布);
                #       强 KL_end 让 N 锁定在 BC 的诚实分布附近, 无法摆烂
                kl_lambda_start=config.stage2_n_kl_lambda_start,
                kl_lambda_end=config.stage2_n_kl_lambda_end,
            )

            trainer_n, log_n, eval_n = _run_one_phase(
                config, env_n, cfg_n, current_state,
                belief_state=current_belief,
                bc_state=bc_state,
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
            # 联合阶段: KL 系数降至 end 值 (策略已稳定, 轻锚定)
            kl_lambda_start=config.stage2_kl_lambda_end,
            kl_lambda_end=config.stage2_kl_lambda_end,
        )

        trainer_j, log_j, eval_j = _run_one_phase(
            config, env_j, cfg_j, current_state,
            belief_state=current_belief,
            bc_state=bc_state,
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
            rnd_label = ri['round']
            s_imp = ri['S_phase']
            n_imp = ri['N_phase']
            if n_imp is None:
                print(f"    {rnd_label}: S→{s_imp:+.2f}  N→(frozen rule)")
            else:
                print(f"    R{rnd_label}: S→{s_imp:+.2f}  N→{n_imp:+.2f}")
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
    """检测 N-S 高花配合情况 (含 4-4 / 5-3 / 双高花)."""
    p1, p2 = declarer_side_players
    h1 = count_suit_length(hands[p1], 2)
    h2 = count_suit_length(hands[p2], 2)
    s1 = count_suit_length(hands[p1], 3)
    s2 = count_suit_length(hands[p2], 3)
    heart_fit_44  = h1 >= 4 and h2 >= 4
    spade_fit_44  = s1 >= 4 and s2 >= 4
    heart_fit_53  = (h1 >= 5 and h2 >= 3) or (h1 >= 3 and h2 >= 5)
    spade_fit_53  = (s1 >= 5 and s2 >= 3) or (s1 >= 3 and s2 >= 5)
    heart_fit     = heart_fit_44 or heart_fit_53
    spade_fit     = spade_fit_44 or spade_fit_53
    return {
        'heart_fit_44': heart_fit_44,
        'spade_fit_44': spade_fit_44,
        'heart_fit_53': heart_fit_53 and not heart_fit_44,
        'spade_fit_53': spade_fit_53 and not spade_fit_44,
        'heart_fit':    heart_fit,
        'spade_fit':    spade_fit,
        'double_fit':   heart_fit and spade_fit,
        'any_fit':      heart_fit or spade_fit,
        'best_suit':    'H' if (heart_fit and not spade_fit) else
                        ('S' if spade_fit else None),
    }


def _run_diagnostics(env, trainer, num_deals: int) -> dict:
    """
    全面训练诊断:
      1. Contract Distribution
      2. Fit Detection — 细分 4-4 / 5-3 / 双高花
      3. Decision Error Matrix — 代价矩阵 (有配合却叫 3NT / 无配合叫 4M)
      4. Reward / IMP Distribution
    """
    agent = trainer.agent

    contract_counts = Counter()
    imp_values, reward_values = [], []

    # Fit 细分计数器
    fit_counts = {
        'heart_44': 0, 'spade_44': 0,
        'heart_53': 0, 'spade_53': 0,
        'double':   0, 'no_fit':   0,
    }
    # 代价矩阵: [情形] -> IMP 列表
    cost_matrix = {
        'fit_bid_4M':    [],   # ✓ 有配合, 叫对了
        'fit_bid_3NT':   [],   # ✗ 有配合, 叫了 3NT (漏配合)
        'nofit_bid_3NT': [],   # ✓ 无配合, 叫对了
        'nofit_bid_4M':  [],   # ✗ 无配合, 叫了 4M (假配合)
        'other':         [],
    }

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
                    obs_t, all_h_t, deterministic=True)
                action = action.item()
            obs, reward, done, info = env.step(action)

        imp_val = info.get('imp', 0)
        imp_values.append(imp_val)
        reward_values.append(reward)

        contract  = env.env.state.final_contract
        category  = _classify_contract(contract)
        contract_counts[category] += 1

        fit_info  = _has_major_fit(hands)
        bid_4M    = (contract is not None and contract.suit in (2, 3) and contract.level >= 4)
        bid_3NT   = (contract is not None and contract.suit == 4 and contract.level == 3)

        # Fit 细分统计
        if fit_info['double_fit']:        fit_counts['double']   += 1
        elif fit_info['heart_fit_44']:    fit_counts['heart_44'] += 1
        elif fit_info['spade_fit_44']:    fit_counts['spade_44'] += 1
        elif fit_info['heart_fit_53']:    fit_counts['heart_53'] += 1
        elif fit_info['spade_fit_53']:    fit_counts['spade_53'] += 1
        else:                             fit_counts['no_fit']   += 1

        # 代价矩阵
        if fit_info['any_fit']:
            if bid_4M:   cost_matrix['fit_bid_4M'].append(imp_val)
            elif bid_3NT:cost_matrix['fit_bid_3NT'].append(imp_val)
            else:        cost_matrix['other'].append(imp_val)
        else:
            if bid_3NT:  cost_matrix['nofit_bid_3NT'].append(imp_val)
            elif bid_4M: cost_matrix['nofit_bid_4M'].append(imp_val)
            else:        cost_matrix['other'].append(imp_val)

    # === Print ===
    total = sum(contract_counts.values())
    print(f"\n  Contract Distribution ({total} deals):")
    display_order = ['passed_out', 'part_score', '3NT', '4M', '5m',
                     'other_game', 'small_slam', 'grand_slam']
    for cat in display_order:
        n = contract_counts.get(cat, 0)
        pct = n / max(1, total)
        bar = "█" * int(pct * 40)
        print(f"    {cat:14s}: {n:4d} ({pct:5.1%}) {bar}")

    print(f"\n  Fit Distribution ({total} deals):")
    for label, key in [('4-4 Heart', 'heart_44'), ('4-4 Spade', 'spade_44'),
                       ('5-3 Heart', 'heart_53'), ('5-3 Spade', 'spade_53'),
                       ('Double Fit', 'double'), ('No Fit', 'no_fit')]:
        n = fit_counts[key]
        print(f"    {label:12s}: {n:4d} ({n/max(1,total):5.1%})")

    print(f"\n  Decision Error Matrix:")
    for label, key in [
        ('✓ fit  → 4M  ', 'fit_bid_4M'),
        ('✗ fit  → 3NT ', 'fit_bid_3NT'),
        ('✓ nofit→ 3NT ', 'nofit_bid_3NT'),
        ('✗ nofit→ 4M  ', 'nofit_bid_4M'),
    ]:
        vals = cost_matrix[key]
        if vals:
            print(f"    {label}: n={len(vals):4d}  "
                  f"IMP={np.mean(vals):+.2f}±{np.std(vals):.2f}  "
                  f"[median={np.median(vals):+.1f}]")
        else:
            print(f"    {label}: n=   0")

    imp_arr = np.array(imp_values)
    rw_arr  = np.array(reward_values)
    print(f"\n  Reward Distribution:")
    print(f"    reward: mean={rw_arr.mean():+.3f}, std={rw_arr.std():.3f}, "
          f"min={rw_arr.min():+.3f}, max={rw_arr.max():+.3f}")
    print(f"    IMP:    mean={imp_arr.mean():+.2f}, median={np.median(imp_arr):+.1f}, "
          f"[p5={np.percentile(imp_arr,5):+.0f}, p95={np.percentile(imp_arr,95):+.0f}]")

    # Build result dict
    def _safe_mean(lst): return float(np.mean(lst)) if lst else 0.0

    diag = {
        'contract_distribution': {k: contract_counts.get(k, 0) / max(1, total)
                                  for k in display_order},
        'fit_distribution': {k: v / max(1, total) for k, v in fit_counts.items()},
        'decision_matrix': {
            k: {'n': len(v), 'mean_imp': _safe_mean(v)}
            for k, v in cost_matrix.items()
        },
        'reward_stats': {
            'mean': float(rw_arr.mean()), 'std': float(rw_arr.std()),
            'min': float(rw_arr.min()),   'max': float(rw_arr.max()),
        },
        'imp_stats': {
            'mean': float(imp_arr.mean()), 'std': float(imp_arr.std()),
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
        config.critic_warmup_rounds = 2      # quick mode: 少跑几轮
        config.critic_warmup_deals = 128
        print("⚡ Quick mode: reduced steps for testing")

    run_phase2(config)


if __name__ == "__main__":
    main()
