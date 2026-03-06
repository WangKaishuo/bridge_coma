#!/usr/bin/env python3
"""
Subgame Validation (Phase 2)
==============================

Phase 2 实验主入口:
  1. Stayman 子博弈: 纯合作, 无 BC, 训练 2 个 agent, 单桌评估
  2. Competitive 子博弈: 1H-1S, BC 预热, 训练 3 个 agent, 双桌交叉对抗

Go/No-Go 决策点:
  - Stayman 后: Info Ratio > 1.0?
  - Competitive 后: B vs A 显著? β 有效果?

Usage:
    cd bridge-coma/

    # 先生成约束数据 (只需一次)
    python -m utils.generate_subgame_data --type both --num_workers 4

    # 运行实验
    python experiments/subgame_validation.py \\
        --stayman_data data/stayman_50k.npz \\
        --competitive_data data/competitive_100k.npz \\
        --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from subgames.stayman_env import StaymanSubgameEnv
from subgames.competitive_env import (
    CompetitiveSubgameEnv, cross_evaluate, make_agent_policy,
)
from subgames.subgame_trainer import SubgameTrainer, SubgameConfig
from algorithms.behavioral_cloning import (
    create_bc_dataset_for_competitive, behavioral_cloning_warmup, evaluate_pass_rate,
)
from env import BridgeBiddingEnv, NORTH, SOUTH


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

    # Stayman (action space ~3-7, converges fast)
    stayman_steps: int = 3000
    stayman_deals_per_step: int = 32
    stayman_eval_deals: int = 200

    # Competitive (action space ~38, needs more data)
    competitive_steps: int = 5000
    competitive_deals_per_step: int = 32
    competitive_eval_deals: int = 500

    # BC (uses competitive_data for BC dataset)
    bc_num_samples: int = 30000
    bc_epochs: int = 10

    # Go/No-Go thresholds
    go_info_ratio: float = 1.0
    go_belief_acc: float = 0.7
    go_imp_p_value: float = 0.05


# ============================================================================
# Stayman Experiment
# ============================================================================

def run_stayman_experiment(config: Phase2Config) -> dict:
    """
    Stayman 子博弈实验.

    训练 2 个 agent:
      - Agent A: MAPPO (control, no info bonus)
      - Agent B: MAPPO + r_info (β=0, partner-only)

    评估: 单桌 actual vs DDS optimal → IMP
    """
    print("=" * 60)
    print("Phase 2a: Stayman Subgame (Pure Cooperative)")
    print("=" * 60)

    env = StaymanSubgameEnv(config.stayman_data)

    results = {}

    for name, use_info, beta in [
        ("A_control", False, 0.0),
        ("B_partner_only", True, 0.0),
    ]:
        print(f"\n--- Training Agent {name} ---")
        sub_config = SubgameConfig(
            num_steps=config.stayman_steps,
            deals_per_step=config.stayman_deals_per_step,
            use_info_bonus=use_info,
            beta=beta,
            device=config.device,
            active_players=[NORTH, SOUTH],  # Stayman: only NS decide
        )
        trainer = SubgameTrainer(env, sub_config)

        # Train
        t0 = time.time()
        log = trainer.train()
        train_time = time.time() - t0

        # Evaluate
        belief_acc = trainer.evaluate_belief_accuracy(
            num_deals=config.stayman_eval_deals
        ) if use_info else 0.0

        # Single-table IMP evaluation
        eval_imps = _evaluate_stayman_single_table(
            env, trainer.agent, config.stayman_eval_deals
        )

        info_metrics = {}
        if use_info and log:
            last = log[-1]
            info_metrics = {
                'info_ratio': last.get('info_ratio', 0),
                'partner_gain': last.get('partner_gain', 0),
                'opponent_leak': last.get('opponent_leak', 0),
            }

        results[name] = {
            'mean_imp': float(np.mean(eval_imps)),
            'std_imp': float(np.std(eval_imps)),
            'belief_accuracy': belief_acc,
            'train_time_sec': train_time,
            'final_log': log[-1] if log else {},
            **info_metrics,
        }

        print(f"  {name}: IMP={results[name]['mean_imp']:+.3f}±{results[name]['std_imp']:.3f}, "
              f"belief_acc={belief_acc:.3f}")

    # Go/No-Go
    results['go_no_go'] = _stayman_go_no_go(results, config)

    return results


def _evaluate_stayman_single_table(env, agent, num_deals: int) -> list:
    """在 Stayman 子博弈中做单桌评估, 返回 IMP list."""
    import torch

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
                action, _, _, _ = agent.model.get_action_and_value(obs_t, all_h_t, deterministic=True)
                action = action.item()

            obs, reward, done, info = env.step(action)

        imps.append(info.get('imp', reward))

    return imps


def _stayman_go_no_go(results: dict, config: Phase2Config) -> dict:
    """Stayman Go/No-Go 检查."""
    b_result = results.get('B_partner_only', {})
    a_result = results.get('A_control', {})

    info_ratio = b_result.get('info_ratio', 0)
    belief_acc = b_result.get('belief_accuracy', 0)
    b_imp = b_result.get('mean_imp', -999)
    a_imp = a_result.get('mean_imp', -999)

    checks = {
        'info_ratio_pass': info_ratio > config.go_info_ratio,
        'belief_acc_pass': belief_acc > config.go_belief_acc,
        'imp_improvement': b_imp > a_imp,
        'info_ratio_value': info_ratio,
        'belief_acc_value': belief_acc,
        'imp_diff': b_imp - a_imp,
    }

    go = checks['info_ratio_pass']  # 核心 Go/No-Go 指标
    checks['decision'] = 'GO' if go else 'NO-GO'

    if go:
        print(f"\n✅ Stayman Go/No-Go: GO (Info Ratio = {info_ratio:.3f} > {config.go_info_ratio})")
    else:
        print(f"\n❌ Stayman Go/No-Go: NO-GO (Info Ratio = {info_ratio:.3f} <= {config.go_info_ratio})")
        print("  → 需要重新审视 Belief Net 架构或训练方式")

    return checks


# ============================================================================
# Competitive Experiment
# ============================================================================

def run_competitive_experiment(config: Phase2Config) -> dict:
    """
    Competitive 子博弈实验.

    训练 3 个 agent:
      - Agent A: MAPPO (control)
      - Agent B: MAPPO + r_info (β=0)
      - Agent C: MAPPO + r_info (β=0.5)

    交叉对抗: 双桌 IMP (B vs A, C vs A, C vs B)
    """
    print("\n" + "=" * 60)
    print("Phase 2b: Competitive Subgame (1H-1S)")
    print("=" * 60)

    env = CompetitiveSubgameEnv(config.competitive_data)

    # BC dataset (shared across all agents)
    print("\n--- Creating BC dataset ---")
    bc_dataset = create_bc_dataset_for_competitive(
        config.competitive_data, num_samples=config.bc_num_samples
    )

    agents = {}

    for name, use_info, beta in [
        ("A_control", False, 0.0),
        ("B_beta0", True, 0.0),
        ("C_beta05", True, 0.5),
    ]:
        print(f"\n--- Training Agent {name} ---")
        sub_config = SubgameConfig(
            num_steps=config.competitive_steps,
            deals_per_step=config.competitive_deals_per_step,
            use_info_bonus=use_info,
            beta=beta,
            device=config.device,
        )
        trainer = SubgameTrainer(env, sub_config)

        # BC warmup
        print(f"  BC warmup for {name}...")
        bc_stats = behavioral_cloning_warmup(
            trainer.agent, bc_dataset,
            epochs=config.bc_epochs,
        )
        pass_rate = evaluate_pass_rate(trainer.agent, BridgeBiddingEnv())
        print(f"  Post-BC pass rate: {pass_rate:.2%}")

        # Train
        t0 = time.time()
        log = trainer.train()
        train_time = time.time() - t0

        agents[name] = {
            'agent': trainer.agent,
            'trainer': trainer,
            'log': log,
            'bc_stats': bc_stats,
            'pass_rate': pass_rate,
            'train_time': train_time,
        }

    # Cross-evaluation
    print("\n--- Cross Evaluation ---")
    cross_results = {}

    matchups = [
        ("B_vs_A", "B_beta0", "A_control"),
        ("C_vs_A", "C_beta05", "A_control"),
        ("C_vs_B", "C_beta05", "B_beta0"),
    ]

    for label, agent_x_name, agent_y_name in matchups:
        agent_x = agents[agent_x_name]['agent']
        agent_y = agents[agent_y_name]['agent']

        # Create policies
        x_policy = make_agent_policy(agent_x, deterministic=True)
        y_policy = make_agent_policy(agent_y, deterministic=True)

        result = cross_evaluate(
            env,
            agent_a_ns_policy=x_policy,
            agent_a_ew_policy=x_policy,
            agent_b_ns_policy=y_policy,
            agent_b_ew_policy=y_policy,
            num_deals=config.competitive_eval_deals,
        )

        cross_results[label] = {
            'mean_imp': result.mean_imp,
            'std_imp': result.std_imp,
            'win_rate': result.win_rate,
            'p_value': result.p_value,
            'significant': result.significant,
            'n_deals': result.n_deals,
        }

        sig_str = "***" if result.p_value < 0.01 else ("**" if result.p_value < 0.05 else "ns")
        print(f"  {label}: IMP={result.mean_imp:+.3f}±{result.std_imp:.3f}, "
              f"win={result.win_rate:.1%}, p={result.p_value:.4f} {sig_str}")

    # Go/No-Go
    go_check = _competitive_go_no_go(cross_results, agents, config)

    return {
        'cross_results': cross_results,
        'agent_stats': {
            name: {
                'bc_stats': info['bc_stats'],
                'pass_rate': info['pass_rate'],
                'train_time': info['train_time'],
                'final_log': info['log'][-1] if info['log'] else {},
            }
            for name, info in agents.items()
        },
        'go_no_go': go_check,
    }


def _competitive_go_no_go(cross_results: dict, agents: dict, config: Phase2Config) -> dict:
    """Competitive Go/No-Go 检查."""
    b_vs_a = cross_results.get('B_vs_A', {})
    c_vs_a = cross_results.get('C_vs_A', {})
    c_vs_b = cross_results.get('C_vs_B', {})

    checks = {
        'b_vs_a_significant': b_vs_a.get('significant', False),
        'c_vs_a_significant': c_vs_a.get('significant', False),
        'b_vs_a_positive': b_vs_a.get('mean_imp', 0) > 0,
        'c_vs_a_positive': c_vs_a.get('mean_imp', 0) > 0,
        'c_vs_b_positive': c_vs_b.get('mean_imp', 0) > 0,
        'beta_has_value': c_vs_b.get('mean_imp', 0) > 0,
    }

    # r_info 有效?
    rinfo_effective = checks['b_vs_a_positive'] and checks['b_vs_a_significant']
    checks['rinfo_effective'] = rinfo_effective

    # β 有价值?
    beta_value = checks['c_vs_b_positive']
    checks['beta_has_marginal_value'] = beta_value

    if rinfo_effective:
        print(f"\n✅ Competitive Go/No-Go: r_info is effective! "
              f"(B vs A: {b_vs_a['mean_imp']:+.3f} IMP, p={b_vs_a['p_value']:.4f})")
        if beta_value:
            print(f"  ✅ β has marginal value (C vs B: {c_vs_b['mean_imp']:+.3f} IMP)")
            checks['decision'] = 'GO_FULL'
        else:
            print(f"  ⚠️  β shows no advantage (C vs B: {c_vs_b['mean_imp']:+.3f} IMP)")
            print("  → Consider Partner-Only (β=0) direction")
            checks['decision'] = 'GO_PARTNER_ONLY'
    else:
        print(f"\n❌ Competitive Go/No-Go: r_info not significant "
              f"(B vs A: {b_vs_a['mean_imp']:+.3f} IMP, p={b_vs_a['p_value']:.4f})")
        print("  → Check r_info scale, annealing, belief quality")
        checks['decision'] = 'NO_GO'

    return checks


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

    # Phase 2a: Stayman
    stayman_results = run_stayman_experiment(config)
    report['stayman'] = stayman_results

    # Go/No-Go check
    if stayman_results['go_no_go']['decision'] == 'NO-GO':
        print("\n⚠️  Stayman Go/No-Go failed. Proceeding to Competitive anyway for diagnosis.")

    # Phase 2b: Competitive
    competitive_results = run_competitive_experiment(config)
    report['competitive'] = competitive_results

    # Save report
    report_path = output_dir / "phase2_report.json"
    _save_json(report, report_path)
    print(f"\nReport saved to {report_path}")

    # Summary
    _print_summary(report)

    return report


def _save_json(data: dict, path: Path):
    """保存 JSON (处理不可序列化的类型)."""
    def default(o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=default)


def _print_summary(report: dict):
    """打印最终总结."""
    print("\n" + "=" * 60)
    print("Phase 2 Summary")
    print("=" * 60)

    # Stayman
    st = report.get('stayman', {})
    st_go = st.get('go_no_go', {})
    print(f"\nStayman: {st_go.get('decision', '?')}")
    print(f"  Info Ratio: {st_go.get('info_ratio_value', '?')}")
    print(f"  Belief Acc: {st_go.get('belief_acc_value', '?')}")
    print(f"  IMP diff (B-A): {st_go.get('imp_diff', '?')}")

    # Competitive
    comp = report.get('competitive', {})
    comp_go = comp.get('go_no_go', {})
    print(f"\nCompetitive: {comp_go.get('decision', '?')}")

    cross = comp.get('cross_results', {})
    for label, res in cross.items():
        sig = "✓" if res.get('significant') else "✗"
        print(f"  {label}: {res.get('mean_imp', 0):+.3f} IMP "
              f"(p={res.get('p_value', 1):.4f}) {sig}")


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Phase 2: Subgame Validation")
    p.add_argument('--stayman_data', default='data/stayman_50k.npz',
                   help='Stayman constrained DDS data (.npz)')
    p.add_argument('--competitive_data', default='data/competitive_100k.npz',
                   help='Competitive constrained DDS data (.npz)')
    p.add_argument('--device', default=None, help='Device (default: auto)')
    p.add_argument('--output_dir', default='results/')
    p.add_argument('--stayman_steps', type=int, default=3000)
    p.add_argument('--competitive_steps', type=int, default=5000)
    p.add_argument('--bc_samples', type=int, default=30000)
    p.add_argument('--eval_deals', type=int, default=500)
    args = p.parse_args()

    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    config = Phase2Config(
        stayman_data=args.stayman_data,
        competitive_data=args.competitive_data,
        device=device,
        output_dir=args.output_dir,
        stayman_steps=args.stayman_steps,
        competitive_steps=args.competitive_steps,
        bc_num_samples=args.bc_samples,
        competitive_eval_deals=args.eval_deals,
    )

    run_phase2(config)


if __name__ == "__main__":
    main()
