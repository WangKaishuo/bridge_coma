"""
P97c Diagnostic: Partner Info Gain comparison (A vs B)

Uses B's belief net as the judge for both agents.
No training, just loads checkpoints and evaluates.

Usage:
    python experiments/diagnose_partner_gain.py \
        --data data/competitive_500k.npz \
        --agent_a results/competitive/agent_a_seed42.pt \
        --agent_b results/competitive/agent_b_seed42.pt \
        --sl results/sl_base.pt \
        --deals 500
"""

import argparse, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from subgames.competitive_env import CompetitiveSubgameEnv
from subgames.subgame_trainer import SubgameTrainer, SubgameConfig
from utils.running_stats import RunningStats
from env import NORTH, EAST, SOUTH, WEST


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/competitive_500k.npz')
    parser.add_argument('--agent_a', default='results/competitive/agent_a_seed42.pt')
    parser.add_argument('--agent_b', default='results/competitive/agent_b_seed42.pt')
    parser.add_argument('--sl', default='results/sl_base.pt')
    parser.add_argument('--deals', type=int, default=500)
    parser.add_argument('--belief_pretrain_rounds', type=int, default=50,
                        help='Rounds for belief pretrain (50=100k deals)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Diagnostic] device={device}  deals={args.deals}")

    # ── Environment ──
    env = CompetitiveSubgameEnv(data_path=args.data)

    # ── Build trainers (for agent wrappers only, no training) ──
    cfg_a = SubgameConfig(use_info_bonus=False, device=device)
    cfg_b = SubgameConfig(use_info_bonus=True, beta=0.05, device=device,
                          freeze_belief=False)

    trainer_a = SubgameTrainer(env, cfg_a, reward_stats=RunningStats())
    trainer_b = SubgameTrainer(env, cfg_b, reward_stats=RunningStats())

    # ── Load SL weights first (as base), then overwrite with checkpoints ──
    sl_ckpt = torch.load(args.sl, map_location=device)
    for trainer in [trainer_a, trainer_b]:
        for player, key in [(NORTH, 'actor_n'), (EAST, 'actor_e'),
                            (SOUTH, 'actor_s'), (WEST, 'actor_w')]:
            if key in sl_ckpt:
                trainer.agent.get_actor(player).load_state_dict(
                    {k: v.to(device) for k, v in sl_ckpt[key].items()})

    # ── Load trained checkpoints ──
    print(f"  Loading Agent A: {args.agent_a}")
    trainer_a.agent.load(args.agent_a)
    print(f"  Loading Agent B: {args.agent_b}")
    trainer_b.agent.load(args.agent_b)

    # ── Pretrain Belief Net (B's belief net, needed as judge) ──
    print(f"\n[Belief Pretrain] {args.belief_pretrain_rounds * 2000} deals...")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    trainer_b.pretrain_belief(
        num_rounds=args.belief_pretrain_rounds,
        deals_per_round=2000,
        max_epochs=50,
    )

    # ── Belief accuracy on B's RL trajectories ──
    print("\n[1] Belief Net accuracy on Agent B trajectories:")
    trainer_b.evaluate_belief(num_deals=min(args.deals, 200))

    # ── Partner Info Gain: A vs B ──
    belief_net = trainer_b.belief_net
    print(f"\n[2] Partner Info Gain ({args.deals} deals, B's belief net as judge)")

    print("\n  Agent A (no r_info):")
    pig_a = trainer_a.evaluate_partner_info_gain(belief_net, num_deals=args.deals)

    print("\n  Agent B (with r_info):")
    pig_b = trainer_b.evaluate_partner_info_gain(belief_net, num_deals=args.deals)

    diff = pig_b['mean_partner_gain'] - pig_a['mean_partner_gain']
    pct = diff / max(pig_a['mean_partner_gain'], 1e-8) * 100

    print(f"\n{'='*60}")
    print(f"  PARTNER INFO GAIN COMPARISON")
    print(f"{'='*60}")
    print(f"  Agent A: {pig_a['mean_partner_gain']:.4f} ± {pig_a['std_partner_gain']:.4f}  (n={pig_a['n_steps']})")
    print(f"  Agent B: {pig_b['mean_partner_gain']:.4f} ± {pig_b['std_partner_gain']:.4f}  (n={pig_b['n_steps']})")
    print(f"  Δ(B-A):  {diff:+.4f}  ({pct:+.1f}%)")
    if diff > 0:
        print(f"  → B communicates MORE to partner than A")
    elif diff < 0:
        print(f"  → A communicates MORE to partner than B")
    else:
        print(f"  → No difference")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
