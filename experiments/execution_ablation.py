"""Paired real-belief versus prior-belief execution ablation.

This script exists for legacy 667-dimensional BCA checkpoints.  It is not part
of the formal deployment path.  No opponent model is shared in either mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from algorithms.mappo import MAPPOAgent, MAPPOConfig
from experiments.evaluation import (
    evaluate_execution_ablation,
    make_belief_conditioned_factory,
    sample_evaluation_deals,
)
from networks.belief_net import BeliefNetwork
from networks.policy_net import BELIEF_OBS_DIM
from subgames.competitive_env import CompetitiveSubgameEnv


def load_checkpoint(path: str, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("obs_dim") != BELIEF_OBS_DIM:
        raise ValueError(
            f"Execution ablation requires a {BELIEF_OBS_DIM}-dimensional BCA "
            f"checkpoint; got {checkpoint.get('obs_dim')}"
        )
    agent = MAPPOAgent(MAPPOConfig(device=device, obs_dim=BELIEF_OBS_DIM))
    agent.load(path)
    if "belief_net" not in checkpoint:
        raise ValueError("Checkpoint does not contain an embedded BeliefNet")
    state = checkpoint["belief_net"]
    hidden_dim = state["trunk.0.weight"].shape[0]
    belief_net = BeliefNetwork(hidden_dim=hidden_dim).to(device)
    belief_net.load_state_dict(state)
    belief_net.eval()
    return agent, belief_net


def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    env = CompetitiveSubgameEnv(args.data)
    agent, belief_net = load_checkpoint(args.checkpoint, device)
    real = make_belief_conditioned_factory(agent, belief_net, use_prior=False)
    prior = make_belief_conditioned_factory(agent, belief_net, use_prior=True)
    deals = sample_evaluation_deals(env, args.deals, args.seed)
    result = evaluate_execution_ablation(env, deals, real, prior, real)

    summary = {
        "checkpoint": args.checkpoint,
        "deals": args.deals,
        "seed": args.seed,
        "mean_imp_delta": result.mean_imp_delta,
        "std_imp_delta": result.std_imp_delta,
        "action_disagreement_rate": result.action_disagreement_rate,
        "auction_disagreement_rate": result.auction_disagreement_rate,
        "contract_disagreement_rate": result.contract_disagreement_rate,
        "score_disagreement_rate": result.score_disagreement_rate,
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/competitive_500k.npz")
    parser.add_argument("--deals", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
