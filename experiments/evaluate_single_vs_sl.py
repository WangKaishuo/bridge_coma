"""Evaluate one unrestricted resume checkpoint against the frozen SL policy."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from algorithms.mappo import MAPPOAgent, MAPPOConfig
from experiments.evaluate_resume_ab import load_resume_agent
from experiments.evaluation import (
    evaluate_match_stratified,
    make_mappo_factory,
    sample_evaluation_deals,
)
from experiments.subgame_validation import load_policy_checkpoint
from networks.policy_net import OBS_DIM
from subgames.unrestricted_env import UnrestrictedBiddingEnv


def compact(result) -> dict:
    return {key: value for key, value in asdict(result).items() if key != "imps"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--sl-checkpoint", type=Path, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--deals", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    agent, checkpoint_round = load_resume_agent(args.agent, device)
    sl_agent = MAPPOAgent(MAPPOConfig(
        device=device,
        obs_dim=OBS_DIM,
        hidden_dim=1024,
        actor_belief_conditioned=False,
    ))
    metadata = load_policy_checkpoint(sl_agent, str(args.sl_checkpoint), device)
    for actor in (
        sl_agent.model.actor_n,
        sl_agent.model.actor_e,
        sl_agent.model.actor_s,
        sl_agent.model.actor_w,
    ):
        actor.eval()

    env = UnrestrictedBiddingEnv(args.data)
    deals = sample_evaluation_deals(env, args.deals, args.seed)
    result = evaluate_match_stratified(
        env, deals, make_mappo_factory(agent), make_mappo_factory(sl_agent)
    )
    payload = {
        "checkpoint_round": checkpoint_round,
        "deals": args.deals,
        "seed": args.seed,
        "sl_iteration": metadata.get("iteration"),
        "agent_vs_sl": {
            "overall": compact(result.overall),
            "competitive": compact(result.competitive),
            "non_competitive": compact(result.non_competitive),
            "mixed": compact(result.mixed),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
