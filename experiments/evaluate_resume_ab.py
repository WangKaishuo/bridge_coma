"""Evaluate round-boundary A/B resume checkpoints without resuming training."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from algorithms.mappo import MAPPOAgent, MAPPOConfig
from experiments.evaluation import (
    evaluate_match_stratified,
    make_mappo_factory,
    sample_evaluation_deals,
)
from networks.policy_net import ACTION_MAPPING_VERSION, OBS_DIM
from subgames.unrestricted_env import UnrestrictedBiddingEnv


def load_resume_agent(path: Path, device: str):
    resume = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint = resume["agent"]
    if checkpoint.get("action_mapping_version") != ACTION_MAPPING_VERSION:
        raise ValueError(f"Unexpected action mapping in {path}")
    agent = MAPPOAgent(MAPPOConfig(
        device=device,
        obs_dim=checkpoint.get("obs_dim", OBS_DIM),
        hidden_dim=checkpoint.get("hidden_dim", 1024),
        actor_belief_conditioned=checkpoint.get(
            "actor_belief_conditioned", False
        ),
        actor_belief_hidden_dim=checkpoint.get(
            "actor_belief_hidden_dim", checkpoint.get("hidden_dim", 1024)
        ),
    ))
    for role in ("actor_n", "actor_e", "actor_s", "actor_w"):
        getattr(agent.model, role).load_state_dict(checkpoint[role])
        getattr(agent.model, role).eval()
    return agent, int(resume["completed_rounds"])


def compact(result) -> dict:
    return {key: value for key, value in asdict(result).items() if key != "imps"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-a", type=Path, required=True)
    parser.add_argument("--agent-b", type=Path, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--deals", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    agent_a, round_a = load_resume_agent(args.agent_a, device)
    agent_b, round_b = load_resume_agent(args.agent_b, device)
    if round_a != round_b:
        raise ValueError(f"Checkpoint rounds differ: A={round_a}, B={round_b}")

    env = UnrestrictedBiddingEnv(args.data)
    deals = sample_evaluation_deals(env, args.deals, args.seed)
    result = evaluate_match_stratified(
        env, deals, make_mappo_factory(agent_b), make_mappo_factory(agent_a)
    )
    overall = result.overall
    if overall.ci_low > 0:
        decision = "WIN"
    elif overall.mean_imp <= 0:
        decision = "NOT_WIN"
    else:
        decision = "INCONCLUSIVE"
    payload = {
        "round": round_a,
        "deals": args.deals,
        "seed": args.seed,
        "decision": decision,
        "overall": compact(overall),
        "competitive": compact(result.competitive),
        "non_competitive": compact(result.non_competitive),
        "mixed": compact(result.mixed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
