"""Run the competitive subgame validation experiment.

Formal design
-------------
Agent A: MAPPO + FSP, task reward only.
Agent B: MAPPO + FSP, task reward + partner information gain.
Agent C: MAPPO + FSP, task reward + partner gain - beta * opponent leakage.

All three agents use the same 571-dimensional policy at training and execution.
Belief networks are training-only communication critics for B and C.  Evaluation
is black-box and never shares an internal model between agents.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from algorithms.mappo import MAPPOAgent, MAPPOConfig
from env import NUM_PLAYERS
from experiments.evaluation import (
    evaluate_match,
    make_mappo_factory,
    sample_evaluation_deals,
)
from networks.belief_net import BeliefNetwork
from networks.policy_net import ACTION_MAPPING_VERSION, OBS_DIM
from subgames.competitive_env import CompetitiveSubgameEnv
from subgames.subgame_trainer import SubgameConfig, SubgameTrainer
from utils.running_stats import RunningStats


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(args, label: str, device: str) -> SubgameConfig:
    if label == "A":
        use_info_bonus, beta = False, 0.0
    elif label == "B":
        use_info_bonus, beta = True, 0.0
    elif label == "C":
        use_info_bonus, beta = True, args.beta
    else:
        raise ValueError(f"Unknown agent label: {label}")

    return SubgameConfig(
        num_rounds=args.rounds,
        steps_per_phase=2 if args.quick else 64,
        deals_per_step=32 if args.quick else 512,
        lr=args.learning_rate,
        batch_size=16 if args.quick else args.batch_size,
        num_epochs=1 if args.quick else 4,
        entropy_coef=args.entropy_coef,
        hidden_dim=1024,
        use_info_bonus=use_info_bonus,
        beta=beta,
        info_reward_weight=args.info_weight if use_info_bonus else 0.0,
        freeze_belief=not use_info_bonus,
        belief_update_epochs=1,
        belief_update_lr=args.belief_learning_rate,
        # KL is an optional baseline regularizer, not part of the main method.
        kl_lambda_start=args.kl_lambda,
        kl_lambda_end=args.kl_lambda,
        kl_anneal_frac=0.0,
        fsp_pool_size=0,
        fsp_add_interval=1,
        self_play=False,
        fsp_quality_gate=args.fsp_quality_gate,
        fsp_gate_eval_deals=50 if args.quick else 200,
        fsp_sl_sample_prob=args.fsp_sl_sample_prob,
        critic_prewarm_deals=64 if args.quick else 2048,
        critic_prewarm_epochs=1 if args.quick else 10,
        bc_warmup_samples=128 if args.quick else 5000,
        bc_warmup_epochs=1 if args.quick else 20,
        device=device,
    )


def load_policy_checkpoint(agent: MAPPOAgent, path: str, device: str) -> dict:
    """Load either the shared SL checkpoint or a four-actor MAPPO checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("obs_dim", OBS_DIM) != OBS_DIM:
        raise ValueError(
            f"Expected a {OBS_DIM}-dimensional policy checkpoint, got "
            f"{checkpoint.get('obs_dim')}. Execution-time BCA checkpoints are "
            "not accepted by the formal experiment."
        )

    if "model_state" in checkpoint:
        state = checkpoint["model_state"]
        for player in range(NUM_PLAYERS):
            agent.get_actor(player).load_state_dict(state)
    else:
        keys = ("actor_n", "actor_e", "actor_s", "actor_w")
        for player, key in enumerate(keys):
            if key in checkpoint:
                agent.get_actor(player).load_state_dict(checkpoint[key])
    return checkpoint


def load_belief_checkpoint(trainer: SubgameTrainer, path: str, device: str) -> None:
    if trainer.belief_net is None:
        return
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("belief_net", checkpoint)
    hidden_dim = state["trunk.0.weight"].shape[0]
    if trainer.belief_net.trunk[0].out_features != hidden_dim:
        trainer.belief_net = BeliefNetwork(hidden_dim=hidden_dim).to(device)
        trainer.dual_info.belief_net = trainer.belief_net
        trainer.belief_optimizer = torch.optim.Adam(
            trainer.belief_net.parameters(), lr=trainer.config.belief_update_lr
        )
    trainer.belief_net.load_state_dict(state)


def save_training_checkpoint(trainer: SubgameTrainer, path: Path, label: str) -> None:
    trainer.agent.save(str(path))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["agent_label"] = label
    checkpoint["execution_uses_belief"] = False
    checkpoint["action_mapping_version"] = ACTION_MAPPING_VERSION
    if trainer.belief_net is not None:
        checkpoint["belief_net"] = {
            key: value.detach().cpu()
            for key, value in trainer.belief_net.state_dict().items()
        }
    torch.save(checkpoint, path)


def load_deployment_agent(path: Path, device: str) -> MAPPOAgent:
    """Load a trained 571-dimensional agent without constructing a trainer."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    obs_dim = checkpoint.get("obs_dim", OBS_DIM)
    if obs_dim != OBS_DIM:
        raise ValueError(f"Expected {OBS_DIM}-dimensional agent, got {obs_dim}: {path}")
    mapping_version = checkpoint.get("action_mapping_version")
    if mapping_version != ACTION_MAPPING_VERSION:
        raise ValueError(
            f"Checkpoint uses an unknown or legacy action mapping: {path}. "
            "Retrain it with the corrected OpenSpiel 52..89 mapping."
        )
    agent = MAPPOAgent(MAPPOConfig(
        device=device,
        obs_dim=OBS_DIM,
        hidden_dim=checkpoint.get("hidden_dim", 1024),
    ))
    agent.load(str(path))
    return agent


def format_result(row: str, column: str, result) -> str:
    return (
        f"{row} vs {column}: {result.mean_imp:+.3f} ± {result.std_imp:.3f} IMP "
        f"(95% CI [{result.ci_low:+.3f}, {result.ci_high:+.3f}]); "
        f"{result.wins}/{result.losses}/{result.ties}"
    )


def run(args) -> None:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    env = CompetitiveSubgameEnv(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [] if args.eval_only else [
        label for label in args.train_agents if label not in args.skip_agent
    ]
    if not args.eval_only and not labels:
        raise ValueError("At least one training agent is required")
    trainers: dict[str, SubgameTrainer] = {}
    for label in labels:
        config = build_config(args, label, device)
        trainers[label] = SubgameTrainer(env, config, reward_stats=RunningStats())

    print(f"Device: {device}")
    print(f"Policy observation: {OBS_DIM} dimensions")
    print("BeliefNet at execution: disabled")

    if labels:
        reference_label = labels[0]
        reference = trainers[reference_label]
        if args.sl_checkpoint and os.path.exists(args.sl_checkpoint):
            load_policy_checkpoint(reference.agent, args.sl_checkpoint, device)
        else:
            print(
                f"Agent {reference_label}: no SL checkpoint; running one shared "
                "rule-based warmup"
            )
            reference.run_bc_warmup()

        shared_initial_state = {
            key: value.detach().clone()
            for key, value in reference.agent.state_dict().items()
        }
        for label, trainer in trainers.items():
            if label != reference_label:
                trainer.agent.load_state_dict(shared_initial_state)
            if args.kl_lambda > 0:
                trainer.set_bc_anchor(trainer.agent)
        print(f"Shared initial Actor+Critic parameters copied to: {', '.join(labels)}")

    info_trainers = [trainers[label] for label in ("B", "C") if label in trainers]
    if info_trainers:
        if args.belief_checkpoint and os.path.exists(args.belief_checkpoint):
            for trainer in info_trainers:
                load_belief_checkpoint(trainer, args.belief_checkpoint, device)
        elif args.belief_pretrain_rounds > 0:
            set_seed(args.seed + 10_000)
            source = info_trainers[0]
            source.pretrain_belief(
                num_rounds=args.belief_pretrain_rounds,
                deals_per_round=200 if args.quick else args.belief_pretrain_deals,
                epochs_per_round=2 if args.quick else 5,
                max_epochs=args.belief_pretrain_max_epochs,
            )
            for trainer in info_trainers[1:]:
                trainer.belief_net.load_state_dict(source.belief_net.state_dict())
                if hasattr(source, "_pretrain_replay"):
                    trainer._pretrain_replay = {
                        key: value.clone()
                        for key, value in source._pretrain_replay.items()
                    }

    logs = {}
    for label, trainer in trainers.items():
        print(f"\n=== Training Agent {label} ===")
        # Common random numbers reduce variance in A/B/C treatment comparisons.
        set_seed(args.seed)
        logs[label] = trainer.run(num_rounds=args.rounds)
        save_training_checkpoint(
            trainer, output_dir / f"agent_{label.lower()}_seed{args.seed}.pt", label
        )

    should_evaluate = (
        args.eval_only or args.evaluate or set(labels) == set("ABC")
    )
    if not should_evaluate:
        print("Training complete. Cross-agent evaluation deferred to an eval-only run.")
        return

    checkpoint_overrides = {
        "A": args.agent_a,
        "B": args.agent_b,
        "C": args.agent_c,
    }
    deployment_agents = {
        label: trainer.agent for label, trainer in trainers.items()
    }
    for label in "ABC":
        if label in deployment_agents:
            continue
        path = Path(checkpoint_overrides[label]) if checkpoint_overrides[label] else (
            output_dir / f"agent_{label.lower()}_seed{args.seed}.pt"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Agent {label} checkpoint for evaluation: {path}"
            )
        deployment_agents[label] = load_deployment_agent(path, device)

    deals = sample_evaluation_deals(env, args.eval_deals, args.eval_seed)
    factories = {
        label: make_mappo_factory(agent)
        for label, agent in deployment_agents.items()
    }
    results = {}
    for row, column in (("B", "A"), ("C", "A"), ("C", "B")):
        if row not in factories or column not in factories:
            continue
        result = evaluate_match(env, deals, factories[row], factories[column])
        results[f"{row}_vs_{column}"] = {
            key: value for key, value in asdict(result).items() if key != "imps"
        }
        print(format_result(row, column, result))

    summary = {
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "execution_uses_belief": False,
        "config": vars(args),
        "matchups": results,
    }
    with open(output_dir / f"summary_seed{args.seed}.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/competitive_100k.npz")
    parser.add_argument("--sl-checkpoint", default="results/sl_base.pt")
    parser.add_argument("--belief-checkpoint", default="results/sl_base_bca.pt")
    parser.add_argument("--output-dir", default="results/competitive_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--eval-deals", type=int, default=5000)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--info-weight", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--belief-learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--kl-lambda", type=float, default=0.0)
    parser.add_argument("--belief-pretrain-rounds", type=int, default=0)
    parser.add_argument("--belief-pretrain-deals", type=int, default=2000)
    parser.add_argument("--belief-pretrain-max-epochs", type=int, default=50)
    parser.add_argument("--fsp-sl-sample-prob", type=float, default=0.30)
    parser.add_argument("--fsp-quality-gate", action="store_true")
    parser.add_argument(
        "--train-agents", nargs="+", choices=list("ABC"), default=list("ABC"),
        help="Agents to train in this Colab session, e.g. --train-agents B",
    )
    parser.add_argument("--skip-agent", choices=list("ABC"), action="append", default=[])
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Evaluate after a partial training run by loading missing checkpoints",
    )
    parser.add_argument("--agent-a", help="Agent A checkpoint for evaluation")
    parser.add_argument("--agent-b", help="Agent B checkpoint for evaluation")
    parser.add_argument("--agent-c", help="Agent C checkpoint for evaluation")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
