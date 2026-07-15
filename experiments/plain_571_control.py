"""Train the corrected-infrastructure plain-571 Agent A control.

This control deliberately removes every belief-related actor path and every
information reward. It keeps the current action mapping, rollout attribution,
GAE, PPO value clipping, and black-box duplicate evaluation, while matching the
key settings of the previously successful plain-571 Agent A run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
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
from experiments.evaluation import (
    evaluate_match,
    make_mappo_factory,
    sample_evaluation_deals,
)
from experiments.subgame_validation import load_policy_checkpoint
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


def save_plain_checkpoint(trainer: SubgameTrainer, path: Path) -> None:
    trainer.agent.save(str(path))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint.update({
        "agent_label": "A_plain_571_control",
        "execution_uses_belief": False,
        "belief_interface": "none_plain_571",
        "action_mapping_version": ACTION_MAPPING_VERSION,
        "control_experiment": "corrected_pipeline_plain_571_v1",
    })
    torch.save(checkpoint, path)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    env = CompetitiveSubgameEnv(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SubgameConfig(
        num_rounds=args.rounds,
        steps_per_phase=2 if args.quick else 64,
        deals_per_step=32 if args.quick else 512,
        lr=args.learning_rate,
        batch_size=16 if args.quick else args.batch_size,
        num_epochs=1 if args.quick else 4,
        entropy_coef=args.entropy_coef,
        hidden_dim=1024,
        use_info_bonus=False,
        beta=0.0,
        info_reward_weight=0.0,
        belief_conditioned=False,
        actor_belief_coef=0.0,
        freeze_belief=True,
        kl_lambda_start=0.0,
        kl_lambda_end=0.0,
        kl_anneal_frac=0.0,
        fsp_pool_size=args.fsp_pool_size,
        fsp_add_interval=1,
        self_play=False,
        fsp_quality_gate=args.fsp_quality_gate,
        fsp_gate_eval_deals=50 if args.quick else 200,
        fsp_sl_sample_prob=args.fsp_sl_sample_prob,
        critic_prewarm_deals=64 if args.quick else 2048,
        critic_prewarm_epochs=1 if args.quick else 10,
        device=device,
    )
    trainer = SubgameTrainer(env, config, reward_stats=RunningStats())
    load_policy_checkpoint(trainer.agent, args.sl_checkpoint, device)

    print(f"Device: {device}")
    print(f"Policy observation: {OBS_DIM} dimensions")
    print("Control: plain 571 actor; no belief head; no Judge; no information reward")
    print(
        f"Control settings: entropy={args.entropy_coef} "
        f"fsp_pool_size={args.fsp_pool_size} "
        f"quality_gate={args.fsp_quality_gate}"
    )

    manifest = {
        "status": "running",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "control": "corrected_pipeline_plain_571_v1",
        "config": vars(args),
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logs = trainer.run(num_rounds=args.rounds)
    checkpoint_path = output_dir / f"agent_a_plain571_seed{args.seed}.pt"
    save_plain_checkpoint(trainer, checkpoint_path)

    sl_agent = MAPPOAgent(MAPPOConfig(
        device=device,
        obs_dim=OBS_DIM,
        hidden_dim=1024,
        actor_belief_conditioned=False,
    ))
    load_policy_checkpoint(sl_agent, args.sl_checkpoint, device)
    deals = sample_evaluation_deals(env, args.eval_deals, args.eval_seed)
    result = evaluate_match(
        env,
        deals,
        make_mappo_factory(trainer.agent),
        make_mappo_factory(sl_agent),
    )
    print(
        f"Plain-571 A vs SL: {result.mean_imp:+.3f} +/- {result.std_imp:.3f} IMP "
        f"(95% CI [{result.ci_low:+.3f}, {result.ci_high:+.3f}]); "
        f"{result.wins}/{result.losses}/{result.ties}"
    )

    summary = {
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "control": "corrected_pipeline_plain_571_v1",
        "execution_uses_belief": False,
        "checkpoint": checkpoint_path.name,
        "rounds_logged": len(logs),
        "config": vars(args),
        "plain_571_a_vs_sl": {
            key: value for key, value in asdict(result).items() if key != "imps"
        },
    }
    (output_dir / f"summary_seed{args.seed}.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    manifest.update({
        "status": "complete",
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": checkpoint_path.name,
        "summary": f"summary_seed{args.seed}.json",
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/competitive_500k.npz")
    parser.add_argument("--sl-checkpoint", default="results/sl_base.pt")
    parser.add_argument(
        "--output-dir", default="results/plain_571_control_seed42"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=20260714)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--eval-deals", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--entropy-coef", type=float, default=0.001,
        help="Matches the old successful plain-571 Agent A run.",
    )
    parser.add_argument(
        "--fsp-pool-size", type=int, default=10,
        help="Matches the observed cap in the old plain-571 Agent A log.",
    )
    parser.add_argument("--fsp-sl-sample-prob", type=float, default=0.30)
    parser.add_argument("--fsp-quality-gate", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
