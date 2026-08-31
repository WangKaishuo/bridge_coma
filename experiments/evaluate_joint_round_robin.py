"""Evaluate a policy set jointly and retain paired deal-level IMP vectors."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.mappo import MAPPOAgent, MAPPOConfig
from experiments.evaluate_resume_ab import load_resume_agent
from experiments.evaluation import (
    EvaluationDeal,
    evaluate_match_stratified,
    make_mappo_factory,
    sample_evaluation_deals,
)
from experiments.subgame_validation import load_policy_checkpoint
from networks.policy_net import OBS_DIM
from subgames.unrestricted_env import UnrestrictedBiddingEnv
from utils.dds_data import deck_to_hands


def compact(result) -> dict:
    return {key: value for key, value in asdict(result).items() if key != "imps"}


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    se = std / np.sqrt(max(len(values), 1))
    mean = float(values.mean()) if len(values) else 0.0
    return {
        "deals": int(len(values)),
        "mean_imp": mean,
        "std_imp": std,
        "standard_error": se,
        "ci_low": mean - 1.96 * se,
        "ci_high": mean + 1.96 * se,
        "wins": int((values > 0).sum()),
        "losses": int((values < 0).sum()),
        "ties": int((values == 0).sum()),
    }


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=CHECKPOINT")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=CHECKPOINT")
    return name, Path(raw_path)


def safe_key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def sample_evaluation_deals_without_replacement(
    env: UnrestrictedBiddingEnv,
    count: int,
    seed: int,
) -> tuple[list[EvaluationDeal], np.ndarray]:
    """Select exact memmap rows once and randomize public board context."""
    records = getattr(env.loader, "records", None)
    if records is None:
        raise ValueError("without-replacement evaluation requires memmap DDS data")
    if count < 1 or count > len(records):
        raise ValueError(
            f"deal count must be in [1,{len(records)}], got {count}"
        )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(records), size=count, replace=False)
    selected = records[indices]
    hands = deck_to_hands(np.asarray(selected["decks"]))
    tricks = np.asarray(selected["tricks"])
    dealers = rng.integers(0, 4, size=count)
    vulnerability_indices = rng.integers(0, 4, size=count)
    vulnerabilities = (
        (False, False), (True, False), (False, True), (True, True)
    )
    deals = [
        EvaluationDeal(
            hands=hands[i],
            dd_table=tricks[i].copy(),
            dealer=int(dealers[i]),
            vulnerability=vulnerabilities[int(vulnerability_indices[i])],
        )
        for i in range(count)
    ]
    return deals, np.asarray(indices, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent",
        action="append",
        type=parse_named_path,
        required=True,
        help="Named resume checkpoint, as NAME=CHECKPOINT; repeat per policy",
    )
    parser.add_argument("--sl-checkpoint", type=Path)
    parser.add_argument("--sl-name", default="SL")
    parser.add_argument("--data", required=True)
    parser.add_argument("--deals", type=int, default=20000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--allow-round-mismatch", action="store_true")
    parser.add_argument(
        "--without-replacement",
        action="store_true",
        help="Sample unique rows from a memmap DDS dataset and retain indices",
    )
    args = parser.parse_args()

    names = [name for name, _ in args.agent]
    if len(names) != len(set(names)):
        raise ValueError("agent names must be unique")
    if args.sl_checkpoint and args.sl_name in names:
        raise ValueError("SL name duplicates a trained-agent name")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    factories = {}
    rounds: dict[str, int | None] = {}
    sources: dict[str, str] = {}
    for name, path in args.agent:
        agent, checkpoint_round = load_resume_agent(path, device)
        factories[name] = make_mappo_factory(agent)
        rounds[name] = checkpoint_round
        sources[name] = str(path)

    trained_rounds = {value for value in rounds.values() if value is not None}
    if len(trained_rounds) > 1 and not args.allow_round_mismatch:
        raise ValueError(f"checkpoint rounds differ: {rounds}")

    if args.sl_checkpoint:
        sl_agent = MAPPOAgent(MAPPOConfig(
            device=device,
            obs_dim=OBS_DIM,
            hidden_dim=1024,
            actor_belief_conditioned=False,
        ))
        metadata = load_policy_checkpoint(
            sl_agent, str(args.sl_checkpoint), device
        )
        for actor in (
            sl_agent.model.actor_n,
            sl_agent.model.actor_e,
            sl_agent.model.actor_s,
            sl_agent.model.actor_w,
        ):
            actor.eval()
        factories[args.sl_name] = make_mappo_factory(sl_agent)
        rounds[args.sl_name] = None
        sources[args.sl_name] = str(args.sl_checkpoint)
        sl_iteration = metadata.get("iteration")
    else:
        sl_iteration = None

    policy_names = list(factories)
    env = UnrestrictedBiddingEnv(args.data)
    if args.without_replacement:
        deals, deal_indices = sample_evaluation_deals_without_replacement(
            env, args.deals, args.seed
        )
    else:
        deals = sample_evaluation_deals(env, args.deals, args.seed)
        deal_indices = None
    vectors: dict[tuple[str, str], np.ndarray] = {}
    pair_payload = {}
    npz_payload = {}

    for first, second in combinations(policy_names, 2):
        result = evaluate_match_stratified(
            env, deals, factories[first], factories[second]
        )
        vector = np.asarray(result.overall.imps, dtype=np.float32)
        vectors[(first, second)] = vector
        key = f"{safe_key(first)}_minus_{safe_key(second)}"
        npz_payload[key] = vector
        pair_payload[key] = {
            "first": first,
            "second": second,
            "orientation": f"{first} minus {second}",
            "overall": compact(result.overall),
            "competitive": compact(result.competitive),
            "non_competitive": compact(result.non_competitive),
            "mixed": compact(result.mixed),
        }
        print(
            f"{first} - {second}: "
            f"{result.overall.mean_imp:+.5f} "
            f"[{result.overall.ci_low:+.5f},"
            f"{result.overall.ci_high:+.5f}]",
            flush=True,
        )

    def vector(first: str, second: str) -> np.ndarray:
        if first == second:
            return np.zeros(args.deals, dtype=np.float32)
        if (first, second) in vectors:
            return vectors[(first, second)]
        return -vectors[(second, first)]

    triple_payload = {}
    for first, second, third in combinations(policy_names, 3):
        residual = (
            vector(first, second)
            + vector(second, third)
            - vector(first, third)
        )
        key = (
            f"{safe_key(first)}__{safe_key(second)}__{safe_key(third)}"
        )
        npz_payload[f"residual__{key}"] = residual
        triple_payload[key] = {
            "first": first,
            "second": second,
            "third": third,
            "formula": (
                f"V({first},{second}) + V({second},{third}) "
                f"- V({first},{third})"
            ),
            **summarize(residual),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "deal_level_imps.npz", **npz_payload)
    if deal_indices is not None:
        np.save(args.output_dir / "deal_indices.npy", deal_indices)
    payload = {
        "deals": args.deals,
        "seed": args.seed,
        "sampling": (
            "memmap_without_replacement"
            if args.without_replacement
            else "legacy_seeded_with_replacement"
        ),
        "deal_indices_file": (
            "deal_indices.npy" if deal_indices is not None else None
        ),
        "device": device,
        "policy_order": policy_names,
        "checkpoint_rounds": rounds,
        "sources": sources,
        "sl_iteration": sl_iteration,
        "pairs": pair_payload,
        "transitivity_residuals": triple_payload,
    }
    temporary = args.output_dir / "round_robin.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output_dir / "round_robin.json")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
