"""Measure frozen-Judge information gains in an unrestricted B-vs-A match.

Every deal is played twice with the A/B partnerships swapped.  For every
public call, the script reconstructs the exact receiver observations used by
training and measures the signed loss reduction for the bidder's hidden hand.
Statistics are clustered by paired deal; individual calls are not treated as
independent observations.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from experiments.evaluate_resume_ab import load_resume_agent
from experiments.evaluation import (
    auction_is_competitive,
    make_mappo_factory,
    sample_evaluation_deals,
)
from networks.belief_net import BeliefNetwork, DualInfoComputer
from networks.policy_net import (
    encode_openspiel_auction_observation,
    physical_to_openspiel_player,
)
from subgames.unrestricted_env import UnrestrictedBiddingEnv
from utils.hand_features import hand_to_belief_target
from utils.imp import score_to_imp


METRICS = ("partner_gain", "opponent_gain", "secrecy_difference", "beta_005_bonus")
STRATA = ("overall", "competitive", "non_competitive", "mixed")


def load_judge(path: Path, device: str) -> BeliefNetwork:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("belief_net", checkpoint)
    hidden_dim = int(state["trunk.0.weight"].shape[0])
    judge = BeliefNetwork(hidden_dim=hidden_dim).to(device)
    judge.load_state_dict(state)
    judge.eval()
    for parameter in judge.parameters():
        parameter.requires_grad_(False)
    return judge


def play_table(env, deal, dealer_factory, other_factory):
    dealer_policy = dealer_factory(deal.hands, deal.dealer, deal.vulnerability)
    other_policy = other_factory(deal.hands, deal.dealer, deal.vulnerability)
    return env.play_mixed(
        deal.hands,
        deal.dd_table,
        opener_policy=dealer_policy.act,
        overcaller_policy=other_policy.act,
        vulnerability=deal.vulnerability,
        dealer=deal.dealer,
    )


def append_table_records(records, deal_index, deal, history, dealer_label, other_label):
    dealer_side = {deal.dealer, (deal.dealer + 2) % 4}
    for step_index, action in enumerate(history):
        player = (deal.dealer + step_index) % 4
        partner = (player + 2) % 4
        opponent = (player + 1) % 4
        before_history = history[:step_index]
        after_history = history[:step_index + 1]
        records.append({
            "deal_index": deal_index,
            "model": dealer_label if player in dealer_side else other_label,
            "action": int(action),
            "partner_before": encode_openspiel_auction_observation(
                deal.hands, deal.dealer, before_history, partner, deal.vulnerability
            ),
            "partner_after": encode_openspiel_auction_observation(
                deal.hands, deal.dealer, after_history, partner, deal.vulnerability
            ),
            "opponent_before": encode_openspiel_auction_observation(
                deal.hands, deal.dealer, before_history, opponent, deal.vulnerability
            ),
            "opponent_after": encode_openspiel_auction_observation(
                deal.hands, deal.dealer, after_history, opponent, deal.vulnerability
            ),
            "target": hand_to_belief_target(deal.hands[player]),
            "target_pos": physical_to_openspiel_player(player, deal.dealer),
        })


def score_records(records, judge, device: str) -> None:
    computer = DualInfoComputer(judge, beta=0.05)
    chunk_size = 2048
    with torch.no_grad():
        for start in range(0, len(records), chunk_size):
            chunk = records[start:start + chunk_size]
            partner_before = torch.as_tensor(
                np.stack([row["partner_before"] for row in chunk]),
                dtype=torch.float32,
                device=device,
            )
            partner_after = torch.as_tensor(
                np.stack([row["partner_after"] for row in chunk]),
                dtype=torch.float32,
                device=device,
            )
            opponent_before = torch.as_tensor(
                np.stack([row["opponent_before"] for row in chunk]),
                dtype=torch.float32,
                device=device,
            )
            opponent_after = torch.as_tensor(
                np.stack([row["opponent_after"] for row in chunk]),
                dtype=torch.float32,
                device=device,
            )
            targets = torch.as_tensor(
                np.stack([row["target"] for row in chunk]),
                dtype=torch.float32,
                device=device,
            )
            target_pos = torch.as_tensor(
                [row["target_pos"] for row in chunk],
                dtype=torch.long,
                device=device,
            )
            before = torch.cat([partner_before, opponent_before], dim=0)
            after = torch.cat([partner_after, opponent_after], dim=0)
            positions = torch.cat([target_pos, target_pos], dim=0)
            probs_before = judge.get_probs(before, positions)
            probs_after = judge.get_probs(after, positions)
            count = len(chunk)
            partner = computer.compute_info_gain(
                probs_before[:count], probs_after[:count], targets
            ).cpu().numpy()
            opponent = computer.compute_info_gain(
                probs_before[count:], probs_after[count:], targets
            ).cpu().numpy()
            for row, partner_gain, opponent_gain in zip(chunk, partner, opponent):
                row["partner_gain"] = float(partner_gain)
                row["opponent_gain"] = float(opponent_gain)
                row["secrecy_difference"] = float(partner_gain - opponent_gain)
                row["beta_005_bonus"] = float(partner_gain - 0.05 * opponent_gain)
                for key in (
                    "partner_before", "partner_after", "opponent_before",
                    "opponent_after", "target",
                ):
                    del row[key]


def summarize(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    mean = float(values.mean()) if count else 0.0
    std = float(values.std(ddof=1)) if count > 1 else 0.0
    se = std / np.sqrt(count) if count else 0.0
    return {
        "count": count,
        "mean": mean,
        "std": std,
        "standard_error": se,
        "ci_low": mean - 1.96 * se,
        "ci_high": mean + 1.96 * se,
    }


def aggregate(records, deal_strata, deal_count: int) -> dict:
    per_deal = {
        model: {metric: np.zeros(deal_count, dtype=np.float64) for metric in METRICS}
        for model in ("A", "B")
    }
    call_counts = {model: np.zeros(deal_count, dtype=np.int64) for model in ("A", "B")}
    positive_counts = {
        model: {metric: 0 for metric in METRICS} for model in ("A", "B")
    }
    call_sums = {model: {metric: 0.0 for metric in METRICS} for model in ("A", "B")}

    for row in records:
        model = row["model"]
        index = row["deal_index"]
        call_counts[model][index] += 1
        for metric in METRICS:
            value = row[metric]
            per_deal[model][metric][index] += value
            call_sums[model][metric] += value
            positive_counts[model][metric] += int(value > 0)

    output = {"by_stratum": {}, "paired_b_minus_a": {}}
    for stratum in STRATA:
        indices = np.asarray([
            i for i, value in enumerate(deal_strata)
            if stratum == "overall" or value == stratum
        ], dtype=np.int64)
        output["by_stratum"][stratum] = {}
        output["paired_b_minus_a"][stratum] = {}
        for model in ("A", "B"):
            model_row = {
                "deals": int(len(indices)),
                "calls": int(call_counts[model][indices].sum()),
                "mean_calls_per_deal": float(call_counts[model][indices].mean())
                if len(indices) else 0.0,
                "metrics": {},
            }
            for metric in METRICS:
                totals = per_deal[model][metric][indices]
                call_count = int(call_counts[model][indices].sum())
                model_row["metrics"][metric] = {
                    "per_deal_total": summarize(totals),
                    "mean_per_call": float(totals.sum() / max(call_count, 1)),
                }
                if stratum == "overall":
                    model_row["metrics"][metric]["positive_call_fraction"] = (
                        positive_counts[model][metric]
                        / max(int(call_counts[model].sum()), 1)
                    )
            output["by_stratum"][stratum][model] = model_row
        for metric in METRICS:
            delta = (
                per_deal["B"][metric][indices]
                - per_deal["A"][metric][indices]
            )
            output["paired_b_minus_a"][stratum][metric] = summarize(delta)

    for model in ("A", "B"):
        p = output["by_stratum"]["overall"][model]["metrics"]["partner_gain"]["mean_per_call"]
        o = output["by_stratum"]["overall"][model]["metrics"]["opponent_gain"]["mean_per_call"]
        output["by_stratum"]["overall"][model]["partner_to_opponent_ratio"] = (
            p / o if abs(o) > 1e-12 else None
        )
    return output


def compact_match(result) -> dict:
    return {key: value for key, value in asdict(result).items() if key != "imps"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-a", type=Path, required=True)
    parser.add_argument("--agent-b", type=Path, required=True)
    parser.add_argument("--judge-checkpoint", type=Path, required=True)
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
    judge = load_judge(args.judge_checkpoint, device)

    env = UnrestrictedBiddingEnv(args.data)
    deals = sample_evaluation_deals(env, args.deals, args.seed)
    factory_a = make_mappo_factory(agent_a)
    factory_b = make_mappo_factory(agent_b)

    records = []
    deal_strata = []
    imp_values = []
    for deal_index, deal in enumerate(deals):
        table_ba = play_table(env, deal, factory_b, factory_a)
        table_ab = play_table(env, deal, factory_a, factory_b)
        append_table_records(records, deal_index, deal, table_ba[2], "B", "A")
        append_table_records(records, deal_index, deal, table_ab[2], "A", "B")

        comp_ba = auction_is_competitive(table_ba[2], deal.dealer)
        comp_ab = auction_is_competitive(table_ab[2], deal.dealer)
        if comp_ba and comp_ab:
            deal_strata.append("competitive")
        elif not comp_ba and not comp_ab:
            deal_strata.append("non_competitive")
        else:
            deal_strata.append("mixed")

        difference = table_ba[1] - table_ab[1]
        if deal.dealer % 2 == 1:
            difference = -difference
        imp_values.append(float(score_to_imp(difference)))

        if (deal_index + 1) % 250 == 0:
            print(f"played {deal_index + 1}/{args.deals} deals", flush=True)

    score_records(records, judge, device)
    values = np.asarray(imp_values, dtype=np.float64)
    match = summarize(values)
    match.update({
        "wins": int((values > 0).sum()),
        "losses": int((values < 0).sum()),
        "ties": int((values == 0).sum()),
    })
    payload = {
        "checkpoint_round": round_a,
        "deals": args.deals,
        "seed": args.seed,
        "judge_checkpoint": str(args.judge_checkpoint),
        "measurement": (
            "Raw signed Judge loss reduction per actual call in duplicate "
            "B-vs-A cross-play; receiver observations match training."
        ),
        "b_vs_a_match_sanity_check": match,
        "judge_information": aggregate(records, deal_strata, args.deals),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
