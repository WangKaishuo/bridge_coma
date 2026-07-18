"""Write duplicate-auction records for unrestricted resume checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import NORTH, EAST, SOUTH, WEST
from experiments.evaluate_resume_ab import load_resume_agent
from experiments.evaluation import make_mappo_factory, sample_evaluation_deals
from subgames.unrestricted_env import UnrestrictedBiddingEnv
from utils.imp import score_to_imp


SEATS = ("N", "E", "S", "W")
SEAT_NAMES = ("North", "East", "South", "West")
RANKS = "23456789TJQKA"
SUITS = ("C", "D", "H", "S")


def bid_name(action: int) -> str:
    if action == 0:
        return "Pass"
    if action == 1:
        return "X"
    if action == 2:
        return "XX"
    level = (action - 3) // 5 + 1
    strain = ("C", "D", "H", "S", "NT")[(action - 3) % 5]
    return f"{level}{strain}"


def hand_name(hand: np.ndarray) -> str:
    chunks = []
    for suit in range(3, -1, -1):
        cards = "".join(
            RANKS[rank]
            for rank in range(12, -1, -1)
            if hand[suit * 13 + rank] > 0.5
        )
        chunks.append(f"{SUITS[suit]}:{cards or '-'}")
    return " ".join(chunks)


def hand_hcp(hand: np.ndarray) -> int:
    points = {12: 4, 11: 3, 10: 2, 9: 1}
    return sum(
        points.get(rank, 0)
        for suit in range(4)
        for rank in range(13)
        if hand[suit * 13 + rank] > 0.5
    )


def hand_shape(hand: np.ndarray) -> str:
    return "-".join(
        str(int(hand[suit * 13:(suit + 1) * 13].sum()))
        for suit in range(3, -1, -1)
    )


def vulnerability_name(vulnerability: tuple[bool, bool]) -> str:
    return {
        (False, False): "None",
        (True, False): "NS",
        (False, True): "EW",
        (True, True): "Both",
    }[vulnerability]


def contract_name(contract) -> str:
    if contract is None:
        return "Passed out"
    strain = ("C", "D", "H", "S", "NT")[contract.suit]
    doubled = ("", "X", "XX")[contract.doubled]
    return f"{contract.level}{strain}{doubled} by {SEATS[contract.declarer]}"


def write_auction(handle, history: list[int], dealer: int) -> None:
    seat_order = [(dealer + offset) % 4 for offset in range(4)]
    handle.write("  Seat   " + "".join(f"{SEATS[s]:^10}" for s in seat_order) + "\n")
    row = [""] * 4
    for index, action in enumerate(history):
        column = index % 4
        row[column] = bid_name(action)
        if column == 3:
            handle.write("         " + "".join(f"{bid:^10}" for bid in row) + "\n")
            row = [""] * 4
    if any(row):
        handle.write("         " + "".join(f"{bid:^10}" for bid in row) + "\n")


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


def write_table(handle, title, result, deal, dealer_label, other_label) -> None:
    contract, score_ns, history = result
    dealer_side = {deal.dealer, (deal.dealer + 2) % 4}
    seat_order = [(deal.dealer + offset) % 4 for offset in range(4)]
    controllers = [dealer_label if s in dealer_side else other_label for s in seat_order]
    handle.write(f"\n  -- {title} --\n")
    handle.write("  Model  " + "".join(f"{label:^10}" for label in controllers) + "\n")
    write_auction(handle, history, deal.dealer)
    handle.write(f"  Contract: {contract_name(contract)}\n")
    if contract is not None:
        tricks = int(deal.dd_table[contract.suit, contract.declarer])
        required = 6 + contract.level
        result_text = (
            "making" if tricks == required
            else f"making +{tricks - required}" if tricks > required
            else f"down {required - tricks}"
        )
        handle.write(f"  DDS: {tricks} tricks ({result_text})\n")
    handle.write(f"  NS score: {score_ns:+d}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-a", type=Path, required=True)
    parser.add_argument("--agent-b", type=Path, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deals", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"
    agent_a, round_a = load_resume_agent(args.agent_a, device)
    agent_b, round_b = load_resume_agent(args.agent_b, device)
    if round_a != round_b:
        raise ValueError(f"Checkpoint rounds differ: A={round_a}, B={round_b}")

    env = UnrestrictedBiddingEnv(args.data)
    deals = sample_evaluation_deals(env, args.deals, args.seed)
    factory_a = make_mappo_factory(agent_a)
    factory_b = make_mappo_factory(agent_b)
    imp_values = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write("UNRESTRICTED B vs A - DUPLICATE AUCTION RECORDS\n")
        handle.write(f"Checkpoint round: {round_a}    Deals: {args.deals}    Seed: {args.seed}\n")
        handle.write("Table 1 gives B the dealer partnership; Table 2 swaps A/B.\n")
        for index, deal in enumerate(deals, 1):
            table_1 = play_table(env, deal, factory_b, factory_a)
            table_2 = play_table(env, deal, factory_a, factory_b)
            score_difference = table_1[1] - table_2[1]
            if deal.dealer % 2 == 1:
                score_difference = -score_difference
            imp_b = float(score_to_imp(score_difference))
            imp_values.append(imp_b)

            handle.write("\n\n" + "=" * 78 + "\n")
            handle.write(
                f"DEAL {index:03d}/{args.deals}    Dealer: {SEAT_NAMES[deal.dealer]}    "
                f"Vulnerability: {vulnerability_name(deal.vulnerability)}\n"
            )
            handle.write("=" * 78 + "\n")
            for seat in (NORTH, EAST, SOUTH, WEST):
                handle.write(
                    f"  {SEAT_NAMES[seat]:<5}: {hand_name(deal.hands[seat]):<40} "
                    f"({hand_hcp(deal.hands[seat]):>2} HCP, {hand_shape(deal.hands[seat])})\n"
                )
            write_table(handle, "TABLE 1: B dealer side vs A", table_1, deal, "B", "A")
            write_table(handle, "TABLE 2: A dealer side vs B", table_2, deal, "A", "B")
            outcome = "tie" if imp_b == 0 else ("B wins" if imp_b > 0 else "A wins")
            handle.write(
                f"\n  IMP (B perspective): {imp_b:+.0f} "
                f"(T1={table_1[1]:+d}, T2={table_2[1]:+d}) -- {outcome}\n"
            )

        values = np.asarray(imp_values, dtype=np.float64)
        handle.write("\n\n" + "=" * 78 + "\n")
        handle.write(f"B vs A TRACE SUMMARY ({args.deals} deals)\n")
        handle.write("=" * 78 + "\n")
        handle.write(f"B-perspective IMP: mean={values.mean():+.3f}, std={values.std():.3f}\n")
        handle.write(
            f"B wins / A wins / ties: {int((values > 0).sum())} / "
            f"{int((values < 0).sum())} / {int((values == 0).sum())}\n"
        )
    print(args.output)


if __name__ == "__main__":
    main()
