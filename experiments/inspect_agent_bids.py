"""Print detailed deterministic bidding traces for current 571-dim agents.

The report compares two agents on the same constrained deals in both roles:
table 1 assigns Agent A to the opener partnership and Agent B to the
overcaller partnership; table 2 swaps them.  BeliefNet and critics are never
loaded because current checkpoints use them only during training.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import BridgeBiddingEnv, NUM_BIDS, NORTH, EAST, SOUTH, WEST, string_to_bid
from networks.policy_net import (
    MLPPolicyNetwork,
    OBS_DIM,
    convert_hands_suit_to_rank,
    get_openspiel_obs,
    hands_to_openspiel_state,
    openspiel_raw_to_ours,
    ours_to_openspiel_raw,
    physical_to_openspiel_player,
)
from subgames.competitive_env import CompetitiveSubgameEnv, FIXED_PREFIX
from utils.imp import score_to_imp


SEATS = ("N", "E", "S", "W")
RANKS = "23456789TJQKA"
SUITS = ("C", "D", "H", "S")
VULNERABILITIES = (
    (False, False),
    (True, False),
    (False, True),
    (True, True),
)


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
    return "  ".join(chunks)


def vulnerability_name(vulnerability: tuple[bool, bool]) -> str:
    return {
        (False, False): "None",
        (True, False): "NS",
        (False, True): "EW",
        (True, True): "Both",
    }[vulnerability]


class ActorPolicy:
    """The four physical-seat actors from one deployment checkpoint."""

    def __init__(self, label: str, checkpoint_path: Path, device: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        obs_dim = int(checkpoint.get("obs_dim", OBS_DIM))
        hidden_dim = int(checkpoint.get("hidden_dim", 1024))
        if obs_dim != OBS_DIM:
            raise ValueError(
                f"{label}: expected a {OBS_DIM}-dimensional checkpoint, got {obs_dim}"
            )

        self.label = label
        self.device = torch.device(device)
        self.actors = []
        for seat, key in enumerate(("actor_n", "actor_e", "actor_s", "actor_w")):
            if key not in checkpoint:
                raise KeyError(f"{checkpoint_path} does not contain {key}")
            actor = MLPPolicyNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim)
            actor.load_state_dict(checkpoint[key])
            actor.to(self.device).eval()
            self.actors.append(actor)
        del checkpoint

    def choose(
        self, state, player: int, dealer: int
    ) -> tuple[int, list[tuple[int, float]]]:
        observer = physical_to_openspiel_player(player, dealer)
        observation = get_openspiel_obs(state, observer)
        legal_mask = np.zeros(NUM_BIDS, dtype=np.float32)
        for raw_action in state.legal_actions():
            action = openspiel_raw_to_ours(raw_action)
            if 0 <= action < NUM_BIDS:
                legal_mask[action] = 1.0

        obs_t = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        legal_t = torch.as_tensor(
            legal_mask, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits = self.actors[player](obs_t, legal_t)
            probabilities = F.softmax(logits, dim=-1).squeeze(0)
            legal_count = int(legal_mask.sum())
            values, indices = torch.topk(probabilities, k=min(3, legal_count))
        top = [
            (int(action), float(probability))
            for action, probability in zip(indices.cpu(), values.cpu())
        ]
        return top[0][0], top


@dataclass
class BidTrace:
    player: int
    controller: str
    action: int
    top3: list[tuple[int, float]] | None
    selected_action: int | None = None


@dataclass
class TableTrace:
    bids: list[BidTrace]
    contract: object
    score_ns: int
    tricks: int | None


def play_table(
    env: CompetitiveSubgameEnv,
    hands: np.ndarray,
    dd_table: np.ndarray,
    dealer: int,
    vulnerability: tuple[bool, bool],
    opener_policy: ActorPolicy,
    overcaller_policy: ActorPolicy,
) -> TableTrace:
    inner = BridgeBiddingEnv(60)
    inner.reset(hands, dealer=dealer, vulnerability=vulnerability)
    state = hands_to_openspiel_state(
        convert_hands_suit_to_rank(hands),
        dealer,
        vulnerability=vulnerability,
    )
    opener_seats = {dealer, (dealer + 2) % 4}
    bids: list[BidTrace] = []

    for prefix_bid in FIXED_PREFIX:
        action = string_to_bid(prefix_bid)
        player = inner.state.current_player
        raw_action = ours_to_openspiel_raw(action)
        if raw_action not in state.legal_actions():
            raise RuntimeError(f"Fixed-prefix action {prefix_bid} is illegal")
        state.apply_action(raw_action)
        _, _, done, _ = inner.step(action)
        bids.append(BidTrace(player, "fixed", action, None))
        if done:
            break

    done = inner.state.final_contract is not None or inner._check_done()
    while not done:
        player = inner.state.current_player
        policy = opener_policy if player in opener_seats else overcaller_policy
        selected_action, top3 = policy.choose(state, player, dealer)
        # Match CompetitiveSubgameEnv.play_mixed exactly: OpenSpiel supplies
        # the policy mask, while the local auction environment has the final
        # validity check and falls back to Pass when the two disagree.
        action = selected_action
        if not inner._is_valid_action(action):
            action = 0
        raw_action = ours_to_openspiel_raw(action)
        if raw_action not in state.legal_actions():
            raise RuntimeError(
                f"Applied action {bid_name(action)} is illegal for {SEATS[player]}"
            )
        bids.append(
            BidTrace(player, policy.label, action, top3, selected_action)
        )
        state.apply_action(raw_action)
        _, _, done, _ = inner.step(action)

    contract = inner.state.final_contract
    score_ns = env._compute_score_ns(contract, dd_table, vulnerability)
    tricks = None if contract is None else int(dd_table[contract.suit, contract.declarer])
    return TableTrace(bids, contract, score_ns, tricks)


def contract_name(contract) -> str:
    if contract is None:
        return "Passed out"
    strain = ("C", "D", "H", "S", "NT")[contract.suit]
    doubled = ("", "X", "XX")[contract.doubled]
    return f"{contract.level}{strain}{doubled} by {SEATS[contract.declarer]}"


def write_table(handle, title: str, trace: TableTrace) -> None:
    handle.write(f"\n{title}\n")
    handle.write("-" * 88 + "\n")
    handle.write(" No. Seat Model  Bid    Top-3 legal-action probabilities\n")
    for index, bid in enumerate(trace.bids, 1):
        if bid.top3 is None:
            top_text = "fixed experimental prefix"
        else:
            top_text = ", ".join(
                f"{bid_name(action)}={probability:.1%}"
                for action, probability in bid.top3
            )
            if bid.selected_action != bid.action:
                top_text = (
                    f"selected {bid_name(bid.selected_action)} -> local-env fallback Pass; "
                    + top_text
                )
        handle.write(
            f" {index:>3}  {SEATS[bid.player]:>2}   {bid.controller:<5} "
            f"{bid_name(bid.action):>5}   {top_text}\n"
        )

    handle.write(f" Contract: {contract_name(trace.contract)}\n")
    if trace.contract is not None:
        required = 6 + trace.contract.level
        result = (
            f"made with {trace.tricks - required} overtrick(s)"
            if trace.tricks >= required
            else f"down {required - trace.tricks}"
        )
        handle.write(f" DDS tricks: {trace.tricks} ({result})\n")
    handle.write(f" NS score: {trace.score_ns:+d}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-a", default="agent_a_seed42.pt")
    parser.add_argument("--agent-b", default="agent_b_seed42.pt")
    parser.add_argument("--data", default="data/competitive_500k.npz")
    parser.add_argument("--output", default="agent_a_vs_b_50_bidding_traces.txt")
    parser.add_argument("--deals", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def within_project(path: Path) -> bool:
    try:
        path.resolve().relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        return False


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    output_path = (PROJECT_ROOT / args.output).resolve()
    if not within_project(output_path):
        raise ValueError("Output must remain inside the current project directory")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    agent_a = ActorPolicy("A", (PROJECT_ROOT / args.agent_a).resolve(), args.device)
    agent_b = ActorPolicy("B", (PROJECT_ROOT / args.agent_b).resolve(), args.device)
    env = CompetitiveSubgameEnv(str((PROJECT_ROOT / args.data).resolve()))

    imp_values = []
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Agent A vs Agent B: deterministic bidding traces\n")
        handle.write(f"Deals: {args.deals}  Seed: {args.seed}\n")
        handle.write("Each deal is played twice with opener/overcaller roles swapped.\n")
        handle.write("Probabilities are normalized over legal actions only.\n")

        for deal_index in range(1, args.deals + 1):
            hands, dd_table = env.generate_deal()
            dealer = int(env._sampled_dealer)
            vulnerability = VULNERABILITIES[np.random.randint(4)]

            table_1 = play_table(
                env, hands, dd_table, dealer, vulnerability, agent_a, agent_b
            )
            table_2 = play_table(
                env, hands, dd_table, dealer, vulnerability, agent_b, agent_a
            )
            score_difference = table_1.score_ns - table_2.score_ns
            if dealer % 2 == 1:
                score_difference = -score_difference
            imp_a = float(score_to_imp(score_difference))
            imp_values.append(imp_a)

            handle.write("\n\n" + "=" * 88 + "\n")
            handle.write(
                f"DEAL {deal_index:02d}/{args.deals}  Dealer={SEATS[dealer]}  "
                f"Vulnerability={vulnerability_name(vulnerability)}\n"
            )
            handle.write("=" * 88 + "\n")
            for seat in (NORTH, EAST, SOUTH, WEST):
                handle.write(f" {SEATS[seat]}: {hand_name(hands[seat])}\n")

            write_table(handle, "TABLE 1: A opener partnership vs B overcaller partnership", table_1)
            write_table(handle, "TABLE 2: B opener partnership vs A overcaller partnership", table_2)
            handle.write(
                f"\n A-perspective result: {imp_a:+.0f} IMP "
                f"(table scores {table_1.score_ns:+d}, {table_2.score_ns:+d})\n"
            )

        values = np.asarray(imp_values, dtype=np.float64)
        handle.write("\n\n" + "=" * 88 + "\n")
        handle.write("SUMMARY\n")
        handle.write("=" * 88 + "\n")
        handle.write(
            f"A-perspective IMP: mean={values.mean():+.3f}, std={values.std():.3f}\n"
        )
        handle.write(
            f"A wins / B wins / ties: "
            f"{int((values > 0).sum())} / {int((values < 0).sum())} / "
            f"{int((values == 0).sum())}\n"
        )

    print(output_path)


if __name__ == "__main__":
    main()
