#!/usr/bin/env python3
"""
Bidding Inspector — P105 (OpenSpiel-native observations)
==========================================================

Compare RL agent vs SL on specific deals, or run SL-only diagnostics.

Uses pyspiel.State.observation_tensor() for all observation generation.
Handles card encoding conversion for competitive_env deals (suit-major → rank-major).

Usage:
  # SL-only mode (diagnose SL quality on competitive deals)
  python bid_inspector.py \
      --sl results/sl_base.pt \
      --data data/competitive_500k.npz \
      --sl_only --num_deals 10

  # Agent vs SL comparison
  python bid_inspector.py \
      --agent results/agent_a.pt \
      --sl results/sl_base.pt \
      --data data/competitive_500k.npz \
      --num_deals 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F
import pyspiel

from env import (
    BridgeBiddingEnv, NUM_BIDS, NUM_PLAYERS,
    BID_PASS, BID_DOUBLE, BID_REDOUBLE, BID_1C,
    bid_to_string, string_to_bid,
    NORTH, EAST, SOUTH, WEST,
)
from networks.policy_net import (
    MLPPolicyNetwork, OBS_DIM,
    openspiel_raw_to_ours, ours_to_openspiel_raw,
    convert_hands_suit_to_rank, hands_to_openspiel_state,
    get_openspiel_obs, advance_openspiel_state,
    load_sl_into_mappo_agent,
)
from subgames.competitive_env import CompetitiveSubgameEnv, FIXED_PREFIX
from subgames.action_mask import count_hcp, count_suit_length
from utils.imp import score_to_imp
from utils.scoring import Contract, calculate_score
from algorithms.mappo import MAPPOAgent, MAPPOConfig


# ── Constants ────────────────────────────────────────────────────────────────
PLAYER_NAMES = ['North', 'East', 'South', 'West']
PLAYER_SHORT = ['N', 'E', 'S', 'W']
SUIT_SYMBOLS = ['♣', '♦', '♥', '♠']
RANK_CHARS   = "23456789TJQKA"


# ==============================================================================
# Hand display (suit-major encoding from competitive_env)
# ==============================================================================

def hand_to_str(hand: np.ndarray) -> str:
    """(52,) binary hand → readable string. Assumes suit-major encoding."""
    parts = []
    for suit in range(3, -1, -1):  # S, H, D, C
        cards = []
        for rank in range(12, -1, -1):  # A, K, Q, ... 2
            if hand[suit * 13 + rank] > 0.5:
                cards.append(RANK_CHARS[rank])
        parts.append(f"{SUIT_SYMBOLS[suit]}{''.join(cards)}")
    return ' '.join(parts)


def hand_summary(hand: np.ndarray) -> str:
    """HCP and shape distribution."""
    hcp = count_hcp(hand)
    lengths = [count_suit_length(hand, s) for s in range(4)]  # C, D, H, S
    dist = f"{lengths[3]}-{lengths[2]}-{lengths[1]}-{lengths[0]}"
    return f"{hcp} HCP, {dist}"


def display_deal(hands: np.ndarray, dealer: int):
    """Print all four hands."""
    print(f"\n{'─' * 55}")
    print(f"  Dealer: {PLAYER_NAMES[dealer]}")
    print(f"{'─' * 55}")
    for p in range(4):
        name = PLAYER_NAMES[p]
        hstr = hand_to_str(hands[p])
        summary = hand_summary(hands[p])
        marker = (" ← opener" if p == dealer else
                  " ← overcaller" if p == (dealer + 1) % 4 else "")
        print(f"  {name:6s}: {hstr}  ({summary}){marker}")
    print(f"{'─' * 55}")


# ==============================================================================
# Load models
# ==============================================================================

def load_agent(path: str, device: str) -> MAPPOAgent:
    """Load RL agent checkpoint."""
    agent = MAPPOAgent(MAPPOConfig(device=device, obs_dim=OBS_DIM))
    agent.load(path)
    return agent


def load_sl(path: str, device: str) -> MAPPOAgent:
    """Load SL checkpoint as MAPPOAgent."""
    agent = MAPPOAgent(MAPPOConfig(device=device, obs_dim=OBS_DIM))
    load_sl_into_mappo_agent(agent, path)
    return agent


# ==============================================================================
# Policy wrappers using OpenSpiel state
# ==============================================================================

def make_policy_with_probs_openspiel(agent: MAPPOAgent, device: str):
    """
    Wrap agent as policy function that uses an OpenSpiel state for observations.

    Returns function: (os_state, player) → (our_action, top3)
    where os_state is a pyspiel.State.
    """
    def policy(os_state, player):
        obs = get_openspiel_obs(os_state)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        # Legal actions from OpenSpiel state
        legal_os = os_state.legal_actions()
        legal_mask = torch.zeros(1, NUM_BIDS, dtype=torch.float32, device=device)
        for os_a in legal_os:
            ours = openspiel_raw_to_ours(os_a)
            if ours >= 0:
                legal_mask[0, ours] = 1.0

        actor = agent.get_actor(player)
        with torch.no_grad():
            logits = actor(obs_t, legal_mask)
            probs  = F.softmax(logits, dim=-1)[0]
            action = logits.argmax(dim=-1).item()

            topk = torch.topk(probs, min(3, NUM_BIDS))
            top3 = [(bid_to_string(topk.indices[i].item()),
                     topk.values[i].item())
                    for i in range(len(topk.indices))]

        return action, top3
    return policy


# ==============================================================================
# Play deal using OpenSpiel state
# ==============================================================================

def play_deal_sl_only(
    hands_sm: np.ndarray,
    dd_table:  np.ndarray,
    dealer:    int,
    sl:        MAPPOAgent,
    device:    str,
    verbose:   bool = True,
):
    """
    Play a deal with SL model for all 4 seats, using OpenSpiel observations.

    hands_sm: (4, 52) in suit-major encoding (from competitive_env)
    """
    # Convert to rank-major and create OpenSpiel state
    hands_rm = convert_hands_suit_to_rank(hands_sm)
    os_state = hands_to_openspiel_state(hands_rm, dealer)

    # Verify dealer matches
    os_dealer = os_state.current_player()

    policy_fn = make_policy_with_probs_openspiel(sl, device)

    # Apply fixed prefix (1H → 1S)
    prefix_bids = [string_to_bid(b) for b in FIXED_PREFIX]
    for bid in prefix_bids:
        os_action = ours_to_openspiel_raw(bid)
        os_state.apply_action(os_action)

    bid_log = []  # [(player, bid_str, top3)]

    while not os_state.is_terminal():
        # --- ADDED: Check if bidding phase has ended and play phase has begun ---
        legal_os = os_state.legal_actions()
        if len(legal_os) > 0 and legal_os[0] < 52:
            break
        # ------------------------------------------------------------------------

        player = os_state.current_player()
        action, top3 = policy_fn(os_state, player)

        bid_log.append((player, bid_to_string(action), top3))
        os_action = ours_to_openspiel_raw(action)
        os_state.apply_action(os_action)

    # Get contract from BridgeBiddingEnv (for scoring compatibility)
    # Replay in our env to extract contract
    inner_env = BridgeBiddingEnv(max_history_len=60)
    obs = inner_env.reset(hands_sm, dealer=dealer)

    all_bids = prefix_bids + [string_to_bid(b[1]) for b in bid_log]
    for bid in all_bids:
        if not inner_env._is_valid_action(bid):
            bid = BID_PASS
        obs, _, done, _ = inner_env.step(bid)
        if done:
            break

    contract = inner_env.state.final_contract
    return {
        'bid_log': bid_log,
        'contract': contract,
        'all_bids': all_bids,
    }


def play_deal(
    hands_sm: np.ndarray,
    dd_table:  np.ndarray,
    dealer:    int,
    agent:     MAPPOAgent,
    sl:        MAPPOAgent,
    device:    str,
    verbose:   bool = True,
):
    """Play deal with both agent and SL, compare results."""
    vul = (False, False)
    results = {}

    for label, model in [("Agent", agent), ("SL", sl)]:
        hands_rm = convert_hands_suit_to_rank(hands_sm)
        os_state = hands_to_openspiel_state(hands_rm, dealer)
        policy_fn = make_policy_with_probs_openspiel(model, device)

        # Apply fixed prefix
        prefix_bids = [string_to_bid(b) for b in FIXED_PREFIX]
        for bid in prefix_bids:
            os_action = ours_to_openspiel_raw(bid)
            os_state.apply_action(os_action)

        bid_log = []
        while not os_state.is_terminal():
            # --- ADDED: Check if bidding phase has ended and play phase has begun ---
            legal_os = os_state.legal_actions()
            if len(legal_os) > 0 and legal_os[0] < 52:
                break
            # ------------------------------------------------------------------------

            player = os_state.current_player()
            action, top3 = policy_fn(os_state, player)
            bid_log.append((player, bid_to_string(action), top3))
            os_action = ours_to_openspiel_raw(action)
            os_state.apply_action(os_action)

        # Extract contract via BridgeBiddingEnv
        inner_env = BridgeBiddingEnv(max_history_len=60)
        obs = inner_env.reset(hands_sm, dealer=dealer, vulnerability=vul)
        all_bids_int = prefix_bids + [string_to_bid(b[1]) for b in bid_log]
        for bid in all_bids_int:
            if not inner_env._is_valid_action(bid):
                bid = BID_PASS
            obs, _, done, _ = inner_env.step(bid)
            if done:
                break

        results[label] = {
            'bid_log': bid_log,
            'history': all_bids_int,
            'contract': inner_env.state.final_contract,
        }

    if verbose:
        _print_comparison(hands_sm, dealer, dd_table, results, vul)

    return results


# ==============================================================================
# Display
# ==============================================================================

def _print_sl_solo(hands, dealer, dd_table, result, vul):
    """Print SL-only result."""
    bid_log = result['bid_log']
    contract = result['contract']

    print(f"\n{'═' * 65}")
    print(f"  SL SOLO TABLE")
    print(f"{'═' * 65}")

    print(f"\n  Fixed prefix: {PLAYER_SHORT[dealer]}:1♥ → "
          f"{PLAYER_SHORT[(dealer+1)%4]}:1♠")

    # Print bidding sequence
    prefix_strs = [bid_to_string(string_to_bid(b)) for b in FIXED_PREFIX]
    all_bids_str = prefix_strs + [b[1] for b in bid_log]

    print(f"\n  ── SL (all 4 seats) ──")
    header = "  "
    for i in range(4):
        p = (dealer + i) % 4
        header += f"{PLAYER_SHORT[p]:>8s}"
    print(header)

    row = "  "
    for i, bid_str in enumerate(all_bids_str):
        row += f"{bid_str:>8s}"
        if (i + 1) % 4 == 0:
            print(row)
            row = "  "
    if len(row.strip()) > 0:
        print(row)

    if contract:
        suit_names = ['♣', '♦', '♥', '♠', 'NT']
        suit_str = suit_names[contract.suit]
        declarer_str = PLAYER_NAMES[contract.declarer]
        doubled_str = ['', ' X', ' XX'][contract.doubled]
        print(f"  Contract: {contract.level}{suit_str}{doubled_str} by {declarer_str}")

        tricks = int(dd_table[contract.suit, contract.declarer])
        target = 6 + contract.level
        result_str = f"making {tricks}" if tricks >= target else f"down {target - tricks}"
        score = calculate_score(contract, tricks, vul)
        print(f"  DDS {tricks} tricks ({result_str}), NS score: {score:+d}")
    else:
        print(f"  Contract: Passed out")

    # Step-by-step top-3
    print(f"\n  ── Step-by-step top-3 ──")
    for i, (player, bid_str, top3) in enumerate(bid_log):
        top3_str = ", ".join(f"{b}({p:.1%})" for b, p in top3)
        print(f"    Step {i+1:2d} {PLAYER_SHORT[player]}:  "
              f"{bid_str:>5s}  ({top3_str})")
    print(f"{'═' * 65}")


def _print_comparison(hands, dealer, dd_table, results, vul):
    """Print comparison between Agent and SL."""
    print(f"\n{'═' * 65}")
    print(f"  BIDDING COMPARISON")
    print(f"{'═' * 65}")

    print(f"\n  Fixed prefix: {PLAYER_SHORT[dealer]}:1♥ → "
          f"{PLAYER_SHORT[(dealer+1)%4]}:1♠")

    for label in ["Agent", "SL"]:
        r = results[label]
        bid_log = r['bid_log']
        contract = r['contract']

        print(f"\n  ── {label} ──")
        prefix_strs = [bid_to_string(string_to_bid(b)) for b in FIXED_PREFIX]
        all_bids = prefix_strs + [b[1] for b in bid_log]

        header = "  "
        for i in range(4):
            p = (dealer + i) % 4
            header += f"{PLAYER_SHORT[p]:>8s}"
        print(header)

        row = "  "
        for i, bid_str in enumerate(all_bids):
            row += f"{bid_str:>8s}"
            if (i + 1) % 4 == 0:
                print(row)
                row = "  "
        if len(row.strip()) > 0:
            print(row)

        if contract:
            suit_names = ['♣', '♦', '♥', '♠', 'NT']
            suit_str = suit_names[contract.suit]
            declarer_str = PLAYER_NAMES[contract.declarer]
            doubled_str = ['', ' X', ' XX'][contract.doubled]
            print(f"  Contract: {contract.level}{suit_str}{doubled_str} "
                  f"by {declarer_str}")
            tricks = int(dd_table[contract.suit, contract.declarer])
            target = 6 + contract.level
            result_str = (f"making {tricks}" if tricks >= target
                          else f"down {target - tricks}")
            score = calculate_score(contract, tricks, vul)
            print(f"  DDS: {tricks} tricks ({result_str}), Score: {score:+d}")
        else:
            print(f"  Contract: Passed out")

    # Divergence check
    agent_log = results["Agent"]["bid_log"]
    sl_log    = results["SL"]["bid_log"]
    diverged = False
    for i in range(max(len(agent_log), len(sl_log))):
        a_bid = agent_log[i][1] if i < len(agent_log) else "—"
        s_bid = sl_log[i][1]    if i < len(sl_log)    else "—"
        if a_bid != s_bid and not diverged:
            diverged = True
            step = i + len(FIXED_PREFIX)
            player = agent_log[i][0] if i < len(agent_log) else sl_log[i][0]
            print(f"\n  ⚡ DIVERGENCE at step {step+1} ({PLAYER_NAMES[player]})")
    if not diverged:
        print("  ✅ Agent and SL made identical bids.")

    # IMP
    a_c = results["Agent"]["contract"]
    s_c = results["SL"]["contract"]
    if a_c and s_c:
        a_t = int(dd_table[a_c.suit, a_c.declarer])
        s_t = int(dd_table[s_c.suit, s_c.declarer])
        a_s = calculate_score(a_c, a_t, vul)
        s_s = calculate_score(s_c, s_t, vul)
        imp = score_to_imp(a_s - s_s)
        print(f"\n  IMP (Agent): {imp:+.0f}  "
              f"(Agent={a_s:+d}, SL={s_s:+d})")
    print(f"{'═' * 65}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Bidding Inspector (P105)")
    parser.add_argument('--agent', default=None, help='RL agent checkpoint')
    parser.add_argument('--sl', required=True, help='SL checkpoint')
    parser.add_argument('--data', required=True, help='Competitive data (npz)')
    parser.add_argument('--num_deals', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sl_only', action='store_true',
                        help='SL-only mode (no agent needed)')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    if not args.sl_only and args.agent is None:
        parser.error("--agent is required unless --sl_only is specified")

    print(f"[Bid Inspector] Loading SL: {args.sl}")
    sl = load_sl(args.sl, device)

    agent = None
    if not args.sl_only:
        print(f"[Bid Inspector] Loading agent: {args.agent}")
        agent = load_agent(args.agent, device)

    print(f"[Bid Inspector] Loading deals: {args.data}")
    env = CompetitiveSubgameEnv(args.data)

    mode_str = "SL solo (4 × SL, single table)" if args.sl_only else "Agent vs SL"
    print(f"[Bid Inspector] Mode: {mode_str}\n")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    vul = (False, False)
    scores = []

    for deal_idx in range(args.num_deals):
        hands, dd_table = env.generate_deal()
        dealer = env._sampled_dealer

        print(f"\n\n{'▓' * 65}")
        print(f"  DEAL {deal_idx + 1}/{args.num_deals}")
        print(f"{'▓' * 65}")
        display_deal(hands, dealer)

        if args.sl_only:
            result = play_deal_sl_only(hands, dd_table, dealer, sl, device)
            _print_sl_solo(hands, dealer, dd_table, result, vul)

            contract = result['contract']
            if contract:
                tricks = int(dd_table[contract.suit, contract.declarer])
                score = calculate_score(contract, tricks, vul)
                scores.append(score)
        else:
            results = play_deal(hands, dd_table, dealer, agent, sl, device)

    # Summary
    print(f"\n\n{'█' * 65}")
    if args.sl_only:
        print(f"  SL SOLO SUMMARY ({args.num_deals} deals)")
        print(f"{'█' * 65}")
        if scores:
            scores_arr = np.array(scores)
            print(f"  NS score: mean={scores_arr.mean():+.1f}  "
                  f"std={scores_arr.std():.1f}  "
                  f"min={scores_arr.min():+.0f}  max={scores_arr.max():+.0f}")
            made = sum(1 for s in scores if s >= 0)
            print(f"  Contracts made: {made}/{len(scores)} "
                  f"({made/len(scores):.0%})")
    else:
        print(f"  SUMMARY ({args.num_deals} deals)")
    print(f"{'█' * 65}")


if __name__ == '__main__':
    main()