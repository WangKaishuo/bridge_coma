#!/usr/bin/env python3
"""
Bidding Inspector — Dual-Table Diagnostic (P104)
==================================================

双桌 IMP 诊断工具：在同一副牌上，Agent 和 SL 分别坐不同方位叫牌。

评估方法（与正式 H2H 一致）：
  Table 1: Agent 坐 NS, SL 坐 EW  → 得到 NS 视角得分 score1
  Table 2: Agent 坐 EW, SL 坐 NS  → 得到 NS 视角得分 score2
  IMP = score_to_imp(score1 - score2)
  正值 = Agent 赢

每桌是一个完整的四人叫牌，Agent 和 SL 在同一拍卖中互动。

用法:
  python bid_inspector.py \
      --agent results/drift_sweep_480/lambda0.0_seed42/agent_a_seed42.pt \
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

from env import (
    BridgeBiddingEnv, NUM_BIDS, NUM_PLAYERS,
    BID_PASS, BID_DOUBLE, BID_REDOUBLE, BID_1C,
    bid_to_string, string_to_bid,
    NORTH, EAST, SOUTH, WEST,
)
from networks.policy_net import encode_obs_flat, MLPPolicyNetwork, OBS_DIM
from subgames.competitive_env import CompetitiveSubgameEnv, FIXED_PREFIX
from subgames.action_mask import count_hcp, count_suit_length
from utils.imp import score_to_imp
from utils.scoring import Contract, calculate_score
from algorithms.mappo import MAPPOAgent, MAPPOConfig


# ── Constants ─────────────────────────────────────────────────────────────────
PLAYER_NAMES = ['North', 'East', 'South', 'West']
PLAYER_SHORT = ['N', 'E', 'S', 'W']
SUIT_SYMBOLS = ['♣', '♦', '♥', '♠']
RANK_CHARS   = "23456789TJQKA"
NS_SEATS = {NORTH, SOUTH}
EW_SEATS = {EAST, WEST}


# ==============================================================================
# Hand display
# ==============================================================================

def hand_to_str(hand: np.ndarray) -> str:
    parts = []
    for suit in range(3, -1, -1):
        cards = []
        for rank in range(12, -1, -1):
            if hand[suit * 13 + rank] > 0.5:
                cards.append(RANK_CHARS[rank])
        parts.append(f"{SUIT_SYMBOLS[suit]}{''.join(cards)}")
    return ' '.join(parts)


def hand_summary(hand: np.ndarray) -> str:
    hcp = count_hcp(hand)
    lengths = [count_suit_length(hand, s) for s in range(4)]
    dist = f"{lengths[3]}-{lengths[2]}-{lengths[1]}-{lengths[0]}"
    return f"{hcp} HCP, {dist}"


def display_deal(hands: np.ndarray, dealer: int):
    print(f"\n{'─' * 55}")
    print(f"  Dealer: {PLAYER_NAMES[dealer]}")
    print(f"{'─' * 55}")
    for p in range(4):
        name = PLAYER_NAMES[p]
        hstr = hand_to_str(hands[p])
        summary = hand_summary(hands[p])
        marker = " ← opener" if p == dealer else (
                 " ← overcaller" if p == (dealer + 1) % 4 else "")
        print(f"  {name:6s}: {hstr}  ({summary}){marker}")
    print(f"{'─' * 55}")


# ==============================================================================
# Load models
# ==============================================================================

def load_agent(path: str, device: str) -> MAPPOAgent:
    agent = MAPPOAgent(MAPPOConfig(device=device))
    agent.load(path)
    return agent


def load_sl(path: str, device: str) -> MAPPOAgent:
    agent = MAPPOAgent(MAPPOConfig(device=device))
    ckpt = torch.load(path, map_location=device, weights_only=False)
    for player, key in [(0, 'actor_n'), (1, 'actor_e'),
                        (2, 'actor_s'), (3, 'actor_w')]:
        if key in ckpt:
            agent.get_actor(player).load_state_dict(
                {k: v.to(device) for k, v in ckpt[key].items()})
    return agent


# ==============================================================================
# Play one table (Agent on one side, SL on the other)
# ==============================================================================

def play_one_table(
    hands:       np.ndarray,
    dd_table:    np.ndarray,
    dealer:      int,
    vul:         tuple,
    agent:       MAPPOAgent,
    sl:          MAPPOAgent,
    agent_seats: set,
    device:      str,
) -> dict:
    """
    Play a single table: Agent controls agent_seats, SL controls the rest.
    Returns contract, NS-perspective score, and bid_log with top-3 probs.
    """
    env = BridgeBiddingEnv(max_history_len=60)
    obs = env.reset(hands, dealer=dealer, vulnerability=vul)
    hist = []
    done = False

    # Fixed prefix 1H-1S
    for bid_str in FIXED_PREFIX:
        bid = string_to_bid(bid_str)
        hist.append(bid)
        obs, _, done, _ = env.step(bid)
        if done:
            break

    bid_log = []   # [(player, bid_str, top3, who)]

    while not done:
        player = env.state.current_player
        model  = agent if player in agent_seats else sl
        who    = "Agent" if player in agent_seats else "SL"

        flat   = encode_obs_flat(obs, dealer, hist)
        flat_t = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(device)
        legal  = torch.tensor(obs['legal_actions'], dtype=torch.float32
                              ).unsqueeze(0).to(device)
        actor  = model.get_actor(player)

        with torch.no_grad():
            logits = actor(flat_t, legal)
            probs  = torch.softmax(logits, dim=-1)[0]
            action = logits.argmax(dim=-1).item()
            topk   = torch.topk(probs, min(3, NUM_BIDS))
            top3   = [(bid_to_string(topk.indices[i].item()),
                       topk.values[i].item())
                      for i in range(len(topk.indices))]

        if not env._is_valid_action(action):
            action = BID_PASS
            top3 = [("Pass", 1.0)]

        bid_log.append((player, bid_to_string(action), top3, who))
        hist.append(action)
        obs, _, done, _ = env.step(action)

    contract = env.state.final_contract
    ns_score = 0
    tricks = None
    if contract:
        tricks = int(dd_table[contract.suit, contract.declarer])
        raw = calculate_score(contract, tricks, vul)
        # NS perspective: positive if NS declares and makes, negative if EW
        if contract.declarer in NS_SEATS:
            ns_score = raw
        else:
            ns_score = -raw

    return {
        'bid_log':  bid_log,
        'contract': contract,
        'tricks':   tricks,
        'ns_score': ns_score,
    }


# ==============================================================================
# Display
# ==============================================================================

def _format_contract(contract, tricks, ns_score):
    """Format contract string for display."""
    if not contract:
        return "Passed out", "NS score: 0"
    suit_names = ['♣', '♦', '♥', '♠', 'NT']
    suit_str = suit_names[contract.suit] if contract.suit < 5 else 'NT'
    doubled_str = ['', ' X', ' XX'][contract.doubled]
    declarer_str = PLAYER_NAMES[contract.declarer]
    contract_str = f"{contract.level}{suit_str}{doubled_str} by {declarer_str}"

    target = 6 + contract.level
    if tricks >= target:
        result_str = f"making {tricks}"
    else:
        result_str = f"down {target - tricks}"
    score_str = f"DDS {tricks} tricks ({result_str}), NS score: {ns_score:+d}"
    return contract_str, score_str


def _print_table(label: str, table: dict, dealer: int):
    """Print one table's bidding sequence."""
    bid_log  = table['bid_log']

    print(f"\n  ── {label} ──")

    # Header
    header = "  "
    for i in range(4):
        p = (dealer + i) % 4
        header += f"{PLAYER_SHORT[p]:>8s}"
    print(header)

    # Bids
    prefix_bids = [bid_to_string(string_to_bid(b)) for b in FIXED_PREFIX]
    all_bids = prefix_bids + [b[1] for b in bid_log]

    row = "  "
    for i, bid_str in enumerate(all_bids):
        row += f"{bid_str:>8s}"
        if (i + 1) % 4 == 0:
            print(row)
            row = "  "
    if len(row.strip()) > 0:
        print(row)

    contract_str, score_str = _format_contract(
        table['contract'], table['tricks'], table['ns_score'])
    print(f"  Contract: {contract_str}")
    print(f"  {score_str}")


def _print_deal_result(hands, dealer, dd_table, t1, t2, vul):
    """Print both tables and IMP result for one deal."""

    print(f"\n{'═' * 65}")
    print(f"  DUAL-TABLE BIDDING")
    print(f"{'═' * 65}")
    print(f"\n  Fixed prefix: {PLAYER_SHORT[dealer]}:1♥ → "
          f"{PLAYER_SHORT[(dealer+1)%4]}:1♠")

    _print_table("Table 1 (Agent=NS, SL=EW)", t1, dealer)
    _print_table("Table 2 (Agent=EW, SL=NS)", t2, dealer)

    # IMP
    imp = score_to_imp(t1['ns_score'] - t2['ns_score'])
    print(f"\n  IMP = score_to_imp({t1['ns_score']:+d} − {t2['ns_score']:+d}) = {imp:+.0f}")
    if imp > 0:
        print(f"  → Agent wins {imp:+.0f} IMP")
    elif imp < 0:
        print(f"  → SL wins {-imp:+.0f} IMP")
    else:
        print(f"  → Push (0 IMP)")

    print(f"{'═' * 65}")
    return imp


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bidding Inspector — Dual-table Agent vs SL diagnostic")
    parser.add_argument('--agent', required=True, help='RL agent checkpoint')
    parser.add_argument('--sl', required=True, help='SL baseline checkpoint')
    parser.add_argument('--data', required=True, help='Competitive data (npz)')
    parser.add_argument('--num_deals', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    print(f"[Bid Inspector] Loading agent: {args.agent}")
    agent = load_agent(args.agent, device)
    print(f"[Bid Inspector] Loading SL: {args.sl}")
    sl = load_sl(args.sl, device)

    print(f"[Bid Inspector] Loading deals: {args.data}")
    env = CompetitiveSubgameEnv(args.data)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    imp_total = 0.0
    agent_wins = 0
    sl_wins = 0

    for deal_idx in range(args.num_deals):
        hands, dd_table = env.generate_deal()
        dealer = env._sampled_dealer
        vul = (False, False)

        print(f"\n\n{'▓' * 65}")
        print(f"  DEAL {deal_idx + 1}/{args.num_deals}")
        print(f"{'▓' * 65}")
        display_deal(hands, dealer)

        # Table 1: Agent=NS, SL=EW
        t1 = play_one_table(hands, dd_table, dealer, vul,
                            agent, sl, agent_seats=NS_SEATS, device=device)
        # Table 2: Agent=EW, SL=NS (swap)
        t2 = play_one_table(hands, dd_table, dealer, vul,
                            agent, sl, agent_seats=EW_SEATS, device=device)

        imp = _print_deal_result(hands, dealer, dd_table, t1, t2, vul)
        imp_total += imp
        if imp > 0:
            agent_wins += 1
        elif imp < 0:
            sl_wins += 1

    # Summary
    n = args.num_deals
    ties = n - agent_wins - sl_wins
    print(f"\n\n{'█' * 65}")
    print(f"  SUMMARY ({n} deals, dual-table IMP)")
    print(f"{'█' * 65}")
    print(f"  Agent wins: {agent_wins}  SL wins: {sl_wins}  Ties: {ties}")
    print(f"  Total IMP: {imp_total:+.1f}  (avg {imp_total/n:+.2f}/deal)")
    print(f"{'█' * 65}")


if __name__ == '__main__':
    main()
