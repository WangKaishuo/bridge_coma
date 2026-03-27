#!/usr/bin/env python3
"""
Bidding Inspector — P105 (OpenSpiel-native observations)
==========================================================

一个模型 → 自我对战（单桌展示叫牌过程）
两个模型 → 双桌对战（model1视角IMP）

支持 agent 和 sl 两种模型类型任意组合。

Usage:
  # SL 自我对战（单桌展示）
  python bid_inspector.py \
      --model1 results/sl_base.pt --type1 sl \
      --data data/competitive_500k.npz --num_deals 10

  # Agent 自我对战
  python bid_inspector.py \
      --model1 results/agent_a.pt --type1 agent \
      --data data/competitive_500k.npz --num_deals 10

  # Agent vs SL（双桌对战，agent视角IMP）
  python bid_inspector.py \
      --model1 results/agent_a.pt --type1 agent \
      --model2 results/sl_base.pt  --type2 sl \
      --data data/competitive_500k.npz --num_deals 20

  # Agent A vs Agent B（双桌对战）
  python bid_inspector.py \
      --model1 results/agent_a.pt --type1 agent \
      --model2 results/agent_b.pt --type2 agent \
      --data data/competitive_500k.npz --num_deals 20

  # SL vs SL（双桌对战）
  python bid_inspector.py \
      --model1 results/sl_base.pt      --type1 sl \
      --model2 results/sl_competitive.pt --type2 sl \
      --data data/competitive_500k.npz --num_deals 20
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


def _make_play_mixed_policy(model, hands_sm, dealer, device):
    """
    Build a (obs, player, history_int) -> action closure for play_mixed.
    Uses OpenSpiel state for obs generation (same as RL inference).

    IMPORTANT: play_mixed passes real seat numbers (0=N,1=E,2=S,3=W) as
    `player`, but hands_to_openspiel_state always uses dealer=0 game with
    hands rolled so opener→index 0. So the OpenSpiel-relative player index
    is (real_player - dealer) % 4, which must be used to select the correct
    actor (matching SL training, where dealer=0 always).
    """
    hands_rm = convert_hands_suit_to_rank(hands_sm)

    def policy(obs, player, history_int):
        os_state = hands_to_openspiel_state(hands_rm, dealer)
        for a in history_int:
            if os_state.is_terminal():
                break
            legal_os = os_state.legal_actions()
            if legal_os and legal_os[0] < 52:
                break
            os_a = ours_to_openspiel_raw(a)
            if os_a >= 0 and os_a in legal_os:
                os_state.apply_action(os_a)
        flat = get_openspiel_obs(os_state)
        flat_t = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(device)
        # Use legal actions from OpenSpiel state (matches solo mode)
        legal_os = os_state.legal_actions()
        legal_mask = torch.zeros(1, NUM_BIDS, dtype=torch.float32, device=device)
        for os_a in legal_os:
            ours = openspiel_raw_to_ours(os_a)
            if ours >= 0:
                legal_mask[0, ours] = 1.0
        # Convert real seat → OpenSpiel-relative seat (dealer=0 convention)
        rel_player = (player - dealer) % 4
        actor = model.get_actor(rel_player)
        with torch.no_grad():
            action, _, _ = actor.get_action(flat_t, legal_mask, deterministic=True)
        return action.item()
    return policy


def play_deal(
    hands_sm,
    dd_table,
    dealer,
    agent,
    sl,
    device,
    env,
    verbose=True,
):
    """
    真正的双桌对战:
      桌1: Agent=NS(开叫方), SL=EW(争叫方) -> score_1
      桌2: SL=NS(开叫方),    Agent=EW(争叫方) -> score_2
      IMP = score_to_imp(score_1 - score_2)  (Agent 视角)

    和 cross_evaluate / evaluate_head_to_head 语义完全一致。
    """
    vul = (False, False)

    agent_policy = _make_play_mixed_policy(agent, hands_sm, dealer, device)
    sl_policy    = _make_play_mixed_policy(sl,    hands_sm, dealer, device)

    # 桌1: Agent NS vs SL EW
    contract_1, score_1, hist_1 = env.play_mixed(
        hands_sm, dd_table,
        ns_policy=agent_policy, ew_policy=sl_policy,
        vulnerability=vul, dealer=dealer)

    # 桌2: SL NS vs Agent EW
    contract_2, score_2, hist_2 = env.play_mixed(
        hands_sm, dd_table,
        ns_policy=sl_policy, ew_policy=agent_policy,
        vulnerability=vul, dealer=dealer)

    imp = float(score_to_imp(score_1 - score_2))

    results = {
        'table1': {'label': 'Agent(NS) vs SL(EW)',
                   'contract': contract_1, 'score': score_1, 'history': hist_1},
        'table2': {'label': 'SL(NS) vs Agent(EW)',
                   'contract': contract_2, 'score': score_2, 'history': hist_2},
        'imp': imp,
        'vul': vul,
    }

    if verbose:
        _print_cross_table(hands_sm, dealer, dd_table, results)

    return results

def play_deal_ab(
    hands_sm,
    dd_table,
    dealer,
    agent_a,
    agent_b,
    device,
    env,
    verbose=True,
):
    """
    Agent A vs Agent B 双桌对战:
      桌1: A=NS,  B=EW  -> score_1  (NS视角)
      桌2: B=NS,  A=EW  -> score_2  (NS视角)
      IMP_A = score_to_imp(score_1 - score_2)  (A 视角)
      IMP_B = -IMP_A                             (B 视角)

    与 evaluate_head_to_head 语义完全一致。
    """
    vul = (False, False)

    policy_a = _make_play_mixed_policy(agent_a, hands_sm, dealer, device)
    policy_b = _make_play_mixed_policy(agent_b, hands_sm, dealer, device)

    # 桌1: A=NS, B=EW
    contract_1, score_1, hist_1 = env.play_mixed(
        hands_sm, dd_table,
        ns_policy=policy_a, ew_policy=policy_b,
        vulnerability=vul, dealer=dealer)

    # 桌2: B=NS, A=EW
    contract_2, score_2, hist_2 = env.play_mixed(
        hands_sm, dd_table,
        ns_policy=policy_b, ew_policy=policy_a,
        vulnerability=vul, dealer=dealer)

    imp_a = float(score_to_imp(score_1 - score_2))

    results = {
        'table1': {'label': 'A(NS) vs B(EW)',
                   'contract': contract_1, 'score': score_1, 'history': hist_1},
        'table2': {'label': 'B(NS) vs A(EW)',
                   'contract': contract_2, 'score': score_2, 'history': hist_2},
        'imp_a': imp_a,
        'vul': vul,
    }

    if verbose:
        _print_cross_table_ab(hands_sm, dealer, dd_table, results)

    return results


def _print_cross_table_ab(hands, dealer, dd_table, results):
    """Print A vs B cross-table results."""
    print(f"\n{'═' * 65}")
    print(f"  AGENT A vs AGENT B (双桌对战)")
    print(f"{'═' * 65}")
    print(f"  Fixed prefix: {PLAYER_SHORT[dealer]}:1♥ → "
          f"{PLAYER_SHORT[(dealer+1)%4]}:1♠")

    t1 = results['table1']
    t2 = results['table2']
    vul = results['vul']

    _print_table(t1['label'], t1['history'], dealer, dd_table,
                 t1['contract'], t1['score'], vul)
    _print_table(t2['label'], t2['history'], dealer, dd_table,
                 t2['contract'], t2['score'], vul)

    imp_a = results['imp_a']
    verdict = ("✅ A wins" if imp_a > 0 else
               "❌ B wins" if imp_a < 0 else "— tie")
    print(f"\n  IMP (A perspective): {imp_a:+.0f}  "
          f"(T1={t1['score']:+d}, T2={t2['score']:+d})  {verdict}")
    print(f"{'═' * 65}")


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
        # EW作庄时NS视角取负
        if contract.declarer % 2 == 1:
            score = -score
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


def _print_table(label, history, dealer, dd_table, contract, score, vul):
    """Print one table's bidding sequence and result."""
    suit_names = ['♣', '♦', '♥', '♠', 'NT']
    print(f"\n  ── {label} ──")
    header = "  "
    for i in range(4):
        p = (dealer + i) % 4
        header += f"{PLAYER_SHORT[p]:>8s}"
    print(header)
    row = "  "
    for i, bid in enumerate(history):
        row += f"{bid_to_string(bid):>8s}"
        if (i + 1) % 4 == 0:
            print(row); row = "  "
    if len(row.strip()) > 0:
        print(row)
    if contract:
        suit_str    = suit_names[contract.suit]
        declarer_str= PLAYER_NAMES[contract.declarer]
        doubled_str = ['', ' X', ' XX'][contract.doubled]
        tricks      = int(dd_table[contract.suit, contract.declarer])
        target      = 6 + contract.level
        result_str  = f"making {tricks}" if tricks >= target else f"down {target - tricks}"
        print(f"  Contract: {contract.level}{suit_str}{doubled_str} by {declarer_str}")
        print(f"  DDS: {tricks} tricks ({result_str}), Score: {score:+d}")
    else:
        print(f"  Contract: Passed out, Score: 0")


def _print_cross_table(hands, dealer, dd_table, results):
    """Print cross-table (双桌对战) results."""
    print(f"\n{'═' * 65}")
    print(f"  CROSS-TABLE (双桌对战)")
    print(f"{'═' * 65}")
    print(f"  Fixed prefix: {PLAYER_SHORT[dealer]}:1♥ → "
          f"{PLAYER_SHORT[(dealer+1)%4]}:1♠")

    t1 = results['table1']
    t2 = results['table2']
    vul = results['vul']

    _print_table(t1['label'], t1['history'], dealer, dd_table,
                 t1['contract'], t1['score'], vul)
    _print_table(t2['label'], t2['history'], dealer, dd_table,
                 t2['contract'], t2['score'], vul)

    imp = results['imp']
    verdict = ("✅ Agent wins" if imp > 0 else
               "❌ SL wins"   if imp < 0 else "— tie")
    print(f"\n  IMP (Agent): {imp:+.0f}  "
          f"(T1={t1['score']:+d}, T2={t2['score']:+d})  {verdict}")
    print(f"{'═' * 65}")


# ==============================================================================
# Main
# ==============================================================================

def _load_model(path: str, model_type: str, device: str):
    """Load a model by type ('agent' or 'sl')."""
    if model_type == 'sl':
        print(f"[Bid Inspector] Loading SL: {path}")
        return load_sl(path, device)
    else:
        print(f"[Bid Inspector] Loading agent: {path}")
        return load_agent(path, device)


def main():
    parser = argparse.ArgumentParser(
        description="Bidding Inspector (P105)\n\n"
                    "一个模型 → 自我对战（单桌展示）\n"
                    "两个模型 → 双桌对战（model1视角IMP）",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--model1', required=True,
                        help='第一个模型路径（必填）')
    parser.add_argument('--type1', default='agent', choices=['agent', 'sl'],
                        help='model1类型: agent 或 sl（默认: agent）')
    parser.add_argument('--model2', default=None,
                        help='第二个模型路径（可选，不填则model1自我对战）')
    parser.add_argument('--type2', default='agent', choices=['agent', 'sl'],
                        help='model2类型: agent 或 sl（默认: agent）')
    parser.add_argument('--data', required=True,
                        help='Competitive data (npz)')
    parser.add_argument('--num_deals', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--quiet', action='store_true',
                        help='只打印最终汇总，不打印每局细节')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    model1 = _load_model(args.model1, args.type1, device)

    vs_self = args.model2 is None
    if vs_self:
        model2 = model1
        mode_str = f"{args.type1.upper()}1 自我对战 (4×同一模型)"
    else:
        model2 = _load_model(args.model2, args.type2, device)
        mode_str = (f"{args.type1.upper()}1 vs {args.type2.upper()}2 "
                    f"(双桌对战, model1视角)")

    print(f"[Bid Inspector] Loading deals: {args.data}")
    env = CompetitiveSubgameEnv(args.data)
    print(f"[Bid Inspector] Mode: {mode_str}\n")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    vul = (False, False)
    imps = []

    for deal_idx in range(args.num_deals):
        hands, dd_table = env.generate_deal()
        dealer = env._sampled_dealer

        if not args.quiet:
            print(f"\n\n{'▓' * 65}")
            print(f"  DEAL {deal_idx + 1}/{args.num_deals}")
            print(f"{'▓' * 65}")
            display_deal(hands, dealer)
        elif (deal_idx + 1) % 500 == 0:
            print(f"  ... {deal_idx + 1}/{args.num_deals} deals done")

        if vs_self:
            result = play_deal_sl_only(hands, dd_table, dealer, model1, device,
                                       verbose=False)
            if not args.quiet:
                _print_sl_solo(hands, dealer, dd_table, result, vul)
            contract = result['contract']
            if contract:
                tricks = int(dd_table[contract.suit, contract.declarer])
                score = calculate_score(contract, tricks, vul)
                if contract.declarer % 2 == 1:
                    score = -score
                imps.append(score)
        else:
            result = play_deal_ab(hands, dd_table, dealer,
                                  model1, model2, device, env,
                                  verbose=not args.quiet)
            imps.append(result['imp_a'])

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n\n{'█' * 65}")
    if vs_self:
        print(f"  自我对战 SUMMARY ({args.num_deals} deals)")
        print(f"{'█' * 65}")
        if imps:
            arr = np.array(imps)
            made = (arr >= 0).sum()
            print(f"  NS score: mean={arr.mean():+.1f} ± {arr.std():.1f}  "
                  f"min={arr.min():+.0f}  max={arr.max():+.0f}")
            print(f"  NS正分局数: {made}/{len(imps)} ({made/len(imps):.0%})")
    else:
        label1 = f"{args.type1.upper()}1"
        label2 = f"{args.type2.upper()}2"
        print(f"  {label1} vs {label2} CROSS-TABLE SUMMARY ({args.num_deals} deals)")
        print(f"{'█' * 65}")
        if imps:
            arr = np.array(imps)
            wins  = (arr > 0).sum()
            losses= (arr < 0).sum()
            ties  = (arr == 0).sum()
            print(f"  IMP ({label1} perspective): mean={arr.mean():+.3f} ± {arr.std():.3f}")
            print(f"  {label1} wins / {label2} wins / Tie: {wins} / {losses} / {ties}")
    print(f"{'█' * 65}")


if __name__ == '__main__':
    main()