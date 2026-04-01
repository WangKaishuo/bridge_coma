#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

  # Agent vs SL_ReFine（双桌对战，ReFine 模式）
  python bid_inspector.py \
      --model1 results/agent_a.pt --type1 agent \
      --model2 results/sl_base_bca_refine.pt --type2 sl \
      --data data/competitive_500k.npz --num_deals 5000

  # Agent vs SL_ReFine_coevolved（消融实验: real vs prior belief）
  python bid_inspector.py \
      --model1 results/agent_a.pt --type1 agent \
      --model2 results/sl_bca_refine_coevolved_a.pt --type2 sl \
      --data data/competitive_500k.npz --num_deals 5000 --ablate_belief
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
    MLPPolicyNetwork, OBS_DIM, BELIEF_OBS_DIM, BELIEF_FEAT_DIM,
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

def _detect_obs_dim(path: str) -> int:
    """Detect obs_dim from checkpoint (571 or 667)."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    # RL agent format: stored obs_dim
    if 'obs_dim' in ckpt:
        return ckpt['obs_dim']
    # P105 SL format: infer from first layer weight shape
    if 'model_state' in ckpt:
        w = ckpt['model_state'].get('net.0.weight')
        if w is not None:
            return w.shape[1]
    return OBS_DIM  # fallback 571


def load_agent(path: str, device: str) -> MAPPOAgent:
    """Load RL agent checkpoint, auto-detecting obs_dim."""
    obs_dim = _detect_obs_dim(path)
    agent = MAPPOAgent(MAPPOConfig(device=device, obs_dim=obs_dim))
    agent.load(path)
    agent._is_refine = False
    print(f"[load_agent] obs_dim={obs_dim}")
    return agent


def load_sl(path: str, device: str):
    """Load SL checkpoint as MAPPOAgent (or ReFineActor wrapper), auto-detecting obs_dim.

    ReFine mode (encoding='openspiel_667_refine'):
      Returns a MAPPOAgent (571-dim) with ._refine_actor set to a ReFineActor.
      _make_play_mixed_policy detects this and routes through the ReFine forward path.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    encoding = ckpt.get('encoding', '')

    if encoding == 'openspiel_667_refine':
        # ── ReFine mode: load frozen actor + adapter ──────────────────────────
        from networks.policy_net import OBS_DIM as _OD
        from utils.sl_pretrain_bca import ReFineActor, BeliefAdapter

        hidden_dim  = ckpt.get('hidden_dim', 1024)
        bottleneck  = ckpt.get('bottleneck', 128)

        frozen_actor = MLPPolicyNetwork(obs_dim=_OD, hidden_dim=hidden_dim).to(device)
        frozen_actor.load_state_dict(
            {k: v.to(device) for k, v in ckpt['model_state'].items()})

        adapter = BeliefAdapter(
            belief_dim=BELIEF_FEAT_DIM,
            hidden_dim=hidden_dim, bottleneck=bottleneck).to(device)
        adapter.load_state_dict(
            {k: v.to(device) for k, v in ckpt['adapter_state'].items()})

        refine_actor = ReFineActor(frozen_actor, adapter).to(device)
        refine_actor.eval()

        gate_val = adapter.gate.item()
        print(f"[load_sl] ReFine mode (encoding={encoding}), gate={gate_val:.4f}")

        # Wrap in a MAPPOAgent shell (571-dim) so the caller can still call
        # _attach_belief, etc.  The actual inference goes through ._refine_actor.
        obs_dim = _OD  # 571 — the frozen actor's input dim
        agent = MAPPOAgent(MAPPOConfig(device=device, obs_dim=obs_dim))
        load_sl_into_mappo_agent(agent, path)  # loads 571-dim weights normally
        agent._refine_actor = refine_actor      # attach ReFine actor as side-car
        agent._is_refine    = True
        print(f"[load_sl] ReFineActor attached (adapter params: "
              f"{sum(p.numel() for p in adapter.parameters()):,})")
        return agent

    # ── Standard SL (571 or 667 legacy) ──────────────────────────────────────
    obs_dim = ckpt.get('obs_dim', OBS_DIM)
    if 'model_state' in ckpt:
        w = ckpt['model_state'].get('net.0.weight')
        if w is not None:
            obs_dim = w.shape[1]
    agent = MAPPOAgent(MAPPOConfig(device=device, obs_dim=obs_dim))
    load_sl_into_mappo_agent(agent, path)
    agent._is_refine = False
    print(f"[load_sl] obs_dim={obs_dim}")
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


def _make_play_mixed_policy(model, hands_sm, dealer, device, belief_net=None,
                            opponent_belief_net=None,
                            ablate_belief=False, vulnerability=None):
    """
    Build a (obs, player, history_int) -> action closure for play_mixed.

    BCA mode: if belief_net is provided, appends 96-dim belief features to
    the 571-dim OpenSpiel obs, producing 667-dim input for BCA actors.

    ReFine mode (P125): if model._is_refine=True, obs_571 and belief_96 are
    passed SEPARATELY to ReFineActor.get_action() instead of concatenating.
    The frozen SL actor receives obs_571 only; the adapter adjusts via belief_96.

    ablate_belief: if True AND (is_bca or is_refine), replace real BeliefNet
    output with prior (0.25 uniform). Isolates belief *content* effect.

    P122: vulnerability passed to hands_to_openspiel_state for vul-aware obs.
    P122: use ABSOLUTE seat for BeliefNet target_pos.

    P117 fixes:
    - legal_mask from OpenSpiel state (not BridgeBiddingEnv obs)
    - actor selected by (player-dealer)%4 relative seat
    """
    from networks.policy_net import BELIEF_OBS_DIM, append_belief_features
    hands_rm = convert_hands_suit_to_rank(hands_sm)
    is_bca = (belief_net is not None)
    if vulnerability is None:
        vulnerability = (False, False)

    # Detect ReFine mode: model has a ._refine_actor side-car
    is_refine = getattr(model, '_is_refine', False) and hasattr(model, '_refine_actor')

    def policy(obs, player, history_int):
        os_state = hands_to_openspiel_state(hands_rm, dealer,
                                            vulnerability=vulnerability)
        for a in history_int:
            if os_state.is_terminal():
                break
            legal_os = os_state.legal_actions()
            if legal_os and legal_os[0] < 52:
                break
            os_a = ours_to_openspiel_raw(a)
            if os_a >= 0 and os_a in legal_os:
                os_state.apply_action(os_a)

        flat = get_openspiel_obs(os_state)  # (571,)

        # Legal mask from OpenSpiel state (shared for all paths)
        legal_os_acts = os_state.legal_actions()
        legal_mask = torch.zeros(1, NUM_BIDS, dtype=torch.float32, device=device)
        for os_a in legal_os_acts:
            ours = openspiel_raw_to_ours(os_a)
            if ours >= 0:
                legal_mask[0, ours] = 1.0

        rel_player = (player - dealer) % 4

        if is_refine:
            # ── ReFine path: obs_571 and belief_96 passed separately ──────────
            # belief_net must be provided for ReFine (it was loaded alongside model)
            obs_t = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(device)

            if is_bca and belief_net is not None:
                if ablate_belief:
                    bf_t = torch.full((1, 96), 0.25, dtype=torch.float32, device=device)
                else:
                    partner = (player + 2) % 4
                    rho     = (player - 1) % 4
                    rho_bn  = opponent_belief_net or belief_net  # P125: Full Disclosure
                    with torch.no_grad():
                        bf_partner = belief_net.get_probs(
                            obs_t, torch.tensor([partner], dtype=torch.long, device=device))
                        bf_rho     = rho_bn.get_probs(
                            obs_t, torch.tensor([rho],     dtype=torch.long, device=device))
                    bf_t = torch.cat([bf_partner, bf_rho], dim=-1)  # (1, 96)
            else:
                # No BeliefNet: use prior
                bf_t = torch.full((1, 96), 0.25, dtype=torch.float32, device=device)

            with torch.no_grad():
                action, _, _ = model._refine_actor.get_action(
                    obs_t, bf_t, legal_mask, deterministic=True)
            return action.item()

        else:
            # ── Standard path (plain SL or RL agent, 571 or 667-dim) ──────────
            if is_bca:
                if ablate_belief:
                    bf = np.full(96, 0.25, dtype=np.float32)
                else:
                    # P122: use ABSOLUTE seat for BeliefNet target_pos.
                    # P125: Full Disclosure — RHO uses opponent's BeliefNet
                    partner = (player + 2) % 4
                    rho     = (player - 1) % 4
                    rho_bn  = opponent_belief_net or belief_net  # P125
                    obs_t   = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        bf_partner = belief_net.get_probs(
                            obs_t, torch.tensor([partner], dtype=torch.long, device=device))
                        bf_rho     = rho_bn.get_probs(
                            obs_t, torch.tensor([rho],     dtype=torch.long, device=device))
                    bf = torch.cat([bf_partner, bf_rho], dim=-1).squeeze(0).cpu().numpy()
                flat = append_belief_features(flat, bf)  # (667,)

            flat_t = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(device)
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
    ablate_belief_model2=False,
    no_full_disclosure=False,
):
    """
    真正的双桌对战:
      桌1: Agent=Opener(开叫方阵营), SL=Overcaller(争叫方阵营) -> score_1
      桌2: SL=Opener(开叫方阵营),    Agent=Overcaller(争叫方阵营) -> score_2
      IMP = score_to_imp(score_1 - score_2) (Dealer为NS时) 
            或 score_to_imp(score_2 - score_1) (Dealer为EW时)

    和 cross_evaluate / evaluate_head_to_head 语义完全一致。

    ablate_belief_model2: if True, SL's belief features are replaced with prior.
    P122: vulnerability randomized per deal.
    """
    # P122: randomize vulnerability
    _vul_choices = [(False, False), (True, False),
                    (False, True),  (True, True)]
    vul = _vul_choices[np.random.randint(4)]

    _agent_bn = getattr(agent, '_belief_net', None)
    _sl_bn    = getattr(sl,    '_belief_net', None)
    _opp_bn_agent = None if no_full_disclosure else _sl_bn
    _opp_bn_sl    = None if no_full_disclosure else _agent_bn
    agent_policy = _make_play_mixed_policy(agent, hands_sm, dealer, device,
                                           belief_net=_agent_bn,
                                           opponent_belief_net=_opp_bn_agent,
                                           vulnerability=vul)
    sl_policy    = _make_play_mixed_policy(sl,    hands_sm, dealer, device,
                                           belief_net=_sl_bn,
                                           opponent_belief_net=_opp_bn_sl,
                                           ablate_belief=ablate_belief_model2,
                                           vulnerability=vul)

    # 桌1: Agent Opener vs SL Overcaller
    contract_1, score_1, hist_1 = env.play_mixed(
        hands_sm, dd_table,
        opener_policy=agent_policy, overcaller_policy=sl_policy,
        vulnerability=vul, dealer=dealer)

    # 桌2: SL Opener vs Agent Overcaller
    contract_2, score_2, hist_2 = env.play_mixed(
        hands_sm, dd_table,
        opener_policy=sl_policy, overcaller_policy=agent_policy,
        vulnerability=vul, dealer=dealer)

    # P123 修复：分数基于物理 NS。如果 Opener 在物理 EW（dealer 是 E 或 W），
    # 那么 Agent 在桌 1 控制 EW 侧，Agent 打得好 -> EW 赢得多 -> NS 分数 score_1 越低。
    # 此时应翻转 IMP 计算公式。
    if dealer % 2 == 1:
        imp = float(score_to_imp(score_2 - score_1))
    else:
        imp = float(score_to_imp(score_1 - score_2))

    opener_str     = f"{PLAYER_SHORT[dealer]}{PLAYER_SHORT[(dealer+2)%4]}"
    overcaller_str = f"{PLAYER_SHORT[(dealer+1)%4]}{PLAYER_SHORT[(dealer+3)%4]}"
    results = {
        'table1': {'label': f'Agent({opener_str}) vs SL({overcaller_str})',
                   'contract': contract_1, 'score': score_1, 'history': hist_1},
        'table2': {'label': f'SL({opener_str}) vs Agent({overcaller_str})',
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
    ablate_belief_model2=False,
    no_full_disclosure=False,
):
    """
    Agent A vs Agent B 双桌对战:
      桌1: A=Opener,  B=Overcaller  -> score_1  (NS视角)
      桌2: B=Opener,  A=Overcaller  -> score_2  (NS视角)
      IMP_A = 根据 dealer 翻转计算的真实 IMP_A
      IMP_B = -IMP_A                             (B 视角)

    与 evaluate_head_to_head 语义完全一致。

    ablate_belief_model2: if True, agent_b's belief features are replaced with prior.
    P122: vulnerability randomized per deal, matching cross_evaluate.
    """
    # P122: randomize vulnerability
    _vul_choices = [(False, False), (True, False),
                    (False, True),  (True, True)]
    vul = _vul_choices[np.random.randint(4)]

    _bn_a = getattr(agent_a, '_belief_net', None)
    _bn_b = getattr(agent_b, '_belief_net', None)
    _opp_bn_a = None if no_full_disclosure else _bn_b
    _opp_bn_b = None if no_full_disclosure else _bn_a
    policy_a = _make_play_mixed_policy(agent_a, hands_sm, dealer, device,
                                       belief_net=_bn_a,
                                       opponent_belief_net=_opp_bn_a,
                                       vulnerability=vul)
    policy_b = _make_play_mixed_policy(agent_b, hands_sm, dealer, device,
                                       belief_net=_bn_b,
                                       opponent_belief_net=_opp_bn_b,
                                       ablate_belief=ablate_belief_model2,
                                       vulnerability=vul)

    # 桌1: A=Opener, B=Overcaller
    contract_1, score_1, hist_1 = env.play_mixed(
        hands_sm, dd_table,
        opener_policy=policy_a, overcaller_policy=policy_b,
        vulnerability=vul, dealer=dealer)

    # 桌2: B=Opener, A=Overcaller
    contract_2, score_2, hist_2 = env.play_mixed(
        hands_sm, dd_table,
        opener_policy=policy_b, overcaller_policy=policy_a,
        vulnerability=vul, dealer=dealer)

    # P123 修复
    if dealer % 2 == 1:
        imp_a = float(score_to_imp(score_2 - score_1))
    else:
        imp_a = float(score_to_imp(score_1 - score_2))

    # Label reflects actual opener/overcaller seats (dealer-dependent)
    opener_str     = f"{PLAYER_SHORT[dealer]}{PLAYER_SHORT[(dealer+2)%4]}"
    overcaller_str = f"{PLAYER_SHORT[(dealer+1)%4]}{PLAYER_SHORT[(dealer+3)%4]}"
    results = {
        'table1': {'label': f'A({opener_str}) vs B({overcaller_str})',
                   'contract': contract_1, 'score': score_1, 'history': hist_1},
        'table2': {'label': f'B({opener_str}) vs A({overcaller_str})',
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
    parser.add_argument('--belief_checkpoint', default=None,
                        help='BeliefNet checkpoint (sl_base_bca.pt) for BCA models (667-dim). '
                             'Applied to all loaded models.')
    parser.add_argument('--no_full_disclosure', action='store_true',
                        help='Disable Full Disclosure (P125): each agent uses only own BeliefNet. '
                             'Compare with default (Full Disclosure on) to quantify convention drift effect.')
    parser.add_argument('--ablate_belief', action='store_true',
                        help='Ablation: replace model2\'s BeliefNet output with uniform prior '
                             '(0.25). Model2 still uses 667-dim input but belief columns carry '
                             'no information. Isolates belief content effect from Stage B '
                             'training effect.')
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

    # ── Load BeliefNet for each model ──────────────────────────────────────
    # Priority per model:
    #   1. belief_net embedded in the model's own checkpoint (co-evolved)
    #   2. --belief_checkpoint (standalone BeliefNet, e.g. sl_base_bca.pt)
    #   3. None (571-dim model, no belief features)
    from networks.belief_net import BeliefNetwork

    # Load standalone fallback if provided
    _fallback_bn_sd = None
    _fallback_bn_hidden = None
    if args.belief_checkpoint:
        _fb_ckpt = torch.load(args.belief_checkpoint, map_location=device, weights_only=False)
        _fallback_bn_sd = _fb_ckpt.get('belief_net', _fb_ckpt)
        _fallback_bn_hidden = _fb_ckpt.get('belief_hidden_dim',
                                            next(iter(_fallback_bn_sd.values())).shape[0])

    def _attach_belief(model, model_path, model_label):
        """Attach BeliefNet to a model if it's 667-dim or ReFine.

        ReFine models (is_refine=True) have obs_dim=571 but still need a BeliefNet
        for the belief_96 adapter input.  Their BeliefNet is embedded in the
        sl_base_bca_refine.pt checkpoint under the 'belief_net' key.
        """
        is_refine_model = getattr(model, '_is_refine', False)
        _obs_dim = getattr(getattr(model, 'config', None), 'obs_dim', OBS_DIM)

        # Non-BCA and non-ReFine: no BeliefNet needed
        if _obs_dim != BELIEF_OBS_DIM and not is_refine_model:
            model._belief_net = None
            return

        # Try loading from model's own checkpoint first
        _ckpt = torch.load(model_path, map_location=device, weights_only=False)
        if 'belief_net' in _ckpt:
            _bn_sd = _ckpt['belief_net']
            _bn_hidden = _ckpt.get('belief_hidden_dim',
                                    next(iter(_bn_sd.values())).shape[0])
            _bn = BeliefNetwork(hidden_dim=_bn_hidden).to(device)
            _bn.load_state_dict({k: v.to(device) for k, v in _bn_sd.items()})
            _bn.eval()
            model._belief_net = _bn
            _mode_tag = ' (ReFine)' if is_refine_model else ''
            print(f"[Bid Inspector] {model_label}: BeliefNet from checkpoint "
                  f"(co-evolved{_mode_tag}, hidden={_bn_hidden})")
            return

        if is_refine_model:
            # ReFine model without embedded BeliefNet — prior only
            model._belief_net = None
            print(f"[Bid Inspector] ⚠️  {model_label}: ReFine model but no BeliefNet in "
                  f"checkpoint. Will use prior features (0.25).")
            return

        # Fallback to --belief_checkpoint
        if _fallback_bn_sd is not None:
            _bn = BeliefNetwork(hidden_dim=_fallback_bn_hidden).to(device)
            _bn.load_state_dict({k: v.to(device) for k, v in _fallback_bn_sd.items()})
            _bn.eval()
            model._belief_net = _bn
            print(f"[Bid Inspector] {model_label}: BeliefNet from --belief_checkpoint "
                  f"(fallback, hidden={_fallback_bn_hidden})")
            return

        # No BeliefNet available for a 667-dim model — this is an error
        print(f"[Bid Inspector] ⚠️  WARNING: {model_label} is 667-dim but no BeliefNet "
              f"found in checkpoint or --belief_checkpoint. Using prior features.")
        model._belief_net = None

    _attach_belief(model1, args.model1, "model1")
    if not vs_self:
        _attach_belief(model2, args.model2, "model2")
    else:
        model2._belief_net = model1._belief_net

    print(f"[Bid Inspector] Loading deals: {args.data}")
    env = CompetitiveSubgameEnv(args.data)
    print(f"[Bid Inspector] Mode: {mode_str}")
    if args.no_full_disclosure:
        print(f"[Bid Inspector] ⚠️  NO FULL DISCLOSURE: each agent uses only own BeliefNet (convention drift measurement)")
    if args.ablate_belief:
        print(f"[Bid Inspector] ⚠️  ABLATION MODE: model2's belief features replaced with prior (0.25)")
    print()

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
                                  verbose=not args.quiet,
                                  ablate_belief_model2=args.ablate_belief,
                                  no_full_disclosure=args.no_full_disclosure)
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
        if args.no_full_disclosure:
            label1 += "(no-FD)"
            label2 += "(no-FD)"
        if args.ablate_belief:
            label2 += "(ablated)"
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