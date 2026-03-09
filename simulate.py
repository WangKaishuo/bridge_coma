#!/usr/bin/env python3
"""
simulate.py — Stayman Subgame Deal Simulator
=============================================

加载训练好的 Agent A / Agent B checkpoint, 对指定牌局 (或随机抽取)
模拟完整叫牌过程, 逐步显示每位 agent 的决策依据.

用法:
    # 随机抽取 3 副牌, 对比 A 和 B 的叫牌
    python simulate.py --data data/stayman_50k.npz --n 3

    # 随机抽取, 指定 random seed 以复现
    python simulate.py --data data/stayman_50k.npz --n 5 --seed 42

    # 手动输入一副牌 (PBN 格式)
    python simulate.py --data data/stayman_50k.npz \\
        --pbn "N:AKQ2.AT2.KJ3.987 .J9876.Q654.T632 J9876.KQ3.A2.AJ4 T543.54.T987.KQ5"

    # 只跑 Agent B
    python simulate.py --data data/stayman_50k.npz --agent b

    # 指定 checkpoint 路径
    python simulate.py --data data/stayman_50k.npz \\
        --ckpt_a results/A_control.pt \\
        --ckpt_b results/B_partner_only.pt

注意: 须先 cd bridge-coma/ 再运行.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from env import (
    BridgeBiddingEnv, NUM_BIDS, NUM_PLAYERS,
    BID_PASS, NORTH, EAST, SOUTH, WEST,
    bid_to_string, string_to_bid,
)
from subgames.stayman_env import StaymanSubgameEnv
from subgames.action_mask import count_hcp, count_suit_length
from algorithms.mappo import MAPPOAgent, MAPPOConfig
from networks import BeliefNetwork
from utils.dds_data import create_loader


# ── Constants ─────────────────────────────────────────────────────────────────
SUIT_SYMBOLS = ['♣', '♦', '♥', '♠', 'NT']
POS_NAMES    = ['N', 'E', 'S', 'W']
RANK_CHARS   = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']


# ── Model loading ─────────────────────────────────────────────────────────────

def _build_agent(device: str = 'cpu') -> MAPPOAgent:
    """构建空白 MAPPOAgent (与训练时相同的超参数)."""
    cfg = MAPPOConfig(
        hand_dim=256, history_dim=256, hidden_dim=256,
        device=device,
    )
    return MAPPOAgent(cfg)


def load_agent(ckpt_path: str, device: str = 'cpu') -> MAPPOAgent:
    """从 checkpoint 加载 agent."""
    ckpt = torch.load(ckpt_path, map_location=device)
    agent = _build_agent(device)
    agent.model.load_state_dict(ckpt['model'])
    agent.model.eval()
    return agent


def load_belief_net(ckpt_path: str, device: str = 'cpu'):
    """从 checkpoint 加载 BeliefNetwork (如果存在)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'belief_net' not in ckpt:
        return None
    net = BeliefNetwork(hand_dim=256, history_dim=256, hidden_dim=256).to(device)
    net.load_state_dict(ckpt['belief_net'])
    net.eval()
    return net


# ── Hand utilities ────────────────────────────────────────────────────────────

def hand_to_str(hand: np.ndarray) -> str:
    """将 52-dim one-hot 手牌转为可读字符串, e.g. ♠AKQ ♥J9 ♦T32 ♣654"""
    parts = []
    for suit_idx in range(3, -1, -1):   # S H D C (高到低)
        sym = SUIT_SYMBOLS[suit_idx]
        base = suit_idx * 13
        cards = [RANK_CHARS[r] for r in range(12, -1, -1) if hand[base + r] > 0.5]
        parts.append(f"{sym}{''.join(cards) if cards else '—'}")
    return '  '.join(parts)


def hand_summary(hand: np.ndarray, position: int) -> str:
    """单行手牌摘要: HCP + 高花张数."""
    hcp = count_hcp(hand)
    h   = count_suit_length(hand, 2)
    s   = count_suit_length(hand, 3)
    d   = count_suit_length(hand, 1)
    c   = count_suit_length(hand, 0)
    return (f"{POS_NAMES[position]}: {hcp:2d}HCP  "
            f"♠{s} ♥{h} ♦{d} ♣{c}  |  {hand_to_str(hand)}")


# ── PBN parser ────────────────────────────────────────────────────────────────

def parse_pbn(pbn: str) -> np.ndarray:
    """
    解析 PBN 格式手牌字符串, 返回 (4, 52) one-hot 数组.

    PBN 格式: "N:HAND_N HAND_E HAND_S HAND_W"
    每手牌格式: "AKQJT98765432.AKQJT98765432.AKQJT98765432.AKQJT98765432"
    (黑桃.红桃.方块.梅花)

    示例:
        "N:AK2.J98.QT3.7654 QJT.AKQ.A54.KJ32 9876.T765.KJ2.AQ8 543.432.9876.T9"
    """
    rank_map = {'A':12,'K':11,'Q':10,'J':9,'T':8,'9':7,'8':6,
                '7':5,'6':4,'5':3,'4':2,'3':1,'2':0}
    suit_map = {0:3, 1:2, 2:1, 3:0}  # PBN 顺序: S H D C → suit index: 3 2 1 0

    pbn = pbn.strip()
    if ':' in pbn:
        first_seat_char, hands_str = pbn.split(':', 1)
        # first_seat 指定 N/E/S/W, 决定手牌顺序起点
        first_seat = {'N':0,'E':1,'S':2,'W':3}.get(first_seat_char.strip().upper(), 0)
    else:
        hands_str = pbn
        first_seat = 0

    hand_strs = hands_str.strip().split()
    if len(hand_strs) != 4:
        raise ValueError(f"PBN 需要 4 手牌, 得到 {len(hand_strs)}")

    hands = np.zeros((4, 52), dtype=np.float32)
    for i, hs in enumerate(hand_strs):
        player = (first_seat + i) % 4
        suits = hs.split('.')
        if len(suits) != 4:
            raise ValueError(f"手牌 {hs} 需要 4 个花色 (用 . 分隔)")
        for pbn_suit_idx, cards in enumerate(suits):
            suit_idx = suit_map[pbn_suit_idx]  # S→3, H→2, D→1, C→0
            for ch in cards:
                if ch == '-':
                    continue
                if ch not in rank_map:
                    raise ValueError(f"未知牌面: {ch!r}")
                rank = rank_map[ch]
                hands[player, suit_idx * 13 + rank] = 1.0

    return hands


# ── Core simulator ────────────────────────────────────────────────────────────

def simulate_deal(agent: MAPPOAgent, agent_name: str, agent_label: str,
                  hands: np.ndarray, data_path: str,
                  show_probs: bool = True, device: str = 'cpu'):
    """
    用给定 agent 对指定手牌模拟完整 Stayman 叫牌过程.

    Args:
        agent:       已加载的 MAPPOAgent
        agent_name:  显示名称, e.g. "A (MAPPO)"
        agent_label: 简短标签用于日志
        hands:       (4, 52) one-hot 手牌
        data_path:   stayman 数据路径 (用于构建 env)
        show_probs:  是否显示每步概率分布
    """
    env = StaymanSubgameEnv(data_path, north_rule=False)

    print(f"\n  ┌{'─'*58}┐")
    print(f"  │  Agent {agent_name:<50s}│")
    print(f"  └{'─'*58}┘")

    # Execute fixed prefix
    obs = env.env.reset(hands, dealer=NORTH, vulnerability=(False, False))
    history_display = []
    for bid_str in env.FIXED_PREFIX:
        bid = string_to_bid(bid_str)
        obs, _, done, _ = env.env.step(bid)
        history_display.append(bid_str)

    print(f"  Prefix : {' — '.join(history_display)}")
    print()

    step = 0
    done = False
    final_info = {}

    while not done:
        player = env.env.state.current_player
        obs['legal_actions'] = env._get_stayman_mask()

        obs_t = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device)
                 for k, v in obs.items()}
        all_h = torch.tensor(hands, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = agent.model.actor(obs_t)
            legal  = torch.tensor(obs['legal_actions'], dtype=torch.float32).to(device)
            masked = logits.squeeze(0) - 1e9 * (1.0 - legal)
            probs  = F.softmax(masked, dim=-1)
            action = int(probs.argmax())

        bid_str_chosen = bid_to_string(action)
        step += 1

        # ── Hand info for this player ──────────────────────────────────────
        ph = hands[player]
        hcp = count_hcp(ph)
        h_len = count_suit_length(ph, 2)
        s_len = count_suit_length(ph, 3)
        d_len = count_suit_length(ph, 1)
        c_len = count_suit_length(ph, 0)
        pos_str = POS_NAMES[player]

        # ── Probability bar for legal actions ─────────────────────────────
        legal_idx = [i for i in range(NUM_BIDS) if obs['legal_actions'][i] > 0.5]
        sorted_legal = sorted(legal_idx, key=lambda i: probs[i].item(), reverse=True)

        chosen_marker = "◀ chosen"
        print(f"  Step {step} — {pos_str} bids  "
              f"[{hcp}HCP ♠{s_len}♥{h_len}♦{d_len}♣{c_len}]")
        if show_probs:
            for idx in sorted_legal:
                p = probs[idx].item()
                if p < 0.005:
                    continue
                bar  = '█' * int(p * 30)
                mark = '  ◀ CHOSEN' if idx == action else ''
                print(f"    {bid_to_string(idx):5s} {p:5.1%}  {bar}{mark}")
        else:
            print(f"    → {bid_str_chosen}")
        print()

        obs, reward, done, info = env.step(action)
        history_display.append(bid_str_chosen)
        final_info = info

    # ── Final contract & score ─────────────────────────────────────────────
    contract = env.env.state.final_contract
    imp_val  = final_info.get('imp', 0.0)
    if contract:
        declarer_str = POS_NAMES[contract.declarer]
        contract_str = f"{contract.level}{SUIT_SYMBOLS[contract.suit]} by {declarer_str}"
    else:
        contract_str = "Passed out"

    full_auction = ' — '.join(history_display)
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"  Full auction   : {full_auction}")
    print(f"  Final contract : {contract_str}")
    print(f"  IMP vs DDS opt : {imp_val:+.1f}")


def run_simulation(args):
    """主流程: 加载模型, 生成/解析牌局, 运行模拟."""
    device = args.device

    # ── Load agents ───────────────────────────────────────────────────────
    agents = {}
    if args.agent in ('a', 'both'):
        ckpt_a = args.ckpt_a or 'results/A_control.pt'
        if not Path(ckpt_a).exists():
            print(f"[ERROR] Checkpoint not found: {ckpt_a}")
            sys.exit(1)
        agents['A (MAPPO)'] = load_agent(ckpt_a, device)
        print(f"✓ Loaded Agent A from {ckpt_a}")

    if args.agent in ('b', 'both'):
        ckpt_b = args.ckpt_b or 'results/B_partner_only.pt'
        if not Path(ckpt_b).exists():
            print(f"[ERROR] Checkpoint not found: {ckpt_b}")
            sys.exit(1)
        agents['B (MAPPO+r_info)'] = load_agent(ckpt_b, device)
        print(f"✓ Loaded Agent B from {ckpt_b}")

    if not agents:
        print("[ERROR] No agents loaded.")
        sys.exit(1)

    # ── Generate / parse deals ────────────────────────────────────────────
    if args.pbn:
        # Single manually specified deal
        try:
            hands_list = [parse_pbn(args.pbn)]
        except ValueError as e:
            print(f"[ERROR] PBN parse failed: {e}")
            sys.exit(1)
        print(f"✓ Using manually specified deal (PBN)")
        deal_labels = ["Manual deal"]
    else:
        # Random deals from dataset
        if args.seed is not None:
            np.random.seed(args.seed)
        loader = create_loader(args.data)
        env_sample = StaymanSubgameEnv(args.data, north_rule=False)
        hands_list = []
        attempts = 0
        while len(hands_list) < args.n and attempts < args.n * 200:
            attempts += 1
            hands, _ = loader.sample_one()
            # Must satisfy Stayman constraints
            from subgames.action_mask import is_balanced
            from subgames.stayman_env import StaymanSubgameEnv as _SE
            n_hcp = count_hcp(hands[NORTH])
            s_hcp = count_hcp(hands[SOUTH])
            s_h   = count_suit_length(hands[SOUTH], 2)
            s_s   = count_suit_length(hands[SOUTH], 3)
            if (15 <= n_hcp <= 17 and is_balanced(hands[NORTH])
                    and s_hcp >= 8 and (s_h >= 4 or s_s >= 4)):
                hands_list.append(hands)
        print(f"✓ Sampled {len(hands_list)} deals "
              f"({attempts} attempts, seed={args.seed})")
        deal_labels = [f"Random deal {i+1}" for i in range(len(hands_list))]

    # ── Run simulations ───────────────────────────────────────────────────
    for deal_idx, (hands, label) in enumerate(zip(hands_list, deal_labels)):
        print(f"\n{'='*62}")
        print(f"  {label}")
        print(f"{'='*62}")

        # Print all four hands
        print("  Hands:")
        for p in range(4):
            print(f"    {hand_summary(hands[p], p)}")

        # Fit info
        h_ns = count_suit_length(hands[NORTH], 2) + count_suit_length(hands[SOUTH], 2)
        s_ns = count_suit_length(hands[NORTH], 3) + count_suit_length(hands[SOUTH], 3)
        fit_parts = []
        if h_ns >= 8: fit_parts.append(f"♥{h_ns}-card fit")
        if s_ns >= 8: fit_parts.append(f"♠{s_ns}-card fit")
        fit_str = ', '.join(fit_parts) if fit_parts else "No 8-card major fit"
        print(f"  NS fit : {fit_str}")

        for agent_name, agent in agents.items():
            simulate_deal(agent, agent_name, agent_name,
                          hands, args.data,
                          show_probs=not args.compact,
                          device=device)

    print(f"\n{'='*62}")
    print("  Simulation complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Simulate Stayman subgame bidding with trained agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--data', default='data/stayman_50k.npz',
                        help='Path to stayman data (default: data/stayman_50k.npz)')
    parser.add_argument('--ckpt_a', default=None,
                        help='Agent A checkpoint (default: results/A_control.pt)')
    parser.add_argument('--ckpt_b', default=None,
                        help='Agent B checkpoint (default: results/B_partner_only.pt)')
    parser.add_argument('--agent', choices=['a', 'b', 'both'], default='both',
                        help='Which agent(s) to simulate (default: both)')
    parser.add_argument('--n', type=int, default=3,
                        help='Number of random deals to simulate (default: 3)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--pbn', type=str, default=None,
                        help='PBN hand string to simulate a specific deal')
    parser.add_argument('--compact', action='store_true',
                        help='Skip probability bars, show only chosen bid')
    parser.add_argument('--device', default='cpu',
                        help='PyTorch device (default: cpu)')
    args = parser.parse_args()
    run_simulation(args)


if __name__ == '__main__':
    main()
