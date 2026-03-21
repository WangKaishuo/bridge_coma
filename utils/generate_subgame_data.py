#!/usr/bin/env python3
"""
Generate Subgame Data
=====================

生成子博弈专用的 DDS 数据 (约束发牌 + DDS 求解).

比从通用 DDS 数据中筛选高效得多:
- 通用数据筛选率 Stayman ~2.7%, Competitive ~5%
- 本脚本直接生成符合约束的牌, 100% 利用率

输出格式与 dds_data.py 兼容:
  - decks: uint8 (N, 52)
  - tricks: int8 (N, 5, 4)

推荐样本量:
  - Stayman:     50,000 (训练 2 agent × 24k deals/agent, 池 ≥ 2x)
  - Competitive: 100,000 (训练 3 agent × 40k deals/agent, 池 ≥ 2x)

Usage:
    cd bridge-coma/

    # 生成两种数据 (推荐)
    python -m utils.generate_subgame_data --type both --num_workers 4

    # 单独生成
    python -m utils.generate_subgame_data --type stayman --num_samples 50000
    python -m utils.generate_subgame_data --type competitive --num_workers 8 --num_samples 500000 data/competitive_500k.npz

    # 自定义输出
    python -m utils.generate_subgame_data --type stayman --num_samples 50000 --output data/stayman_50k.npz
"""

import argparse
import time
import multiprocessing as mp
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

# 尝试导入 endplay (DDS 求解)
try:
    import endplay
    HAS_ENDPLAY = True
except ImportError:
    HAS_ENDPLAY = False
    print("WARNING: endplay not installed. DDS solving disabled.")


# ============================================================================
# Hand analysis (standalone, no package dependency)
# ============================================================================

def count_hcp(hand_52: np.ndarray) -> int:
    """HCP: A=4, K=3, Q=2, J=1. hand_52: (52,) one-hot."""
    hcp = 0
    for suit in range(4):
        base = suit * 13
        if hand_52[base + 12] > 0.5: hcp += 4  # A
        if hand_52[base + 11] > 0.5: hcp += 3  # K
        if hand_52[base + 10] > 0.5: hcp += 2  # Q
        if hand_52[base + 9] > 0.5:  hcp += 1  # J
    return hcp


def suit_length(hand_52: np.ndarray, suit: int) -> int:
    return int(hand_52[suit * 13:(suit + 1) * 13].sum())


def is_balanced(hand_52: np.ndarray) -> bool:
    """无单缺 (所有花色 >= 2). 允许 5M/6m."""
    return all(suit_length(hand_52, s) >= 2 for s in range(4))


def hands_to_deck(hands: np.ndarray) -> np.ndarray:
    """(4, 52) one-hot → (52,) uint8 deck[card] = player."""
    deck = np.zeros(52, dtype=np.uint8)
    for player in range(4):
        for card in range(52):
            if hands[player, card] > 0.5:
                deck[card] = player
    return deck


def deck_to_hands_local(deck: np.ndarray) -> np.ndarray:
    """(52,) uint8 → (4, 52) float32 one-hot. Standalone, no package dep."""
    hands = np.zeros((4, 52), dtype=np.float32)
    for card in range(52):
        hands[deck[card], card] = 1.0
    return hands


# ============================================================================
# Constrained dealing
# ============================================================================

def deal_constrained_stayman(rng: np.random.Generator) -> Optional[np.ndarray]:
    """
    生成符合 Stayman 约束的牌.

    N: 15-17 HCP, 均型 (无单缺)
    S: 8-10 HCP, 至少有 4 张高花 (H 或 S)

    S 点力上限 10 的理由:
    NS 合计 23-27 HCP 时, 将牌张数对定约选择影响最显著:
    - 4-4 配合: 4M 明显优于 3NT
    - 无配合: 3NT 几乎必然最优
    高点力局面 (S 11+) 下 4-3 配合也可成局, DDS 标注 4M, 污染标签分布.

    Returns:
        hands: (4, 52) one-hot, 或 None 如果尝试失败
    """
    deck = np.arange(52)
    
    for _ in range(2000):  # 接受率约 1.7%, 期望约 59 次; 加大上限保险
        rng.shuffle(deck)
        hands = np.zeros((4, 52), dtype=np.float32)
        for i, card in enumerate(deck):
            hands[i // 13, card] = 1.0

        n_hand = hands[0]  # North
        s_hand = hands[2]  # South

        # Check N: 15-17 HCP, balanced
        n_hcp = count_hcp(n_hand)
        if not (15 <= n_hcp <= 17):
            continue
        if not is_balanced(n_hand):
            continue

        # Check S: 8-10 HCP, 4+ major
        # 上限 10: 排除高点力局面 (NS 合计 25-27+ HCP 时将牌约束失效)
        s_hcp = count_hcp(s_hand)
        if not (8 <= s_hcp <= 10):
            continue
        s_h = suit_length(s_hand, 2)
        s_s = suit_length(s_hand, 3)
        if s_h < 4 and s_s < 4:
            continue

        return hands

    return None


def deal_constrained_competitive(rng: np.random.Generator) -> Optional[np.ndarray]:
    """
    生成符合 Competitive (1H-1S) 约束的牌.

    N: 5+ hearts, 12-21 HCP
    E: 5+ spades, 8-16 HCP

    Returns:
        hands: (4, 52) one-hot, 或 None 如果尝试失败
    """
    deck = np.arange(52)
    
    for _ in range(5000):  # ~0.7% acceptance → ~150 expected tries, 5000 for safety
        rng.shuffle(deck)
        hands = np.zeros((4, 52), dtype=np.float32)
        for i, card in enumerate(deck):
            hands[i // 13, card] = 1.0

        n_hand = hands[0]
        e_hand = hands[1]

        # N: 5+ H, 12-21 HCP
        n_hcp = count_hcp(n_hand)
        if not (12 <= n_hcp <= 21):
            continue
        n_h = suit_length(n_hand, 2)
        if n_h < 5:
            continue

        # E: 5+ S, 8-16 HCP
        e_hcp = count_hcp(e_hand)
        if not (8 <= e_hcp <= 16):
            continue
        e_s = suit_length(e_hand, 3)
        if e_s < 5:
            continue

        return hands

    return None


# ============================================================================
# DDS solving
# ============================================================================

def solve_dds(hands: np.ndarray) -> np.ndarray:
    """
    用 endplay DDS 求解 double-dummy tricks.

    Args:
        hands: (4, 52) one-hot
    Returns:
        tricks: (5, 4) int8, tricks[suit][player]
              suit: 0=C, 1=D, 2=H, 3=S, 4=NT
              player: 0=N, 1=E, 2=S, 3=W
    """
    if not HAS_ENDPLAY:
        # Fallback: random tricks (for testing without endplay)
        return np.random.randint(0, 14, (5, 4)).astype(np.int8)

    deal = _hands_to_endplay_deal(hands)

    # Use endplay.dds.par / solve_board or dd_table
    # dd_table solves all 20 combinations at once (5 strains × 4 declarers)
    try:
        from endplay.dds import calc_dd_table
        from endplay.types import Player, Denom

        table = calc_dd_table(deal)

        tricks = np.zeros((5, 4), dtype=np.int8)

        # endplay Denom order and our suit order mapping:
        # Our: 0=C, 1=D, 2=H, 3=S, 4=NT
        denom_list = [Denom.clubs, Denom.diamonds, Denom.hearts,
                      Denom.spades, Denom.nt]
        player_list = [Player.north, Player.east, Player.south, Player.west]

        for our_suit, denom in enumerate(denom_list):
            for our_player, player in enumerate(player_list):
                tricks[our_suit, our_player] = table[denom, player]

        return tricks

    except Exception as e:
        print(f"DDS solve error: {e}")
        return np.zeros((5, 4), dtype=np.int8)


def _hands_to_endplay_deal(hands: np.ndarray):
    """
    Convert (4, 52) one-hot → endplay Deal via PBN string.

    Card index: suit * 13 + rank, rank 0=2, 1=3, ..., 12=A
    PBN format: "N:SAKQ.HAKQ.DAKQ.CAKQ E:... S:... W:..."
    PBN suit order: S.H.D.C (high to low)
    """
    from endplay.types import Deal

    RANK_CHARS = "23456789TJQKA"  # index 0='2', ..., 12='A'

    pbn_hands = []
    for p in range(4):  # N, E, S, W
        suits_str = []
        for s in [3, 2, 1, 0]:  # PBN order: S, H, D, C
            cards = []
            for r in range(12, -1, -1):  # High to low: A, K, Q, ..., 2
                if hands[p, s * 13 + r] > 0.5:
                    cards.append(RANK_CHARS[r])
            suits_str.append("".join(cards))
        pbn_hands.append(".".join(suits_str))

    pbn = "N:" + " ".join(pbn_hands)
    return Deal(pbn)


# ============================================================================
# Worker function for parallel generation
# ============================================================================

def _generate_worker(args: Tuple[int, int, str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Worker: 生成一批约束牌 + DDS 求解.

    Args:
        (batch_size, seed, deal_type)
    Returns:
        (decks, tricks)
    """
    batch_size, seed, deal_type = args
    rng = np.random.default_rng(seed)

    dealer_fn = deal_constrained_stayman if deal_type == 'stayman' else deal_constrained_competitive

    decks = np.zeros((batch_size, 52), dtype=np.uint8)
    tricks = np.zeros((batch_size, 5, 4), dtype=np.int8)

    generated = 0
    while generated < batch_size:
        hands = dealer_fn(rng)
        if hands is None:
            continue

        dd = solve_dds(hands)
        decks[generated] = hands_to_deck(hands)
        tricks[generated] = dd
        generated += 1

    return decks, tricks


# ============================================================================
# Main generation pipeline
# ============================================================================

def generate_subgame_data(
    deal_type: str,
    num_samples: int,
    output_path: str,
    num_workers: int = 4,
    seed: int = 42,
):
    """
    生成子博弈数据.

    Args:
        deal_type: 'stayman' or 'competitive'
        num_samples: 总样本数
        output_path: 输出 .npz 路径
        num_workers: 并行 worker 数
        seed: 随机种子
    """
    from tqdm import tqdm

    print(f"Generating {num_samples} {deal_type} deals with {num_workers} workers...")
    t0 = time.time()

    # 拆成小chunk（每chunk 500局），方便进度条更新
    chunk_size = 500
    n_chunks = (num_samples + chunk_size - 1) // chunk_size
    tasks = []
    remaining = num_samples
    for i in range(n_chunks):
        n = min(chunk_size, remaining)
        tasks.append((n, seed + i * 137, deal_type))
        remaining -= n

    # Run with progress bar
    all_decks_list = []
    all_tricks_list = []

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for decks, tricks in tqdm(
                pool.imap_unordered(_generate_worker, tasks),
                total=n_chunks,
                desc=f"{deal_type}",
                unit="chunk",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ):
                all_decks_list.append(decks)
                all_tricks_list.append(tricks)
    else:
        for task in tqdm(tasks, desc=f"{deal_type}", unit="chunk"):
            decks, tricks = _generate_worker(task)
            all_decks_list.append(decks)
            all_tricks_list.append(tricks)

    # Merge
    all_decks = np.concatenate(all_decks_list, axis=0)[:num_samples]
    all_tricks = np.concatenate(all_tricks_list, axis=0)[:num_samples]

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), decks=all_decks, tricks=all_tricks)

    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Saved {len(all_decks)} samples to {output_path} "
          f"({size_mb:.1f} MB, {elapsed:.1f}s, "
          f"{len(all_decks)/elapsed:.0f} samples/sec)")

    # Quick stats
    _print_stats(all_decks, all_tricks, deal_type)


def _print_stats(decks, tricks, deal_type):
    """打印数据统计."""
    n = min(1000, len(decks))
    sample_hands = np.array([deck_to_hands_local(decks[i]) for i in range(n)])

    hcps_n = [count_hcp(sample_hands[i, 0]) for i in range(n)]
    hcps_e = [count_hcp(sample_hands[i, 1]) for i in range(n)]
    hcps_s = [count_hcp(sample_hands[i, 2]) for i in range(n)]

    print(f"\nStats (first {n} samples):")
    print(f"  N HCP: {np.mean(hcps_n):.1f} ± {np.std(hcps_n):.1f} "
          f"(range {np.min(hcps_n)}-{np.max(hcps_n)})")

    if deal_type == 'stayman':
        print(f"  S HCP: {np.mean(hcps_s):.1f} ± {np.std(hcps_s):.1f}")
        # Check 4M rate
        has_4h = sum(1 for i in range(n) if suit_length(sample_hands[i, 2], 2) >= 4)
        has_4s = sum(1 for i in range(n) if suit_length(sample_hands[i, 2], 3) >= 4)
        print(f"  S has 4+H: {has_4h/n:.1%}, 4+S: {has_4s/n:.1%}")
    else:
        print(f"  E HCP: {np.mean(hcps_e):.1f} ± {np.std(hcps_e):.1f}")
        n_h = [suit_length(sample_hands[i, 0], 2) for i in range(n)]
        e_s = [suit_length(sample_hands[i, 1], 3) for i in range(n)]
        print(f"  N hearts: {np.mean(n_h):.1f} ± {np.std(n_h):.1f}")
        print(f"  E spades: {np.mean(e_s):.1f} ± {np.std(e_s):.1f}")

    # DD tricks distribution (NT by N)
    nt_n = tricks[:n, 4, 0]
    print(f"  NT by N tricks: {np.mean(nt_n):.1f} ± {np.std(nt_n):.1f}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate subgame DDS data")
    parser.add_argument('--type', choices=['stayman', 'competitive', 'both'],
                        default='both', help='Which subgame type (default: both)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples per type. '
                        'Default: stayman=50000, competitive=100000')
    parser.add_argument('--output', default=None,
                        help='Output .npz path (only for single type)')
    parser.add_argument('--output_dir', default='data/',
                        help='Output directory (used when --output not set)')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Default sample counts per type
    default_samples = {'stayman': 50000, 'competitive': 100000}

    if args.type in ('stayman', 'both'):
        n = args.num_samples or default_samples['stayman']
        out = args.output if (args.output and args.type == 'stayman') else \
            str(output_dir / f'stayman_{n // 1000}k.npz')
        generate_subgame_data('stayman', n, out, args.num_workers, args.seed)

    if args.type in ('competitive', 'both'):
        n = args.num_samples or default_samples['competitive']
        out = args.output if (args.output and args.type == 'competitive') else \
            str(output_dir / f'competitive_{n // 1000}k.npz')
        generate_subgame_data('competitive', n, out, args.num_workers,
                              args.seed + 99999)


if __name__ == '__main__':
    main()
