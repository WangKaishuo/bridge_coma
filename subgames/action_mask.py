"""
Action Mask
===========

Level 1: 硬性合法 mask (复用 BridgeBiddingEnv 逻辑)
Level 2: 软性启发式 mask (HCP-based, 加速收敛)

用法:
    mask = get_combined_mask(hand, history, position, dealer)
    logits = logits - 1e9 * (1 - mask)
"""

import numpy as np

# 复用 env 常量
from env import (
    NUM_BIDS, BID_PASS, BID_DOUBLE, BID_REDOUBLE, BID_1C,
)


# ============================================================================
# Hand analysis helpers
# ============================================================================

def count_hcp(hand: np.ndarray) -> int:
    """
    计算 High Card Points (A=4, K=3, Q=2, J=1)

    Args:
        hand: (52,) one-hot, card index = suit*13 + rank, rank 0=2 .. 12=A
    """
    hcp = 0
    for suit in range(4):
        base = suit * 13
        if hand[base + 12] > 0.5:  # Ace
            hcp += 4
        if hand[base + 11] > 0.5:  # King
            hcp += 3
        if hand[base + 10] > 0.5:  # Queen
            hcp += 2
        if hand[base + 9] > 0.5:   # Jack
            hcp += 1
    return hcp


def count_suit_length(hand: np.ndarray, suit: int) -> int:
    """某花色的牌张数  suit: 0=C, 1=D, 2=H, 3=S"""
    base = suit * 13
    return int(hand[base:base + 13].sum())


def suit_lengths(hand: np.ndarray):
    """返回 (clubs, diamonds, hearts, spades)"""
    return tuple(count_suit_length(hand, s) for s in range(4))


def is_balanced(hand: np.ndarray) -> bool:
    """
    均型判断: 无单缺 (所有花色 >= 2)
    允许 5M / 6m (如 5332, 6322 等)
    """
    lengths = suit_lengths(hand)
    return all(l >= 2 for l in lengths)


def has_suit_length(hand: np.ndarray, suit: int, min_len: int) -> bool:
    return count_suit_length(hand, suit) >= min_len


# ============================================================================
# Level 1: Legal mask (mirrors BridgeBiddingEnv._get_legal_actions)
# ============================================================================

def get_legal_mask(history: list, current_player: int, dealer: int) -> np.ndarray:
    """
    硬性合法 mask — 叫品规则。

    直接从 history 计算, 不需要 env 实例。
    """
    legal = np.zeros(NUM_BIDS, dtype=np.float32)
    legal[BID_PASS] = 1.0

    # 找最高实质叫品
    highest_bid = None
    last_real_bid_idx = -1
    for i, bid in enumerate(history):
        if bid >= BID_1C:
            highest_bid = bid
            last_real_bid_idx = i

    # 新叫品必须更高
    min_bid = BID_1C if highest_bid is None else highest_bid + 1
    for bid in range(min_bid, NUM_BIDS):
        legal[bid] = 1.0

    # Double: 对手最后叫了实质叫品, 且未被加倍
    if last_real_bid_idx >= 0:
        bidder = (dealer + last_real_bid_idx) % 4
        if bidder % 2 != current_player % 2:
            doubled = False
            for bid in history[last_real_bid_idx + 1:]:
                if bid == BID_DOUBLE:
                    doubled = True
                elif bid == BID_REDOUBLE or bid >= BID_1C:
                    doubled = False
            if not doubled:
                legal[BID_DOUBLE] = 1.0

    # Redouble: 对手加倍了我方叫品
    if last_real_bid_idx >= 0:
        bidder = (dealer + last_real_bid_idx) % 4
        if bidder % 2 == current_player % 2:
            doubled_by_opp = False
            for j, bid in enumerate(history[last_real_bid_idx + 1:]):
                bp = (dealer + last_real_bid_idx + 1 + j) % 4
                if bid == BID_DOUBLE and bp % 2 != current_player % 2:
                    doubled_by_opp = True
                elif bid == BID_REDOUBLE or bid >= BID_1C:
                    doubled_by_opp = False
            if doubled_by_opp:
                legal[BID_REDOUBLE] = 1.0

    return legal


# ============================================================================
# Level 2: Soft heuristic mask (domain knowledge, optional)
# ============================================================================

def get_soft_mask(hand: np.ndarray, history: list, position: int,
                  dealer: int) -> np.ndarray:
    """
    软性合理 mask — 启发式规则, 减少无意义探索。

    规则:
    - 极弱牌 (<5 HCP) 在开叫位不开叫 (保留 Pass)
    - 极弱牌 (<5 HCP) 不做新花色叫品 (只保留 Pass / raise)
    """
    soft = np.ones(NUM_BIDS, dtype=np.float32)

    hcp = count_hcp(hand)

    # 判断是否为开叫位: history 中无实质叫品
    has_real = any(b >= BID_1C for b in history)

    if not has_real and hcp < 5:
        # 极弱牌不开叫
        for bid in range(BID_1C, NUM_BIDS):
            soft[bid] = 0.0

    return soft


# ============================================================================
# Combined mask
# ============================================================================

def get_combined_mask(hand: np.ndarray, history: list, current_player: int,
                      dealer: int, use_soft: bool = True) -> np.ndarray:
    """
    合并 legal + soft mask, 确保至少保留 Pass。
    """
    legal = get_legal_mask(history, current_player, dealer)

    if use_soft:
        soft = get_soft_mask(hand, history, current_player, dealer)
        combined = legal * soft
    else:
        combined = legal

    # 保底: 至少有 Pass
    if combined.sum() < 0.5:
        combined[BID_PASS] = 1.0

    return combined
