"""Small, deterministic bridge hand-feature helpers.

Cards use suit-major indexing: ``card = suit * 13 + rank`` with suits ordered
clubs, diamonds, hearts, spades and ranks ordered from two to ace.
"""

from __future__ import annotations

import numpy as np


def count_hcp(hand: np.ndarray) -> int:
    """Return Milton Work high-card points (A=4, K=3, Q=2, J=1)."""
    points = 0
    for suit in range(4):
        offset = suit * 13
        points += 4 * int(hand[offset + 12] > 0.5)
        points += 3 * int(hand[offset + 11] > 0.5)
        points += 2 * int(hand[offset + 10] > 0.5)
        points += int(hand[offset + 9] > 0.5)
    return points


def count_suit_length(hand: np.ndarray, suit: int) -> int:
    """Return the number of cards held in ``suit`` (0=C, 1=D, 2=H, 3=S)."""
    if suit not in range(4):
        raise ValueError(f"suit must be in [0, 3], got {suit}")
    offset = suit * 13
    return int(hand[offset:offset + 13].sum())


def suit_lengths(hand: np.ndarray) -> tuple[int, int, int, int]:
    """Return suit lengths in clubs, diamonds, hearts, spades order."""
    return tuple(count_suit_length(hand, suit) for suit in range(4))


def is_balanced(hand: np.ndarray) -> bool:
    """Return whether the hand has at least two cards in every suit."""
    return all(length >= 2 for length in suit_lengths(hand))
