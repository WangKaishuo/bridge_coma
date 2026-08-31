"""Convert a 52-card hand into the 48-dimensional belief target.

The first 16 entries identify ownership of A, K, Q, and J in each suit. The
remaining 32 entries are four independent eight-bin one-hot suit lengths.
Lengths of seven or more cards share the final bin.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

BELIEF_DIM = 48
HONOR_DIM = 16
LENGTH_DIM = 32
NUM_SUITS = 4
HONOR_RANKS = [12, 11, 10, 9]
LENGTH_BINS = 8

HONOR_PRIOR = 13.0 / 52.0
HONOR_BIAS_INIT = math.log(HONOR_PRIOR / (1.0 - HONOR_PRIOR))
LENGTH_PRIOR = [0.01, 0.08, 0.24, 0.29, 0.21, 0.12, 0.04, 0.01]
LENGTH_BIAS_INIT = [math.log(probability + 1e-8) for probability in LENGTH_PRIOR]

# These aliases preserve the public interface of historical checkpoints.
HONOR_POS_WEIGHT = 1.0
LENGTH_POS_WEIGHT = 1.0
POS_WEIGHT = 1.0


def build_pos_weight() -> torch.Tensor:
    """Return unit weights; weighted BCE would distort probability calibration."""
    return torch.ones(BELIEF_DIM)


def hand_to_belief_target(hand: np.ndarray) -> np.ndarray:
    """Convert one suit-major 52-card one-hot hand to 48 binary features."""
    features = np.zeros(BELIEF_DIM, dtype=np.float32)
    index = 0
    for suit in range(NUM_SUITS):
        for rank in HONOR_RANKS:
            features[index] = hand[suit * 13 + rank]
            index += 1
    for suit in range(NUM_SUITS):
        suit_length = int(hand[suit * 13 : (suit + 1) * 13].sum())
        length_bin = min(suit_length, 7)
        features[HONOR_DIM + suit * LENGTH_BINS + length_bin] = 1.0
    return features


def batch_hand_to_belief_target(hands: np.ndarray) -> np.ndarray:
    """Vectorize belief-target conversion over a batch of hands."""
    batch_size = hands.shape[0]
    features = np.zeros((batch_size, BELIEF_DIM), dtype=np.float32)
    index = 0
    for suit in range(NUM_SUITS):
        for rank in HONOR_RANKS:
            features[:, index] = hands[:, suit * 13 + rank]
            index += 1
    rows = np.arange(batch_size)
    for suit in range(NUM_SUITS):
        suit_lengths = hands[:, suit * 13 : (suit + 1) * 13].sum(axis=1).astype(int)
        bins = np.minimum(suit_lengths, 7)
        features[rows, HONOR_DIM + suit * LENGTH_BINS + bins] = 1.0
    return features


def belief_accuracy(probs: torch.Tensor, targets: torch.Tensor) -> dict:
    """Return classification and calibration diagnostics for belief outputs."""
    honor_probs = probs[:, :HONOR_DIM]
    honor_targets = targets[:, :HONOR_DIM]
    honor_predictions = (honor_probs > 0.5).float()
    honor_acc = float((honor_predictions == honor_targets).float().mean())
    honor_brier = float(((honor_probs - honor_targets) ** 2).mean())
    honor_nll = float(
        F.binary_cross_entropy(
            honor_probs.clamp(1e-7, 1 - 1e-7),
            honor_targets,
            reduction="mean",
        )
    )

    length_probs = probs[:, HONOR_DIM:].view(-1, NUM_SUITS, LENGTH_BINS)
    length_targets = targets[:, HONOR_DIM:].view(-1, NUM_SUITS, LENGTH_BINS)
    predicted_bins = length_probs.argmax(dim=-1)
    true_bins = length_targets.argmax(dim=-1)
    length_acc = float((predicted_bins == true_bins).float().mean())
    length_nll = float(
        F.cross_entropy(
            length_probs.reshape(-1, LENGTH_BINS),
            true_bins.reshape(-1),
            reduction="mean",
        )
    )
    overall_acc = float(((probs > 0.5).float() == targets).float().mean())
    return {
        "honor_acc": honor_acc,
        "length_acc": length_acc,
        "overall_acc": overall_acc,
        "honor_brier": honor_brier,
        "honor_nll": honor_nll,
        "length_nll": length_nll,
    }


def decode_belief_target(features: np.ndarray) -> dict:
    """Decode the 48 features into readable honor and length dictionaries."""
    suit_names = ["C", "D", "H", "S"]
    honor_names = ["A", "K", "Q", "J"]
    honors = {}
    index = 0
    for suit in range(NUM_SUITS):
        honors[suit_names[suit]] = [
            honor_names[honor]
            for honor in range(4)
            if features[index + honor] > 0.5
        ]
        index += 4
    lengths = {}
    for suit in range(NUM_SUITS):
        section = features[
            HONOR_DIM + suit * LENGTH_BINS : HONOR_DIM + (suit + 1) * LENGTH_BINS
        ]
        length_bin = int(section.argmax())
        lengths[suit_names[suit]] = f"{length_bin}+" if length_bin == 7 else str(length_bin)
    return {"honors": honors, "lengths": lengths}
