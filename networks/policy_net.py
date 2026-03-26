"""
Policy and Value Networks — MLP Architecture (P104: 480-dim OpenSpiel encoding)
==================================================================================

P104: Observation encoding rewritten to match the OpenSpiel/Pgx/Lockhart+20
standard (480-dim), replacing the previous 301-dim custom encoding.

The old 301-dim encoding had three fatal flaws:
  1. Absolute player positions (N/E/S/W) instead of relative (self/LHO/partner/RHO)
  2. Complete loss of Pass information (only substance bids were recorded)
  3. No "pass before opening" indicator

The new 480-dim encoding matches the standard used by:
  - OpenSpiel (Lanctot et al., 2019)
  - Pgx (Koyamada et al., 2023)
  - JPS (Tian et al., NeurIPS 2020)
  - Lockhart et al. (NeurIPS 2020 Workshop)
  - Kita et al. (CoG 2024)

480-dim layout (all binary):
    obs[  0:  4]  Vulnerability (one-hot, 4 combos)
    obs[  4:  8]  "Pass before opening" per relative player (4 bits)
    obs[  8:428]  Bidding history: 35 bids × 12 bits each = 420
                    Per bid b (12 bits):
                      [self_bid, LHO_bid, partner_bid, RHO_bid,    ← who made this bid
                       self_dbl, LHO_dbl, partner_dbl, RHO_dbl,    ← who doubled it
                       self_rdbl, LHO_rdbl, partner_rdbl, RHO_rdbl] ← who redoubled it
    obs[428:480]  My hand (52-dim, 13-hot)
    ─────────────────────────────────────────────────
    Total: 480 binary features

P101 BCA extension (576-dim):
    480 (base) + 48 (partner belief) + 48 (RHO belief) = 576

Note on relative player mapping:
    The current player is always player 0 (self).
    Relative positions rotate based on who is currently acting:
      rel_player = (absolute_player - current_player) % 4
      0 = self, 1 = LHO, 2 = partner, 3 = RHO
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from env import NUM_BIDS, NUM_PLAYERS, BID_PASS, BID_DOUBLE, BID_REDOUBLE, BID_1C

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_REAL_BIDS  = 35      # 1C–7NT (bid index 3–37)
OBS_DIM        = 480     # P104: OpenSpiel/Pgx standard
BASE_INPUT_DIM = OBS_DIM

# P101: Belief-conditioned Actor
BELIEF_FEAT_DIM = 96     # 48 (partner) + 48 (RHO)
BELIEF_OBS_DIM  = OBS_DIM + BELIEF_FEAT_DIM  # 480 + 96 = 576

# Legacy alias
OBS_DIM_OLD = 301


# ==============================================================================
# Observation Encoding (P104: 480-dim OpenSpiel standard)
# ==============================================================================

def encode_obs_flat(obs: Dict[str, np.ndarray],
                    dealer: int,
                    history_int: list) -> np.ndarray:
    """
    Encode observation to 480-dim binary vector (OpenSpiel/Pgx standard).

    This matches the encoding used by Lockhart+20, Tian+20, Kita+24, and
    the OpenSpiel/Pgx bridge_bidding environments.

    Args:
        obs         : BridgeBiddingEnv._get_observation() dict
        dealer      : dealer position (0=N, 1=E, 2=S, 3=W)
        history_int : complete bidding history (int list, incl. Pass/X/XX/bids)

    Returns:
        flat: (480,) float32 binary vector
    """
    hand = obs['hand']                # (52,)
    vul  = obs['vulnerability']       # (2,) → NS_vul, EW_vul

    # ── Determine current player ─────────────────────────────────────────
    # Current player = (dealer + len(history)) % 4
    current_player = (dealer + len(history_int)) % NUM_PLAYERS

    # ── Vulnerability: 4-dim one-hot ─────────────────────────────────────
    ns_vul = bool(vul[0] > 0.5)
    ew_vul = bool(vul[1] > 0.5)
    vul4 = np.zeros(4, dtype=np.float32)
    vul4[int(ns_vul) * 2 + int(ew_vul)] = 1.0

    # ── "Pass before opening" (4 bits, relative positions) ───────────────
    # For each relative player, did they pass before the first substance bid?
    pass_before_opening = np.zeros(4, dtype=np.float32)
    for step_idx, bid in enumerate(history_int):
        if bid >= BID_1C:
            # First substance bid found — stop
            break
        if bid == BID_PASS:
            abs_player = (dealer + step_idx) % NUM_PLAYERS
            rel_player = (abs_player - current_player) % NUM_PLAYERS
            pass_before_opening[rel_player] = 1.0

    # ── Bidding history: 35 bids × 12 bits ──────────────────────────────
    # For each real bid (1C..7NT), track who bid/doubled/redoubled it
    # in relative positions.
    #
    # Layout per bid b (12 bits):
    #   [self_bid, LHO_bid, partner_bid, RHO_bid,
    #    self_dbl, LHO_dbl, partner_dbl, RHO_dbl,
    #    self_rdbl, LHO_rdbl, partner_rdbl, RHO_rdbl]
    bid_history = np.zeros((NUM_REAL_BIDS, 12), dtype=np.float32)

    # Track what the last real bid was (for associating doubles)
    last_real_bid_idx = -1  # index into 0..34

    for step_idx, bid in enumerate(history_int):
        abs_player = (dealer + step_idx) % NUM_PLAYERS
        rel_player = (abs_player - current_player) % NUM_PLAYERS

        if bid >= BID_1C:
            # Substance bid: record who made it
            real_idx = bid - BID_1C  # 0..34
            bid_history[real_idx, 0 + rel_player] = 1.0  # bid slot
            last_real_bid_idx = real_idx

        elif bid == BID_DOUBLE and last_real_bid_idx >= 0:
            # Double: record who doubled the last real bid
            bid_history[last_real_bid_idx, 4 + rel_player] = 1.0

        elif bid == BID_REDOUBLE and last_real_bid_idx >= 0:
            # Redouble: record who redoubled the last real bid
            bid_history[last_real_bid_idx, 8 + rel_player] = 1.0

        # Pass: no explicit recording here (handled by pass_before_opening
        # and implicitly by the absence of bid markers)

    # ── Assemble 480-dim vector ──────────────────────────────────────────
    flat = np.concatenate([
        vul4,                           #   4
        pass_before_opening,            #   4
        bid_history.flatten(),          # 420  (35 × 12)
        hand,                           #  52
    ])
    assert flat.shape == (OBS_DIM,), f"Expected {OBS_DIM}, got {flat.shape}"
    return flat


def batch_encode_obs(obs_list, dealers, history_ints):
    """Batch encode: returns (B, 480) float32 ndarray."""
    return np.stack([
        encode_obs_flat(o, d, h)
        for o, d, h in zip(obs_list, dealers, history_ints)
    ])


# ==============================================================================
# Legacy 301-dim encoder (for backward compatibility during migration)
# ==============================================================================

def encode_obs_flat_legacy(obs: Dict[str, np.ndarray],
                           dealer: int,
                           history_int: list) -> np.ndarray:
    """
    Legacy 301-dim encoding (pre-P104). Kept for reference and migration testing.
    DO NOT USE for new experiments.
    """
    vul  = obs['vulnerability']
    hand = obs['hand']

    ns_vul, ew_vul = bool(vul[0] > 0.5), bool(vul[1] > 0.5)
    vul4 = np.zeros(4, dtype=np.float32)
    vul4[int(ns_vul) * 2 + int(ew_vul)] = 1.0

    who_called   = np.zeros((NUM_REAL_BIDS, 4), dtype=np.float32)
    double_state = np.zeros((NUM_REAL_BIDS, 3), dtype=np.float32)
    last_real_bid_real_idx = -1

    for step_idx, bid in enumerate(history_int):
        caller = (dealer + step_idx) % NUM_PLAYERS
        if bid >= 3:
            real_idx = bid - 3
            who_called[real_idx, caller] = 1.0
            double_state[real_idx, 0] = 1.0
            last_real_bid_real_idx = real_idx
        elif bid == 1 and last_real_bid_real_idx >= 0:
            ri = last_real_bid_real_idx
            double_state[ri, 0] = 0.0
            double_state[ri, 1] = 1.0
        elif bid == 2 and last_real_bid_real_idx >= 0:
            ri = last_real_bid_real_idx
            double_state[ri, 1] = 0.0
            double_state[ri, 2] = 1.0

    flat = np.concatenate([vul4, hand, who_called.flatten(), double_state.flatten()])
    assert flat.shape == (OBS_DIM_OLD,), f"Expected {OBS_DIM_OLD}, got {flat.shape}"
    return flat


# ==============================================================================
# Belief feature utilities (unchanged from P101)
# ==============================================================================

def make_belief_features_prior() -> np.ndarray:
    """
    P101: Return belief prior feature vector (96,) = partner (48) + RHO (48).
    """
    honor_prior = np.full(16, 0.25, dtype=np.float32)
    length_prior = np.full(32, 0.125, dtype=np.float32)
    single_prior = np.concatenate([honor_prior, length_prior])
    return np.concatenate([single_prior, single_prior])


def append_belief_features(flat_obs: np.ndarray,
                           belief_feats: np.ndarray) -> np.ndarray:
    """
    P101: Concatenate base obs (480) + belief features (96) → (576,).
    """
    return np.concatenate([flat_obs, belief_feats])


def encode_history_flat(history: torch.Tensor) -> torch.Tensor:
    """
    Compress bidding history to fixed-dim vector for BeliefNetwork.
    (Unchanged — BeliefNetwork uses its own encoding.)
    """
    bid_presence = history.max(dim=1).values
    return bid_presence.repeat(1, NUM_PLAYERS)


# ==============================================================================
# AllHandsEncoder (Critic only, unchanged)
# ==============================================================================

class AllHandsEncoder(nn.Module):
    """Centralized Critic: encode all 4 hands → 256-dim."""
    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * 52, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.ReLU(),
        )

    def forward(self, all_hands: torch.Tensor) -> torch.Tensor:
        return self.net(all_hands.view(all_hands.size(0), -1))


# ==============================================================================
# MLPPolicyNetwork (Actor)
# ==============================================================================

class MLPPolicyNetwork(nn.Module):
    """
    Actor: 4 × 1024 MLP.
    P104: Input is 480-dim (or 576-dim with belief features).
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = 1024,
                 num_actions: int = NUM_BIDS):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def _masked_logits(self, flat_obs, legal_actions):
        logits = self.net(flat_obs)
        return logits - 1e9 * (1.0 - legal_actions)

    def forward(self, flat_obs, legal_actions):
        return self._masked_logits(flat_obs, legal_actions)

    def get_action(self, flat_obs, legal_actions, deterministic=False):
        logits = self._masked_logits(flat_obs, legal_actions)
        probs  = F.softmax(logits, dim=-1)
        dist   = torch.distributions.Categorical(probs)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(self, flat_obs, legal_actions, actions):
        logits = self._masked_logits(flat_obs, legal_actions)
        probs  = F.softmax(logits, dim=-1)
        dist   = torch.distributions.Categorical(probs)
        return dist.log_prob(actions), dist.entropy()


# ==============================================================================
# MLPValueNetwork (Critic)
# ==============================================================================

class MLPValueNetwork(nn.Module):
    """Critic: 4 × 1024 MLP + optional AllHandsEncoder (CTDE)."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = 1024,
                 centralized: bool = True):
        super().__init__()
        self.centralized = centralized

        if centralized:
            self.all_hands_encoder = AllHandsEncoder(output_dim=256)
            input_dim = obs_dim + 256
        else:
            input_dim = obs_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, flat_obs, all_hands=None):
        if self.centralized and all_hands is not None:
            hand_feat = self.all_hands_encoder(all_hands)
            x = torch.cat([flat_obs, hand_feat], dim=-1)
        else:
            x = flat_obs
        return self.net(x).squeeze(-1)


# ==============================================================================
# Backward-compatible aliases
# ==============================================================================

PolicyNetwork = MLPPolicyNetwork
ValueNetwork  = MLPValueNetwork


class ActorCritic(nn.Module):
    """Backward-compatible container."""
    def __init__(self, obs_dim=OBS_DIM, hidden_dim=1024,
                 centralized_critic=True, **_ignored):
        super().__init__()
        self.actor  = MLPPolicyNetwork(obs_dim, hidden_dim)
        self.critic = MLPValueNetwork(obs_dim, hidden_dim, centralized_critic)

    def get_action_and_value(self, flat_obs, legal_actions,
                             all_hands=None, deterministic=False):
        action, log_prob, entropy = self.actor.get_action(
            flat_obs, legal_actions, deterministic)
        value = self.critic(flat_obs, all_hands)
        return action, log_prob, entropy, value
