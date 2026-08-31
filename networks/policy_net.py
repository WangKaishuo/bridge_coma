"""
Policy and Value Networks - MLP Architecture (P105: OpenSpiel-native 571-dim)
================================================================================

P105: Observation generation now uses pyspiel.State.observation_tensor() directly.
All custom encode_obs_flat functions have been REMOVED - they accumulated three
independent bugs (dealer assignment, encoding layout, hand encoding).

571-dim observation comes directly from:
    pyspiel.load_game('bridge(use_double_dummy_result=false)')
    state.observation_tensor()

Belief-conditioned actor:
    external API: 571 (OpenSpiel base)
    internal activation: 571 + 48 (partner belief) + 48 (RHO belief) = 667

Action mapping (our ordering <-> OpenSpiel):
    Our:      Pass=0, Dbl=1, Rdbl=2, 1C=3, 1D=4, ..., 7NT=37
    OpenSpiel: Pass=52, Dbl=53, Rdbl=54, 1C=55, ..., 7NT=89

Card encoding:
    OpenSpiel/SAYC data: rank-major (card_id = rank * 4 + suit)
    competitive_env:     suit-major (card_id = suit * 13 + rank)
    Use suit_major_to_rank_major() when crossing the boundary.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from utils.hand_features import HONOR_DIM, LENGTH_BINS, LENGTH_DIM, NUM_SUITS

try:
    import pyspiel
    _GAME = pyspiel.load_game('bridge(use_double_dummy_result=false)')
    OBS_DIM = _GAME.observation_tensor_shape()[0]  # 571
except ImportError:
    # Fallback if pyspiel not installed (e.g. for unit tests on networks only)
    OBS_DIM = 571
    _GAME = None

# Game cache for different dealers (populated lazily by get_openspiel_game)
_GAMES = {}  # will be populated in get_openspiel_game()

from env import NUM_BIDS, NUM_PLAYERS, BID_PASS, BID_DOUBLE, BID_REDOUBLE, BID_1C

# -- Constants ------------------------------------------------------------------
BASE_INPUT_DIM = OBS_DIM
ACTION_MAPPING_VERSION = "openspiel_native_52_89_v1"

# P101: Belief-conditioned Actor
BELIEF_FEAT_DIM = 96                            # 48 (partner) + 48 (RHO)
BELIEF_OBS_DIM  = OBS_DIM + BELIEF_FEAT_DIM     # 571 + 96 = 667

# OpenSpiel action constants
OS_MIN_ACTION = 52
OS_PASS       = 52
OS_DOUBLE     = 53
OS_REDOUBLE   = 54
OS_1C         = 55
OS_7NT        = 89

# Legacy aliases (for code that references old dims - will fail loudly if misused)
OBS_DIM_OLD    = 301
OBS_DIM_P104   = 480


# ==============================================================================
# Action Mapping (our ordering <-> OpenSpiel)
# ==============================================================================

def openspiel_raw_to_ours(os_action: int) -> int:
    """Convert OpenSpiel raw action (52-89) to our action (0-37)."""
    if os_action == OS_PASS:      return BID_PASS        # 0
    if os_action == OS_DOUBLE:    return BID_DOUBLE       # 1
    if os_action == OS_REDOUBLE:  return BID_REDOUBLE     # 2
    if OS_1C <= os_action <= OS_7NT:
        return BID_1C + (os_action - OS_1C)  # 3..37
    return -1


def ours_to_openspiel_raw(our_action: int) -> int:
    """Convert our action (0-37) to OpenSpiel raw action (52-89)."""
    if our_action == BID_PASS:      return OS_PASS
    if our_action == BID_DOUBLE:    return OS_DOUBLE
    if our_action == BID_REDOUBLE:  return OS_REDOUBLE
    if BID_1C <= our_action <= 37:  return OS_1C + (our_action - BID_1C)
    return -1


def physical_to_openspiel_player(player: int, dealer: int) -> int:
    """Map a physical N/E/S/W seat to the dealer-relative rolled state."""
    return (int(player) - int(dealer)) % NUM_PLAYERS


def encode_openspiel_auction_observation(
    hands_sm: np.ndarray,
    dealer: int,
    history_int: list,
    player: int,
    vulnerability: tuple = (False, False),
) -> np.ndarray:
    """Build OpenSpiel's 571-dim auction observation without replaying a state.

    This is a direct translation of BridgeState::WriteObservationTensor for
    the auction phase.  The unused play-phase tail remains zero.  Keeping this
    small, pure NumPy path avoids dealing 52 cards into a new pyspiel state for
    every rollout table.
    """
    values = np.zeros(OBS_DIM, dtype=np.float32)
    has_contract = any(int(call) >= BID_1C for call in history_int)
    opening_lead = (
        has_contract and len(history_int) >= 3
        and all(int(call) == BID_PASS for call in history_int[-3:])
    )
    values[1 if opening_lead else 0] = 1.0
    offset = 4

    ns_vul, ew_vul = bool(vulnerability[0]), bool(vulnerability[1])
    own_vul = ns_vul if player % 2 == 0 else ew_vul
    opp_vul = ew_vul if player % 2 == 0 else ns_vul
    values[offset + int(own_vul)] = 1.0
    offset += 2
    values[offset + int(opp_vul)] = 1.0
    offset += 2

    os_player = physical_to_openspiel_player(player, dealer)
    last_bid = 0
    for auction_idx, call in enumerate(history_int):
        call = int(call)
        relative_bidder = (auction_idx - os_player) % NUM_PLAYERS
        if last_bid == 0 and call == BID_PASS:
            values[offset + relative_bidder] = 1.0
        if call == BID_DOUBLE:
            base = offset + NUM_PLAYERS + (last_bid - BID_1C) * 12
            values[base + NUM_PLAYERS + relative_bidder] = 1.0
        elif call == BID_REDOUBLE:
            base = offset + NUM_PLAYERS + (last_bid - BID_1C) * 12
            values[base + 2 * NUM_PLAYERS + relative_bidder] = 1.0
        elif call != BID_PASS:
            last_bid = call
            base = offset + NUM_PLAYERS + (last_bid - BID_1C) * 12
            values[base + relative_bidder] = 1.0

    offset += NUM_PLAYERS * (1 + 3 * 35)
    values[offset:offset + 52] = convert_hands_suit_to_rank(hands_sm)[player]
    return values


# ==============================================================================
# Card Encoding Conversion
# ==============================================================================

def suit_major_to_rank_major(card_sm: int) -> int:
    """Convert suit-major card_id to rank-major card_id.
    suit-major: card = suit * 13 + rank  (used by competitive_env)
    rank-major: card = rank * 4 + suit   (used by OpenSpiel/SAYC)
    """
    suit = card_sm // 13
    rank = card_sm % 13
    return rank * 4 + suit


def rank_major_to_suit_major(card_rm: int) -> int:
    """Convert rank-major card_id to suit-major card_id."""
    rank = card_rm // 4
    suit = card_rm % 4
    return suit * 13 + rank


def convert_hands_suit_to_rank(hands_sm: np.ndarray) -> np.ndarray:
    """Convert (4, 52) hand matrix from suit-major to rank-major encoding.

    Input:  hands_sm[p, suit*13+rank] = 1.0
    Output: hands_rm[p, rank*4+suit]  = 1.0

    Perf (P109): vectorized via precomputed permutation index - avoids
    the O(4x52) Python loop that was a bottleneck in _encode_for_actor.
    """
    return hands_sm[:, _SM_TO_RM_IDX]


# Precomputed permutation: hands_rm = hands_sm[:, _SM_TO_RM_IDX]
# _SM_TO_RM_IDX[rm_idx] = sm_idx  such that hands_rm[p, rm_idx] = hands_sm[p, sm_idx]
# i.e. for each rank-major index, what is the suit-major index of the same card?
def _build_sm_to_rm_idx():
    idx = np.zeros(52, dtype=np.intp)
    for sm_idx in range(52):
        suit = sm_idx // 13
        rank = sm_idx % 13
        rm_idx = rank * 4 + suit
        idx[rm_idx] = sm_idx   # hands_rm[:, rm_idx] = hands_sm[:, sm_idx]
    return idx

_SM_TO_RM_IDX = _build_sm_to_rm_idx()


# ==============================================================================
# OpenSpiel State Helpers
# ==============================================================================


def get_openspiel_game(dealer: int = 0,
                       dealer_vul: bool = False,
                       non_dealer_vul: bool = False):
    """Get a cached OpenSpiel bridge game instance for the given parameters.

    P122: Added dealer_vul and non_dealer_vul support. Cache key expanded
    to (dealer, dealer_vul, non_dealer_vul).
    """
    key = (dealer, dealer_vul, non_dealer_vul)
    if key not in _GAMES:
        import pyspiel
        _GAMES[key] = pyspiel.load_game(
            f'bridge(use_double_dummy_result=false,dealer={dealer},'
            f'dealer_vul={"true" if dealer_vul else "false"},'
            f'non_dealer_vul={"true" if non_dealer_vul else "false"})')
    return _GAMES[key]


def hands_to_openspiel_state(hands_rm: np.ndarray, dealer: int = 0,
                             vulnerability: tuple = None):
    """
    Create an OpenSpiel state from a (4, 52) rank-major hand matrix.

    Args:
        hands_rm: (4, 52) float32, rank-major encoding (card_id = rank*4+suit).
                  If from competitive_env, convert first with convert_hands_suit_to_rank().
        dealer:   dealer seat (0=N, 1=E, 2=S, 3=W).
        vulnerability: (ns_vul, ew_vul) tuple of bools, or None for (False, False).

    Returns: pyspiel.State after dealing (ready for bidding)

    P107 fix: SAYC training data always has dealer=North (dealer=0). The SL model
    has only ever seen observations generated by GAME(dealer=0). Using a different
    dealer game produces observations with different semantics, causing the model
    to bid incoherently.

    Fix: always use GAME(dealer=0). When dealer != 0, roll the hands so that
    hands_rm[dealer] (the actual opener) is placed at index 0 before dealing.
    This makes the observation identical in structure to training data.
    The SL model is seat-agnostic: it only sees bidding history, not seat labels.

    P122: Added vulnerability support. The vul is specified as (ns_vul, ew_vul)
    and converted to OpenSpiel's (dealer_vul, non_dealer_vul) relative to the
    rolled dealer (always seat 0). When dealer is NS (d%2==0), dealer_vul=ns_vul.
    When dealer is EW (d%2==1), dealer_vul=ew_vul.

    P109 perf: cache the interleaved deal-action sequence keyed on the bytes of
    hands_to_deal. Building cards_per_player via np.where+sorted is O(4x52) and
    was called once per rollout step - this reduces it to O(1) on cache hit.
    """
    if vulnerability is None:
        vulnerability = (False, False)
    ns_vul, ew_vul = vulnerability

    # Convert (ns_vul, ew_vul) to OpenSpiel's (dealer_vul, non_dealer_vul)
    # After rolling, seat 0 = actual dealer. If actual dealer is NS, dealer_vul=ns_vul.
    if dealer % 2 == 0:  # dealer is N or S (NS)
        dealer_vul = ns_vul
        non_dealer_vul = ew_vul
    else:  # dealer is E or W (EW)
        dealer_vul = ew_vul
        non_dealer_vul = ns_vul

    # Always use dealer=0 game (matches SL training distribution)
    game = get_openspiel_game(0, dealer_vul, non_dealer_vul)

    # Roll hands so the actual opener sits at index 0
    if dealer == 0:
        hands_to_deal = hands_rm
    else:
        hands_to_deal = np.roll(hands_rm, -dealer, axis=0)

    # P109: cache the deal sequence - hands_to_deal is constant for a given deal
    cache_key = hands_to_deal.tobytes()
    deal_actions = _deal_action_cache.get(cache_key)
    if deal_actions is None:
        # P108 fix: OpenSpiel deals cards interleaved, not per-player consecutive.
        # Order: p0[0], p1[0], p2[0], p3[0], p0[1], p1[1], ..., p3[12]
        cards_per_player = [
            sorted(np.where(hands_to_deal[p] > 0.5)[0]) for p in range(4)
        ]
        for p in range(4):
            assert len(cards_per_player[p]) == 13, \
                f"Player {p} has {len(cards_per_player[p])} cards, expected 13"
        deal_actions = [
            int(cards_per_player[p][i])
            for i in range(13) for p in range(4)
        ]
        # Keep cache bounded to ~8000 unique deals
        # (covers Stage 2 training deals + Stage 3 eval 3000 deals)
        if len(_deal_action_cache) >= 8192:
            # Evict oldest entry (insertion-ordered dict in Python 3.7+)
            _deal_action_cache.pop(next(iter(_deal_action_cache)))
        _deal_action_cache[cache_key] = deal_actions

    state = game.new_initial_state()
    for card in deal_actions:
        state.apply_action(card)

    assert not state.is_chance_node(), "Dealing not complete"
    return state


# LRU-style cache for deal action sequences (P109 perf)
_deal_action_cache: dict = {}


def get_openspiel_obs(state, player: int | None = None) -> np.ndarray:
    """Return the 571-dimensional observation for a specific player.

    Passing the observer explicitly is essential for belief modelling: the
    public auction is shared, but each receiver conditions on a different
    private hand.  Policy calls may omit ``player`` when it is the current turn.
    """
    if player is None:
        tensor = state.observation_tensor()
    else:
        tensor = state.observation_tensor(player)
    return np.array(tensor, dtype=np.float32)


def advance_openspiel_state(state, our_action: int):
    """Apply our action (0-37) to an OpenSpiel state."""
    os_action = ours_to_openspiel_raw(our_action)
    state.apply_action(os_action)
    return state


# ==============================================================================
# Belief feature utilities (BCA extension)
# ==============================================================================

def make_belief_features_prior() -> np.ndarray:
    """
    Return belief prior feature vector (96,) = partner (48) + RHO (48).
    """
    honor_prior  = np.full(16, 0.25, dtype=np.float32)
    length_prior = np.full(32, 0.125, dtype=np.float32)
    single_prior = np.concatenate([honor_prior, length_prior])
    return np.concatenate([single_prior, single_prior])


def append_belief_features(flat_obs: np.ndarray,
                           belief_feats: np.ndarray) -> np.ndarray:
    """
    Concatenate base obs (571) + belief features (96) -> (667,).
    """
    return np.concatenate([flat_obs, belief_feats])


# ==============================================================================
# AllHandsEncoder (Critic only, unchanged)
# ==============================================================================

class AllHandsEncoder(nn.Module):
    """Centralized Critic: encode all 4 hands -> 256-dim."""
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

class ActorBeliefHead(nn.Module):
    """Trainable partner/RHO decoder used inside the deployed actor.

    Unlike the frozen training-time Judge, this module predicts two relative
    seats directly from the acting player's legal 571-dimensional observation:
    partner first, then right-hand opponent (RHO).  Its probabilities are an
    internal activation, not part of the policy's external API.
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = 1024):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.partner_honor_head = nn.Linear(hidden_dim, HONOR_DIM)
        self.partner_length_head = nn.Linear(hidden_dim, LENGTH_DIM)
        self.rho_honor_head = nn.Linear(hidden_dim, HONOR_DIM)
        self.rho_length_head = nn.Linear(hidden_dim, LENGTH_DIM)

    @staticmethod
    def _to_probs(honor_logits: torch.Tensor,
                  length_logits: torch.Tensor) -> torch.Tensor:
        honor = torch.sigmoid(honor_logits)
        length = F.softmax(
            length_logits.view(-1, NUM_SUITS, LENGTH_BINS), dim=-1
        ).view(-1, LENGTH_DIM)
        return torch.cat([honor, length], dim=-1)

    def forward(self, obs_571: torch.Tensor) -> torch.Tensor:
        h = self.trunk(obs_571)
        partner = self._to_probs(
            self.partner_honor_head(h), self.partner_length_head(h)
        )
        rho = self._to_probs(self.rho_honor_head(h), self.rho_length_head(h))
        return torch.cat([partner, rho], dim=-1)

    @staticmethod
    def _target_loss(honor_logits: torch.Tensor,
                     length_logits: torch.Tensor,
                     target: torch.Tensor) -> torch.Tensor:
        honor_loss = F.binary_cross_entropy_with_logits(
            honor_logits, target[:, :HONOR_DIM]
        )
        labels = target[:, HONOR_DIM:].view(
            -1, NUM_SUITS, LENGTH_BINS
        ).argmax(dim=-1)
        length_loss = F.cross_entropy(
            length_logits.view(-1, LENGTH_BINS), labels.reshape(-1)
        )
        return honor_loss + length_loss

    def compute_loss(self, obs_571: torch.Tensor,
                     targets: torch.Tensor) -> torch.Tensor:
        """Supervise relative partner/RHO predictions.

        ``targets`` has shape ``(batch, 2, 48)`` in partner, RHO order.
        """
        h = self.trunk(obs_571)
        partner_loss = self._target_loss(
            self.partner_honor_head(h), self.partner_length_head(h), targets[:, 0]
        )
        rho_loss = self._target_loss(
            self.rho_honor_head(h), self.rho_length_head(h), targets[:, 1]
        )
        return 0.5 * (partner_loss + rho_loss)

    def initialize_from_judge(self, judge) -> None:
        """Warm-start both relative heads from a pretrained frozen Judge."""
        with torch.no_grad():
            judge_first = judge.trunk[0]
            actor_first = self.trunk[0]
            actor_first.weight.copy_(judge_first.weight[:, :actor_first.in_features])
            actor_first.bias.copy_(judge_first.bias)
            self.trunk[2].load_state_dict(judge.trunk[2].state_dict())
            for honor_head in (self.partner_honor_head, self.rho_honor_head):
                honor_head.load_state_dict(judge.honor_head.state_dict())
            for length_head in (self.partner_length_head, self.rho_length_head):
                length_head.load_state_dict(judge.length_head.state_dict())


class MLPPolicyNetwork(nn.Module):
    """
    Actor: 4 x 1024 MLP.
    External input is always 571-dim (OpenSpiel native).
    In belief-conditioned mode, a trainable internal decoder supplies the
    additional 96 features before the policy MLP.

    Compatible with SL checkpoint (sl_pretrain.py MLPPolicy):
    Both use self.net = nn.Sequential(...) with identical layer structure,
    so state_dict keys match ('net.0.weight', 'net.0.bias', etc.).
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = 1024,
                 num_actions: int = NUM_BIDS,
                 belief_conditioned: bool = False,
                 belief_hidden_dim: Optional[int] = None):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.belief_conditioned = belief_conditioned
        self.belief_head = None
        policy_input_dim = obs_dim
        if belief_conditioned:
            if obs_dim != OBS_DIM:
                raise ValueError(
                    "Belief-conditioned actors require the 571-dimensional "
                    "OpenSpiel observation API"
                )
            self.belief_head = ActorBeliefHead(
                obs_dim=obs_dim,
                hidden_dim=belief_hidden_dim or hidden_dim,
            )
            policy_input_dim = BELIEF_OBS_DIM

        self.net = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        if belief_conditioned:
            # Preserve the exact 571-only policy at initialization.  PPO can
            # learn to use the supervised decoder features from a clean zero
            # contribution rather than receiving random belief-driven logits.
            with torch.no_grad():
                self.net[0].weight[:, OBS_DIM:].zero_()

    def compute_belief_features(self, flat_obs: torch.Tensor) -> torch.Tensor:
        """Generate the actor's native 96-dimensional belief activation."""

        if self.belief_head is None:
            raise RuntimeError("actor is not belief-conditioned")
        return self.belief_head(flat_obs)

    def _policy_input(self, flat_obs, belief_features=None):
        if self.belief_head is None:
            if belief_features is not None:
                raise ValueError(
                    "belief_features override requires a belief-conditioned actor"
                )
            return flat_obs
        if belief_features is None:
            belief_features = self.compute_belief_features(flat_obs)
        expected_shape = flat_obs.shape[:-1] + (BELIEF_FEAT_DIM,)
        if belief_features.shape != expected_shape:
            raise ValueError(
                f"belief_features must have shape {expected_shape}, "
                f"got {tuple(belief_features.shape)}"
            )
        if belief_features.device != flat_obs.device:
            raise ValueError("belief_features must be on the same device as flat_obs")
        if belief_features.dtype != flat_obs.dtype:
            raise ValueError("belief_features must have the same dtype as flat_obs")
        return torch.cat([flat_obs, belief_features], dim=-1)

    def _masked_logits(self, flat_obs, legal_actions, belief_features=None):
        logits = self.net(self._policy_input(flat_obs, belief_features))
        return logits - 1e9 * (1.0 - legal_actions)

    def compute_belief_loss(self, flat_obs, targets):
        if self.belief_head is None:
            return torch.zeros((), dtype=flat_obs.dtype, device=flat_obs.device)
        return self.belief_head.compute_loss(flat_obs, targets)

    def initialize_belief_from_judge(self, judge) -> None:
        if self.belief_head is not None:
            self.belief_head.initialize_from_judge(judge)

    def forward(self, flat_obs, legal_actions, belief_features=None):
        """Return masked logits, optionally under an explicit belief intervention.

        Omitting ``belief_features`` preserves the deployed behavior exactly:
        the actor's own belief head generates the internal activation.
        """

        return self._masked_logits(flat_obs, legal_actions, belief_features)

    def forward_with_belief_features(
        self, flat_obs, legal_actions, belief_features
    ):
        """Explicit inference-only-style entry point for heard/deaf overrides."""

        return self._masked_logits(flat_obs, legal_actions, belief_features)

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
    """Critic: 4 x 1024 MLP + optional AllHandsEncoder (CTDE)."""

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


# ==============================================================================
# SL Checkpoint Loading Helper
# ==============================================================================

def load_sl_into_mappo_agent(agent, sl_checkpoint_path: str):
    """
    Load a P105 SL checkpoint into a MAPPOAgent.

    P105 SL format: {'model_state': state_dict, 'obs_dim': 571, ...}
    Old SL format:  {'actor_n': state_dict, 'actor_s': state_dict, ...}

    The same weights are loaded into all 4 actor networks.
    """
    ckpt = torch.load(sl_checkpoint_path, map_location=agent.device,
                      weights_only=False)

    def _load_actor_compat(actor, source_state):
        """Load a legacy 571 actor into the 667-dim internal actor safely."""
        target = actor.state_dict()
        for key, value in source_state.items():
            if key not in target:
                continue
            value = value.to(agent.device)
            if target[key].shape == value.shape:
                target[key] = value
            elif (key == 'net.0.weight'
                  and target[key].ndim == 2
                  and value.ndim == 2
                  and target[key].shape[0] == value.shape[0]
                  and target[key].shape[1] > value.shape[1]):
                expanded = torch.zeros_like(target[key])
                expanded[:, :value.shape[1]] = value
                target[key] = expanded
            else:
                raise ValueError(
                    f"Unsupported SL tensor shape for {key}: "
                    f"checkpoint={tuple(value.shape)} "
                    f"target={tuple(target[key].shape)}"
                )
        actor.load_state_dict(target)

    if 'model_state' in ckpt:
        # P105 format
        sd = {k: v.to(agent.device) for k, v in ckpt['model_state'].items()}
        for player in range(NUM_PLAYERS):
            _load_actor_compat(agent.get_actor(player), sd)
    elif 'actor_n' in ckpt:
        # Old format (actor_n, actor_s, actor_e, actor_w)
        for player, key in [(0, 'actor_n'), (1, 'actor_e'),
                            (2, 'actor_s'), (3, 'actor_w')]:
            if key in ckpt:
                sd = {k: v.to(agent.device) for k, v in ckpt[key].items()}
                _load_actor_compat(agent.get_actor(player), sd)
    else:
        raise ValueError(f"Unknown SL checkpoint format. Keys: {list(ckpt.keys())}")

    encoding = ckpt.get('encoding', 'unknown')
    obs_dim  = ckpt.get('obs_dim', 'unknown')
    acc      = ckpt.get('non_pass_acc', ckpt.get('val_acc', '?'))
    print(f"[load_sl] Loaded SL checkpoint: encoding={encoding}, "
          f"obs_dim={obs_dim}, acc={acc}")
