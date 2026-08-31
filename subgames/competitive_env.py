"""Competitive bridge-bidding subgame with a fixed 1H-1S prefix.

All four seats continue the auction.  Dealer rotation preserves the constrained
opener/overcaller deal distribution.  ``play_mixed`` assigns the opener and
overcaller partnerships to independent black-box policies.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

from env import (
    BridgeBiddingEnv, NUM_BIDS, NUM_PLAYERS,
    BID_PASS, BID_DOUBLE, BID_1C,
    bid_to_string, string_to_bid,
    NORTH, EAST, SOUTH, WEST,
)
from utils.scoring import Contract, calculate_score
from utils.imp import score_to_imp
from utils.dds_data import create_loader
from utils.bridge_features import count_hcp, count_suit_length


FIXED_PREFIX = ["1H", "1S"]
_SUIT_H = 2   # hearts index
_SUIT_S = 3   # spades index


# ==============================================================================
# ==============================================================================

def _rule_based_action(obs: Dict[str, np.ndarray], player: int,
                        history: list, dealer: int) -> int:
    """Return a conservative legal action for fallback BC initialization."""
    hand         = obs['hand']             # (52,) float
    legal        = obs['legal_actions']    # (38,) float

    def _legal(bid: int) -> bool:
        return bool(legal[bid] > 0.5)

    def _bid_if_legal(bid: int) -> int:
        return bid if _legal(bid) else BID_PASS

    hcp  = int(count_hcp(hand))
    h    = int(count_suit_length(hand, _SUIT_H))
    s    = int(count_suit_length(hand, _SUIT_S))

    hist_len = len(history)

    if player == SOUTH and hist_len == 2:
        bid_2h = string_to_bid("2H")
        bid_2s = string_to_bid("2S")
        bid_2c = string_to_bid("2C")
        bid_2d = string_to_bid("2D")

        if h >= 3 and hcp >= 8:
            return _bid_if_legal(bid_2h)
        elif hcp >= 10:
            d = int(count_suit_length(hand, 1))
            c = int(count_suit_length(hand, 0))
            if d >= 4 and _legal(bid_2d):
                return bid_2d
            if c >= 4 and _legal(bid_2c):
                return bid_2c
            return _bid_if_legal(bid_2s)
        else:
            return BID_PASS

    elif player == WEST and hist_len == 3:
        bid_2s = string_to_bid("2S")
        bid_3s = string_to_bid("3S")

        if s >= 4 and hcp >= 10:
            if s >= 6 or hcp >= 12:
                return _bid_if_legal(bid_3s)
            return _bid_if_legal(bid_2s)
        return BID_PASS

    # -- N rebid(hist_len == 4)----------------------------------------------
    elif player == NORTH and hist_len == 4:
        bid_2h = string_to_bid("2H")
        bid_3h = string_to_bid("3H")
        bid_4h = string_to_bid("4H")

        if hcp >= 18:
            return _bid_if_legal(bid_4h)
        elif hcp >= 15:
            return _bid_if_legal(bid_3h)
        elif hcp >= 12:
            return _bid_if_legal(bid_2h)
        return BID_PASS

    # -- E rebid(hist_len == 5)----------------------------------------------
    elif player == EAST and hist_len == 5:
        bid_2s = string_to_bid("2S")
        bid_3s = string_to_bid("3S")
        bid_4s = string_to_bid("4S")

        if hcp >= 14:
            return _bid_if_legal(bid_4s)
        elif hcp >= 12:
            return _bid_if_legal(bid_3s)
        elif hcp >= 9:
            return _bid_if_legal(bid_2s)
        return BID_PASS

    return BID_PASS


def generate_rule_based_bc_data(
    env: "CompetitiveSubgameEnv",
    num_samples: int = 5000,
) -> List[Dict]:
    """See the formal README for the current behavior contract."""
    from networks.policy_net import (
        convert_hands_suit_to_rank, hands_to_openspiel_state,
        get_openspiel_obs, ours_to_openspiel_raw,
    )

    data = []
    attempts = 0
    max_attempts = num_samples * 20

    while len(data) < num_samples and attempts < max_attempts:
        attempts += 1
        hands, dd_table = env.generate_deal()
        dealer = env._sampled_dealer
        vul = (False, False)
        hands_rm = convert_hands_suit_to_rank(hands)

        inner_env = BridgeBiddingEnv(max_history_len=60)
        obs = inner_env.reset(hands, dealer=dealer, vulnerability=vul)
        done = False
        history_ours = []

        prefix_ok = True
        for bid_str in FIXED_PREFIX:
            bid = string_to_bid(bid_str)
            if not inner_env._is_valid_action(bid):
                prefix_ok = False
                break
            history_ours.append(bid)
            obs, _, done, _ = inner_env.step(bid)
            if done:
                break

        if not prefix_ok or done:
            continue

        while not done:
            player  = inner_env.state.current_player
            history = inner_env.state.history[:]

            action = _rule_based_action(obs, player, history, dealer)
            if not inner_env._is_valid_action(action):
                action = BID_PASS

            os_state = hands_to_openspiel_state(hands_rm, dealer)
            for a in history_ours:
                os_state.apply_action(ours_to_openspiel_raw(a))
            flat = get_openspiel_obs(os_state)
            data.append({'flat_obs': flat, 'action': action})

            history_ours.append(action)
            obs, _, done, _ = inner_env.step(action)

    print(f"[BC Data] Generated {len(data)} samples "
          f"from {attempts} attempts (acceptance: {len(data)/max(1,attempts):.1%})")
    return data


# ==============================================================================
# CompetitiveSubgameEnv
# ==============================================================================

class CompetitiveSubgameEnv:
    """See the formal README for the current behavior contract."""

    def __init__(self, data_path: str, max_history_len: int = 60):
        self.loader          = create_loader(data_path)
        self.env             = BridgeBiddingEnv(max_history_len)
        self.max_history_len = max_history_len
        self.dealer          = NORTH
        self._sampled_dealer: int = NORTH  # set by generate_deal(), used by reset()

        self._is_constrained_data = self._check_if_constrained()
        if not self._is_constrained_data:
            self._filtered_deals: list = []
            self._prefetch(min_deals=500, max_attempts=50000)
        else:
            self._filtered_deals = None
            print(f"[CompetitiveEnv] Pre-generated constrained data: "
                  f"{len(self.loader)} samples")

        self._current_hands: Optional[np.ndarray] = None
        self._current_dd:    Optional[np.ndarray] = None
        self._vulnerability: Tuple[bool, bool]    = (False, False)
        self.history_int:    list                 = []

    @property
    def initial_history_length(self) -> int:
        """Number of forced public calls present after ``reset``."""
        return len(FIXED_PREFIX)

    @property
    def initial_history_actions(self) -> List[int]:
        return [string_to_bid(call) for call in FIXED_PREFIX]

    def clone_for_worker(self):
        """Create a lightweight rollout worker sharing the read-only loader."""
        clone = type(self).__new__(type(self))
        clone.loader = self.loader
        clone.env = BridgeBiddingEnv(self.max_history_len)
        clone.max_history_len = self.max_history_len
        clone.dealer = NORTH
        clone._sampled_dealer = NORTH
        if hasattr(self, "_is_constrained_data"):
            clone._is_constrained_data = self._is_constrained_data
        if hasattr(self, "_filtered_deals"):
            clone._filtered_deals = self._filtered_deals
        clone._current_hands = None
        clone._current_dd = None
        clone._vulnerability = (False, False)
        clone.history_int = []
        return clone

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def _check_if_constrained(self, sample_size: int = 20) -> bool:
        passed = 0
        for _ in range(sample_size):
            hands, _ = self.loader.sample_one()
            # Use NORTH as reference dealer for the check (pre-generated data is N-opener)
            if self._satisfies_constraints(hands, dealer=NORTH):
                passed += 1
        return passed >= sample_size * 0.9

    def _prefetch(self, min_deals: int, max_attempts: int):
        attempts = 0
        rng = np.random.default_rng(42)
        while len(self._filtered_deals) < min_deals and attempts < max_attempts:
            attempts += 1
            hands, dd_table = self.loader.sample_one()
            # Check all 4 rotations - store as (hands, dd_table, dealer)
            for dealer in range(NUM_PLAYERS):
                if self._satisfies_constraints(hands, dealer=dealer):
                    self._filtered_deals.append((hands, dd_table, dealer))
                    break  # one rotation per deal to avoid bias
        rate = len(self._filtered_deals) / max(1, attempts)
        print(f"[CompetitiveEnv] Prefetched {len(self._filtered_deals)} deals "
              f"({rate:.1%} acceptance rate)")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def _satisfies_constraints(self, hands: np.ndarray,
                               dealer: int = NORTH) -> bool:
        opener_seat     = dealer
        overcaller_seat = (dealer + 1) % NUM_PLAYERS
        return (self._satisfies_opener(hands[opener_seat]) and
                self._satisfies_overcaller(hands[overcaller_seat]))

    @staticmethod
    def _satisfies_opener(hand: np.ndarray) -> bool:
        """N: 5+H, 12-21 HCP."""
        return 12 <= count_hcp(hand) <= 21 and count_suit_length(hand, _SUIT_H) >= 5

    @staticmethod
    def _satisfies_overcaller(hand: np.ndarray) -> bool:
        """E: 5+S, 8-16 HCP."""
        return 8 <= count_hcp(hand) <= 16 and count_suit_length(hand, _SUIT_S) >= 5

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def generate_deal(self) -> Tuple[np.ndarray, np.ndarray]:
        """See the formal README for the current behavior contract."""
        if self._is_constrained_data:
            # Pre-generated data: N(pos0)=opener, E(pos1)=overcaller.
            # To make dealer=rotation be the opener, we need opener(pos0) -> pos(rotation).
            # np.roll(+k) shifts pos0 -> posk. (NOT -k, which was a bug!)
            hands, dd_table = self.loader.sample_one()
            rotation = np.random.randint(NUM_PLAYERS)
            hands    = np.roll(hands, rotation, axis=0)      # opener(0) -> pos(rotation)
            dd_table = np.roll(dd_table, rotation, axis=1)   # declarer axis matches
            self._sampled_dealer = rotation
            return hands, dd_table

        if self._filtered_deals:
            idx = np.random.randint(len(self._filtered_deals))
            entry = self._filtered_deals[idx]
            if len(entry) == 3:
                hands, dd_table, dealer = entry
            else:
                hands, dd_table = entry; dealer = NORTH
            self._sampled_dealer = dealer
            return hands, dd_table

        # Fallback: search with random dealer rotation
        for _ in range(10000):
            hands, dd_table = self.loader.sample_one()
            dealer = np.random.randint(NUM_PLAYERS)
            rotated = np.roll(hands, -dealer, axis=0)
            if self._satisfies_constraints(rotated, dealer=NORTH):
                self._sampled_dealer = dealer
                return np.roll(hands, -dealer, axis=0), np.roll(dd_table, -dealer, axis=1)

        raise RuntimeError("Cannot generate valid competitive deal after 10000 attempts")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def reset(
        self,
        hands:         Optional[np.ndarray]   = None,
        dd_table:      Optional[np.ndarray]   = None,
        vulnerability: Tuple[bool, bool]       = (False, False),
        dealer:        Optional[int]           = None,
    ) -> Dict[str, np.ndarray]:
        """Reset the constrained auction and apply the fixed ``1H-1S`` prefix.

        Generated deals carry a sampled dealer so that the constrained N/E
        dataset can be rotated without changing its relative roles.  External
        callers remain backward compatible: when no dealer is supplied, an
        explicitly provided deal is interpreted using North as dealer.
        """
        if hands is None or dd_table is None:
            hands, dd_table = self.generate_deal()
            generated_dealer = self._sampled_dealer
        else:
            generated_dealer = NORTH

        if dealer is None:
            dealer = generated_dealer
        dealer = int(dealer)

        self._current_hands = hands
        self._current_dd    = dd_table
        self._vulnerability = vulnerability
        self.dealer         = dealer
        self.history_int    = []

        obs = self.env.reset(hands, dealer=dealer, vulnerability=vulnerability)

        # Fixed prefix: opener (dealer) bids 1H; overcaller (dealer+1) bids 1S
        for bid_str in FIXED_PREFIX:
            bid = string_to_bid(bid_str)
            self.history_int.append(bid)
            obs, _, done, _ = self.env.step(bid)
            if done:
                break

        return obs

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """See the formal README for the current behavior contract."""
        self.history_int.append(action)
        obs, _, done, info = self.env.step(action)

        reward = 0.0
        if done:
            reward = self._compute_terminal_reward()
            info['imp'] = reward

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def _compute_terminal_reward(self) -> float:
        """See the formal README for the current behavior contract."""
        contract = self.env.state.final_contract
        score_ns = self._compute_score_ns(
            contract, self._current_dd, self._vulnerability)

        opt_score = self._compute_dds_optimal_score_ns(
            self._current_dd, self._vulnerability, self.dealer)

        return float(score_to_imp(score_ns - opt_score))

    def _compute_dds_optimal_score_ns(
        self,
        dd_table:      np.ndarray,
        vulnerability: Tuple[bool, bool],
        dealer: int,
    ) -> int:
        """Return the audited dealer-par DDS reference in NS score units."""
        from utils.dds_reference import dds_par_score_ns

        return dds_par_score_ns(dd_table, vulnerability, dealer)

    def _compute_score_ns(
        self,
        contract:      Optional[Contract],
        dd_table:      np.ndarray,
        vulnerability: Tuple[bool, bool],
    ) -> int:
        """See the formal README for the current behavior contract."""
        if contract is None:
            return 0
        tricks = int(dd_table[contract.suit, contract.declarer])
        vul    = vulnerability[contract.declarer % 2]
        score  = calculate_score(contract, tricks, vul)
        if contract.declarer % 2 == 1:
            score = -score
        return score

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    @property
    def current_player(self) -> int:
        return self.env.state.current_player

    @property
    def history(self) -> List[int]:
        return self.env.state.history.copy()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def play_mixed(
        self,
        hands:         np.ndarray,
        dd_table:      np.ndarray,
        opener_policy:     Optional[Callable[[Dict, int, list], int]] = None,
        overcaller_policy: Optional[Callable[[Dict, int, list], int]] = None,
        vulnerability: Tuple[bool, bool] = (False, False),
        dealer:        Optional[int] = None,
        **kwargs
    ) -> Tuple[Optional[Contract], int, List[int]]:
        """See the formal README for the current behavior contract."""
        if opener_policy is None:
            opener_policy = kwargs.get('ns_policy')
        if overcaller_policy is None:
            overcaller_policy = kwargs.get('ew_policy')
        
        if opener_policy is None or overcaller_policy is None:
            raise ValueError("Must provide opener_policy and overcaller_policy")

        dealer = dealer if dealer is not None else self.dealer
        self.dealer = dealer  # P93 fix: policy closures read env.dealer for encode_obs_flat
        self._vulnerability = vulnerability  # P122: policy closures read env._vulnerability
        opener_seats = {dealer, (dealer + 2) % NUM_PLAYERS}

        inner = BridgeBiddingEnv(self.max_history_len)
        obs   = inner.reset(hands, dealer=dealer, vulnerability=vulnerability)
        hist  = []
        done  = False

        for bid_str in FIXED_PREFIX:
            bid = string_to_bid(bid_str)
            hist.append(bid)
            obs, _, done, _ = inner.step(bid)
            if done:
                break

        while not done:
            player = inner.state.current_player
            if player in opener_seats:
                action = opener_policy(obs, player, hist[:])
            else:
                action = overcaller_policy(obs, player, hist[:])

            if not inner._is_valid_action(action):
                action = BID_PASS

            hist.append(action)
            obs, _, done, _ = inner.step(action)

        contract = inner.state.final_contract
        score    = self._compute_score_ns(contract, dd_table, vulnerability)
        return contract, score, list(inner.state.history)


# ==============================================================================
# ==============================================================================

def make_rule_policy() -> Callable[[Dict, int, list], int]:
    """See the formal README for the current behavior contract."""
    def policy(obs: Dict, player: int, history_int: list) -> int:
        return _rule_based_action(obs, player, history_int, NORTH)
    return policy
