"""Unrestricted complete-auction environment for the main experiment."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from env import BridgeBiddingEnv, BID_PASS, NORTH, NUM_PLAYERS
from subgames.competitive_env import CompetitiveSubgameEnv
from utils.dds_data import create_loader
from utils.scoring import Contract


class UnrestrictedBiddingEnv(CompetitiveSubgameEnv):
    """Random DDS deals with no hand filtering or forced auction prefix."""

    def __init__(self, data_path: str, max_history_len: int = 60):
        # Do not call the competitive constructor: it detects and filters deals
        # for the fixed 1H-1S validation distribution.
        self.loader = create_loader(data_path)
        self.env = BridgeBiddingEnv(max_history_len)
        self.max_history_len = max_history_len
        self.dealer = NORTH
        self._sampled_dealer = NORTH
        self._current_hands: Optional[np.ndarray] = None
        self._current_dd: Optional[np.ndarray] = None
        self._vulnerability: Tuple[bool, bool] = (False, False)
        self.history_int: list[int] = []
        print(f"[MainEnv] Unrestricted DDS data: {len(self.loader):,} samples")

    @property
    def initial_history_length(self) -> int:
        return 0

    @property
    def initial_history_actions(self) -> List[int]:
        return []

    def generate_deal(self) -> Tuple[np.ndarray, np.ndarray]:
        hands, dd_table = self.loader.sample_one()
        self._sampled_dealer = int(np.random.randint(NUM_PLAYERS))
        return hands, dd_table

    def reset(
        self,
        hands: Optional[np.ndarray] = None,
        dd_table: Optional[np.ndarray] = None,
        vulnerability: Tuple[bool, bool] = (False, False),
        dealer: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Start at the dealer's opening call with an empty public history."""
        if hands is None or dd_table is None:
            hands, dd_table = self.generate_deal()
            generated_dealer = self._sampled_dealer
        else:
            generated_dealer = NORTH
        dealer = generated_dealer if dealer is None else int(dealer)

        self._current_hands = hands
        self._current_dd = dd_table
        self._vulnerability = vulnerability
        self.dealer = dealer
        self.history_int = []
        return self.env.reset(hands, dealer=dealer, vulnerability=vulnerability)

    def play_mixed(
        self,
        hands: np.ndarray,
        dd_table: np.ndarray,
        opener_policy: Optional[Callable[[Dict, int, list], int]] = None,
        overcaller_policy: Optional[Callable[[Dict, int, list], int]] = None,
        vulnerability: Tuple[bool, bool] = (False, False),
        dealer: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Optional[Contract], int, List[int]]:
        """Play a full auction with one black-box policy per partnership."""
        if opener_policy is None:
            opener_policy = kwargs.get("ns_policy")
        if overcaller_policy is None:
            overcaller_policy = kwargs.get("ew_policy")
        if opener_policy is None or overcaller_policy is None:
            raise ValueError("Must provide both partnership policies")

        dealer = self.dealer if dealer is None else int(dealer)
        self.dealer = dealer
        self._vulnerability = vulnerability
        dealer_side = {dealer, (dealer + 2) % NUM_PLAYERS}

        inner = BridgeBiddingEnv(self.max_history_len)
        obs = inner.reset(hands, dealer=dealer, vulnerability=vulnerability)
        history: list[int] = []
        done = False
        while not done:
            player = inner.state.current_player
            policy = opener_policy if player in dealer_side else overcaller_policy
            action = policy(obs, player, history[:])
            if not inner._is_valid_action(action):
                action = BID_PASS
            history.append(action)
            obs, _, done, _ = inner.step(action)

        contract = inner.state.final_contract
        score = self._compute_score_ns(contract, dd_table, vulnerability)
        return contract, score, list(inner.state.history)
