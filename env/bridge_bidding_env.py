"""A four-player contract-bridge bidding environment."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.scoring import Contract

NUM_PLAYERS = 4
NUM_SUITS = 5
NUM_LEVELS = 7
NUM_BIDS = 38
BID_PASS = 0
BID_DOUBLE = 1
BID_REDOUBLE = 2
BID_1C = 3
NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3


def bid_to_string(bid: int) -> str:
    """Convert an action index to a bridge call."""
    if bid == BID_PASS:
        return "Pass"
    if bid == BID_DOUBLE:
        return "X"
    if bid == BID_REDOUBLE:
        return "XX"
    level = (bid - BID_1C) // 5 + 1
    strain = (bid - BID_1C) % 5
    return f"{level}{['C', 'D', 'H', 'S', 'NT'][strain]}"


def string_to_bid(value: str) -> int:
    """Convert a bridge call to its action index."""
    value = value.upper().strip()
    if value in ("PASS", "P"):
        return BID_PASS
    if value in ("X", "DBL", "DOUBLE"):
        return BID_DOUBLE
    if value in ("XX", "RDBL", "REDOUBLE"):
        return BID_REDOUBLE
    strain_map = {"C": 0, "D": 1, "H": 2, "S": 3, "N": 4, "NT": 4}
    level = int(value[0])
    strain = strain_map[value[1:].replace("NT", "N")]
    return BID_1C + (level - 1) * 5 + strain


@dataclass
class BiddingState:
    """Mutable state for one auction."""

    hands: np.ndarray
    dealer: int = 0
    vulnerability: Tuple[bool, bool] = (False, False)
    history: List[int] = field(default_factory=list)
    current_player: int = 0
    final_contract: Optional[Contract] = None

    def __post_init__(self) -> None:
        self.current_player = self.dealer


class BridgeBiddingEnv:
    """A Dec-POMDP auction with 38 legal-action outputs."""

    def __init__(self, max_history_len: int = 60):
        self.max_history_len = max_history_len
        self.state: Optional[BiddingState] = None

    def reset(
        self,
        hands: Optional[np.ndarray] = None,
        dealer: int = 0,
        vulnerability: Tuple[bool, bool] = (False, False),
    ) -> Dict[str, np.ndarray]:
        """Start a new auction and return its first observation."""
        if hands is None:
            hands = self._deal_random()
        self.state = BiddingState(
            hands=hands.astype(np.float32),
            dealer=dealer,
            vulnerability=vulnerability,
        )
        return self._get_observation()

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Apply one legal call."""
        assert self.state is not None
        assert self._is_valid_action(action), f"Invalid action: {action}"
        self.state.history.append(action)
        done = self._check_done()
        if done:
            self.state.final_contract = self._get_final_contract()
        else:
            self.state.current_player = (self.state.current_player + 1) % NUM_PLAYERS
        return self._get_observation(), 0.0, done, {
            "bid": action,
            "bid_str": bid_to_string(action),
        }

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Return the current player's local observation."""
        assert self.state is not None
        player = self.state.current_player
        history = np.zeros((self.max_history_len, NUM_BIDS), dtype=np.float32)
        for index, bid in enumerate(self.state.history[-self.max_history_len :]):
            history[index, bid] = 1.0
        position = np.zeros(NUM_PLAYERS, dtype=np.float32)
        position[player] = 1.0
        return {
            "hand": self.state.hands[player].copy(),
            "history": history,
            "legal_actions": self._get_legal_actions(),
            "position": position,
            "vulnerability": np.asarray(self.state.vulnerability, dtype=np.float32),
        }

    def _get_legal_actions(self) -> np.ndarray:
        """Return the action mask under auction and doubling rules."""
        assert self.state is not None
        legal = np.zeros(NUM_BIDS, dtype=np.float32)
        legal[BID_PASS] = 1.0
        highest_bid = None
        last_contract_index = -1
        for index, bid in enumerate(self.state.history):
            if bid >= BID_1C:
                highest_bid = bid
                last_contract_index = index
        minimum_bid = BID_1C if highest_bid is None else highest_bid + 1
        legal[minimum_bid:NUM_BIDS] = 1.0
        if last_contract_index < 0:
            return legal

        doubling_state = 0
        last_bidder = (self.state.dealer + last_contract_index) % NUM_PLAYERS
        for bid in self.state.history[last_contract_index + 1 :]:
            if bid == BID_DOUBLE:
                doubling_state = 1
            elif bid == BID_REDOUBLE:
                doubling_state = 2
        current = self.state.current_player
        if doubling_state == 0 and last_bidder % 2 != current % 2:
            legal[BID_DOUBLE] = 1.0
        if doubling_state == 1 and last_bidder % 2 == current % 2:
            legal[BID_REDOUBLE] = 1.0
        return legal

    def _is_valid_action(self, action: int) -> bool:
        return bool(self._get_legal_actions()[action] > 0.5)

    def _check_done(self) -> bool:
        """End after four opening passes or three passes after a contract call."""
        assert self.state is not None
        history = self.state.history
        if len(history) < NUM_PLAYERS:
            return False
        consecutive_passes = 0
        for bid in reversed(history):
            if bid != BID_PASS:
                break
            consecutive_passes += 1
        if len(history) == NUM_PLAYERS and consecutive_passes == NUM_PLAYERS:
            return True
        return any(bid >= BID_1C for bid in history) and consecutive_passes >= 3

    def _get_final_contract(self) -> Optional[Contract]:
        """Construct the final contract and first-bidder declarer."""
        assert self.state is not None
        history = self.state.history
        if all(bid == BID_PASS for bid in history):
            return None
        last_bid = None
        last_bid_index = -1
        for index, bid in enumerate(history):
            if bid >= BID_1C:
                last_bid = bid
                last_bid_index = index
        if last_bid is None:
            return None
        level = (last_bid - BID_1C) // 5 + 1
        strain = (last_bid - BID_1C) % 5
        doubled = 0
        for bid in history[last_bid_index + 1 :]:
            if bid == BID_DOUBLE:
                doubled = 1
            elif bid == BID_REDOUBLE:
                doubled = 2
        last_bidder = (self.state.dealer + last_bid_index) % NUM_PLAYERS
        partnership = last_bidder % 2
        declarer = last_bidder
        for index, bid in enumerate(history):
            if bid < BID_1C:
                continue
            bidder = (self.state.dealer + index) % NUM_PLAYERS
            if (bid - BID_1C) % 5 == strain and bidder % 2 == partnership:
                declarer = bidder
                break
        return Contract(level=level, suit=strain, doubled=doubled, declarer=declarer)

    @staticmethod
    def _deal_random() -> np.ndarray:
        deck = np.arange(52)
        np.random.shuffle(deck)
        hands = np.zeros((NUM_PLAYERS, 52), dtype=np.float32)
        for index, card in enumerate(deck):
            hands[index // 13, card] = 1.0
        return hands

    def render(self) -> None:
        """Print hands and auction state using ASCII suit names."""
        if self.state is None:
            print("Environment not initialized")
            return
        suit_names = ["C", "D", "H", "S"]
        rank_names = "23456789TJQKA"
        player_names = ["North", "East", "South", "West"]
        print("\n" + "=" * 50)
        for player in range(NUM_PLAYERS):
            groups = []
            for suit in range(3, -1, -1):
                cards = [
                    rank_names[rank]
                    for rank in range(12, -1, -1)
                    if self.state.hands[player, suit * 13 + rank] > 0.5
                ]
                groups.append(f"{suit_names[suit]}{''.join(cards)}")
            print(f"{player_names[player]:6s}: {' '.join(groups)}")
        print("-" * 50)
        print("Bidding:", " ".join(bid_to_string(bid) for bid in self.state.history))
        print(f"Current: {player_names[self.state.current_player]}")
        print("=" * 50)
