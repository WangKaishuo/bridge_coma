import unittest

import numpy as np

from env import BridgeBiddingEnv
from networks.policy_net import (
    convert_hands_suit_to_rank,
    encode_openspiel_auction_observation,
    get_openspiel_obs,
    hands_to_openspiel_state,
    ours_to_openspiel_raw,
    physical_to_openspiel_player,
)


class FastObservationTests(unittest.TestCase):
    def test_matches_openspiel_for_random_auctions(self):
        rng = np.random.default_rng(20260715)

        for _ in range(40):
            deck = rng.permutation(52)
            hands = np.zeros((4, 52), dtype=np.float32)
            for player in range(4):
                hands[player, deck[player * 13:(player + 1) * 13]] = 1.0
            dealer = int(rng.integers(4))
            vulnerability = (bool(rng.integers(2)), bool(rng.integers(2)))
            env = BridgeBiddingEnv(60)
            obs = env.reset(hands, dealer=dealer, vulnerability=vulnerability)
            history = []

            while True:
                state = hands_to_openspiel_state(
                    convert_hands_suit_to_rank(hands), dealer, vulnerability
                )
                for action in history:
                    state.apply_action(ours_to_openspiel_raw(action))

                for player in range(4):
                    expected = get_openspiel_obs(
                        state, physical_to_openspiel_player(player, dealer)
                    )
                    actual = encode_openspiel_auction_observation(
                        hands, dealer, history, player, vulnerability
                    )
                    np.testing.assert_array_equal(actual, expected)

                legal = np.flatnonzero(obs["legal_actions"] > 0.5)
                action = int(rng.choice(legal))
                history.append(action)
                obs, _, done, _ = env.step(action)
                if done:
                    terminal_state = hands_to_openspiel_state(
                        convert_hands_suit_to_rank(hands), dealer, vulnerability
                    )
                    for terminal_action in history:
                        terminal_state.apply_action(
                            ours_to_openspiel_raw(terminal_action)
                        )
                    for terminal_player in range(4):
                        np.testing.assert_array_equal(
                            encode_openspiel_auction_observation(
                                hands, dealer, history, terminal_player,
                                vulnerability,
                            ),
                            get_openspiel_obs(
                                terminal_state,
                                physical_to_openspiel_player(
                                    terminal_player, dealer
                                ),
                            ),
                        )
                    break


if __name__ == "__main__":
    unittest.main()
