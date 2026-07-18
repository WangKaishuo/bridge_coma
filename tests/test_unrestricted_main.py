import tempfile
import unittest
from pathlib import Path

import numpy as np

from env import BID_1C, BID_PASS
from experiments.evaluation import auction_is_competitive
from subgames.unrestricted_env import UnrestrictedBiddingEnv


def _write_dds(path: Path, count: int = 8) -> None:
    decks = np.empty((count, 52), dtype=np.uint8)
    for index in range(count):
        order = np.random.default_rng(index).permutation(52)
        deck = np.empty(52, dtype=np.uint8)
        for position, card in enumerate(order):
            deck[card] = position // 13
        decks[index] = deck
    tricks = np.full((count, 5, 4), 7, dtype=np.int8)
    np.savez_compressed(path, decks=decks, tricks=tricks)


class UnrestrictedMainEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.data = Path(self.tempdir.name) / "dds_test.npz"
        _write_dds(self.data)
        self.env = UnrestrictedBiddingEnv(str(self.data))

    def tearDown(self):
        self.tempdir.cleanup()

    def test_reset_has_no_forced_prefix(self):
        hands, dd = self.env.generate_deal()
        obs = self.env.reset(hands, dd, dealer=3)
        self.assertEqual(self.env.history, [])
        self.assertEqual(self.env.current_player, 3)
        self.assertEqual(self.env.initial_history_length, 0)
        self.assertEqual(obs["legal_actions"][BID_PASS], 1.0)

    def test_all_pass_is_complete_four_call_auction(self):
        hands, dd = self.env.generate_deal()

        def pass_policy(_obs, _player, _history):
            return BID_PASS

        contract, score, history = self.env.play_mixed(
            hands, dd, pass_policy, pass_policy, dealer=2
        )
        self.assertIsNone(contract)
        self.assertEqual(score, 0)
        self.assertEqual(history, [BID_PASS] * 4)

    def test_worker_clone_preserves_unrestricted_type(self):
        clone = self.env.clone_for_worker()
        self.assertIsInstance(clone, UnrestrictedBiddingEnv)
        self.assertIs(clone.loader, self.env.loader)
        hands, dd = clone.generate_deal()
        clone.reset(hands, dd, dealer=1)
        self.assertEqual(clone.history, [])

    def test_competition_classifier_uses_both_partnerships(self):
        one_side = [BID_1C, BID_PASS, BID_1C + 5, BID_PASS, BID_PASS, BID_PASS]
        both_sides = [BID_1C, BID_1C + 1, BID_PASS, BID_PASS, BID_PASS]
        self.assertFalse(auction_is_competitive(one_side, dealer=0))
        self.assertTrue(auction_is_competitive(both_sides, dealer=0))


if __name__ == "__main__":
    unittest.main()
