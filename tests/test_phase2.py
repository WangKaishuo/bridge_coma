#!/usr/bin/env python3
"""
Phase 2 Tests
=============

Tests for subgame environments, action masks, BC, and trainer.

Usage:
    cd bridge-coma/
    python tests/test_phase2.py
    python tests/test_phase2.py --no-torch   # skip GPU/slow tests
"""

import sys
import os
import argparse
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Action Mask Tests
# ============================================================================

class TestActionMask(unittest.TestCase):
    """Test action_mask.py"""

    def test_count_hcp(self):
        from subgames.action_mask import count_hcp
        hand = np.zeros(52)
        # Give all aces (index 12, 25, 38, 51)
        hand[12] = 1.0   # Ace of clubs
        hand[25] = 1.0   # Ace of diamonds
        hand[38] = 1.0   # Ace of hearts
        hand[51] = 1.0   # Ace of spades
        self.assertEqual(count_hcp(hand), 16)

    def test_count_hcp_zero(self):
        from subgames.action_mask import count_hcp
        hand = np.zeros(52)
        # Only low cards
        hand[0] = 1.0   # 2 of clubs
        hand[13] = 1.0  # 2 of diamonds
        self.assertEqual(count_hcp(hand), 0)

    def test_suit_length(self):
        from subgames.action_mask import count_suit_length
        hand = np.zeros(52)
        hand[0:5] = 1.0  # 5 clubs
        self.assertEqual(count_suit_length(hand, 0), 5)
        self.assertEqual(count_suit_length(hand, 1), 0)

    def test_is_balanced(self):
        from subgames.action_mask import is_balanced
        # 4-3-3-3
        hand = np.zeros(52)
        hand[0:4] = 1.0   # 4 clubs
        hand[13:16] = 1.0 # 3 diamonds
        hand[26:29] = 1.0 # 3 hearts
        hand[39:42] = 1.0 # 3 spades
        self.assertTrue(is_balanced(hand))

    def test_is_not_balanced_singleton(self):
        from subgames.action_mask import is_balanced
        # 6-1-3-3 = not balanced (singleton)
        hand = np.zeros(52)
        hand[0:6] = 1.0   # 6 clubs
        hand[13:14] = 1.0 # 1 diamond (singleton!)
        hand[26:29] = 1.0 # 3 hearts
        hand[39:42] = 1.0 # 3 spades
        self.assertFalse(is_balanced(hand))

    def test_balanced_5m(self):
        from subgames.action_mask import is_balanced
        # 5-3-3-2 = balanced (允许 5M/6m, 无单缺)
        hand = np.zeros(52)
        hand[0:5] = 1.0   # 5 clubs
        hand[13:16] = 1.0 # 3 diamonds
        hand[26:29] = 1.0 # 3 hearts
        hand[39:41] = 1.0 # 2 spades
        self.assertTrue(is_balanced(hand))

    def test_legal_mask_initial(self):
        from subgames.action_mask import get_legal_mask
        from env import BID_PASS, BID_1C, NUM_BIDS
        mask = get_legal_mask([], current_player=0, dealer=0)
        self.assertEqual(mask[BID_PASS], 1.0)
        # All bids from 1C should be legal
        for bid in range(BID_1C, NUM_BIDS):
            self.assertEqual(mask[bid], 1.0)
        # No double/redouble at start
        self.assertEqual(mask[1], 0.0)  # Double
        self.assertEqual(mask[2], 0.0)  # Redouble

    def test_combined_mask_has_pass(self):
        from subgames.action_mask import get_combined_mask
        hand = np.zeros(52)
        hand[0:13] = 1.0  # All clubs, 0 HCP
        mask = get_combined_mask(hand, [], current_player=0, dealer=0, use_soft=True)
        # Pass should always be available
        self.assertEqual(mask[0], 1.0)


# ============================================================================
# BC Rules Tests
# ============================================================================

class TestBCRules(unittest.TestCase):
    """Test behavioral_cloning.py rules."""

    def _make_hand(self, clubs=0, diamonds=0, hearts=0, spades=0, hcp_cards=None):
        """
        Create a hand with specified suit lengths.
        hcp_cards: list of (suit, rank) for HCP cards, e.g. [(2, 12)] = Ace of hearts
        """
        hand = np.zeros(52)
        idx = 0
        for suit, count in enumerate([clubs, diamonds, hearts, spades]):
            for i in range(count):
                hand[suit * 13 + i] = 1.0
                idx += 1
        # Override with specific HCP cards
        if hcp_cards:
            for suit, rank in hcp_cards:
                hand[suit * 13 + rank] = 1.0
        return hand

    def test_select_simple_opening_pass(self):
        from algorithms.behavioral_cloning import select_simple_opening
        from env import BID_PASS
        hand = np.zeros(52)
        hand[0:13] = 1.0  # 13 clubs, all low → 0 HCP
        self.assertEqual(select_simple_opening(hand), BID_PASS)

    def test_select_simple_opening_1h(self):
        from algorithms.behavioral_cloning import select_simple_opening
        from env import string_to_bid
        # 5 hearts with A, K = 7 HCP in hearts, need 12+ total
        hand = np.zeros(52)
        # Hearts: A, K, Q, J, T (5 cards, 10 HCP)
        for r in [12, 11, 10, 9, 8]:
            hand[26 + r] = 1.0
        # Clubs: A, K (2 HCP each = 7 HCP)
        hand[12] = 1.0   # Ace clubs
        # Diamonds: few low
        hand[13] = 1.0; hand[14] = 1.0; hand[15] = 1.0
        # Spades: few low
        hand[39] = 1.0; hand[40] = 1.0; hand[41] = 1.0; hand[42] = 1.0
        # Total HCP = 10 (AKQJ hearts) + 4 (A clubs) = 14
        self.assertEqual(select_simple_opening(hand), string_to_bid("1H"))

    def test_competitive_response_pass_weak(self):
        from algorithms.behavioral_cloning import competitive_response_after_1h_1s
        from env import BID_PASS
        # Very weak hand, 2 hearts
        hand = np.zeros(52)
        hand[0:4] = 1.0    # 4 clubs
        hand[13:17] = 1.0  # 4 diamonds
        hand[26:28] = 1.0  # 2 hearts
        hand[39:42] = 1.0  # 3 spades
        # 0 HCP
        self.assertEqual(competitive_response_after_1h_1s(hand), BID_PASS)


# ============================================================================
# Stayman Env Tests
# ============================================================================

class TestStaymanEnvConstraints(unittest.TestCase):
    """Test Stayman constraint functions (without DDS data)."""

    def test_opener_constraint(self):
        from subgames.stayman_env import StaymanSubgameEnv
        # 15 HCP balanced hand
        hand = np.zeros(52)
        # Spades: A, K, x, x (4 cards, 7 HCP)
        hand[51] = 1; hand[50] = 1; hand[39] = 1; hand[40] = 1
        # Hearts: Q, x, x (3 cards, 2 HCP)
        hand[36] = 1; hand[26] = 1; hand[27] = 1
        # Diamonds: K, x, x (3 cards, 3 HCP)
        hand[24] = 1; hand[13] = 1; hand[14] = 1
        # Clubs: A, x, x (3 cards, 4 HCP) → total = 16
        hand[12] = 1; hand[0] = 1; hand[1] = 1
        # That's 13 cards, 16 HCP, 4-3-3-3 balanced
        self.assertTrue(StaymanSubgameEnv._satisfies_opener(hand))

    def test_responder_constraint(self):
        from subgames.stayman_env import StaymanSubgameEnv
        # 10 HCP with 4 hearts
        hand = np.zeros(52)
        # Hearts: A, x, x, x (4 cards, 4 HCP)
        hand[38] = 1; hand[26] = 1; hand[27] = 1; hand[28] = 1
        # Spades: K, x, x (3 cards, 3 HCP)
        hand[50] = 1; hand[39] = 1; hand[40] = 1
        # Diamonds: Q, x, x (3 cards, 2 HCP)
        hand[23] = 1; hand[13] = 1; hand[14] = 1
        # Clubs: x, x, x (3 cards, 0 HCP) → total = 9
        hand[0] = 1; hand[1] = 1; hand[2] = 1
        self.assertTrue(StaymanSubgameEnv._satisfies_responder(hand))

    def test_responder_fails_no_4m(self):
        from subgames.stayman_env import StaymanSubgameEnv
        # 10 HCP but only 3 hearts, 3 spades
        hand = np.zeros(52)
        hand[38] = 1; hand[26] = 1; hand[27] = 1  # 3 hearts
        hand[50] = 1; hand[39] = 1; hand[40] = 1  # 3 spades
        hand[24] = 1; hand[13] = 1; hand[14] = 1; hand[15] = 1  # 4 diamonds
        hand[0] = 1; hand[1] = 1; hand[2] = 1  # 3 clubs
        # HCP = 4 (AH) + 3 (KS) + 3 (KD) = 10, but no 4-card major
        self.assertFalse(StaymanSubgameEnv._satisfies_responder(hand))


# ============================================================================
# Competitive Env Tests
# ============================================================================

class TestCompetitiveEnvConstraints(unittest.TestCase):

    def test_opener_constraint(self):
        from subgames.competitive_env import CompetitiveSubgameEnv
        # N: 5 hearts, 14 HCP
        hand = np.zeros(52)
        # 5 hearts: A, K, x, x, x
        hand[38] = 1; hand[37] = 1; hand[26] = 1; hand[27] = 1; hand[28] = 1
        # 3 spades
        hand[39] = 1; hand[40] = 1; hand[41] = 1
        # 3 diamonds: Q, x, x
        hand[23] = 1; hand[13] = 1; hand[14] = 1
        # 2 clubs
        hand[0] = 1; hand[1] = 1
        # HCP = 7 (AK hearts) + 2 (Q diamonds) = 9... need more
        # Add K of spades
        hand[50] = 1; hand[41] = 0  # replace low spade with K
        # HCP = 7 + 2 + 3 = 12
        self.assertTrue(CompetitiveSubgameEnv._satisfies_opener(hand))

    def test_overcaller_constraint(self):
        from subgames.competitive_env import CompetitiveSubgameEnv
        # E: 5 spades, 10 HCP
        hand = np.zeros(52)
        # 5 spades: K, Q, x, x, x
        hand[50] = 1; hand[49] = 1; hand[39] = 1; hand[40] = 1; hand[41] = 1
        # 3 hearts
        hand[26] = 1; hand[27] = 1; hand[28] = 1
        # 3 diamonds: A, x, x
        hand[25] = 1; hand[13] = 1; hand[14] = 1
        # 2 clubs
        hand[0] = 1; hand[1] = 1
        # HCP = 5 (KQ spades) + 4 (A diamonds) = 9
        self.assertTrue(CompetitiveSubgameEnv._satisfies_overcaller(hand))

    def test_overcaller_fails_too_few_spades(self):
        from subgames.competitive_env import CompetitiveSubgameEnv
        hand = np.zeros(52)
        hand[50] = 1; hand[49] = 1; hand[39] = 1; hand[40] = 1  # only 4 spades
        hand[26:30] = 1  # 4 hearts
        hand[13:16] = 1  # 3 diamonds
        hand[0:2] = 1    # 2 clubs
        self.assertFalse(CompetitiveSubgameEnv._satisfies_overcaller(hand))


# ============================================================================
# Optimal contract test
# ============================================================================

class TestOptimalContract(unittest.TestCase):

    def test_optimal_contract_basic(self):
        from subgames.stayman_env import StaymanSubgameEnv
        # dd_table: (5 suits, 4 players), tricks
        dd_table = np.zeros((5, 4), dtype=np.int8)
        # NT by N can make 9 tricks → 3NT
        dd_table[4, 0] = 9
        # Hearts by N can make 10 tricks → 4H
        dd_table[2, 0] = 10

        optimal = StaymanSubgameEnv._get_optimal_contract_ns(dd_table)
        # 4H by N: 420 (10 tricks, non-vul) vs 3NT: 400
        # Actually 4H = 420, 3NT = 400, so 4H is better
        self.assertIsNotNone(optimal)
        self.assertEqual(optimal.suit, 2)  # Hearts
        self.assertEqual(optimal.level, 4)


# ============================================================================
# Integration Tests (require torch)
# ============================================================================

class TestSubgameTrainerSmoke(unittest.TestCase):
    """Smoke test — requires torch but not DDS data."""

    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def test_make_agent_policy(self):
        """Test that make_agent_policy returns callable."""
        from algorithms.mappo import MAPPOAgent, MAPPOConfig
        from subgames.competitive_env import make_agent_policy

        agent = MAPPOAgent(MAPPOConfig(device='cpu'))
        policy = make_agent_policy(agent)
        self.assertTrue(callable(policy))

        # Should work with a dummy obs
        obs = {
            'hand': np.random.rand(52).astype(np.float32),
            'history': np.zeros((60, 38), dtype=np.float32),
            'legal_actions': np.ones(38, dtype=np.float32),
            'position': np.array([1, 0, 0, 0], dtype=np.float32),
            'vulnerability': np.array([0, 0], dtype=np.float32),
        }
        action = policy(obs)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 38)


# ============================================================================
# Runner
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-torch', action='store_true', help='Skip torch tests')
    args = parser.parse_args()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Always run these
    suite.addTests(loader.loadTestsFromTestCase(TestActionMask))
    suite.addTests(loader.loadTestsFromTestCase(TestBCRules))
    suite.addTests(loader.loadTestsFromTestCase(TestStaymanEnvConstraints))
    suite.addTests(loader.loadTestsFromTestCase(TestCompetitiveEnvConstraints))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimalContract))

    if not args.no_torch:
        suite.addTests(loader.loadTestsFromTestCase(TestSubgameTrainerSmoke))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
