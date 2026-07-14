"""Focused tests for the formal dual-information reward definition."""

import unittest

import torch
import numpy as np

from networks.belief_net import DualInfoComputer
from networks.policy_net import get_openspiel_obs
from subgames.subgame_trainer import SubgameTrainer
from utils.hand_features import BELIEF_DIM, HONOR_DIM, LENGTH_BINS, NUM_SUITS


def target_with_length(length_bin: int = 3) -> torch.Tensor:
    target = torch.zeros((1, BELIEF_DIM), dtype=torch.float32)
    for suit in range(NUM_SUITS):
        target[0, HONOR_DIM + suit * LENGTH_BINS + length_bin] = 1.0
    return target


class DualInformationRewardTests(unittest.TestCase):
    def test_observation_uses_explicit_receiver(self):
        class FakeState:
            def __init__(self):
                self.received_player = None

            def observation_tensor(self, player=None):
                self.received_player = player
                return [float(player)] * 571

        state = FakeState()
        observation = get_openspiel_obs(state, player=2)
        self.assertEqual(state.received_player, 2)
        self.assertEqual(observation.shape, (571,))
        self.assertTrue(np.all(observation == 2.0))

    def test_partner_and_opponent_samples_share_bidder_target(self):
        target = np.arange(48, dtype=np.float32)
        step = {
            '_rinfo': True,
            'partner_obs_after': np.full(571, 1.0, dtype=np.float32),
            'opponent_obs_after': np.full(571, 2.0, dtype=np.float32),
            'target_pos': 0,
            'belief_target': target,
        }
        obs, target_pos, targets = SubgameTrainer._extract_receiver_belief_data([[step]])
        self.assertEqual(tuple(obs.shape), (2, 571))
        self.assertEqual(target_pos.tolist(), [0, 0])
        self.assertTrue(torch.equal(targets[0], targets[1]))
        self.assertTrue(torch.equal(targets[0], torch.tensor(target)))

    def test_better_posterior_has_positive_gain(self):
        target = target_with_length()
        before = torch.full((1, BELIEF_DIM), 0.25)
        after = before.clone()
        after[:, :HONOR_DIM] = target[:, :HONOR_DIM] * 0.9 + 0.05
        for suit in range(NUM_SUITS):
            start = HONOR_DIM + suit * LENGTH_BINS
            after[:, start:start + LENGTH_BINS] = 0.01
            after[:, start + 3] = 0.93

        computer = DualInfoComputer(belief_net=None, beta=0.05)
        gain = computer.compute_info_gain(before, after, target)
        self.assertGreater(float(gain.item()), 0.0)

    def test_worse_posterior_has_negative_gain(self):
        target = target_with_length()
        good = torch.full((1, BELIEF_DIM), 0.25)
        bad = good.clone()
        for suit in range(NUM_SUITS):
            start = HONOR_DIM + suit * LENGTH_BINS
            good[:, start:start + LENGTH_BINS] = 0.01
            good[:, start + 3] = 0.93
            bad[:, start:start + LENGTH_BINS] = 0.01
            bad[:, start] = 0.93

        computer = DualInfoComputer(belief_net=None, beta=0.05)
        gain = computer.compute_info_gain(good, bad, target)
        self.assertLess(float(gain.item()), 0.0)

    def test_opponent_leakage_is_subtracted(self):
        computer = DualInfoComputer(belief_net=None, beta=0.25)
        partner = torch.tensor([2.0])
        opponent = torch.tensor([1.2])
        reward, parts = computer.compute_dual_info_bonus(partner, opponent)
        self.assertAlmostEqual(float(reward.item()), 1.7)
        self.assertAlmostEqual(parts["partner_gain"], 2.0)
        self.assertAlmostEqual(parts["opponent_leak"], 1.2)


if __name__ == "__main__":
    unittest.main()
