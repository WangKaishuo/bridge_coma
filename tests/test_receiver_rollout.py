"""Integration test for receiver-specific information-reward attribution."""

from pathlib import Path
import unittest

import numpy as np

try:
    import pyspiel  # noqa: F401
    HAS_OPEN_SPIEL = True
except ImportError:
    HAS_OPEN_SPIEL = False

from subgames.competitive_env import CompetitiveSubgameEnv
from subgames.subgame_trainer import SubgameConfig, SubgameTrainer
from utils.hand_features import hand_to_belief_target
from utils.running_stats import RunningStats


@unittest.skipUnless(HAS_OPEN_SPIEL, "OpenSpiel is not installed")
class ReceiverRolloutTests(unittest.TestCase):
    def test_rotated_dealer_preserves_fixed_prefix_semantics(self):
        data = Path(__file__).resolve().parents[1] / "data" / "competitive_100k.npz"
        env = CompetitiveSubgameEnv(str(data))
        config = SubgameConfig(
            hidden_dim=32,
            deals_per_step=8,
            steps_per_phase=1,
            batch_size=8,
            device="cpu",
        )
        trainer = SubgameTrainer(env, config, RunningStats())
        np.random.seed(2026)
        episodes, _ = trainer._collect_episodes_batch(
            64,
            train_side="NS",
            fsp_sd=None,
            batch_size=8,
            skip_dual_table=True,
        )
        rollout_dealers = set()
        for episode in episodes:
            first = episode[0]
            dealer = first["dealer"]
            hands = first["all_hands"]
            rollout_dealers.add(dealer)
            self.assertTrue(env._satisfies_opener(hands[dealer]))
            self.assertTrue(env._satisfies_overcaller(hands[(dealer + 1) % 4]))
            self.assertEqual(first["player"], (dealer + 2) % 4)
        self.assertEqual(rollout_dealers, {0, 1, 2, 3})

    def test_bidder_target_and_receiver_views(self):
        data = Path(__file__).resolve().parents[1] / "data" / "competitive_100k.npz"
        env = CompetitiveSubgameEnv(str(data))
        config = SubgameConfig(
            use_info_bonus=True,
            hidden_dim=32,
            deals_per_step=2,
            steps_per_phase=1,
            batch_size=2,
            freeze_belief=False,
            device="cpu",
        )
        trainer = SubgameTrainer(env, config, RunningStats())
        episodes, reward_data = trainer._collect_episodes_batch(
            2,
            train_side="NS",
            fsp_sd=None,
            batch_size=2,
            skip_dual_table=True,
        )
        steps = [step for episode in episodes for step in episode if step.get("_rinfo")]
        self.assertTrue(steps)
        for step in steps:
            self.assertEqual(step["target_pos"], step["player"])
            expected = hand_to_belief_target(step["all_hands"][step["player"]])
            self.assertTrue(np.array_equal(step["belief_target"], expected))
            self.assertEqual(step["partner_obs_before"].shape, (571,))
            self.assertEqual(step["opponent_obs_before"].shape, (571,))
            self.assertFalse(np.shares_memory(
                step["partner_obs_before"], step["opponent_obs_before"]
            ))
        self.assertEqual(len(reward_data["target"]), len(steps))
        self.assertTrue(np.array_equal(
            reward_data["target"],
            np.stack([step["belief_target"] for step in steps]),
        ))


if __name__ == "__main__":
    unittest.main()
