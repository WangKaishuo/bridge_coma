import unittest

import numpy as np

from env import BID_PASS, NUM_BIDS
from experiments.published_agent_adapter import load_published_factory


SEEN = {}


def build_test_policy(dealer, vulnerability):
    SEEN["builder"] = (dealer, vulnerability)

    def policy(observation, player, history):
        SEEN["keys"] = set(observation)
        SEEN["player"] = player
        SEEN["history"] = history
        return BID_PASS

    return policy


class PublishedAdapterTest(unittest.TestCase):
    def test_builder_never_receives_deal_and_policy_sees_only_observation(self):
        factory = load_published_factory(
            f"{__name__}:build_test_policy"
        )
        policy = factory(np.ones((4, 52)), 2, (True, False))
        observation = {
            "hand": np.zeros(52),
            "history": np.zeros((60, NUM_BIDS)),
            "legal_actions": np.r_[1.0, np.zeros(NUM_BIDS - 1)],
            "position": np.eye(4)[2],
            "vulnerability": np.array([1.0, 0.0]),
            "secret": np.ones(99),
        }
        self.assertEqual(policy.act(observation, 2, []), BID_PASS)
        self.assertEqual(SEEN["builder"], (2, (True, False)))
        self.assertNotIn("secret", SEEN["keys"])


if __name__ == "__main__":
    unittest.main()
