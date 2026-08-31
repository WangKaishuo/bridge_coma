"""Observation-only adapters for published bridge bidding baselines.

An adapter target is specified as ``python.module:builder``.  The builder is
called once per deal as ``builder(dealer, vulnerability)`` and must return an
object with ``act(observation, player, history)`` or an equivalent callable.
The builder never receives the deal or any DDS information.
"""

from __future__ import annotations

import importlib
from typing import Callable

import numpy as np

from experiments.evaluation import BiddingPolicy, PolicyFactory


PUBLIC_OBSERVATION_KEYS = (
    "hand",
    "history",
    "legal_actions",
    "position",
    "vulnerability",
)


class ObservationOnlyPolicy:
    """Validate and isolate calls to an external black-box policy."""

    def __init__(self, policy):
        self.policy = policy

    def act(self, observation: dict, player: int, history: list[int]) -> int:
        public_observation = {
            key: np.asarray(observation[key]).copy()
            for key in PUBLIC_OBSERVATION_KEYS
            if key in observation
        }
        if hasattr(self.policy, "act"):
            action = self.policy.act(public_observation, int(player), list(history))
        elif callable(self.policy):
            action = self.policy(public_observation, int(player), list(history))
        else:
            raise TypeError("Published policy must be callable or expose act()")

        action = int(action)
        legal = public_observation.get("legal_actions")
        if legal is None or not 0 <= action < len(legal) or legal[action] <= 0.5:
            raise ValueError(f"Published policy returned illegal action {action}")
        return action


def load_published_factory(spec: str) -> PolicyFactory:
    """Load an observation-only policy builder from ``module:attribute``."""
    if ":" not in spec:
        raise ValueError("Published agent spec must be module:attribute")
    module_name, attribute = spec.split(":", 1)
    builder = getattr(importlib.import_module(module_name), attribute)
    if not callable(builder):
        raise TypeError(f"Published policy builder is not callable: {spec}")

    def factory(_hands, dealer, vulnerability) -> BiddingPolicy:
        # `_hands` is intentionally ignored: the external builder receives only
        # public deal metadata and each acting player's legal observation.
        policy = builder(int(dealer), tuple(vulnerability))
        return ObservationOnlyPolicy(policy)

    return factory
