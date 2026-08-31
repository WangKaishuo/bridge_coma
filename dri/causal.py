"""Executable structural definition of the heard/deaf DRI intervention.

The intervention deliberately separates two roles of an observed call:

* its mechanical role in the public auction (history, contract state and legal
  actions), which is identical in both worlds; and
* its likelihood contribution as evidence about one bidder's private state,
  which is omitted in the deaf world for one explicitly identified event.

This module is intentionally independent of a particular neural listener.  A
later amortized listener can implement the same contract, while the explicit
finite-hypothesis implementation here provides an auditable reference model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


@dataclass(frozen=True)
class PublicDecisionState:
    """Mechanical state at a receiver decision, shared by heard and deaf.

    Tuples make the snapshot immutable and make accidental differences between
    intervention arms easy to detect.  ``contract_state`` is optional because
    an auction implementation may encode it entirely in ``history``.
    """

    history: tuple[int, ...]
    dealer: int
    current_player: int
    vulnerability: tuple[bool, bool]
    legal_actions: tuple[bool, ...]
    contract_state: Hashable | None = None


def snapshot_public_state(
    env, contract_state: Hashable | None = None
) -> PublicDecisionState:
    """Take an immutable public snapshot from ``BridgeBiddingEnv``-like state."""

    if env.state is None:
        raise ValueError("environment must be reset before taking a snapshot")
    legal = np.asarray(env._get_legal_actions())
    return PublicDecisionState(
        history=tuple(int(action) for action in env.state.history),
        dealer=int(env.state.dealer),
        current_player=int(env.state.current_player),
        vulnerability=tuple(bool(value) for value in env.state.vulnerability),
        legal_actions=tuple(bool(value > 0.5) for value in legal),
        contract_state=contract_state,
    )


@dataclass(frozen=True)
class LikelihoodEvent:
    """One public event's likelihood under bidder-private-state hypotheses.

    ``event_id`` identifies the particular public occurrence, rather than just
    its call value (two players may make the same call).  Likelihoods may be
    unnormalised but must be finite and non-negative.
    """

    event_id: Hashable
    bidder: int
    message: Hashable
    likelihood: tuple[float, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.likelihood, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("likelihood must be a non-empty one-dimensional vector")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("likelihood entries must be finite and non-negative")
        if not np.any(values > 0.0):
            raise ValueError("likelihood cannot be zero for every hypothesis")


@dataclass(frozen=True)
class CausalListener:
    """Finite-hypothesis reference implementation of a causal listener."""

    public_state: PublicDecisionState
    prior: tuple[float, ...]
    events: tuple[LikelihoodEvent, ...]

    def __post_init__(self) -> None:
        prior = np.asarray(self.prior, dtype=np.float64)
        if prior.ndim != 1 or prior.size == 0:
            raise ValueError("prior must be a non-empty one-dimensional vector")
        if not np.all(np.isfinite(prior)) or np.any(prior < 0.0) or prior.sum() <= 0.0:
            raise ValueError("prior must be finite, non-negative and have positive mass")
        for event in self.events:
            if len(event.likelihood) != prior.size:
                raise ValueError("all likelihood vectors must match the prior size")
        ids = [event.event_id for event in self.events]
        if len(set(ids)) != len(ids):
            raise ValueError("event_id must uniquely identify a public event")

    def heard(self) -> np.ndarray:
        """Posterior using every observed public event as private-state evidence."""

        return self._posterior(suppressed_event=None)

    def deaf(self, target_event: Hashable, target_bidder: int) -> np.ndarray:
        """Posterior omitting exactly one target bidder/message likelihood factor.

        The public state is not copied or edited: callers receive beliefs for
        the same ``public_state`` object.  Requiring both event and bidder IDs
        prevents accidentally suppressing an identical call made by another
        player.
        """

        matches = [
            event for event in self.events
            if event.event_id == target_event and event.bidder == target_bidder
        ]
        if len(matches) != 1:
            raise ValueError("target_event and target_bidder must identify exactly one event")
        return self._posterior(suppressed_event=(target_event, int(target_bidder)))

    def _posterior(self, suppressed_event: tuple[Hashable, int] | None) -> np.ndarray:
        weights = np.asarray(self.prior, dtype=np.float64).copy()
        weights /= weights.sum()
        for event in self.events:
            if suppressed_event == (event.event_id, event.bidder):
                continue
            weights *= np.asarray(event.likelihood, dtype=np.float64)
            total = weights.sum()
            if total <= 0.0:
                raise ValueError("observed likelihood factors leave zero posterior mass")
            weights /= total
        return weights


def decision_distribution(
    posterior: Sequence[float],
    policy_given_private_state: Sequence[Sequence[float]],
    legal_actions: Sequence[bool],
) -> np.ndarray:
    """Marginalise a receiver policy over bidder-private-state hypotheses.

    Rows of ``policy_given_private_state`` correspond to private hypotheses and
    columns to actions.  Illegal action probability is required to be zero;
    silently masking it would hide a violation of the causal invariant.
    """

    belief = np.asarray(posterior, dtype=np.float64)
    policy = np.asarray(policy_given_private_state, dtype=np.float64)
    legal = np.asarray(legal_actions, dtype=bool)
    if policy.shape != (belief.size, legal.size):
        raise ValueError("policy shape must be (number of hypotheses, number of actions)")
    if np.any(policy < 0.0) or not np.all(np.isfinite(policy)):
        raise ValueError("policy probabilities must be finite and non-negative")
    if not np.isclose(belief.sum(), 1.0) or np.any(belief < 0.0):
        raise ValueError("posterior must be a probability vector")
    row_sums = policy.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError("each conditional policy row must sum to one")
    if np.any(policy[:, ~legal] > 1e-12):
        raise ValueError("conditional policy assigns mass to an illegal action")
    distribution = belief @ policy
    distribution[~legal] = 0.0
    return distribution


@dataclass(frozen=True)
class DecisionRegretResult:
    heard_regret: float
    deaf_regret: float
    dri: float


def decision_regret_reduction(
    q_task: Sequence[float],
    heard_policy: Sequence[float],
    deaf_policy: Sequence[float],
    legal_actions: Sequence[bool],
) -> DecisionRegretResult:
    """Compute R(deaf)-R(heard) from task-only legal-action values."""

    q_values = np.asarray(q_task, dtype=np.float64)
    heard = np.asarray(heard_policy, dtype=np.float64)
    deaf = np.asarray(deaf_policy, dtype=np.float64)
    legal = np.asarray(legal_actions, dtype=bool)
    if (
        q_values.shape != heard.shape
        or q_values.shape != deaf.shape
        or q_values.shape != legal.shape
    ):
        raise ValueError("Q, policy distributions and legal mask must have identical shapes")
    if not np.any(legal):
        raise ValueError("at least one action must be legal")
    for name, policy in (("heard", heard), ("deaf", deaf)):
        if np.any(policy < 0.0) or not np.isclose(policy.sum(), 1.0):
            raise ValueError(f"{name} policy must be a probability vector")
        if np.any(policy[~legal] > 1e-12):
            raise ValueError(f"{name} policy assigns mass to an illegal action")
    best = float(np.max(q_values[legal]))
    heard_regret = best - float(np.dot(heard, q_values))
    deaf_regret = best - float(np.dot(deaf, q_values))
    return DecisionRegretResult(
        heard_regret=heard_regret,
        deaf_regret=deaf_regret,
        dri=deaf_regret - heard_regret,
    )
