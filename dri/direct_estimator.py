"""Direct DRI estimates from persisted policy probabilities and CF outcomes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DirectDRIStateEstimate:
    state_id: str
    deaf_distribution: str
    dri_imp: float
    heard_expected_q_imp: float
    deaf_expected_q_imp: float
    selected_action_count: int
    heard_covered_mass: float
    deaf_covered_mass: float
    tail_error_bound_imp: float
    continuation_budget: int | None


def weighted_q(estimate: Mapping[str, Any]) -> float:
    """Return the declared finite-mixture expectation with strict alignment."""

    samples = estimate.get("q_samples")
    if samples is None:
        value = float(estimate["q_mean"])
        if not math.isfinite(value):
            raise ValueError("q_mean must be finite")
        return value
    values = np.asarray(samples, dtype=np.float64)
    weights = np.asarray(
        estimate.get("sample_weights", np.ones(values.size)), dtype=np.float64
    )
    if (
        values.ndim != 1 or values.size == 0 or weights.shape != values.shape
        or not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights))
        or np.any(weights < 0) or weights.sum() <= 0
    ):
        raise ValueError("invalid aligned q_samples/sample_weights")
    # Exact mixture weights sum to one; adaptive nested records carry explicit
    # Horvitz-Thompson coefficients whose realized sum need not. Normalizing
    # those coefficients would turn the frozen unbiased estimator into a
    # self-normalized ratio estimator and disagree with the collector's q_mean.
    value = float(np.dot(values, weights))
    declared = estimate.get("q_mean")
    if declared is not None and not np.isclose(
        value, float(declared), atol=1e-6, rtol=1e-6
    ):
        raise ValueError("q_mean disagrees with weighted samples")
    return value


def direct_dri_from_label_state(
    state: Mapping[str, Any],
    *,
    deaf_distribution: str,
    population_id: str = "PI60",
    q_clip_imp: tuple[float, float] = (-24.0, 24.0),
) -> DirectDRIStateEstimate:
    """Compute `E_heard[Q] - E_deaf[Q]` over persisted selected actions."""

    if deaf_distribution not in {"deaf_partner", "deaf_rho"}:
        raise ValueError("deaf_distribution must be deaf_partner or deaf_rho")
    persisted = state.get("policy_distributions")
    if not isinstance(persisted, Mapping):
        raise ValueError("state lacks persisted policy_distributions")
    order = [int(action) for action in persisted.get("action_order", [])]
    probabilities = persisted.get("probabilities")
    if not order or not isinstance(probabilities, Mapping):
        raise ValueError("invalid persisted policy distribution record")
    if order != list(range(len(order))):
        raise ValueError("policy action order must be canonical project action IDs")
    if "heard" not in probabilities or deaf_distribution not in probabilities:
        raise ValueError("requested heard/deaf distributions are unavailable")
    heard = np.asarray(probabilities["heard"], dtype=np.float64)
    deaf = np.asarray(probabilities[deaf_distribution], dtype=np.float64)
    if heard.shape != (len(order),) or deaf.shape != heard.shape:
        raise ValueError("policy distribution width differs from action order")
    if not np.isclose(heard.sum(), 1.0, atol=1e-6) or not np.isclose(
        deaf.sum(), 1.0, atol=1e-6
    ):
        raise ValueError("policy distributions must sum to one")

    q_by_action: dict[int, float] = {}
    budgets = set()
    for action_record in state.get("actions", []):
        action = int(action_record["action"])
        populations = action_record.get("populations")
        if not isinstance(populations, Mapping) or population_id not in populations:
            raise ValueError(f"action lacks population {population_id!r}")
        q_by_action[action] = float(np.clip(
            weighted_q(populations[population_id]), q_clip_imp[0], q_clip_imp[1]
        ))
        adaptive = action_record.get("adaptive_nested_finite_mixture")
        if isinstance(adaptive, Mapping) and "final_budget" in adaptive:
            budgets.add(int(adaptive["final_budget"]))
    if not q_by_action:
        raise ValueError("state has no selected action values")
    selected = np.asarray(sorted(q_by_action), dtype=np.int64)
    q_values = np.asarray([q_by_action[int(action)] for action in selected])
    heard_expected = float(np.dot(heard[selected], q_values))
    deaf_expected = float(np.dot(deaf[selected], q_values))
    heard_mass = float(heard[selected].sum())
    deaf_mass = float(deaf[selected].sum())
    q_span = float(q_clip_imp[1] - q_clip_imp[0])
    tail_bound = q_span * ((1.0 - heard_mass) + (1.0 - deaf_mass))
    return DirectDRIStateEstimate(
        state_id=str(state.get("state_id", "")),
        deaf_distribution=deaf_distribution,
        dri_imp=heard_expected - deaf_expected,
        heard_expected_q_imp=heard_expected,
        deaf_expected_q_imp=deaf_expected,
        selected_action_count=int(selected.size),
        heard_covered_mass=heard_mass,
        deaf_covered_mass=deaf_mass,
        tail_error_bound_imp=max(0.0, tail_bound),
        continuation_budget=(next(iter(budgets)) if len(budgets) == 1 else None),
    )
