"""Deterministic heard/deaf policy-mass action coverage for Task-Q labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from dri.evidence_removal import remove_target_evidence
from networks.policy_net import encode_openspiel_auction_observation


@dataclass(frozen=True)
class PolicyMassSelection:
    actions: tuple[int, ...]
    covered_mass: Mapping[str, float]
    tail_mass: Mapping[str, float]
    tail_q_error_bound_imp: float
    target: float


def serialize_policy_distributions(
    distributions: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    """Persist full action probabilities needed for direct DRI recomputation."""

    if not distributions:
        raise ValueError("at least one policy distribution is required")
    normalized: dict[str, list[float]] = {}
    action_count = None
    for name, raw in distributions.items():
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"distribution {name!r} must be one-dimensional")
        if action_count is None:
            action_count = int(values.size)
        elif values.size != action_count:
            raise ValueError("policy distributions have inconsistent action widths")
        if np.any(values < 0) or not np.all(np.isfinite(values)):
            raise ValueError(f"distribution {name!r} is invalid")
        total = float(values.sum())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"distribution {name!r} must sum to one")
        normalized[str(name)] = [float(value / total) for value in values]
    return {
        "action_order": list(range(int(action_count))),
        "probabilities": normalized,
    }


def receiver_heard_deaf_distributions(
    actor,
    hands_suit_major: np.ndarray,
    dealer: int,
    vulnerability: Sequence[bool],
    public_history: Sequence[int],
    acting_seat: int,
    legal_action_mask: Sequence[int | float | bool],
) -> dict[str, np.ndarray]:
    """Return native heard plus every mechanically defined deaf intervention.

    At a receiver decision, the immediately previous call is RHO evidence and
    the call two positions back is partner evidence. Early auction states retain
    only the interventions whose target call exists.
    """
    history = [int(action) for action in public_history]
    hands = np.asarray(hands_suit_major, dtype=np.float32)
    legal = np.asarray(legal_action_mask, dtype=np.float32)
    if hands.shape != (4, 52):
        raise ValueError("hands_suit_major must have shape (4, 52)")
    if legal.shape != (actor.num_actions,):
        raise ValueError("legal_action_mask has the wrong shape")
    if not np.any(legal > 0.5):
        raise ValueError("at least one action must be legal")
    receiver_obs = encode_openspiel_auction_observation(
        hands, int(dealer), history, int(acting_seat), tuple(vulnerability)
    )
    device = next(actor.parameters()).device
    obs_t = torch.as_tensor(
        receiver_obs, dtype=torch.float32, device=device
    ).unsqueeze(0)
    legal_t = torch.as_tensor(
        legal, dtype=torch.float32, device=device
    ).unsqueeze(0)
    with torch.no_grad():
        heard_features = actor.compute_belief_features(obs_t)
        heard_logits = actor.forward_with_belief_features(
            obs_t, legal_t, heard_features
        )
        distributions = {
            "heard": F.softmax(heard_logits, dim=-1).squeeze(0).cpu().numpy()
        }
    targets = []
    if len(history) >= 2:
        targets.append(("deaf_partner", history[:-2], history[:-1], "partner"))
    if len(history) >= 1:
        targets.append(("deaf_rho", history[:-1], history, "rho"))
    for name, before_history, after_history, slot in targets:
        before = encode_openspiel_auction_observation(
            hands, int(dealer), before_history, int(acting_seat), tuple(vulnerability)
        )
        after = encode_openspiel_auction_observation(
            hands, int(dealer), after_history, int(acting_seat), tuple(vulnerability)
        )
        removed = remove_target_evidence(
            actor, receiver_obs[None, :], before[None, :], after[None, :],
            target_slot=slot,
        )
        with torch.no_grad():
            logits = actor.forward_with_belief_features(
                obs_t, legal_t, removed.deaf_features
            )
            distributions[name] = (
                F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            )
    return distributions


def select_policy_mass_union(
    distributions: Mapping[str, Sequence[float]],
    legal_action_mask: Sequence[int | float | bool],
    *,
    forced_actions: Sequence[int] = (),
    target: float = 0.9995,
    q_error_span_imp: float = 48.0,
) -> PolicyMassSelection:
    """Select the deterministic minimal per-policy mass sets and their union."""
    if not 0.0 < target <= 1.0:
        raise ValueError("target must lie in (0, 1]")
    legal = np.asarray(legal_action_mask, dtype=bool)
    if legal.ndim != 1 or not np.any(legal):
        raise ValueError("legal_action_mask must be a non-empty vector")
    legal_ids = set(np.flatnonzero(legal).tolist())
    chosen = {int(action) for action in forced_actions}
    if not chosen.issubset(legal_ids):
        raise ValueError("forced actions must be legal")
    normalized = {}
    for name, raw in distributions.items():
        values = np.asarray(raw, dtype=np.float64)
        if values.shape != legal.shape:
            raise ValueError(f"distribution {name!r} has the wrong shape")
        if np.any(values < 0) or not np.all(np.isfinite(values)):
            raise ValueError(f"distribution {name!r} is invalid")
        if np.any(values[~legal] > 1e-10):
            raise ValueError(f"distribution {name!r} assigns mass to illegal actions")
        total = float(values[legal].sum())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"distribution {name!r} legal mass must sum to one")
        values = values / total
        normalized[name] = values
        ranked = sorted(legal_ids, key=lambda action: (-values[action], action))
        cumulative = 0.0
        for action in ranked:
            chosen.add(action)
            cumulative += float(values[action])
            if cumulative + 1e-12 >= target:
                break
    covered = {
        name: float(values[list(chosen)].sum())
        for name, values in normalized.items()
    }
    tails = {name: max(0.0, 1.0 - mass) for name, mass in covered.items()}
    heard_tail = tails.get("heard", 0.0)
    deaf_tails = [value for name, value in tails.items() if name.startswith("deaf")]
    worst_pair_tail = heard_tail + (max(deaf_tails) if deaf_tails else heard_tail)
    return PolicyMassSelection(
        actions=tuple(sorted(chosen)),
        covered_mass=covered,
        tail_mass=tails,
        tail_q_error_bound_imp=float(q_error_span_imp * worst_pair_tail),
        target=float(target),
    )
