"""Neural causal removal of one target bid's evidence at receiver time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from networks.policy_net import BELIEF_FEAT_DIM, OBS_DIM
from utils.hand_features import BELIEF_DIM, HONOR_DIM, LENGTH_BINS, NUM_SUITS


@dataclass(frozen=True)
class EvidenceRemovalResult:
    heard_features: torch.Tensor
    deaf_features: torch.Tensor
    target_event_partner_natural_delta: torch.Tensor
    target_slot: str = "partner"

    @property
    def target_event_natural_delta(self) -> torch.Tensor:
        """Slot-neutral alias for the retained legacy field name."""

        return self.target_event_partner_natural_delta


def _validate_features(value: torch.Tensor, name: str) -> None:
    if value.shape[-1] != BELIEF_FEAT_DIM:
        raise ValueError(
            f"{name} must end in {BELIEF_FEAT_DIM} partner+RHO features"
        )
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite probabilities")


def remove_target_evidence_from_features(
    heard_features: torch.Tensor,
    target_evidence_before_features: torch.Tensor,
    target_evidence_after_features: torch.Tensor,
    *,
    eps: float = 1e-6,
    target_slot: str = "partner",
) -> EvidenceRemovalResult:
    """Subtract one event's natural-parameter increment from one explicit slot.

    Inputs are actor-native 96-dimensional probabilities (partner 48, then RHO
    48). ``target_slot`` must be partner or RHO; the other half is copied
    exactly. Honor dimensions use Bernoulli logits. Each suit-length
    distribution uses log probabilities followed by a softmax after subtraction.
    """
    for value, name in (
        (heard_features, "heard_features"),
        (target_evidence_before_features, "target_evidence_before_features"),
        (target_evidence_after_features, "target_evidence_after_features"),
    ):
        _validate_features(value, name)
    if heard_features.shape != target_evidence_before_features.shape or (
        heard_features.shape != target_evidence_after_features.shape
    ):
        raise ValueError("heard/before/after belief feature shapes must match")
    if not 0.0 < float(eps) < 0.5:
        raise ValueError("eps must lie in (0, 0.5)")

    if target_slot not in ("partner", "rho"):
        raise ValueError("target_slot must be 'partner' or 'rho'")
    start = 0 if target_slot == "partner" else BELIEF_DIM
    end = start + BELIEF_DIM
    heard_partner = heard_features[..., start:end]
    before_partner = target_evidence_before_features[..., start:end]
    after_partner = target_evidence_after_features[..., start:end]

    def logit(probability: torch.Tensor) -> torch.Tensor:
        p = probability.clamp(eps, 1.0 - eps)
        return torch.log(p) - torch.log1p(-p)

    heard_honor_nat = logit(heard_partner[..., :HONOR_DIM])
    honor_delta = (
        logit(after_partner[..., :HONOR_DIM])
        - logit(before_partner[..., :HONOR_DIM])
    )
    deaf_honor = torch.sigmoid(heard_honor_nat - honor_delta).clamp(
        eps, 1.0 - eps
    )

    def normalized_log_probs(features: torch.Tensor) -> torch.Tensor:
        probabilities = features[..., HONOR_DIM:].reshape(
            features.shape[:-1] + (NUM_SUITS, LENGTH_BINS)
        ).clamp_min(eps)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        return torch.log(probabilities)

    heard_length_nat = normalized_log_probs(heard_partner)
    before_length_nat = normalized_log_probs(before_partner)
    after_length_nat = normalized_log_probs(after_partner)
    length_delta = after_length_nat - before_length_nat
    deaf_length = torch.softmax(heard_length_nat - length_delta, dim=-1)
    deaf_length = deaf_length.clamp_min(eps)
    deaf_length = deaf_length / deaf_length.sum(dim=-1, keepdim=True)
    deaf_length = deaf_length.reshape(
        heard_partner.shape[:-1] + (BELIEF_DIM - HONOR_DIM,)
    )
    deaf_partner = torch.cat((deaf_honor, deaf_length), dim=-1)
    deaf = (
        torch.cat((deaf_partner, heard_features[..., BELIEF_DIM:].clone()), dim=-1)
        if target_slot == "partner"
        else torch.cat((heard_features[..., :BELIEF_DIM].clone(), deaf_partner), dim=-1)
    )
    natural_delta = torch.cat(
        (honor_delta, length_delta.reshape(
            honor_delta.shape[:-1] + (BELIEF_DIM - HONOR_DIM,)
        )),
        dim=-1,
    )
    return EvidenceRemovalResult(
        heard_features=heard_features,
        deaf_features=deaf,
        target_event_partner_natural_delta=natural_delta,
        target_slot=target_slot,
    )


def _actor_device(actor: Any) -> torch.device:
    try:
        return next(actor.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def remove_target_evidence(
    actor: Any,
    receiver_mechanical_full_obs: np.ndarray | torch.Tensor,
    target_evidence_query_before: np.ndarray | torch.Tensor,
    target_evidence_query_after: np.ndarray | torch.Tensor,
    *,
    eps: float = 1e-6,
    target_slot: str = "partner",
) -> EvidenceRemovalResult:
    """Query the actor at receiver time, then remove only the target evidence."""
    device = _actor_device(actor)

    def observation(value: np.ndarray | torch.Tensor, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if tensor.shape[-1] != OBS_DIM:
            raise ValueError(f"{name} must end in {OBS_DIM} observation features")
        return tensor

    receiver = observation(receiver_mechanical_full_obs, "receiver_mechanical_full_obs")
    before = observation(target_evidence_query_before, "target_evidence_query_before")
    after = observation(target_evidence_query_after, "target_evidence_query_after")
    if receiver.shape != before.shape or receiver.shape != after.shape:
        raise ValueError("receiver/before/after observation shapes must match")
    with torch.no_grad():
        heard = actor.compute_belief_features(receiver)
        before_features = actor.compute_belief_features(before)
        after_features = actor.compute_belief_features(after)
        return remove_target_evidence_from_features(
            heard, before_features, after_features, eps=eps,
            target_slot=target_slot,
        )
