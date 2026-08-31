"""Audited one-use batches for direct-counterfactual DRI policy updates.

Direct counterfactual values do not generalize to unseen states.  This module
therefore models a counterfactual batch as an immutable, policy-bound set of
events, not as an evaluator that can be refreshed and reused.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import copy
import hashlib
import json
import math
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


AuxiliaryMode = Literal["true", "shuffled", "zero"]
TreatmentKind = Literal[
    "legacy_combined", "partner", "partner_minus_beta_opponent"
]


@dataclass(frozen=True)
class DirectDRIEvent:
    """One target-bid reward record from an immutable CF batch."""

    event_id: str
    episode_index: int
    target_step_index: int
    dri_imp: float
    inclusion_probability: float
    role: str
    acting_seat: int
    depth_bucket: str
    competitive: bool
    action_type: str
    partner_dri_imp: float = 0.0
    opponent_dri_imp: float = 0.0
    partner_inclusion_probability: float | None = None
    opponent_inclusion_probability: float | None = None

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must be non-empty")
        if self.episode_index < 0 or self.target_step_index < 0:
            raise ValueError("episode and step indices must be non-negative")
        if not math.isfinite(self.dri_imp):
            raise ValueError("dri_imp must be finite")
        if not math.isfinite(self.partner_dri_imp):
            raise ValueError("partner_dri_imp must be finite")
        if not math.isfinite(self.opponent_dri_imp):
            raise ValueError("opponent_dri_imp must be finite")
        if not 0 < self.inclusion_probability <= 1:
            raise ValueError("inclusion_probability must lie in (0, 1]")
        for name in (
            "partner_inclusion_probability",
            "opponent_inclusion_probability",
        ):
            probability = getattr(self, name)
            if probability is not None and not 0 < probability <= 1:
                raise ValueError(f"{name} must lie in (0, 1] when present")
        if self.role not in {"partner", "opponent", "combined"}:
            raise ValueError("unsupported DRI role")
        if self.acting_seat not in range(4):
            raise ValueError("acting_seat must lie in [0, 4)")
        if not self.depth_bucket or not self.action_type:
            raise ValueError("shuffle strata must be explicit")

    @property
    def shuffle_stratum(self) -> tuple[object, ...]:
        return (
            self.role,
            self.acting_seat,
            self.depth_bucket,
            self.competitive,
            self.action_type,
        )


@dataclass(frozen=True)
class DirectDRIBatch:
    """Immutable identity and records for one policy-snapshot-bound batch."""

    batch_id: str
    policy_version: int
    policy_snapshot_hash: str
    population_hash: str
    label_hash: str
    events: tuple[DirectDRIEvent, ...]
    treatment: TreatmentKind = "legacy_combined"
    beta: float = 1.0
    filter_hash: str = "none"

    def __post_init__(self) -> None:
        for name in (
            "batch_id", "policy_snapshot_hash", "population_hash", "label_hash"
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        if self.treatment not in {
            "legacy_combined", "partner", "partner_minus_beta_opponent"
        }:
            raise ValueError("unsupported direct DRI treatment")
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be finite and non-negative")
        if self.treatment != "partner_minus_beta_opponent" and self.beta != 1.0:
            raise ValueError("beta is only configurable for C-prime treatment")
        if not self.filter_hash:
            raise ValueError("filter_hash must be explicit")
        if not self.events:
            raise ValueError("a direct DRI batch must contain events")
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event_id values must be unique within a batch")

    def content_hash(self) -> str:
        payload = {
            "batch_id": self.batch_id,
            "policy_version": self.policy_version,
            "policy_snapshot_hash": self.policy_snapshot_hash,
            "population_hash": self.population_hash,
            "label_hash": self.label_hash,
            "treatment": self.treatment,
            "beta": self.beta,
            "filter_hash": self.filter_hash,
            "events": [asdict(event) for event in self.events],
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AuxiliaryAssignment:
    event_id: str
    episode_index: int
    target_step_index: int
    source_dri_imp: float
    assigned_dri_imp: float
    inclusion_probability: float
    objective_weight: float
    shuffle_stratum: tuple[object, ...]


@dataclass(frozen=True)
class AuxiliaryAssignmentAudit:
    mode: AuxiliaryMode
    seed: int
    batch_content_hash: str
    event_count: int
    stratum_count: int
    source_mean: float
    assigned_mean: float
    source_std: float
    assigned_std: float
    assignments: tuple[AuxiliaryAssignment, ...]


def _derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    if size < 2:
        raise ValueError("shuffled placebo requires at least two events per stratum")
    base = np.arange(size)
    # A random cyclic shift is always a derangement and avoids an unbounded
    # rejection loop.  It also preserves every stratum's exact value multiset.
    shift = int(rng.integers(1, size))
    order = rng.permutation(size)
    return order[np.roll(np.arange(size), shift)][np.argsort(order)]


def build_auxiliary_assignments(
    batch: DirectDRIBatch,
    *,
    mode: AuxiliaryMode,
    seed: int,
) -> AuxiliaryAssignmentAudit:
    """Create a non-mutating true/shuffled/zero reward assignment.

    Horvitz-Thompson inclusion correction is represented as an explicit
    objective weight rather than silently multiplying the reward value.
    """

    if mode not in {"true", "shuffled", "zero"}:
        raise ValueError("mode must be true, shuffled, or zero")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    source = np.asarray([event.dri_imp for event in batch.events], dtype=np.float64)
    assigned = source.copy()
    strata: dict[tuple[object, ...], list[int]] = {}
    for index, event in enumerate(batch.events):
        strata.setdefault(event.shuffle_stratum, []).append(index)

    if mode == "zero":
        assigned.fill(0.0)
    elif mode == "shuffled":
        rng = np.random.default_rng(seed)
        for stratum in sorted(strata, key=repr):
            indices = np.asarray(strata[stratum], dtype=np.int64)
            if indices.size < 2:
                raise ValueError(
                    "shuffled placebo has a singleton stratum; revise the "
                    "predeclared bucketing before commissioning"
                )
            permutation = _derangement(indices.size, rng)
            assigned[indices] = source[indices[permutation]]

    assignments = tuple(
        AuxiliaryAssignment(
            event_id=event.event_id,
            episode_index=event.episode_index,
            target_step_index=event.target_step_index,
            source_dri_imp=float(source[index]),
            assigned_dri_imp=float(assigned[index]),
            inclusion_probability=event.inclusion_probability,
            objective_weight=1.0 / event.inclusion_probability,
            shuffle_stratum=event.shuffle_stratum,
        )
        for index, event in enumerate(batch.events)
    )
    return AuxiliaryAssignmentAudit(
        mode=mode,
        seed=seed,
        batch_content_hash=batch.content_hash(),
        event_count=len(assignments),
        stratum_count=len(strata),
        source_mean=float(source.mean()),
        assigned_mean=float(assigned.mean()),
        source_std=float(source.std()),
        assigned_std=float(assigned.std()),
        assignments=assignments,
    )


@dataclass(frozen=True)
class DirectAuxiliarySchedule:
    """Hard counterfactual-compute budget; KL cannot add collection rounds."""

    total_rounds: int
    collection_rounds: tuple[int, ...]
    max_batches: int
    max_auxiliary_epochs: int
    kl_ceiling: float = 0.015
    projected_hours_limit: float = 33.0
    hard_hours_limit: float = 36.0

    def __post_init__(self) -> None:
        if self.total_rounds <= 0:
            raise ValueError("total_rounds must be positive")
        if self.max_batches < 0:
            raise ValueError("max_batches cannot be negative")
        if len(self.collection_rounds) > self.max_batches:
            raise ValueError("collection rounds exceed the hard batch budget")
        if tuple(sorted(set(self.collection_rounds))) != self.collection_rounds:
            raise ValueError("collection_rounds must be unique and increasing")
        if any(not 1 <= round_index <= self.total_rounds
               for round_index in self.collection_rounds):
            raise ValueError("collection round lies outside the training run")
        if self.max_auxiliary_epochs <= 0:
            raise ValueError("max_auxiliary_epochs must be positive")
        if not 0 < self.kl_ceiling <= 0.1:
            raise ValueError("kl_ceiling must be a safety bound in (0, 0.1]")
        if not 0 < self.projected_hours_limit < self.hard_hours_limit:
            raise ValueError("projected limit must be below the hard limit")

    def should_collect(self, completed_round: int) -> bool:
        return completed_round in self.collection_rounds


class DirectAuxiliaryLifecycle:
    """Resume-safe consumption ledger for the immutable collection schedule."""

    def __init__(self, schedule: DirectAuxiliarySchedule):
        self.schedule = schedule
        self._consumed: dict[int, str] = {}

    @property
    def consumed_batches(self) -> int:
        return len(self._consumed)

    def authorize(self, completed_round: int) -> bool:
        return (
            self.schedule.should_collect(completed_round)
            and completed_round not in self._consumed
            and self.consumed_batches < self.schedule.max_batches
        )

    def consume(self, completed_round: int, batch_content_hash: str) -> None:
        if not self.authorize(completed_round):
            raise RuntimeError("counterfactual batch is not authorized")
        if not batch_content_hash:
            raise ValueError("batch_content_hash must be non-empty")
        self._consumed[completed_round] = batch_content_hash

    def state_dict(self) -> dict[str, object]:
        return {
            "schedule": asdict(self.schedule),
            "consumed": {str(key): value for key, value in self._consumed.items()},
        }

    @classmethod
    def from_state_dict(
        cls,
        schedule: DirectAuxiliarySchedule,
        state: Mapping[str, object],
    ) -> "DirectAuxiliaryLifecycle":
        expected = asdict(schedule)
        if state.get("schedule") != expected:
            raise RuntimeError("direct auxiliary schedule changed across resume")
        raw_consumed = state.get("consumed")
        if not isinstance(raw_consumed, Mapping):
            raise ValueError("consumed lifecycle state must be a mapping")
        lifecycle = cls(schedule)
        for raw_round, raw_hash in raw_consumed.items():
            round_index = int(raw_round)
            if not schedule.should_collect(round_index):
                raise RuntimeError("resume contains an undeclared collection round")
            if not isinstance(raw_hash, str) or not raw_hash:
                raise ValueError("resume contains an invalid batch hash")
            lifecycle._consumed[round_index] = raw_hash
        if lifecycle.consumed_batches > schedule.max_batches:
            raise RuntimeError("resume exceeds the hard counterfactual batch budget")
        return lifecycle


def projected_training_hours(
    *,
    ordinary_rounds: int,
    ordinary_round_minutes_p95: float,
    counterfactual_batches: int,
    counterfactual_minutes_p95: float,
    auxiliary_update_minutes_p95: float,
    fixed_overhead_minutes: float,
) -> float:
    values: Iterable[float] = (
        ordinary_round_minutes_p95,
        counterfactual_minutes_p95,
        auxiliary_update_minutes_p95,
        fixed_overhead_minutes,
    )
    if ordinary_rounds < 0 or counterfactual_batches < 0:
        raise ValueError("round and batch counts cannot be negative")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("timing inputs must be finite and non-negative")
    minutes = (
        ordinary_rounds * ordinary_round_minutes_p95
        + counterfactual_batches
        * (counterfactual_minutes_p95 + auxiliary_update_minutes_p95)
        + fixed_overhead_minutes
    )
    return minutes / 60.0


@dataclass(frozen=True)
class DirectAuxiliaryPPOConfig:
    """Frozen actor-only auxiliary PPO controls.

    ``kl_ceiling`` is a rollback boundary, never a requested displacement.
    Advantages are deliberately not standardized per batch: doing so would
    erase the scientific meaning of ``lambda_dri`` and the IMP calibration.
    """

    lambda_dri: float
    advantage_scale_imp: float
    max_epochs: int
    minibatch_size: int
    clip_ratio: float = 0.2
    kl_ceiling: float = 0.015
    max_grad_norm: float = 0.5
    entropy_coefficient: float = 0.0
    center_advantages: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.lambda_dri) or self.lambda_dri < 0:
            raise ValueError("lambda_dri must be finite and non-negative")
        if not math.isfinite(self.advantage_scale_imp) or self.advantage_scale_imp <= 0:
            raise ValueError("advantage_scale_imp must be finite and positive")
        if self.max_epochs <= 0 or self.minibatch_size <= 0:
            raise ValueError("epoch and minibatch counts must be positive")
        if not 0 < self.clip_ratio < 1:
            raise ValueError("clip_ratio must lie in (0, 1)")
        if not 0 < self.kl_ceiling <= 0.1:
            raise ValueError("kl_ceiling must be a safety bound in (0, 0.1]")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.entropy_coefficient < 0:
            raise ValueError("entropy_coefficient cannot be negative")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class DirectAuxiliaryTensorBatch:
    observations: torch.Tensor
    legal_actions: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_action_probs: torch.Tensor
    dri_imp: torch.Tensor
    objective_weights: torch.Tensor
    event_ids: tuple[str, ...]
    batch_content_hash: str
    policy_snapshot_hash: str

    def validate(self) -> None:
        count = self.observations.shape[0]
        if count <= 0:
            raise ValueError("auxiliary tensor batch must be non-empty")
        if self.observations.ndim != 2:
            raise ValueError("observations must be a two-dimensional tensor")
        action_count = self.legal_actions.shape[-1]
        expected = {
            "legal_actions": (count, action_count),
            "old_action_probs": (count, action_count),
            "actions": (count,),
            "old_log_probs": (count,),
            "dri_imp": (count,),
            "objective_weights": (count,),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if len(self.event_ids) != count or len(set(self.event_ids)) != count:
            raise ValueError("event_ids must be unique and match the tensor batch")
        if not self.batch_content_hash or not self.policy_snapshot_hash:
            raise ValueError("batch and policy hashes must be explicit")
        tensors = (
            self.observations, self.legal_actions, self.old_log_probs,
            self.old_action_probs, self.dri_imp, self.objective_weights,
        )
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("auxiliary tensors must be finite")
        legal = self.legal_actions.to(torch.bool)
        if not torch.all((self.legal_actions == 0) | (self.legal_actions == 1)):
            raise ValueError("legal_actions must contain only 0/1 values")
        if not torch.all(legal.any(dim=-1)):
            raise ValueError("every event must have a legal action")
        if torch.any(self.actions < 0) or torch.any(self.actions >= action_count):
            raise ValueError("action index is out of range")
        chosen_legal = legal.gather(1, self.actions.long().unsqueeze(1)).squeeze(1)
        if not torch.all(chosen_legal):
            raise ValueError("stored action must be legal")
        if torch.any(self.old_action_probs < 0):
            raise ValueError("old_action_probs cannot be negative")
        if not torch.allclose(
            self.old_action_probs.sum(dim=-1),
            torch.ones(count, dtype=self.old_action_probs.dtype,
                       device=self.old_action_probs.device),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("old_action_probs must sum to one")
        if torch.any(self.old_action_probs.masked_select(~legal) > 1e-7):
            raise ValueError("illegal old-action probability must be zero")
        chosen_probability = self.old_action_probs.gather(
            1, self.actions.long().unsqueeze(1)
        ).squeeze(1).clamp_min(1e-12)
        if not torch.allclose(
            chosen_probability.log(), self.old_log_probs, atol=1e-5, rtol=1e-5
        ):
            raise ValueError("old_log_probs do not match old_action_probs")
        if torch.any(self.objective_weights <= 0):
            raise ValueError("objective_weights must be positive")


@dataclass(frozen=True)
class DirectAuxiliaryUpdateAudit:
    batch_content_hash: str
    policy_snapshot_hash: str
    event_count: int
    epochs_attempted: int
    optimizer_steps: int
    rolled_back: bool
    rollback_reason: str | None
    approximate_kl: float
    exact_categorical_kl: float
    clip_fraction: float
    effective_sample_size: float
    mean_policy_loss: float
    mean_entropy: float
    mean_gradient_norm: float
    gradient_cosine_with_task: float | None
    dri_advantage_mean: float
    dri_advantage_std: float


def tensor_batch_with_assignments(
    base: DirectAuxiliaryTensorBatch,
    assignment_audit: AuxiliaryAssignmentAudit,
) -> DirectAuxiliaryTensorBatch:
    """Bind an audited placebo/true assignment to stored on-policy tensors."""

    if assignment_audit.batch_content_hash != base.batch_content_hash:
        raise RuntimeError("assignment and tensor batch content hashes differ")
    assigned_ids = tuple(item.event_id for item in assignment_audit.assignments)
    if assigned_ids != base.event_ids:
        raise RuntimeError("assignment event order differs from the tensor batch")
    return replace(
        base,
        dri_imp=torch.as_tensor(
            [item.assigned_dri_imp for item in assignment_audit.assignments],
            dtype=base.dri_imp.dtype,
            device=base.dri_imp.device,
        ),
        objective_weights=torch.as_tensor(
            [item.objective_weight for item in assignment_audit.assignments],
            dtype=base.objective_weights.dtype,
            device=base.objective_weights.device,
        ),
    )


def tensor_batch_with_assignment_subset(
    base: DirectAuxiliaryTensorBatch,
    assignment_audit: AuxiliaryAssignmentAudit,
) -> DirectAuxiliaryTensorBatch:
    """Bind the seat-specific subset of one global assignment audit."""

    if assignment_audit.batch_content_hash != base.batch_content_hash:
        raise RuntimeError("assignment and tensor batch content hashes differ")
    by_id = {item.event_id: item for item in assignment_audit.assignments}
    if not set(base.event_ids).issubset(by_id):
        raise RuntimeError("seat tensor batch contains an unassigned event")
    selected = [by_id[event_id] for event_id in base.event_ids]
    return replace(
        base,
        dri_imp=torch.as_tensor(
            [item.assigned_dri_imp for item in selected],
            dtype=base.dri_imp.dtype, device=base.dri_imp.device,
        ),
        objective_weights=torch.as_tensor(
            [item.objective_weight for item in selected],
            dtype=base.objective_weights.dtype,
            device=base.objective_weights.device,
        ),
    )


def _flatten_gradients(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    pieces = [
        parameter.grad.detach().reshape(-1)
        for parameter in parameters if parameter.grad is not None
    ]
    if not pieces:
        device = parameters[0].device if parameters else torch.device("cpu")
        return torch.zeros(0, device=device)
    return torch.cat(pieces)


@dataclass(frozen=True)
class GradientRatioLambdaCalibration:
    rho: float
    lambda_dri: float
    task_gradient_norms: Mapping[int, float]
    unit_dri_gradient_norms: Mapping[int, float]
    seat_lambda_candidates: Mapping[int, float]
    delivered_gradient_ratios: Mapping[int, float]


@dataclass(frozen=True)
class JointDisplacementSGDCalibration:
    rho: float
    task_displacement_norms: Mapping[int, float]
    unit_dri_gradient_norms: Mapping[int, float]
    joint_task_displacement_norm: float
    joint_unit_dri_gradient_norm: float
    learning_rate: float
    target_joint_aux_displacement: float
    predicted_seat_aux_displacements: Mapping[int, float]
    predicted_seat_displacement_ratios: Mapping[int, float]


@dataclass(frozen=True)
class FullBatchSGDUpdateAudit:
    batch_content_hash: str
    policy_snapshot_hash: str
    event_count: int
    learning_rate: float
    policy_loss: float
    gradient_norm: float
    parameter_displacement_before_rollback: float
    exact_categorical_kl: float
    effective_sample_size: float
    dri_advantage_mean: float
    dri_advantage_std: float
    rolled_back: bool
    rollback_reason: str | None


def unit_dri_policy_gradient(
    actor: torch.nn.Module,
    batch: DirectAuxiliaryTensorBatch,
    *,
    advantage_scale_imp: float,
) -> torch.Tensor:
    """Return the pure policy gradient for unit standardized DRI advantage.

    This is a read-only calibration diagnostic: it performs no optimizer step
    and does not populate ``parameter.grad``.
    """

    batch.validate()
    if not math.isfinite(advantage_scale_imp) or advantage_scale_imp <= 0:
        raise ValueError("advantage_scale_imp must be finite and positive")
    parameters = [
        parameter for parameter in actor.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("actor has no trainable parameters")
    device = parameters[0].device
    observations = batch.observations.to(device)
    legal = batch.legal_actions.to(device)
    actions = batch.actions.to(device).long()
    old_log_probs = batch.old_log_probs.to(device)
    raw_weights = batch.objective_weights.to(device)
    weights = raw_weights / raw_weights.mean()
    advantages = batch.dri_imp.to(device) / advantage_scale_imp
    advantages = advantages - (advantages * weights).sum() / weights.sum()
    log_probs, _entropy = actor.evaluate_actions(observations, legal, actions)
    ratio = torch.exp(log_probs - old_log_probs)
    policy_loss = -(ratio * advantages * weights).sum() / weights.sum()
    if not torch.isfinite(policy_loss):
        raise FloatingPointError("non-finite unit DRI policy loss")
    gradients = torch.autograd.grad(
        policy_loss, parameters, allow_unused=True
    )
    pieces = [
        torch.zeros_like(parameter).reshape(-1)
        if gradient is None else gradient.detach().reshape(-1)
        for parameter, gradient in zip(parameters, gradients, strict=True)
    ]
    return torch.cat(pieces).cpu()


def calibrate_lambda_by_gradient_ratio(
    task_gradients: Mapping[int, torch.Tensor],
    unit_dri_gradients: Mapping[int, torch.Tensor],
    *,
    rho: float = 0.20,
) -> GradientRatioLambdaCalibration:
    """Freeze one global lambda from the median per-seat norm ratio."""

    if not math.isfinite(rho) or not 0 < rho <= 1:
        raise ValueError("rho must lie in (0, 1]")
    seats = sorted(set(task_gradients) | set(unit_dri_gradients))
    if seats != list(range(4)):
        raise ValueError("gradient calibration requires seats 0,1,2,3")
    task_norms: dict[int, float] = {}
    unit_norms: dict[int, float] = {}
    candidates: dict[int, float] = {}
    for seat in seats:
        task = task_gradients[seat].detach().float().reshape(-1)
        unit = unit_dri_gradients[seat].detach().float().reshape(-1)
        if task.shape != unit.shape:
            raise ValueError(f"seat {seat} gradient width mismatch")
        if not torch.isfinite(task).all() or not torch.isfinite(unit).all():
            raise ValueError(f"seat {seat} gradient is non-finite")
        task_norm = float(torch.linalg.vector_norm(task).item())
        unit_norm = float(torch.linalg.vector_norm(unit).item())
        if task_norm <= 0 or unit_norm <= 0:
            raise ValueError(f"seat {seat} gradient norm must be positive")
        task_norms[seat] = task_norm
        unit_norms[seat] = unit_norm
        candidates[seat] = rho * task_norm / unit_norm
    lambda_dri = float(np.median(list(candidates.values())))
    if not math.isfinite(lambda_dri) or not 0 < lambda_dri <= 1:
        raise RuntimeError("gradient-ratio lambda lies outside frozen (0, 1] range")
    delivered = {
        seat: lambda_dri * unit_norms[seat] / task_norms[seat]
        for seat in seats
    }
    return GradientRatioLambdaCalibration(
        rho=rho,
        lambda_dri=lambda_dri,
        task_gradient_norms=task_norms,
        unit_dri_gradient_norms=unit_norms,
        seat_lambda_candidates=candidates,
        delivered_gradient_ratios=delivered,
    )


def calibrate_joint_displacement_sgd(
    task_displacements: Mapping[int, float],
    unit_dri_gradients: Mapping[int, torch.Tensor],
    *,
    rho: float = 0.20,
) -> JointDisplacementSGDCalibration:
    """Derive one scale-faithful SGD step from joint four-actor L2 norms."""

    if not math.isfinite(rho) or not 0 < rho <= 1:
        raise ValueError("rho must lie in (0, 1]")
    seats = sorted(set(task_displacements) | set(unit_dri_gradients))
    if seats != list(range(4)):
        raise ValueError("joint displacement calibration requires seats 0,1,2,3")
    task_norms: dict[int, float] = {}
    gradient_norms: dict[int, float] = {}
    for seat in seats:
        task_norm = float(task_displacements[seat])
        gradient = unit_dri_gradients[seat].detach().float().reshape(-1)
        if not math.isfinite(task_norm) or task_norm <= 0:
            raise ValueError(f"seat {seat} task displacement must be positive")
        if not torch.isfinite(gradient).all():
            raise ValueError(f"seat {seat} DRI gradient is non-finite")
        gradient_norm = float(torch.linalg.vector_norm(gradient).item())
        if gradient_norm <= 0:
            raise ValueError(f"seat {seat} DRI gradient norm must be positive")
        task_norms[seat] = task_norm
        gradient_norms[seat] = gradient_norm
    joint_task = math.sqrt(sum(value * value for value in task_norms.values()))
    joint_gradient = math.sqrt(
        sum(value * value for value in gradient_norms.values())
    )
    target = rho * joint_task
    learning_rate = target / joint_gradient
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise RuntimeError("joint displacement calibration produced invalid SGD rate")
    seat_displacements = {
        seat: learning_rate * gradient_norms[seat] for seat in seats
    }
    seat_ratios = {
        seat: seat_displacements[seat] / task_norms[seat] for seat in seats
    }
    return JointDisplacementSGDCalibration(
        rho=rho,
        task_displacement_norms=task_norms,
        unit_dri_gradient_norms=gradient_norms,
        joint_task_displacement_norm=joint_task,
        joint_unit_dri_gradient_norm=joint_gradient,
        learning_rate=learning_rate,
        target_joint_aux_displacement=target,
        predicted_seat_aux_displacements=seat_displacements,
        predicted_seat_displacement_ratios=seat_ratios,
    )


def run_full_batch_sgd_actor_update(
    actor: torch.nn.Module,
    batch: DirectAuxiliaryTensorBatch,
    *,
    learning_rate: float,
    advantage_scale_imp: float,
    kl_ceiling: float = 0.015,
) -> FullBatchSGDUpdateAudit:
    """Apply one scale-faithful full-batch SGD step with transactional rollback."""

    batch.validate()
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(advantage_scale_imp) or advantage_scale_imp <= 0:
        raise ValueError("advantage_scale_imp must be finite and positive")
    if not math.isfinite(kl_ceiling) or not 0 < kl_ceiling <= 0.1:
        raise ValueError("kl_ceiling must lie in (0, 0.1]")
    parameters = [
        parameter for parameter in actor.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("actor has no trainable parameters")
    device = parameters[0].device
    observations = batch.observations.to(device)
    legal = batch.legal_actions.to(device)
    actions = batch.actions.to(device).long()
    old_log_probs = batch.old_log_probs.to(device)
    old_action_probs = batch.old_action_probs.to(device)
    raw_weights = batch.objective_weights.to(device)
    weights = raw_weights / raw_weights.mean()
    advantages = batch.dri_imp.to(device) / advantage_scale_imp
    advantages = advantages - (advantages * weights).sum() / weights.sum()
    entry_state = copy.deepcopy(actor.state_dict())
    try:
        log_probs, _entropy = actor.evaluate_actions(
            observations, legal, actions
        )
        ratio = torch.exp(log_probs - old_log_probs)
        policy_loss = -(ratio * advantages * weights).sum() / weights.sum()
        if not torch.isfinite(policy_loss):
            raise FloatingPointError("non-finite full-batch DRI policy loss")
        gradients = torch.autograd.grad(
            policy_loss, parameters, allow_unused=True
        )
        gradient_pieces = []
        for parameter, gradient in zip(parameters, gradients, strict=True):
            if gradient is None:
                gradient = torch.zeros_like(parameter)
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("non-finite full-batch DRI gradient")
            gradient_pieces.append(gradient.detach().reshape(-1))
        flat_gradient = torch.cat(gradient_pieces)
        gradient_norm = float(torch.linalg.vector_norm(flat_gradient).item())
        with torch.no_grad():
            for parameter, gradient in zip(parameters, gradients, strict=True):
                if gradient is not None:
                    parameter.add_(gradient, alpha=-learning_rate)
        displacement = learning_rate * gradient_norm
        exact_kl = _exact_policy_kl(
            actor, observations, legal, old_action_probs, weights
        )
        rollback_reason = None
        if not math.isfinite(exact_kl):
            rollback_reason = "non-finite exact categorical KL"
        elif exact_kl > kl_ceiling:
            rollback_reason = (
                f"exact categorical KL {exact_kl:.8f} exceeded ceiling "
                f"{kl_ceiling:.8f}"
            )
        rolled_back = rollback_reason is not None
        if rolled_back:
            actor.load_state_dict(entry_state)
    except Exception:
        actor.load_state_dict(entry_state)
        raise
    weight_sum = float(raw_weights.sum().item())
    effective_sample_size = weight_sum * weight_sum / float(
        torch.square(raw_weights).sum().item()
    )
    return FullBatchSGDUpdateAudit(
        batch_content_hash=batch.batch_content_hash,
        policy_snapshot_hash=batch.policy_snapshot_hash,
        event_count=int(observations.shape[0]),
        learning_rate=learning_rate,
        policy_loss=float(policy_loss.item()),
        gradient_norm=gradient_norm,
        parameter_displacement_before_rollback=displacement,
        exact_categorical_kl=exact_kl,
        effective_sample_size=effective_sample_size,
        dri_advantage_mean=float(advantages.mean().item()),
        dri_advantage_std=float(advantages.std(unbiased=False).item()),
        rolled_back=rolled_back,
        rollback_reason=rollback_reason,
    )


def _nested_state_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _nested_state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _nested_state_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _exact_policy_kl(
    actor: torch.nn.Module,
    observations: torch.Tensor,
    legal_actions: torch.Tensor,
    old_action_probs: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    with torch.no_grad():
        logits = actor(observations, legal_actions)
        new_log_probs = F.log_softmax(logits, dim=-1)
        old = old_action_probs.clamp_min(1e-12)
        per_event = (old_action_probs * (old.log() - new_log_probs)).sum(dim=-1)
        return float((per_event * weights).sum().item() / weights.sum().item())


def run_direct_auxiliary_actor_update(
    actor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: DirectAuxiliaryTensorBatch,
    config: DirectAuxiliaryPPOConfig,
    *,
    adjacent_task_gradient: torch.Tensor | None = None,
) -> DirectAuxiliaryUpdateAudit:
    """Run one transactional actor-only PPO update.

    Any KL breach or non-finite diagnostic restores both actor and optimizer to
    their exact entry state.  The caller owns checkpoint persistence around the
    transaction.
    """

    batch.validate()
    parameters = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("actor has no trainable parameters")
    parameter_ids = {id(parameter) for group in optimizer.param_groups
                     for parameter in group["params"]}
    if parameter_ids != {id(parameter) for parameter in parameters}:
        raise ValueError("optimizer must own exactly the trainable actor parameters")

    device = parameters[0].device
    observations = batch.observations.to(device)
    legal = batch.legal_actions.to(device)
    actions = batch.actions.to(device).long()
    old_log_probs = batch.old_log_probs.to(device)
    old_action_probs = batch.old_action_probs.to(device)
    raw_weights = batch.objective_weights.to(device)
    weights = raw_weights / raw_weights.mean()
    advantages = (
        config.lambda_dri * batch.dri_imp.to(device) / config.advantage_scale_imp
    )
    if config.center_advantages:
        advantages = advantages - (advantages * weights).sum() / weights.sum()

    actor_state = copy.deepcopy(actor.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    rng = torch.Generator(device="cpu")
    rng.manual_seed(config.seed)
    count = observations.shape[0]
    loss_values: list[float] = []
    entropy_values: list[float] = []
    gradient_norms: list[float] = []
    gradient_cosines: list[float] = []
    approximate_kls: list[float] = []
    clipped = 0
    compared = 0
    optimizer_steps = 0
    epochs_attempted = 0
    rollback_reason: str | None = None

    try:
        for epoch in range(config.max_epochs):
            epochs_attempted = epoch + 1
            order = torch.randperm(count, generator=rng)
            for start in range(0, count, config.minibatch_size):
                indices = order[start:start + config.minibatch_size].to(device)
                b_obs = observations[indices]
                b_legal = legal[indices]
                b_actions = actions[indices]
                b_old_log_probs = old_log_probs[indices]
                b_advantages = advantages[indices]
                b_weights = weights[indices]

                log_probs, entropy = actor.evaluate_actions(
                    b_obs, b_legal, b_actions
                )
                ratio = torch.exp(log_probs - b_old_log_probs)
                unclipped = ratio * b_advantages
                clipped_objective = torch.clamp(
                    ratio, 1 - config.clip_ratio, 1 + config.clip_ratio
                ) * b_advantages
                surrogate = torch.minimum(unclipped, clipped_objective)
                policy_loss = -(surrogate * b_weights).sum() / b_weights.sum()
                entropy_mean = (entropy * b_weights).sum() / b_weights.sum()
                loss = policy_loss - config.entropy_coefficient * entropy_mean
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite auxiliary loss")

                optimizer.zero_grad()
                loss.backward()
                flat_gradient = _flatten_gradients(parameters)
                gradient_norm = float(torch.linalg.vector_norm(flat_gradient).item())
                if not math.isfinite(gradient_norm):
                    raise FloatingPointError("non-finite auxiliary gradient")
                if adjacent_task_gradient is not None and flat_gradient.numel():
                    task = adjacent_task_gradient.to(device).reshape(-1)
                    if task.numel() != flat_gradient.numel():
                        raise ValueError("adjacent task gradient has the wrong width")
                    denominator = (
                        torch.linalg.vector_norm(flat_gradient)
                        * torch.linalg.vector_norm(task)
                    )
                    if denominator > 0:
                        gradient_cosines.append(float(
                            torch.dot(flat_gradient, task).item() / denominator.item()
                        ))
                torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
                optimizer.step()
                optimizer_steps += 1

                with torch.no_grad():
                    approximate_kls.append(float(
                        (b_old_log_probs - log_probs).mean().item()
                    ))
                    clipped += int((torch.abs(ratio - 1) > config.clip_ratio).sum().item())
                    compared += int(ratio.numel())
                loss_values.append(float(policy_loss.item()))
                entropy_values.append(float(entropy_mean.item()))
                gradient_norms.append(gradient_norm)

            exact_kl = _exact_policy_kl(
                actor, observations, legal, old_action_probs, weights
            )
            if not math.isfinite(exact_kl):
                raise FloatingPointError("non-finite exact categorical KL")
            if exact_kl > config.kl_ceiling:
                rollback_reason = (
                    f"exact categorical KL {exact_kl:.8f} exceeded ceiling "
                    f"{config.kl_ceiling:.8f}"
                )
                break
    except Exception:
        actor.load_state_dict(actor_state)
        optimizer.load_state_dict(optimizer_state)
        raise

    exact_kl = _exact_policy_kl(actor, observations, legal, old_action_probs, weights)
    rolled_back = rollback_reason is not None
    if rolled_back:
        actor.load_state_dict(actor_state)
        optimizer.load_state_dict(optimizer_state)

    weight_sum = float(raw_weights.sum().item())
    effective_sample_size = weight_sum * weight_sum / float(
        torch.square(raw_weights).sum().item()
    )
    return DirectAuxiliaryUpdateAudit(
        batch_content_hash=batch.batch_content_hash,
        policy_snapshot_hash=batch.policy_snapshot_hash,
        event_count=count,
        epochs_attempted=epochs_attempted,
        optimizer_steps=optimizer_steps,
        rolled_back=rolled_back,
        rollback_reason=rollback_reason,
        approximate_kl=float(np.mean(approximate_kls)) if approximate_kls else 0.0,
        exact_categorical_kl=exact_kl,
        clip_fraction=clipped / max(1, compared),
        effective_sample_size=effective_sample_size,
        mean_policy_loss=float(np.mean(loss_values)) if loss_values else 0.0,
        mean_entropy=float(np.mean(entropy_values)) if entropy_values else 0.0,
        mean_gradient_norm=float(np.mean(gradient_norms)) if gradient_norms else 0.0,
        gradient_cosine_with_task=(
            float(np.mean(gradient_cosines)) if gradient_cosines else None
        ),
        dri_advantage_mean=float(advantages.mean().item()),
        dri_advantage_std=float(advantages.std(unbiased=False).item()),
    )


@dataclass(frozen=True)
class DirectAuxiliaryForkAudit:
    batch_content_hash: str
    branch_audits: Mapping[str, DirectAuxiliaryUpdateAudit]
    parameter_delta_norms: Mapping[str, float]
    probe_probabilities: Mapping[str, tuple[tuple[float, ...], ...]]
    true_vs_shuffled_probe_l1: float
    true_vs_zero_probe_l1: float
    entry_state_restored: bool


def commission_three_way_fork(
    actor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    branch_batches: Mapping[str, DirectAuxiliaryTensorBatch],
    config: DirectAuxiliaryPPOConfig,
    *,
    probe_observations: torch.Tensor,
    probe_legal_actions: torch.Tensor,
    adjacent_task_gradient: torch.Tensor | None = None,
) -> DirectAuxiliaryForkAudit:
    """Run true/shuffled/zero from one entry state and restore that state.

    This is intentionally a commissioning-only fork. It does not select or
    commit one branch to continued training.
    """

    required = {"true", "shuffled", "zero"}
    if set(branch_batches) != required:
        raise ValueError("commissioning fork requires true/shuffled/zero batches")
    reference = branch_batches["true"]
    reference.validate()
    for name, candidate in branch_batches.items():
        candidate.validate()
        identity = (
            candidate.batch_content_hash,
            candidate.policy_snapshot_hash,
            candidate.event_ids,
        )
        expected = (
            reference.batch_content_hash,
            reference.policy_snapshot_hash,
            reference.event_ids,
        )
        if identity != expected:
            raise RuntimeError(f"{name} branch does not share immutable batch identity")
        tensor_pairs = (
            (candidate.observations, reference.observations),
            (candidate.legal_actions, reference.legal_actions),
            (candidate.actions, reference.actions),
            (candidate.old_log_probs, reference.old_log_probs),
            (candidate.old_action_probs, reference.old_action_probs),
        )
        if any(not torch.equal(left, right) for left, right in tensor_pairs):
            raise RuntimeError(f"{name} branch changed stored on-policy tensors")

    parameters = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    entry_actor = copy.deepcopy(actor.state_dict())
    entry_optimizer = copy.deepcopy(optimizer.state_dict())
    entry_flat = torch.cat([
        value.detach().reshape(-1).cpu() for value in entry_actor.values()
        if torch.is_floating_point(value)
    ])
    device = parameters[0].device
    probe_observations = probe_observations.to(device)
    probe_legal_actions = probe_legal_actions.to(device)
    if probe_observations.ndim != 2 or probe_legal_actions.ndim != 2:
        raise ValueError("probe tensors must be two-dimensional")
    if probe_observations.shape[0] != probe_legal_actions.shape[0]:
        raise ValueError("probe observation/legal batch sizes differ")

    branch_audits: dict[str, DirectAuxiliaryUpdateAudit] = {}
    parameter_delta_norms: dict[str, float] = {}
    probe_probabilities: dict[str, tuple[tuple[float, ...], ...]] = {}
    try:
        for name in ("true", "shuffled", "zero"):
            actor.load_state_dict(entry_actor)
            optimizer.load_state_dict(copy.deepcopy(entry_optimizer))
            branch_audits[name] = run_direct_auxiliary_actor_update(
                actor,
                optimizer,
                branch_batches[name],
                config,
                adjacent_task_gradient=adjacent_task_gradient,
            )
            current = actor.state_dict()
            current_flat = torch.cat([
                value.detach().reshape(-1).cpu() for value in current.values()
                if torch.is_floating_point(value)
            ])
            parameter_delta_norms[name] = float(
                torch.linalg.vector_norm(current_flat - entry_flat).item()
            )
            with torch.no_grad():
                probabilities = torch.softmax(
                    actor(probe_observations, probe_legal_actions), dim=-1
                ).detach().cpu()
            probe_probabilities[name] = tuple(
                tuple(float(value) for value in row) for row in probabilities
            )
    finally:
        actor.load_state_dict(entry_actor)
        optimizer.load_state_dict(entry_optimizer)

    true_probe = np.asarray(probe_probabilities["true"], dtype=np.float64)
    shuffled_probe = np.asarray(probe_probabilities["shuffled"], dtype=np.float64)
    zero_probe = np.asarray(probe_probabilities["zero"], dtype=np.float64)
    restored = all(
        torch.equal(value, actor.state_dict()[key])
        for key, value in entry_actor.items()
    ) and _nested_state_equal(optimizer.state_dict(), entry_optimizer)
    return DirectAuxiliaryForkAudit(
        batch_content_hash=reference.batch_content_hash,
        branch_audits=branch_audits,
        parameter_delta_norms=parameter_delta_norms,
        probe_probabilities=probe_probabilities,
        true_vs_shuffled_probe_l1=float(np.abs(true_probe - shuffled_probe).mean()),
        true_vs_zero_probe_l1=float(np.abs(true_probe - zero_probe).mean()),
        entry_state_restored=restored,
    )
