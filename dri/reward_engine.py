"""Read-only episode engine for partner/opponent neural DRI contributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from dri.evidence_removal import remove_target_evidence_from_features
from dri.neural import neural_decision_regret_reduction
from dri.receiver_opportunity import (
    ReceiverOpportunityExtraction,
    extract_receiver_opportunity,
)
from networks.policy_net import MLPPolicyNetwork
from networks.task_q import TaskQNetwork, StructuredTaskQNetwork


@dataclass(frozen=True)
class ReceiverDRIAudit:
    role: str
    receiver_seat: int | None
    dri: float
    reward_eligible: bool
    reason: str | None
    receiver_step_offset: int | None = None
    actor_batch_key: str | None = None
    actor_batch_size: int = 0
    target_slot: str | None = None


@dataclass(frozen=True)
class TargetBidDRIContribution:
    target_step_index: int
    target_bidder: int
    target_action: int
    partner_dri: float
    opponent_dri: float
    combined_dri: float
    beta: float
    partner_audit: ReceiverDRIAudit
    opponent_audit: ReceiverDRIAudit


@dataclass(frozen=True)
class EpisodeDRIResult:
    q_checkpoint_hash: str
    policy_version: int
    beta: float
    contributions: tuple[TargetBidDRIContribution, ...]
    ctde_seat_order: str = "self_lho_partner_rho"


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _validate_engine_identity(
    task_q: TaskQNetwork, q_checkpoint_hash: str, policy_version: int
) -> None:
    if not isinstance(q_checkpoint_hash, str) or not q_checkpoint_hash.strip():
        raise ValueError("q_checkpoint_hash must be explicit and non-empty")
    if not isinstance(policy_version, int) or isinstance(policy_version, bool) or policy_version < 0:
        raise ValueError("policy_version must be an explicit non-negative integer")
    if not getattr(task_q, "_task_q_frozen", False):
        raise RuntimeError("Task-Q must be lifecycle-frozen before reward evaluation")
    if task_q.training or any(parameter.requires_grad for parameter in task_q.parameters()):
        raise RuntimeError("frozen Task-Q must be eval-only with gradients disabled")


def _target_slot(extraction: ReceiverOpportunityExtraction, role: str) -> str:
    bidder, receiver = extraction.target_bidder, extraction.receiver_seat
    if extraction.receiver_role != role:
        raise RuntimeError("receiver extraction role/engine role mismatch")
    if role == "partner":
        if receiver != (bidder + 2) % 4:
            raise RuntimeError("partner receiver/target seat contract mismatch")
        return "partner"
    if role == "opponent":
        if receiver != (bidder + 1) % 4 or bidder != (receiver - 1) % 4:
            raise RuntimeError("opponent receiver/target RHO seat contract mismatch")
        return "rho"
    raise RuntimeError(f"unsupported receiver role: {role!r}")


def _task_q_hands(
    hands_nesw: np.ndarray, acting_seat: int, ctde_seat_order: str
) -> np.ndarray:
    hands = np.asarray(hands_nesw, dtype=np.float32)
    if hands.shape != (4, 52):
        raise ValueError("Task-Q hands must have shape (4, 52)")
    if ctde_seat_order == "absolute_nesw":
        return hands.copy()
    if ctde_seat_order != "self_lho_partner_rho":
        raise ValueError("unsupported ctde_seat_order")
    indices = (
        acting_seat,
        (acting_seat + 1) % 4,
        (acting_seat + 2) % 4,
        (acting_seat - 1) % 4,
    )
    return hands[list(indices)].copy()


def compute_episode_dri_contributions(
    episode_steps: Sequence[Mapping[str, Any]],
    actors_by_seat: Mapping[int, MLPPolicyNetwork],
    task_q: TaskQNetwork,
    *,
    q_checkpoint_hash: str,
    policy_version: int,
    beta: float = 0.0,
    include_opponent: bool = False,
    population_id: int | None = None,
    ctde_seat_order: str = "self_lho_partner_rho",
    receiver_extractor: Callable[..., ReceiverOpportunityExtraction] = (
        extract_receiver_opportunity
    ),
) -> EpisodeDRIResult:
    """Compute auditable DRI values without modifying episode step rewards.

    B-prime uses ``beta=0``. C-prime verifies that the next opponent sees the
    target bidder in its RHO slot. Every terminal-before-receiver arm contributes
    zero and retains the extractor's reason in its audit record.
    """

    _validate_engine_identity(task_q, q_checkpoint_hash, policy_version)
    if beta < 0:
        raise ValueError("beta cannot be negative")
    if beta > 0 and not include_opponent:
        raise ValueError("positive beta requires include_opponent=True")
    if task_q.config.num_populations:
        if population_id is None:
            raise ValueError("conditioned Task-Q requires an explicit population_id")
        if not 0 <= population_id < task_q.config.num_populations:
            raise ValueError("population_id is out of range")
    elif population_id is not None:
        raise ValueError("population_id supplied to an unconditioned Task-Q")

    mutable: list[dict[str, Any]] = []
    pending: list[tuple[int, str, ReceiverOpportunityExtraction]] = []
    for step_index, target_step in enumerate(episode_steps):
        if not target_step.get("_rinfo"):
            continue
        record: dict[str, Any] = {
            "target_step_index": step_index,
            "target_bidder": int(target_step["player"]),
            "target_action": int(target_step["action"]),
        }
        mutable.append(record)
        record_index = len(mutable) - 1
        roles = ("partner", "opponent") if include_opponent else ("partner",)
        for role in roles:
            extraction = receiver_extractor(
                target_step, episode_steps[step_index + 1:], role=role
            )
            target_slot = _target_slot(extraction, role)
            if extraction.reward_eligible:
                pending.append((record_index, role, extraction))
            else:
                record[role] = ReceiverDRIAudit(
                    role=role,
                    receiver_seat=extraction.receiver_seat,
                    dri=0.0,
                    reward_eligible=False,
                    reason=extraction.no_reward_reason,
                    target_slot=target_slot,
                )
        if not include_opponent:
            record["opponent"] = ReceiverDRIAudit(
                role="opponent",
                receiver_seat=None,
                dri=0.0,
                reward_eligible=False,
                reason="opponent_dri_disabled",
            )

    # Evidence decoding and actor inference are batched by receiver actor.
    groups: dict[int, list[tuple[int, str, ReceiverOpportunityExtraction]]] = {}
    group_actors: dict[int, MLPPolicyNetwork] = {}
    for item in pending:
        extraction = item[2]
        if extraction.receiver_seat not in actors_by_seat:
            raise ValueError(f"missing actor for receiver seat {extraction.receiver_seat}")
        actor = actors_by_seat[extraction.receiver_seat]
        key = id(actor)
        groups.setdefault(key, []).append(item)
        group_actors[key] = actor

    q_device = _model_device(task_q)
    for group_index, (key, items) in enumerate(groups.items()):
        actor = group_actors[key]
        if _model_device(actor) != q_device:
            raise ValueError("receiver actor and frozen Task-Q must share a device")
        receiver_obs = np.stack([
            item[2].receiver_mechanical_full_obs for item in items
        ])
        before_obs = np.stack([
            item[2].target_evidence_query_before for item in items
        ])
        after_obs = np.stack([
            item[2].target_evidence_query_after for item in items
        ])
        observations = torch.as_tensor(receiver_obs, dtype=torch.float32, device=q_device)
        before = torch.as_tensor(before_obs, dtype=torch.float32, device=q_device)
        after = torch.as_tensor(after_obs, dtype=torch.float32, device=q_device)
        with torch.no_grad():
            heard_features = actor.compute_belief_features(observations)
            before_features = actor.compute_belief_features(before)
            after_features = actor.compute_belief_features(after)
            deaf_features = torch.empty_like(heard_features)
            slots = [_target_slot(item[2], item[1]) for item in items]
            for slot in ("partner", "rho"):
                indices = [index for index, value in enumerate(slots) if value == slot]
                if not indices:
                    continue
                index_tensor = torch.tensor(indices, dtype=torch.int64, device=q_device)
                removed = remove_target_evidence_from_features(
                    heard_features[index_tensor],
                    before_features[index_tensor],
                    after_features[index_tensor],
                    target_slot=slot,
                )
                deaf_features[index_tensor] = removed.deaf_features
        all_hands = torch.as_tensor(np.stack([
            _task_q_hands(
                item[2].all_hands_ctde,
                int(item[2].acting_seat),
                ctde_seat_order,
            ) for item in items
        ]), dtype=torch.float32, device=q_device)
        dd_table_ctde = torch.as_tensor(np.stack([
            item[2].dd_table_ctde for item in items
        ]), dtype=torch.float32, device=q_device)
        reference_score_ctde = torch.as_tensor(np.stack([
            item[2].reference_score_ctde for item in items
        ]), dtype=torch.float32, device=q_device)
        legal = torch.as_tensor(np.stack([
            item[2].legal_action_mask for item in items
        ]), dtype=torch.bool, device=q_device)
        population_ids = (
            torch.full(
                (len(items),), population_id, dtype=torch.int64, device=q_device
            ) if population_id is not None else None
        )
        action_features_ctde = None
        if isinstance(task_q, StructuredTaskQNetwork):
            if any(item[2].action_features_ctde is None for item in items):
                raise ValueError(
                    "structured Task-Q receiver steps require public history, "
                    "dealer, vulnerability, and action features"
                )
            action_features_ctde = torch.as_tensor(np.stack([
                item[2].action_features_ctde for item in items
            ]), dtype=torch.float32, device=q_device)
        neural = neural_decision_regret_reduction(
            actor,
            task_q,
            observations,
            all_hands,
            legal,
            heard_features,
            deaf_features,
            population_ids,
            dd_table_ctde=dd_table_ctde,
            reference_score_ctde=reference_score_ctde,
            action_features_ctde=action_features_ctde,
        )
        batch_key = f"receiver_actor_{group_index}"
        values = neural.decision_regret_reduction.detach().cpu().tolist()
        for value, slot, (record_index, role, extraction) in zip(
            values, slots, items, strict=True
        ):
            mutable[record_index][role] = ReceiverDRIAudit(
                role=role,
                receiver_seat=extraction.receiver_seat,
                dri=float(value),
                reward_eligible=True,
                reason=None,
                receiver_step_offset=extraction.receiver_step_offset,
                actor_batch_key=batch_key,
                actor_batch_size=len(items),
                target_slot=slot,
            )

    contributions = []
    for record in mutable:
        partner = record["partner"]
        opponent = record["opponent"]
        combined = partner.dri - beta * opponent.dri
        contributions.append(TargetBidDRIContribution(
            target_step_index=record["target_step_index"],
            target_bidder=record["target_bidder"],
            target_action=record["target_action"],
            partner_dri=partner.dri,
            opponent_dri=opponent.dri,
            combined_dri=combined,
            beta=float(beta),
            partner_audit=partner,
            opponent_audit=opponent,
        ))
    return EpisodeDRIResult(
        q_checkpoint_hash=q_checkpoint_hash,
        policy_version=policy_version,
        beta=float(beta),
        contributions=tuple(contributions),
        ctde_seat_order=ctde_seat_order,
    )
