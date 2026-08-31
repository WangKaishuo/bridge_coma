"""Map receiver-state direct DRI labels back to current-policy target calls."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from env import BridgeBiddingEnv
from dri.direct_auxiliary import DirectDRIBatch, DirectDRIEvent, TreatmentKind
from dri.direct_auxiliary import DirectAuxiliaryTensorBatch
from dri.direct_estimator import direct_dri_from_label_state
from networks.policy_net import encode_openspiel_auction_observation


SEAT_NAMES = "NESW"
SINGLETON_MAX_EVENT_FRACTION = 0.005
SINGLETON_MAX_ABS_DRI_MASS_FRACTION = 0.01


def _action_type(action: int) -> str:
    if action == 0:
        return "pass"
    if action == 1:
        return "double"
    if action == 2:
        return "redouble"
    return "contract"


def _depth_bucket(step: int) -> str:
    if step < 4:
        return "0-3"
    if step < 8:
        return "4-7"
    return "8+"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _coarsen_shuffle_event(event: DirectDRIEvent, level: int) -> DirectDRIEvent:
    if level not in range(4):
        raise ValueError("shuffle coarsening level must lie in [0, 3]")
    action_type = event.action_type
    depth_bucket = event.depth_bucket
    competitive = event.competitive
    if level >= 1:
        action_type = "pass" if action_type == "pass" else "nonpass"
    if level >= 2:
        depth_bucket = "0-3" if depth_bucket == "0-3" else "4+"
    if level >= 3:
        competitive = False
    return replace(
        event, action_type=action_type, depth_bucket=depth_bucket,
        competitive=competitive,
    )


def _shuffle_counts(events: Sequence[DirectDRIEvent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        key = json.dumps(event.shuffle_stratum, separators=(",", ":"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _deal_episode_indices(payload: Mapping[str, Any]) -> dict[str, int]:
    deals = {str(state["deal_id"]) for state in payload.get("states", [])}
    by_row = {}
    for deal in deals:
        match = re.search(r"heldout-row-(\d+)", deal)
        if match is None:
            raise ValueError(f"cannot extract held-out row from {deal!r}")
        by_row[int(match.group(1))] = deal
    declared = payload.get("sampling_metadata", {}).get(
        "deal_sampling", {}
    ).get("row_indices", [])
    order = [by_row[row] for row in declared if row in by_row]
    order.extend(sorted(deals - set(order)))
    return {deal: index for index, deal in enumerate(order)}


def build_direct_dri_batch(
    payload: Mapping[str, Any],
    *,
    batch_id: str,
    policy_version: int,
    policy_snapshot_hash: str,
    label_hash: str,
    population_id: str = "PI60",
    current_member_id: str = "PI60:current",
    treatment: TreatmentKind = "partner_minus_beta_opponent",
    beta: float = 1.0,
) -> tuple[DirectDRIBatch, dict[str, Any]]:
    """Aggregate partner benefit and opponent leakage on each current call.

    A current-policy target call receives `+DRI_partner` when its partner is
    the receiver and `-DRI_rho` when the receiver is an opponent. Multiple
    receiver records for the same trajectory call are summed once so PPO never
    sees duplicate samples for one action.
    """

    if payload.get("schema_version") != "direct-counterfactual-dri-samples-v1":
        raise ValueError("payload is not a direct counterfactual label batch")
    if not all((batch_id, policy_snapshot_hash, label_hash, current_member_id)):
        raise ValueError("batch and provenance identities must be non-empty")
    episodes = _deal_episode_indices(payload)
    aggregated: dict[tuple[str, int], dict[str, Any]] = {}
    contribution_count = 0
    max_tail = 0.0
    for state in payload.get("states", []):
        history = [int(action) for action in state["public_history"]]
        dealer = int(state["dealer"])
        behavior = state.get("behavior", {})
        member_ids = behavior.get("member_ids_by_seat")
        if not isinstance(member_ids, Mapping):
            raise ValueError("state behavior lacks realized member IDs by seat")
        inclusion = float(state.get("direct_dri_state_inclusion_probability", 1.0))
        if not math.isfinite(inclusion) or not 0 < inclusion <= 1:
            raise ValueError("invalid direct DRI state inclusion probability")
        competitive = (
            str(state.get("strata", {}).get("competitive_status", ""))
            != "non_competitive"
        )
        candidates = []
        if len(history) >= 1:
            candidates.append(("deaf_rho", len(history) - 1, "opponent"))
        if len(history) >= 2:
            candidates.append(("deaf_partner", len(history) - 2, "partner"))
        available = state.get("policy_distributions", {}).get("probabilities", {})
        for deaf_name, target_step, role in candidates:
            if deaf_name not in available:
                continue
            target_seat = (dealer + target_step) % 4
            if str(member_ids.get(SEAT_NAMES[target_seat])) != current_member_id:
                continue
            estimate = direct_dri_from_label_state(
                state, deaf_distribution=deaf_name, population_id=population_id
            )
            key = (str(state["deal_id"]), target_step)
            target_action = history[target_step]
            record = aggregated.setdefault(key, {
                "episode_index": episodes[key[0]],
                "target_step_index": target_step,
                "acting_seat": target_seat,
                "target_action": target_action,
                "partner_dri_imp": 0.0,
                "opponent_dri_imp": 0.0,
                "partner_tail_bound_imp": 0.0,
                "opponent_tail_bound_imp": 0.0,
                "partner_inclusion_probability": None,
                "opponent_inclusion_probability": None,
                "roles": set(),
                "inclusion_probability": inclusion,
                "competitive": competitive,
                "source_state_ids": [],
            })
            if (
                record["acting_seat"] != target_seat
                or record["target_action"] != target_action
            ):
                raise RuntimeError("target-call aggregation identity conflict")
            if not math.isclose(record["inclusion_probability"], inclusion):
                raise RuntimeError(
                    "combined target contributions have unequal inclusion probabilities"
                )
            record[f"{role}_dri_imp"] += estimate.dri_imp
            record[f"{role}_tail_bound_imp"] += estimate.tail_error_bound_imp
            component_probability = f"{role}_inclusion_probability"
            if record[component_probability] is None:
                record[component_probability] = inclusion
            elif not math.isclose(record[component_probability], inclusion):
                raise RuntimeError(
                    f"{role} target contributions have unequal inclusion probabilities"
                )
            record["roles"].add(role)
            record["competitive"] = record["competitive"] or competitive
            record["source_state_ids"].append(str(state["state_id"]))
            contribution_count += 1
            max_tail = max(max_tail, estimate.tail_error_bound_imp)

    events = []
    audit_events = []
    for (deal_id, target_step), record in sorted(
        aggregated.items(), key=lambda item: (
            item[1]["episode_index"], item[1]["target_step_index"]
        )
    ):
        roles = record["roles"]
        role = "combined" if len(roles) > 1 else next(iter(roles))
        partner_dri = float(record["partner_dri_imp"])
        opponent_dri = float(record["opponent_dri_imp"])
        if treatment == "partner":
            dri_imp = partner_dri
            treatment_tail = float(record["partner_tail_bound_imp"])
        elif treatment == "partner_minus_beta_opponent":
            dri_imp = partner_dri - beta * opponent_dri
            treatment_tail = float(
                record["partner_tail_bound_imp"]
                + beta * record["opponent_tail_bound_imp"]
            )
        elif treatment == "legacy_combined":
            dri_imp = partner_dri - opponent_dri
            treatment_tail = float(
                record["partner_tail_bound_imp"]
                + record["opponent_tail_bound_imp"]
            )
        else:
            raise ValueError(f"unsupported treatment {treatment!r}")
        event_id = hashlib.sha256(
            f"{batch_id}:{deal_id}:{target_step}".encode("utf-8")
        ).hexdigest()
        event = DirectDRIEvent(
            event_id=f"direct-dri-event-{event_id[:32]}",
            episode_index=int(record["episode_index"]),
            target_step_index=int(target_step),
            dri_imp=dri_imp,
            inclusion_probability=float(record["inclusion_probability"]),
            role=role,
            acting_seat=int(record["acting_seat"]),
            depth_bucket=_depth_bucket(target_step),
            competitive=bool(record["competitive"]),
            action_type=_action_type(int(record["target_action"])),
            partner_dri_imp=partner_dri,
            opponent_dri_imp=opponent_dri,
            partner_inclusion_probability=record[
                "partner_inclusion_probability"
            ],
            opponent_inclusion_probability=record[
                "opponent_inclusion_probability"
            ],
        )
        events.append(event)
        audit_events.append({
            **asdict(event),
            "deal_id": deal_id,
            "target_action": int(record["target_action"]),
            "source_state_ids": sorted(record["source_state_ids"]),
            "partner_tail_bound_imp": float(record["partner_tail_bound_imp"]),
            "opponent_tail_bound_imp": float(record["opponent_tail_bound_imp"]),
            "treatment_tail_bound_imp": treatment_tail,
        })
    if not events:
        raise RuntimeError("no current-policy target calls were recovered")
    raw_events = tuple(events)
    selected_level = 0
    shuffle_ready = False
    filter_applied = False
    excluded_indices: list[int] = []
    excluded_event_fraction = 0.0
    excluded_abs_dri_mass_fraction = 0.0
    pre_filter_counts: dict[str, int] = {}
    for level in range(4):
        candidate = tuple(_coarsen_shuffle_event(event, level) for event in raw_events)
        counts = _shuffle_counts(candidate)
        if counts and min(counts.values()) >= 2:
            events = list(candidate)
            selected_level = level
            shuffle_ready = True
            pre_filter_counts = counts
            break
    else:
        selected_level = 3
        candidate = tuple(
            _coarsen_shuffle_event(event, selected_level)
            for event in raw_events
        )
        pre_filter_counts = _shuffle_counts(candidate)
        singleton_keys = {
            key for key, count in pre_filter_counts.items() if count < 2
        }
        excluded_indices = [
            index for index, event in enumerate(candidate)
            if json.dumps(event.shuffle_stratum, separators=(",", ":"))
            in singleton_keys
        ]
        retained = [
            event for index, event in enumerate(candidate)
            if index not in set(excluded_indices)
        ]
        excluded_event_fraction = len(excluded_indices) / len(candidate)
        total_abs_mass = sum(abs(event.dri_imp) for event in candidate)
        excluded_abs_mass = sum(
            abs(candidate[index].dri_imp) for index in excluded_indices
        )
        excluded_abs_dri_mass_fraction = (
            excluded_abs_mass / total_abs_mass
            if total_abs_mass > 0 else float(excluded_abs_mass > 0)
        )
        retained_counts = _shuffle_counts(retained)
        filter_applied = bool(excluded_indices)
        filter_within_limits = (
            bool(retained)
            and excluded_event_fraction <= SINGLETON_MAX_EVENT_FRACTION
            and excluded_abs_dri_mass_fraction
            <= SINGLETON_MAX_ABS_DRI_MASS_FRACTION
            and {event.acting_seat for event in retained} == set(range(4))
            and min(retained_counts.values()) >= 2
        )
        if filter_within_limits:
            events = retained
            counts = retained_counts
            shuffle_ready = True
        else:
            events = list(raw_events)
            counts = _shuffle_counts(events)
            excluded_indices = []
            filter_applied = False

    retained_ids = {event.event_id for event in events}
    effective_level = selected_level if shuffle_ready else 0
    selected_events = tuple(
        _coarsen_shuffle_event(event, effective_level) for event in raw_events
    )
    retained_audit_events = []
    excluded_audit_events = []
    for raw_event, event, audit_event in zip(
        raw_events, selected_events, audit_events, strict=True
    ):
        audit_event["raw_shuffle_stratum"] = list(raw_event.shuffle_stratum)
        audit_event.update({
            "role": event.role,
            "acting_seat": event.acting_seat,
            "depth_bucket": event.depth_bucket,
            "competitive": event.competitive,
            "action_type": event.action_type,
            "excluded_by_singleton_filter": event.event_id not in retained_ids,
        })
        if event.event_id in retained_ids:
            retained_audit_events.append(audit_event)
        else:
            excluded_audit_events.append(audit_event)

    filter_record = {
        "hierarchy_level": selected_level,
        "applied": filter_applied,
        "excluded_event_ids": sorted(
            row["event_id"] for row in excluded_audit_events
        ),
        "excluded_strata": sorted({
            json.dumps(
                (
                    row["role"], row["acting_seat"], row["depth_bucket"],
                    row["competitive"], row["action_type"],
                ),
                separators=(",", ":"),
            )
            for row in excluded_audit_events
        }),
        "max_event_fraction": SINGLETON_MAX_EVENT_FRACTION,
        "max_abs_dri_mass_fraction": (
            SINGLETON_MAX_ABS_DRI_MASS_FRACTION
        ),
    }
    filter_hash = _canonical_hash(filter_record)
    population_hash = _canonical_hash(payload.get("populations", []))
    batch = DirectDRIBatch(
        batch_id=batch_id,
        policy_version=policy_version,
        policy_snapshot_hash=policy_snapshot_hash,
        population_hash=population_hash,
        label_hash=label_hash,
        events=tuple(events),
        treatment=treatment,
        beta=beta,
        filter_hash=filter_hash,
    )
    stratum_counts = _shuffle_counts(events)
    audit = {
        "batch_content_hash": batch.content_hash(),
        "event_count": len(events),
        "receiver_contribution_count": contribution_count,
        "role_counts": {
            role: sum(event.role == role for event in events)
            for role in ("partner", "opponent", "combined")
        },
        "dri_mean_imp": float(sum(event.dri_imp for event in events) / len(events)),
        "dri_std_imp": float(np.std([event.dri_imp for event in events])),
        "dri_abs_max_imp": float(max(abs(event.dri_imp) for event in events)),
        "max_single_receiver_tail_bound_imp": max_tail,
        "max_target_event_treatment_tail_bound_imp": float(max(
            row["treatment_tail_bound_imp"] for row in retained_audit_events
        )),
        "treatment": treatment,
        "beta": beta,
        "component_nonzero_counts": {
            "partner": sum(event.partner_dri_imp != 0.0 for event in events),
            "opponent": sum(event.opponent_dri_imp != 0.0 for event in events),
        },
        "shuffle_stratum_count": len(stratum_counts),
        "shuffle_singleton_stratum_count": sum(
            count == 1 for count in stratum_counts.values()
        ),
        "minimum_shuffle_stratum_size": min(stratum_counts.values()),
        "shuffle_stratum_sizes": stratum_counts,
        "shuffle_ready": shuffle_ready,
        "shuffle_coarsening_level": (
            selected_level if shuffle_ready else None
        ),
        "shuffle_attempted_max_level": selected_level,
        "shuffle_pre_filter_stratum_sizes": pre_filter_counts,
        "singleton_filter_applied": filter_applied,
        "singleton_filter_hash": filter_hash,
        "singleton_filter_record": filter_record,
        "singleton_excluded_event_count": len(excluded_audit_events),
        "singleton_excluded_event_fraction": excluded_event_fraction,
        "singleton_excluded_abs_dri_mass_fraction": (
            excluded_abs_dri_mass_fraction
        ),
        "shuffle_coarsening_hierarchy": [
            "exact role/seat/depth/competitive/action-type",
            "merge non-pass action types",
            "merge depth 4-7 and 8+",
            "drop competitive status",
        ],
        "sign_convention": {
            "partner": "+receiver_DRI",
            "opponent": "-receiver_DRI",
            "combined": "+partner_DRI-opponent_DRI",
        },
        "events": retained_audit_events,
        "excluded_events": excluded_audit_events,
    }
    return batch, audit


def build_direct_auxiliary_tensor_batches(
    payload: Mapping[str, Any],
    batch: DirectDRIBatch,
    audit: Mapping[str, Any],
    actors: Mapping[int, torch.nn.Module],
) -> dict[int, DirectAuxiliaryTensorBatch]:
    """Reconstruct target-call observations and bind snapshot probabilities."""

    states = {str(state["state_id"]): state for state in payload.get("states", [])}
    audit_events = {
        str(row["event_id"]): row for row in audit.get("events", [])
    }
    if set(audit_events) != {event.event_id for event in batch.events}:
        raise ValueError("batch events and builder audit events disagree")
    rows_by_seat: dict[int, list[dict[str, Any]]] = {}
    for event in batch.events:
        row = audit_events[event.event_id]
        sources = row.get("source_state_ids", [])
        if not sources:
            raise ValueError("target event lacks a receiver-state reconstruction source")
        source = states[str(sources[0])]
        hands = np.asarray(source["private_hands_ctde"], dtype=np.float32)
        history = [int(action) for action in source["public_history"]]
        prefix = history[: event.target_step_index]
        target_action = int(row["target_action"])
        if history[event.target_step_index] != target_action:
            raise RuntimeError("target action differs from its trajectory history")
        env = BridgeBiddingEnv(max_history_len=max(60, len(history) + 4))
        env.reset(
            hands, dealer=int(source["dealer"]),
            vulnerability=tuple(bool(value) for value in source["vulnerability"]),
        )
        for action in prefix:
            _, _, done, _ = env.step(action)
            if done:
                raise RuntimeError("target-call prefix terminates before the target")
        if env.state.current_player != event.acting_seat:
            raise RuntimeError("reconstructed target seat differs from the event")
        legal = env._get_legal_actions().astype(np.float32)
        if legal[target_action] <= 0.5:
            raise RuntimeError("recorded target action is illegal at reconstruction")
        observation = encode_openspiel_auction_observation(
            hands, int(source["dealer"]), prefix, event.acting_seat,
            tuple(bool(value) for value in source["vulnerability"]),
        ).astype(np.float32)
        rows_by_seat.setdefault(event.acting_seat, []).append({
            "event": event,
            "observation": observation,
            "legal": legal,
            "action": target_action,
        })

    tensor_batches = {}
    for seat, rows in sorted(rows_by_seat.items()):
        if seat not in actors:
            raise ValueError(f"no frozen actor supplied for seat {seat}")
        actor = actors[seat]
        device = next(actor.parameters()).device
        observations = torch.as_tensor(
            np.stack([row["observation"] for row in rows]),
            dtype=torch.float32, device=device,
        )
        legal = torch.as_tensor(
            np.stack([row["legal"] for row in rows]),
            dtype=torch.float32, device=device,
        )
        actions = torch.as_tensor(
            [row["action"] for row in rows], dtype=torch.long, device=device,
        )
        with torch.no_grad():
            probabilities = torch.softmax(actor(observations, legal), dim=-1)
            old_log_probs = probabilities.gather(
                1, actions.unsqueeze(1)
            ).squeeze(1).clamp_min(1e-12).log()
        tensor_batch = DirectAuxiliaryTensorBatch(
            observations=observations,
            legal_actions=legal,
            actions=actions,
            old_log_probs=old_log_probs,
            old_action_probs=probabilities,
            dri_imp=torch.as_tensor(
                [row["event"].dri_imp for row in rows],
                dtype=torch.float32, device=device,
            ),
            objective_weights=torch.as_tensor(
                [1.0 / row["event"].inclusion_probability for row in rows],
                dtype=torch.float32, device=device,
            ),
            event_ids=tuple(row["event"].event_id for row in rows),
            batch_content_hash=batch.content_hash(),
            policy_snapshot_hash=batch.policy_snapshot_hash,
        )
        tensor_batch.validate()
        tensor_batches[seat] = tensor_batch
    return tensor_batches
