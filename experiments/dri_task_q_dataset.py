"""Build policy-versioned Task-Q datasets from audited Stage-2 rollouts.

One output row represents one sampled state/action under one declared
continuation population.  Labels are weighted means of terminal duplicate
DDS-IMP samples.  Deals, rather than states, are assigned to train or
calibration so multiple auction depths from one deal can never leak across the
split.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from env import BID_1C, BID_DOUBLE, BID_PASS, BID_REDOUBLE, NUM_BIDS

from experiments.dri_stage2_sensitivity import validate_input
from networks.policy_net import encode_openspiel_auction_observation
from networks.task_q import (
    DD_TABLE_CTDE_NORMALIZATION,
    REFERENCE_SCORE_CTDE_NORMALIZATION,
    TASK_Q_LABEL_SOURCE,
    TaskQDataset,
    TaskQGroupedDataset,
    normalize_dd_table_ctde,
    normalize_reference_score_ctde,
)
from utils.scoring import Contract, calculate_score
from utils.dds_reference import actor_duplicate_imp


STRUCTURED_ACTION_FEATURE_VERSION = "bridge_action_context_v1"
STRUCTURED_ACTION_FEATURE_DIM = 20


def build_analytic_stop_baseline(
    public_history: Sequence[int],
    dealer: int,
    acting_seat: int,
    vulnerability: Sequence[bool],
    dd_table: np.ndarray,
    reference_score_ns: int,
    legal_action_mask: np.ndarray,
) -> np.ndarray:
    """Return IMP if the candidate call is followed only by passes."""
    history = [int(action) for action in public_history]
    dealer = int(dealer)
    acting_seat = int(acting_seat)
    table = np.asarray(dd_table)
    legal = np.asarray(legal_action_mask, dtype=np.float32)
    if table.shape != (5, 4):
        raise ValueError("dd_table must have shape (5, 4)")
    if legal.shape != (NUM_BIDS,):
        raise ValueError("legal_action_mask must have length 38")
    if len(vulnerability) != 2:
        raise ValueError("vulnerability must contain (NS, EW)")

    baseline = np.zeros(NUM_BIDS, dtype=np.float32)
    prior_contracts = [
        (index, action) for index, action in enumerate(history) if action >= BID_1C
    ]
    for action in range(NUM_BIDS):
        if not legal[action]:
            continue
        contracts = list(prior_contracts)
        if action >= BID_1C:
            contracts.append((len(history), action))
        if not contracts:
            terminal_score_ns = 0
        else:
            final_index, final_action = contracts[-1]
            final_bidder = (
                acting_seat if final_index == len(history)
                else (dealer + final_index) % 4
            )
            final_side = final_bidder % 2
            strain = (final_action - BID_1C) % 5
            level = (final_action - BID_1C) // 5 + 1
            declarer = final_bidder
            for index, prior_action in contracts:
                bidder = (
                    acting_seat if index == len(history)
                    else (dealer + index) % 4
                )
                if bidder % 2 == final_side and (prior_action - BID_1C) % 5 == strain:
                    declarer = bidder
                    break

            doubled = 0
            for later_call in history[final_index + 1:]:
                if later_call == BID_DOUBLE:
                    doubled = 1
                elif later_call == BID_REDOUBLE:
                    doubled = 2
            if action == BID_DOUBLE:
                doubled = 1
            elif action == BID_REDOUBLE:
                doubled = 2
            elif action >= BID_1C:
                doubled = 0
            contract = Contract(
                level=level, suit=strain, doubled=doubled, declarer=declarer
            )
            declarer_score = calculate_score(
                contract,
                int(table[strain, declarer]),
                bool(vulnerability[final_side]),
            )
            terminal_score_ns = declarer_score if final_side == 0 else -declarer_score
        baseline[action] = float(actor_duplicate_imp(
            int(terminal_score_ns), int(reference_score_ns), acting_seat
        ))
    return baseline


def build_structured_action_features(
    public_history: Sequence[int],
    dealer: int,
    acting_seat: int,
    vulnerability: Sequence[bool],
    dd_table: np.ndarray,
    legal_action_mask: np.ndarray,
) -> np.ndarray:
    """Build shared, audited ``phi(a, auction, payoff)`` for all 38 calls."""
    history = [int(action) for action in public_history]
    dealer = int(dealer)
    acting_seat = int(acting_seat)
    if not 0 <= dealer < 4 or not 0 <= acting_seat < 4:
        raise ValueError("dealer and acting_seat must lie in [0, 4)")
    if len(vulnerability) != 2:
        raise ValueError("vulnerability must contain (NS, EW)")
    table = np.asarray(dd_table)
    if table.shape != (5, 4):
        raise ValueError("dd_table must have shape (5, 4)")
    legal = np.asarray(legal_action_mask, dtype=np.float32)
    if legal.shape != (NUM_BIDS,) or not np.all((legal == 0) | (legal == 1)):
        raise ValueError("legal_action_mask must be a 0/1 vector of length 38")

    contract_calls = [
        (index, action) for index, action in enumerate(history) if action >= BID_1C
    ]
    highest_index, highest_action = (
        contract_calls[-1] if contract_calls else (-1, None)
    )
    highest_bidder = (
        None if highest_index < 0 else (dealer + highest_index) % 4
    )
    bidding_partnerships = {
        (dealer + index) % 2 for index, _ in contract_calls
    }
    competitive = len(bidding_partnerships) == 2
    actor_side = acting_seat % 2
    actor_vul = float(bool(vulnerability[actor_side]))
    opponent_vul = float(bool(vulnerability[1 - actor_side]))

    features = np.zeros((NUM_BIDS, STRUCTURED_ACTION_FEATURE_DIM), dtype=np.float32)
    for action in range(NUM_BIDS):
        # type: pass, double, redouble, contract
        type_index = (
            0 if action == BID_PASS else
            1 if action == BID_DOUBLE else
            2 if action == BID_REDOUBLE else 3
        )
        features[action, type_index] = 1.0
        features[action, 13] = actor_vul
        features[action, 14] = opponent_vul
        features[action, 19] = legal[action]
        if action < BID_1C:
            continue

        level = (action - BID_1C) // 5 + 1
        strain = (action - BID_1C) % 5
        features[action, 4] = level / 7.0
        features[action, 5 + strain] = 1.0
        features[action, 10] = (
            action - (highest_action if highest_action is not None else BID_1C - 1)
        ) / 35.0
        features[action, 11] = float(
            highest_bidder is not None and highest_bidder % 2 != actor_side
        )
        features[action, 12] = float(competitive)

        # The first player of the acting partnership to bid this strain would
        # declare; if none has, the current actor would declare.
        declarer = acting_seat
        for index, prior_action in contract_calls:
            bidder = (dealer + index) % 4
            if (
                bidder % 2 == actor_side
                and (prior_action - BID_1C) % 5 == strain
            ):
                declarer = bidder
                break
        partner = (declarer + 2) % 4
        opponent_seats = [seat for seat in range(4) if seat % 2 != actor_side]
        actor_tricks = max(int(table[strain, declarer]), int(table[strain, partner]))
        opponent_tricks = max(int(table[strain, seat]) for seat in opponent_seats)
        features[action, 15] = actor_tricks / 13.0
        features[action, 16] = opponent_tricks / 13.0
        features[action, 17] = (actor_tricks - (6 + level)) / 13.0
        contract = Contract(level=level, suit=strain, doubled=0, declarer=declarer)
        declarer_score = calculate_score(
            contract, int(table[strain, declarer]), bool(vulnerability[actor_side])
        )
        features[action, 18] = declarer_score / 7600.0
    return features


CTDE_SEAT_ORDER_ACTING_RELATIVE = "self_lho_partner_rho"
CTDE_SEAT_ORDER_ABSOLUTE = "absolute_nesw"
CTDE_SEAT_ORDERS = (
    CTDE_SEAT_ORDER_ACTING_RELATIVE,
    CTDE_SEAT_ORDER_ABSOLUTE,
)


def normalize_ctde_hands(
    hands_nesw: np.ndarray, acting_seat: int, seat_order: str
) -> np.ndarray:
    """Return CTDE hands in the declared absolute or actor-relative order."""

    hands = np.asarray(hands_nesw, dtype=np.float32)
    if hands.shape != (4, 52):
        raise ValueError("private_hands_ctde must have shape (4, 52)")
    if not 0 <= acting_seat < 4:
        raise ValueError("acting_seat must lie in [0, 4)")
    if seat_order == CTDE_SEAT_ORDER_ABSOLUTE:
        return hands.copy()
    if seat_order != CTDE_SEAT_ORDER_ACTING_RELATIVE:
        raise ValueError(f"unknown ctde_seat_order: {seat_order!r}")
    indices = (
        acting_seat,
        (acting_seat + 1) % 4,
        (acting_seat + 2) % 4,
        (acting_seat - 1) % 4,
    )
    return hands[list(indices)].copy()


@dataclass(frozen=True)
class TaskQDatasetArtifact:
    policy_version: str
    policy_snapshot_hash: str
    population_ids: Mapping[str, int]
    train: TaskQDataset
    calibration: TaskQDataset
    grouped_train: TaskQGroupedDataset
    grouped_calibration: TaskQGroupedDataset
    train_deal_ids: tuple[str, ...]
    calibration_deal_ids: tuple[str, ...]
    ctde_seat_order: str = CTDE_SEAT_ORDER_ACTING_RELATIVE
    condition_on_population: bool = False
    label_source: str = TASK_Q_LABEL_SOURCE
    dd_table_normalization: str = DD_TABLE_CTDE_NORMALIZATION
    reference_score_normalization: str = REFERENCE_SCORE_CTDE_NORMALIZATION

    def __post_init__(self) -> None:
        if not self.policy_version or not self.policy_snapshot_hash:
            raise ValueError("policy version and snapshot hash must be non-empty")
        if set(self.train_deal_ids) & set(self.calibration_deal_ids):
            raise ValueError("train and calibration deals must be disjoint")
        if self.label_source != TASK_Q_LABEL_SOURCE:
            raise ValueError("Task-Q artifacts accept task-only DDS-IMP labels")
        if self.ctde_seat_order not in CTDE_SEAT_ORDERS:
            raise ValueError("unsupported ctde_seat_order")
        if self.dd_table_normalization != DD_TABLE_CTDE_NORMALIZATION:
            raise ValueError("unsupported dd_table normalization")
        if self.reference_score_normalization != REFERENCE_SCORE_CTDE_NORMALIZATION:
            raise ValueError("unsupported reference score normalization")


def _weighted_q(estimate: Mapping[str, Any]) -> float:
    samples = estimate.get("q_samples")
    declared_mean = estimate.get("q_mean")
    if samples is None:
        if "sample_weights" in estimate:
            raise ValueError("sample_weights require q_samples")
        value = float(estimate["q_mean"])
        if not math.isfinite(value):
            raise ValueError("q_mean must be finite")
        return value
    values = np.asarray(samples, dtype=np.float64)
    weights = np.asarray(
        estimate.get("sample_weights", np.ones(values.size)), dtype=np.float64
    )
    if (
        values.ndim != 1
        or values.size == 0
        or weights.shape != values.shape
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError("invalid aligned Q samples/weights")
    if declared_mean is not None:
        value = float(declared_mean)
        if not math.isfinite(value):
            raise ValueError("q_mean must be finite")
        return value
    return float(np.dot(values, weights) / weights.sum())


def _calibration_deal(deal_id: str, *, split_seed: int, fraction: float) -> bool:
    material = f"dri-task-q-deal-split-v1:{split_seed}:{deal_id}".encode("utf-8")
    quantile = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") / 2**64
    return quantile < fraction


def build_task_q_dataset_artifact(
    payload: Mapping[str, Any],
    *,
    policy_version: str,
    policy_snapshot_hash: str,
    populations: Sequence[str] | None = None,
    condition_on_population: bool = False,
    calibration_fraction: float = 0.2,
    split_seed: int = 0,
    ctde_seat_order: str = CTDE_SEAT_ORDER_ACTING_RELATIVE,
) -> TaskQDatasetArtifact:
    """Convert a validated Stage-2 payload into deal-disjoint tensors."""

    validate_input(payload)
    if ctde_seat_order not in CTDE_SEAT_ORDERS:
        raise ValueError("unsupported ctde_seat_order")
    if not policy_version or not policy_snapshot_hash:
        raise ValueError("policy version and snapshot hash must be non-empty")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must lie strictly between zero and one")
    declared = [str(row["population_id"]) for row in payload["populations"]]
    chosen = declared if populations is None else [str(value) for value in populations]
    if not chosen or len(set(chosen)) != len(chosen):
        raise ValueError("populations must be non-empty and unique")
    if len(chosen) == 1 and condition_on_population:
        raise ValueError(
            "a single population must use an unconditioned Task-Q without an embedding"
        )
    if len(chosen) > 1 and not condition_on_population:
        raise ValueError(
            "multiple populations require condition_on_population=True; "
            "an unconditional near-on-policy Q must use one policy version/population"
        )
    unknown = set(chosen) - set(declared)
    if unknown:
        raise ValueError(f"unknown populations: {sorted(unknown)}")
    population_ids = {population_id: index for index, population_id in enumerate(chosen)}

    rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for state in payload["states"]:
        acting_seat = int(state["acting_seat"])
        raw_hands = np.asarray(state["private_hands_ctde"], dtype=np.float32)
        hands = normalize_ctde_hands(raw_hands, acting_seat, ctde_seat_order)
        dd_table = normalize_dd_table_ctde(
            torch.as_tensor(state["dd_table"], dtype=torch.float32)
        ).numpy()
        reference_table = state.get("reference_table")
        if not isinstance(reference_table, Mapping) or "reference_score_ns" not in reference_table:
            raise ValueError("every state must declare reference_table.reference_score_ns")
        raw_reference_score = float(reference_table["reference_score_ns"])
        for action_record in state["actions"]:
            for population_id in chosen:
                estimate = action_record["populations"].get(population_id)
                if estimate is None:
                    raise ValueError(
                        f"state/action is missing selected population {population_id}"
                    )
                sampling = estimate.get("sampling_metadata")
                sampled_reference = (
                    sampling.get("reference_table")
                    if isinstance(sampling, Mapping) else None
                )
                if not isinstance(sampled_reference, Mapping) or (
                    "reference_score_ns" not in sampled_reference
                ):
                    raise ValueError(
                        "every action/population sample must declare reference_score_ns"
                    )
                if float(sampled_reference["reference_score_ns"]) != raw_reference_score:
                    raise ValueError(
                        "reference_score_ns must be identical across every "
                        "state/action/population sample"
                    )
        reference_score = normalize_reference_score_ctde(
            torch.tensor([raw_reference_score], dtype=torch.float32)
        ).numpy()
        observation = encode_openspiel_auction_observation(
            raw_hands,
            int(state["dealer"]),
            [int(action) for action in state["public_history"]],
            acting_seat,
            tuple(bool(value) for value in state["vulnerability"]),
        )
        legal_mask = np.asarray(state["legal_action_mask"], dtype=np.float32)
        action_features = build_structured_action_features(
            state["public_history"],
            int(state["dealer"]),
            acting_seat,
            state["vulnerability"],
            np.asarray(state["dd_table"]),
            legal_mask,
        )
        analytic_stop_baseline = build_analytic_stop_baseline(
            state["public_history"],
            int(state["dealer"]),
            acting_seat,
            state["vulnerability"],
            np.asarray(state["dd_table"]),
            int(raw_reference_score),
            legal_mask,
        )
        for action_record in state["actions"]:
            action = int(action_record["action"])
            for population_id in chosen:
                estimate = action_record["populations"].get(population_id)
                if estimate is None:
                    raise ValueError(
                        f"state/action is missing selected population {population_id}"
                    )
                rows.append({
                    "deal_id": str(state["deal_id"]),
                    "observation": observation,
                    "hands": hands,
                    "dd_table": dd_table,
                    "reference_score": reference_score,
                    "legal_mask": legal_mask,
                    "action_features": action_features,
                    "analytic_stop_baseline": analytic_stop_baseline,
                    "action": action,
                    "label": _weighted_q(estimate),
                    "population_id": population_ids[population_id],
                })
        for population_id in chosen:
            group_rows.append({
                "deal_id": str(state["deal_id"]),
                "observation": observation,
                "hands": hands,
                "dd_table": dd_table,
                "reference_score": reference_score,
                "legal_mask": legal_mask,
                "action_features": action_features,
                "analytic_stop_baseline": analytic_stop_baseline,
                "actions": [int(row["action"]) for row in state["actions"]],
                "labels": [
                    _weighted_q(row["populations"][population_id])
                    for row in state["actions"]
                ],
                "population_id": population_ids[population_id],
            })
    if not rows:
        raise ValueError("payload contains no Task-Q rows")

    deal_ids = sorted({row["deal_id"] for row in rows})
    calibration_deals = {
        deal_id for deal_id in deal_ids
        if _calibration_deal(
            deal_id, split_seed=split_seed, fraction=calibration_fraction
        )
    }
    # Tiny pilots must still exercise both paths without splitting a deal.
    if not calibration_deals:
        calibration_deals.add(deal_ids[-1])
    if calibration_deals == set(deal_ids):
        calibration_deals.remove(deal_ids[0])
    train_deals = set(deal_ids) - calibration_deals
    if not train_deals or not calibration_deals:
        raise ValueError("at least two distinct deals are required for a split")

    def make_dataset(selected_deals: set[str]) -> TaskQDataset:
        selected = [row for row in rows if row["deal_id"] in selected_deals]
        encoded_population_ids = (
            torch.tensor(
                [row["population_id"] for row in selected], dtype=torch.int64
            )
            if condition_on_population else None
        )
        return TaskQDataset(
            torch.from_numpy(np.stack([row["observation"] for row in selected])),
            torch.from_numpy(np.stack([row["hands"] for row in selected])),
            torch.from_numpy(np.stack([row["legal_mask"] for row in selected])),
            torch.tensor([row["action"] for row in selected], dtype=torch.int64),
            torch.tensor([row["label"] for row in selected], dtype=torch.float32),
            encoded_population_ids,
            dd_table_ctde=torch.from_numpy(
                np.stack([row["dd_table"] for row in selected])
            ),
            reference_score_ctde=torch.from_numpy(
                np.stack([row["reference_score"] for row in selected])
            ),
            action_features_ctde=torch.from_numpy(
                np.stack([row["action_features"] for row in selected])
            ),
            analytic_stop_baseline_ctde=torch.from_numpy(
                np.stack([row["analytic_stop_baseline"] for row in selected])
            ),
            label_source=TASK_Q_LABEL_SOURCE,
        )

    def make_grouped_dataset(selected_deals: set[str]) -> TaskQGroupedDataset:
        selected = [row for row in group_rows if row["deal_id"] in selected_deals]
        encoded_population_ids = (
            torch.tensor(
                [row["population_id"] for row in selected], dtype=torch.int64
            )
            if condition_on_population else None
        )
        return TaskQGroupedDataset(
            torch.from_numpy(np.stack([row["observation"] for row in selected])),
            torch.from_numpy(np.stack([row["hands"] for row in selected])),
            torch.from_numpy(np.stack([row["legal_mask"] for row in selected])),
            [torch.tensor(row["actions"], dtype=torch.int64) for row in selected],
            [torch.tensor(row["labels"], dtype=torch.float32) for row in selected],
            encoded_population_ids,
            dd_table_ctde=torch.from_numpy(
                np.stack([row["dd_table"] for row in selected])
            ),
            reference_score_ctde=torch.from_numpy(
                np.stack([row["reference_score"] for row in selected])
            ),
            action_features_ctde=torch.from_numpy(
                np.stack([row["action_features"] for row in selected])
            ),
            analytic_stop_baseline_ctde=torch.from_numpy(
                np.stack([row["analytic_stop_baseline"] for row in selected])
            ),
        )

    return TaskQDatasetArtifact(
        policy_version=policy_version,
        policy_snapshot_hash=policy_snapshot_hash,
        population_ids=population_ids,
        train=make_dataset(train_deals),
        calibration=make_dataset(calibration_deals),
        grouped_train=make_grouped_dataset(train_deals),
        grouped_calibration=make_grouped_dataset(calibration_deals),
        train_deal_ids=tuple(sorted(train_deals)),
        calibration_deal_ids=tuple(sorted(calibration_deals)),
        condition_on_population=condition_on_population,
        ctde_seat_order=ctde_seat_order,
    )
