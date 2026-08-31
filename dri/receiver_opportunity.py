"""Extract the next real receiver decision for one target bid.

The extractor is deliberately task-Q agnostic.  The target-message before/after
observations estimate only that event's evidence.  They are not deaf/heard
receiver states: both downstream worlds share the later real receiver decision
observation, legal mask, acting seat, and CTDE deal returned here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Any

import numpy as np
import torch

from env import NUM_BIDS, NUM_PLAYERS
from networks.policy_net import OBS_DIM
from networks.task_q import (
    normalize_dd_table_ctde,
    normalize_reference_score_ctde,
)
from experiments.dri_task_q_dataset import build_structured_action_features


@dataclass(frozen=True)
class ReceiverOpportunityExtraction:
    """A receiver decision, or an explicit no-reward terminal outcome."""

    reward_eligible: bool
    target_bidder: int
    target_action: int
    receiver_seat: int
    receiver_role: str
    no_reward_reason: str | None = None
    receiver_step_offset: int | None = None
    receiver_mechanical_full_obs: np.ndarray | None = None
    target_evidence_query_before: np.ndarray | None = None
    target_evidence_query_after: np.ndarray | None = None
    legal_action_mask: np.ndarray | None = None
    acting_seat: int | None = None
    all_hands_ctde: np.ndarray | None = None
    dd_table_ctde: np.ndarray | None = None
    reference_score_ctde: np.ndarray | None = None
    action_features_ctde: np.ndarray | None = None


def _required(step: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in step:
        raise ValueError(f"{context} is missing required field {key!r}")
    return step[key]


def _vector(value: Any, length: int, label: str, *, dtype=None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != (length,):
        raise ValueError(f"{label} must have shape ({length},), got {array.shape}")
    return array.copy()


def extract_receiver_opportunity(
    target_step: Mapping[str, Any],
    subsequent_steps: Sequence[Mapping[str, Any]],
    *,
    role: str = "partner",
) -> ReceiverOpportunityExtraction:
    """Find the selected receiver's next actual decision state.

    ``role='partner'`` uses the target bidder's partner and remains the default
    DRI path. ``role='opponent'`` uses the right-hand opponent recorded by
    ``_rinfo`` for the C-prime leakage path. The selected role's existing
    before/after fields are target-evidence queries only; mechanical state and
    legality come from that receiver's later real episode step.
    """
    if not target_step.get("_rinfo"):
        raise ValueError("target_step must be an _rinfo bid step")
    if role not in {"partner", "opponent"}:
        raise ValueError("role must be 'partner' or 'opponent'")
    bidder = int(_required(target_step, "player", "target_step"))
    receiver = int(_required(target_step, f"{role}_pos", "target_step"))
    action = int(_required(target_step, "action", "target_step"))
    if not 0 <= bidder < NUM_PLAYERS or not 0 <= receiver < NUM_PLAYERS:
        raise ValueError("target bidder and receiver seats must lie in [0, 4)")
    expected_receiver = (
        (bidder + 2) % NUM_PLAYERS
        if role == "partner"
        else (bidder + 1) % NUM_PLAYERS
    )
    if receiver != expected_receiver:
        raise ValueError(
            f"_rinfo {role}_pos is not the target bidder's expected {role}"
        )

    heard = _vector(
        _required(target_step, f"{role}_obs_after", "target_step"),
        OBS_DIM,
        f"target_step.{role}_obs_after",
        dtype=np.float32,
    )
    deaf = _vector(
        _required(target_step, f"{role}_obs_before", "target_step"),
        OBS_DIM,
        f"target_step.{role}_obs_before",
        dtype=np.float32,
    )
    target_hands = np.asarray(
        _required(target_step, "all_hands", "target_step"), dtype=np.float32
    )
    if target_hands.shape != (NUM_PLAYERS, 52):
        raise ValueError(
            f"target_step.all_hands must have shape (4, 52), got {target_hands.shape}"
        )
    target_dd_table = np.asarray(
        _required(target_step, "dd_table", "target_step"), dtype=np.float32
    )
    if target_dd_table.shape != (5, 4):
        raise ValueError(
            f"target_step.dd_table must have shape (5, 4), got {target_dd_table.shape}"
        )
    target_reference_score = float(
        _required(target_step, "reference_score_ns", "target_step")
    )

    terminal_seen = bool(target_step.get("done", False))
    for offset, step in enumerate(subsequent_steps, start=1):
        player = int(_required(step, "player", f"subsequent_steps[{offset - 1}]"))
        if player == receiver:
            mechanical_obs = _vector(
                _required(step, "obs_571", "receiver step"),
                OBS_DIM,
                "receiver step obs_571",
                dtype=np.float32,
            )
            legal_mask = _vector(
                _required(step, "legal_actions", "receiver step"),
                NUM_BIDS,
                "receiver legal_actions",
                dtype=np.bool_,
            )
            all_hands = np.asarray(
                _required(step, "all_hands", "receiver step"), dtype=np.float32
            )
            if all_hands.shape != (NUM_PLAYERS, 52):
                raise ValueError(
                    "receiver all_hands must have shape (4, 52), "
                    f"got {all_hands.shape}"
                )
            if not np.array_equal(all_hands, target_hands):
                raise ValueError("target and receiver steps do not contain the same CTDE deal")
            receiver_dd_table = np.asarray(
                _required(step, "dd_table", "receiver step"), dtype=np.float32
            )
            if receiver_dd_table.shape != (5, 4):
                raise ValueError(
                    "receiver dd_table must have shape (5, 4), "
                    f"got {receiver_dd_table.shape}"
                )
            if not np.array_equal(receiver_dd_table, target_dd_table):
                raise ValueError(
                    "target and receiver steps do not contain the same dd_table"
                )
            receiver_reference_score = float(
                _required(step, "reference_score_ns", "receiver step")
            )
            if receiver_reference_score != target_reference_score:
                raise ValueError(
                    "target and receiver steps do not contain the same reference_score_ns"
                )
            dd_table_ctde = normalize_dd_table_ctde(
                torch.from_numpy(receiver_dd_table)
            ).numpy()
            reference_score_ctde = normalize_reference_score_ctde(
                torch.tensor([receiver_reference_score], dtype=torch.float32)
            ).numpy()
            action_features_ctde = None
            if all(
                key in step for key in (
                    "public_history_before", "dealer", "vulnerability"
                )
            ):
                action_features_ctde = build_structured_action_features(
                    step["public_history_before"],
                    int(step["dealer"]),
                    player,
                    step["vulnerability"],
                    receiver_dd_table,
                    legal_mask.astype(np.float32),
                )
            return ReceiverOpportunityExtraction(
                reward_eligible=True,
                target_bidder=bidder,
                target_action=action,
                receiver_seat=receiver,
                receiver_role=role,
                receiver_step_offset=offset,
                receiver_mechanical_full_obs=mechanical_obs,
                target_evidence_query_before=deaf,
                target_evidence_query_after=heard,
                legal_action_mask=legal_mask,
                acting_seat=player,
                all_hands_ctde=all_hands.copy(),
                dd_table_ctde=dd_table_ctde,
                reference_score_ctde=reference_score_ctde,
                action_features_ctde=action_features_ctde,
            )
        terminal_seen = terminal_seen or bool(step.get("done", False))
        if terminal_seen:
            break

    reason = (
        "auction_ended_before_receiver_decision"
        if terminal_seen
        else "episode_truncated_before_receiver_decision"
    )
    return ReceiverOpportunityExtraction(
        reward_eligible=False,
        target_bidder=bidder,
        target_action=action,
        receiver_seat=receiver,
        receiver_role=role,
        no_reward_reason=reason,
    )
