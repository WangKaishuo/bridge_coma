"""Task-only action-Q infrastructure for the DRI Stage 2 gate.

The targets accepted by this module are terminal duplicate DDS-IMP outcomes.
Information bonuses, shaped rewards, returns, and PPO advantages are not valid
supervision for this network.  Keeping this evaluator independent from MAPPO
also prevents accidental reuse of the policy critic's mixed reward target.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from env import NUM_BIDS
from networks.policy_net import ACTION_MAPPING_VERSION, OBS_DIM


TASK_Q_CHECKPOINT_VERSION = "task_q_terminal_duplicate_dds_imp_v2"
LEGACY_TASK_Q_CHECKPOINT_VERSION = "task_q_terminal_duplicate_dds_imp_v1"
TASK_Q_LABEL_SOURCE = "terminal_duplicate_dds_imp"
TASK_Q_LABEL_OBJECTIVE = "expected_terminal_duplicate_dds_imp_under_frozen_snapshot"
DD_TABLE_CTDE_NORMALIZATION = "double_dummy_tricks_div_13_v1"
DD_TABLE_CTDE_WIDTH = 5 * 4
REFERENCE_SCORE_CTDE_NORMALIZATION = "duplicate_reference_score_ns_div_7600_v1"
REFERENCE_SCORE_ABS_MAX = 7600.0
MASKED_Q_VALUE = -1.0e9
STRUCTURED_ACTION_FEATURE_DIM = 20
ALL_HANDS_ENCODER_RAW_FLAT_MLP = "raw_flat_mlp_v1"
ALL_HANDS_ENCODER_SHARED_BRIDGE = "shared_bridge_hand_features_v1"
ALL_HANDS_ENCODER_KINDS = (
    ALL_HANDS_ENCODER_RAW_FLAT_MLP,
    ALL_HANDS_ENCODER_SHARED_BRIDGE,
)
OBSERVATION_ENCODER_RAW = "raw_openspiel_observation_v1"
OBSERVATION_ENCODER_STRUCTURED_AUCTION = "structured_relative_auction_v1"
OBSERVATION_ENCODER_KINDS = (
    OBSERVATION_ENCODER_RAW,
    OBSERVATION_ENCODER_STRUCTURED_AUCTION,
)


@dataclass(frozen=True)
class TaskQConfig:
    """Architecture and continuation-population conditioning contract."""

    obs_dim: int = OBS_DIM
    num_actions: int = NUM_BIDS
    hidden_dim: int = 512
    num_hidden_layers: int = 3
    num_populations: int = 0
    population_embedding_dim: int = 16
    ctde_full_deal_required: bool = True
    all_hands_embedding_dim: int = 256
    all_hands_encoder_kind: str = ALL_HANDS_ENCODER_RAW_FLAT_MLP
    observation_encoder_kind: str = OBSERVATION_ENCODER_RAW
    observation_embedding_dim: int = 128
    analytic_stop_baseline_required: bool = False
    dd_table_ctde_required: bool = True
    dd_table_normalization: str = DD_TABLE_CTDE_NORMALIZATION
    reference_score_ctde_required: bool = True
    reference_score_normalization: str = REFERENCE_SCORE_CTDE_NORMALIZATION

    def __post_init__(self) -> None:
        if self.obs_dim <= 0 or self.num_actions <= 0:
            raise ValueError("obs_dim and num_actions must be positive")
        if self.hidden_dim <= 0 or self.num_hidden_layers <= 0:
            raise ValueError("hidden_dim and num_hidden_layers must be positive")
        if self.num_populations < 0:
            raise ValueError("num_populations cannot be negative")
        if self.num_populations and self.population_embedding_dim <= 0:
            raise ValueError("population_embedding_dim must be positive")
        if self.ctde_full_deal_required and self.all_hands_embedding_dim <= 0:
            raise ValueError("all_hands_embedding_dim must be positive")
        if self.all_hands_encoder_kind not in ALL_HANDS_ENCODER_KINDS:
            raise ValueError("unsupported all-hands encoder kind")
        if self.observation_encoder_kind not in OBSERVATION_ENCODER_KINDS:
            raise ValueError("unsupported observation encoder kind")
        if self.observation_embedding_dim <= 0:
            raise ValueError("observation_embedding_dim must be positive")
        if (
            self.observation_encoder_kind == OBSERVATION_ENCODER_STRUCTURED_AUCTION
            and self.obs_dim != OBS_DIM
        ):
            raise ValueError("structured auction encoder requires the OpenSpiel observation")
        if (
            self.all_hands_encoder_kind == ALL_HANDS_ENCODER_SHARED_BRIDGE
            and self.all_hands_embedding_dim % 4 != 0
        ):
            raise ValueError("shared bridge hand embedding must divide across four seats")
        if self.dd_table_normalization != DD_TABLE_CTDE_NORMALIZATION:
            raise ValueError(
                f"dd_table_normalization must be {DD_TABLE_CTDE_NORMALIZATION!r}"
            )
        if self.reference_score_normalization != REFERENCE_SCORE_CTDE_NORMALIZATION:
            raise ValueError(
                "reference_score_normalization must be "
                f"{REFERENCE_SCORE_CTDE_NORMALIZATION!r}"
            )


def normalize_dd_table_ctde(dd_table: torch.Tensor) -> torch.Tensor:
    """Flatten a raw 5x4 DDS trick table to the frozen [0, 1] contract."""
    if dd_table.shape[-2:] == (5, 4):
        flattened = dd_table.reshape(*dd_table.shape[:-2], DD_TABLE_CTDE_WIDTH)
    elif dd_table.shape[-1:] == (DD_TABLE_CTDE_WIDTH,):
        flattened = dd_table
    else:
        raise ValueError("dd_table must have shape (..., 5, 4) or (..., 20)")
    flattened = flattened.to(dtype=torch.float32)
    if not torch.isfinite(flattened).all():
        raise ValueError("dd_table must contain only finite trick counts")
    if torch.any(flattened < 0) or torch.any(flattened > 13):
        raise ValueError("dd_table trick counts must lie in [0, 13]")
    return flattened / 13.0


def normalize_reference_score_ctde(reference_score_ns: torch.Tensor) -> torch.Tensor:
    """Normalize the other-table NS score needed by duplicate IMP labels."""
    score = reference_score_ns.to(dtype=torch.float32)
    if score.ndim == 0:
        score = score.unsqueeze(0)
    if score.shape[-1:] != (1,):
        raise ValueError("reference_score_ns must have shape (..., 1)")
    if not torch.isfinite(score).all():
        raise ValueError("reference_score_ns must contain only finite values")
    if torch.any(score.abs() > REFERENCE_SCORE_ABS_MAX):
        raise ValueError("reference_score_ns exceeds the bridge score bound 7600")
    return score / REFERENCE_SCORE_ABS_MAX


class SharedBridgeAllHandsEncoder(nn.Module):
    """Shared acting-relative hand encoder over AKQJ and suit-length features."""

    def __init__(self, output_dim: int):
        super().__init__()
        seat_dim = output_dim // 4
        self.register_buffer("honor_indices", torch.tensor([
            suit * 13 + rank
            for suit in range(4) for rank in (12, 11, 10, 9)
        ], dtype=torch.int64))
        self.hand_encoder = nn.Sequential(
            nn.Linear(48, seat_dim), nn.ReLU(),
            nn.Linear(seat_dim, seat_dim), nn.ReLU(),
        )

    def forward(self, hands: torch.Tensor) -> torch.Tensor:
        if hands.shape[-2:] != (4, 52):
            raise ValueError("shared bridge encoder requires (...,4,52) hands")
        honors = hands.index_select(-1, self.honor_indices)
        lengths = hands.reshape(*hands.shape[:-1], 4, 13).sum(dim=-1)
        length_features = F.one_hot(
            lengths.round().long().clamp(0, 7), num_classes=8
        ).to(dtype=hands.dtype).reshape(*hands.shape[:-2], 4, 32)
        bridge_features = torch.cat((honors, length_features), dim=-1)
        encoded = self.hand_encoder(bridge_features)
        return encoded.reshape(*hands.shape[:-2], -1)


class StructuredAuctionObservationEncoder(nn.Module):
    """Encode relative auction events without the duplicate raw own-hand channel."""

    def __init__(self, output_dim: int):
        super().__init__()
        bid_static = torch.zeros(35, 6)
        for bid in range(35):
            bid_static[bid, 0] = float(bid // 5 + 1) / 7.0
            bid_static[bid, 1 + bid % 5] = 1.0
        self.register_buffer("bid_static", bid_static)
        self.event_encoder = nn.Sequential(
            nn.Linear(18, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.auction_gru = nn.GRU(64, 64, batch_first=True)
        self.output = nn.Sequential(nn.Linear(76, output_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != OBS_DIM:
            raise ValueError("structured auction encoder requires 571 features")
        leading = observations.shape[:-1]
        context = observations[..., :8]
        opening_passes = observations[..., 8:12]
        events = observations[..., 12:432].reshape(*leading, 35, 12)
        present = events.abs().sum(dim=-1) > 0
        static = self.bid_static.to(observations.dtype).expand(*leading, 35, 6)
        event_features = torch.cat((events, static), dim=-1)
        event_features = event_features * present.unsqueeze(-1).to(observations.dtype)
        encoded = self.event_encoder(event_features).reshape(-1, 35, 64)
        sequence, _ = self.auction_gru(encoded)
        present_flat = present.reshape(-1, 35)
        positions = torch.arange(35, device=observations.device).expand_as(present_flat)
        last_index = positions.masked_fill(~present_flat, -1).amax(dim=-1)
        auction = torch.zeros(
            present_flat.shape[0], 64,
            dtype=observations.dtype, device=observations.device,
        )
        has_contract = last_index >= 0
        if has_contract.any():
            rows = torch.arange(sequence.shape[0], device=observations.device)[has_contract]
            auction[has_contract] = sequence[rows, last_index[has_contract]]
        auction = auction.reshape(*leading, 64)
        return self.output(torch.cat((context, opening_passes, auction), dim=-1))


@dataclass(frozen=True)
class TaskQRunConfig:
    """Independent run identity for parallel local/server jobs."""

    device: str = "cpu"
    seed: int = 0
    shard_index: int = 0
    num_shards: int = 1

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if self.num_shards <= 0:
            raise ValueError("num_shards must be positive")
        if not 0 <= self.shard_index < self.num_shards:
            raise ValueError("shard_index must be in [0, num_shards)")


@dataclass(frozen=True)
class TaskQTrainingBinding:
    """Identity of the frozen continuation policy that defines one Task-Q."""

    policy_version: int
    snapshot_hash: str
    label_objective: str = TASK_Q_LABEL_OBJECTIVE

    def __post_init__(self) -> None:
        if self.policy_version < 0:
            raise ValueError("policy_version cannot be negative")
        if not self.snapshot_hash.strip():
            raise ValueError("snapshot_hash cannot be empty")
        if self.label_objective != TASK_Q_LABEL_OBJECTIVE:
            raise ValueError(
                f"label_objective must be {TASK_Q_LABEL_OBJECTIVE!r}"
            )


def build_task_q_model(
    config: TaskQConfig, run: TaskQRunConfig, *, structured: bool = False
) -> "TaskQNetwork":
    """Deterministically initialize one independently seeded Task-Q model."""

    # Parameters are initialized on CPU, so identical seeds are reproducible
    # regardless of whether the resulting model is assigned to CPU, GPU0, etc.
    # fork_rng avoids changing another parallel experiment's caller RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(run.seed)
        model = StructuredTaskQNetwork(config) if structured else TaskQNetwork(config)
    return model.to(torch.device(run.device))


class TaskQNetwork(nn.Module):
    """Predict Q_task for every project-order bid in one forward pass.

    Formal/default mode is CTDE: ``all_hands_ctde`` supplies the complete deal
    as ``(..., 4, 52)`` or ``(..., 208)``. ``legal_action_mask`` must be a
    boolean or 0/1 tensor with shape
    ``(..., 38)``. Illegal bids are returned as ``MASKED_Q_VALUE``. When a
    continuation population is configured, its integer id is embedded and
    concatenated to the public/private decision observation.
    """

    def __init__(self, config: TaskQConfig = TaskQConfig()):
        super().__init__()
        self.config = config
        self.population_embedding: Optional[nn.Embedding] = None
        self.all_hands_encoder: Optional[nn.Module] = None
        self.observation_encoder: Optional[nn.Module] = (
            StructuredAuctionObservationEncoder(config.observation_embedding_dim)
            if config.observation_encoder_kind == OBSERVATION_ENCODER_STRUCTURED_AUCTION
            else None
        )
        input_dim = (
            config.observation_embedding_dim
            if self.observation_encoder is not None else config.obs_dim
        )
        if config.ctde_full_deal_required:
            self.all_hands_encoder = (
                SharedBridgeAllHandsEncoder(config.all_hands_embedding_dim)
                if config.all_hands_encoder_kind == ALL_HANDS_ENCODER_SHARED_BRIDGE
                else nn.Sequential(
                    nn.Linear(4 * 52, config.all_hands_embedding_dim),
                    nn.ReLU(),
                    nn.Linear(
                        config.all_hands_embedding_dim, config.all_hands_embedding_dim
                    ),
                    nn.ReLU(),
                )
            )
            input_dim += config.all_hands_embedding_dim
        if config.dd_table_ctde_required:
            input_dim += DD_TABLE_CTDE_WIDTH
        if config.reference_score_ctde_required:
            input_dim += 1
        if config.num_populations:
            self.population_embedding = nn.Embedding(
                config.num_populations, config.population_embedding_dim
            )
            input_dim += config.population_embedding_dim

        layers: list[nn.Module] = []
        for _ in range(config.num_hidden_layers):
            layers.extend((nn.Linear(input_dim, config.hidden_dim), nn.ReLU()))
            input_dim = config.hidden_dim
        layers.append(nn.Linear(input_dim, config.num_actions))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        observations: torch.Tensor,
        all_hands_ctde: Optional[torch.Tensor],
        legal_action_mask: torch.Tensor,
        population_ids: Optional[torch.Tensor] = None,
        dd_table_ctde: Optional[torch.Tensor] = None,
        reference_score_ctde: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if observations.shape[-1] != self.config.obs_dim:
            raise ValueError(
                f"Expected observation width {self.config.obs_dim}, "
                f"got {observations.shape[-1]}"
            )
        expected_mask_shape = observations.shape[:-1] + (self.config.num_actions,)
        if legal_action_mask.shape != expected_mask_shape:
            raise ValueError(
                f"Expected legal mask shape {expected_mask_shape}, "
                f"got {tuple(legal_action_mask.shape)}"
            )
        if not torch.all((legal_action_mask == 0) | (legal_action_mask == 1)):
            raise ValueError("legal_action_mask must contain only 0/1 values")
        legal = legal_action_mask.to(dtype=torch.bool)
        if not torch.all(legal.any(dim=-1)):
            raise ValueError("Every decision must have at least one legal action")

        features = (
            self.observation_encoder(observations)
            if self.observation_encoder is not None else observations
        )
        if self.all_hands_encoder is None:
            if all_hands_ctde is not None:
                raise ValueError(
                    "all_hands_ctde supplied to an explicitly CTDE-disabled Task-Q"
                )
        else:
            if all_hands_ctde is None:
                raise ValueError("all_hands_ctde is required in formal CTDE mode")
            leading = observations.shape[:-1]
            valid_shapes = (leading + (4, 52), leading + (4 * 52,))
            if all_hands_ctde.shape not in valid_shapes:
                raise ValueError(
                    "all_hands_ctde must have shape (..., 4, 52) or (..., 208) "
                    "matching the observation batch"
                )
            encoder_input = (
                all_hands_ctde.reshape(*leading, 4, 52)
                if self.config.all_hands_encoder_kind == ALL_HANDS_ENCODER_SHARED_BRIDGE
                else all_hands_ctde.reshape(*leading, 4 * 52)
            )
            features = torch.cat(
                (features, self.all_hands_encoder(encoder_input)), dim=-1
            )
        leading = observations.shape[:-1]
        if self.config.dd_table_ctde_required:
            if dd_table_ctde is None:
                raise ValueError("dd_table_ctde is required by this Task-Q")
            if dd_table_ctde.shape != leading + (DD_TABLE_CTDE_WIDTH,):
                raise ValueError(
                    "dd_table_ctde must have normalized shape (..., 20) "
                    "matching the observation batch"
                )
            if not torch.isfinite(dd_table_ctde).all():
                raise ValueError("dd_table_ctde must contain only finite values")
            if torch.any(dd_table_ctde < 0) or torch.any(dd_table_ctde > 1):
                raise ValueError(
                    "dd_table_ctde must use double_dummy_tricks_div_13_v1"
                )
            features = torch.cat(
                (features, dd_table_ctde.to(dtype=features.dtype)), dim=-1
            )
        elif dd_table_ctde is not None:
            raise ValueError(
                "dd_table_ctde supplied to an explicit legacy compatibility Task-Q"
            )
        if self.config.reference_score_ctde_required:
            if reference_score_ctde is None:
                raise ValueError("reference_score_ctde is required by this Task-Q")
            if reference_score_ctde.shape != leading + (1,):
                raise ValueError(
                    "reference_score_ctde must have normalized shape (..., 1)"
                )
            if not torch.isfinite(reference_score_ctde).all():
                raise ValueError("reference_score_ctde must contain finite values")
            if torch.any(reference_score_ctde < -1) or torch.any(
                reference_score_ctde > 1
            ):
                raise ValueError(
                    "reference_score_ctde must use "
                    "duplicate_reference_score_ns_div_7600_v1"
                )
            features = torch.cat(
                (features, reference_score_ctde.to(dtype=features.dtype)), dim=-1
            )
        elif reference_score_ctde is not None:
            raise ValueError(
                "reference_score_ctde supplied to an explicit legacy compatibility Task-Q"
            )
        if self.population_embedding is None:
            if population_ids is not None:
                raise ValueError("population_ids supplied to an unconditioned Task-Q")
        else:
            if population_ids is None:
                raise ValueError("population_ids are required by this Task-Q")
            if population_ids.shape != observations.shape[:-1]:
                raise ValueError("population_ids must match observation batch shape")
            if torch.any(population_ids < 0) or torch.any(
                population_ids >= self.config.num_populations
            ):
                raise ValueError("population_ids contains an out-of-range id")
            population_features = self.population_embedding(population_ids.long())
            features = torch.cat((features, population_features), dim=-1)

        raw_q = self.net(features)
        return raw_q.masked_fill(~legal, MASKED_Q_VALUE)


class StructuredTaskQNetwork(TaskQNetwork):
    """Shared scalar action evaluator ``Q(s,a)=V(s)+A(s,a)``.

    All actions use the same advantage network.  Advantages are centered over
    the declared legal action set, preventing an independent-head/state-mean
    shortcut while retaining an identifiable value/advantage decomposition.
    """

    def __init__(self, config: TaskQConfig = TaskQConfig()):
        nn.Module.__init__(self)
        self.config = config
        self.population_embedding: Optional[nn.Embedding] = None
        self.all_hands_encoder: Optional[nn.Module] = None
        self.observation_encoder: Optional[nn.Module] = (
            StructuredAuctionObservationEncoder(config.observation_embedding_dim)
            if config.observation_encoder_kind == OBSERVATION_ENCODER_STRUCTURED_AUCTION
            else None
        )
        input_dim = (
            config.observation_embedding_dim
            if self.observation_encoder is not None else config.obs_dim
        )
        if config.ctde_full_deal_required:
            self.all_hands_encoder = (
                SharedBridgeAllHandsEncoder(config.all_hands_embedding_dim)
                if config.all_hands_encoder_kind == ALL_HANDS_ENCODER_SHARED_BRIDGE
                else nn.Sequential(
                    nn.Linear(4 * 52, config.all_hands_embedding_dim), nn.ReLU(),
                    nn.Linear(config.all_hands_embedding_dim, config.all_hands_embedding_dim),
                    nn.ReLU(),
                )
            )
            input_dim += config.all_hands_embedding_dim
        if config.dd_table_ctde_required:
            input_dim += DD_TABLE_CTDE_WIDTH
        if config.reference_score_ctde_required:
            input_dim += 1
        if config.num_populations:
            self.population_embedding = nn.Embedding(
                config.num_populations, config.population_embedding_dim
            )
            input_dim += config.population_embedding_dim

        state_layers: list[nn.Module] = []
        for _ in range(config.num_hidden_layers):
            state_layers.extend((nn.Linear(input_dim, config.hidden_dim), nn.ReLU()))
            input_dim = config.hidden_dim
        self.state_encoder = nn.Sequential(*state_layers)
        self.value_head = nn.Linear(config.hidden_dim, 1)
        self.advantage_net = nn.Sequential(
            nn.Linear(config.hidden_dim + STRUCTURED_ACTION_FEATURE_DIM, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, max(64, config.hidden_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(64, config.hidden_dim // 2), 1),
        )

    def forward(
        self,
        observations: torch.Tensor,
        all_hands_ctde: Optional[torch.Tensor],
        legal_action_mask: torch.Tensor,
        population_ids: Optional[torch.Tensor] = None,
        dd_table_ctde: Optional[torch.Tensor] = None,
        reference_score_ctde: Optional[torch.Tensor] = None,
        action_features_ctde: Optional[torch.Tensor] = None,
        analytic_stop_baseline_ctde: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if observations.shape[-1] != self.config.obs_dim:
            raise ValueError("unexpected observation width")
        leading = observations.shape[:-1]
        if legal_action_mask.shape != leading + (self.config.num_actions,):
            raise ValueError("legal_action_mask does not match observation batch")
        legal = legal_action_mask.to(torch.bool)
        if not torch.all((legal_action_mask == 0) | (legal_action_mask == 1)):
            raise ValueError("legal_action_mask must contain only 0/1 values")
        if not torch.all(legal.any(dim=-1)):
            raise ValueError("every decision must have at least one legal action")
        expected_action_features = leading + (
            self.config.num_actions, STRUCTURED_ACTION_FEATURE_DIM
        )
        if action_features_ctde is None or action_features_ctde.shape != expected_action_features:
            raise ValueError(
                f"action_features_ctde must have shape {expected_action_features}"
            )
        if not torch.isfinite(action_features_ctde).all():
            raise ValueError("action_features_ctde must contain finite values")

        features = (
            self.observation_encoder(observations)
            if self.observation_encoder is not None else observations
        )
        if self.all_hands_encoder is not None:
            if all_hands_ctde is None or all_hands_ctde.shape not in (
                leading + (4, 52), leading + (208,)
            ):
                raise ValueError("all_hands_ctde is required with acting-relative order")
            encoder_input = (
                all_hands_ctde.reshape(*leading, 4, 52)
                if self.config.all_hands_encoder_kind == ALL_HANDS_ENCODER_SHARED_BRIDGE
                else all_hands_ctde.reshape(*leading, 208)
            )
            features = torch.cat((
                features, self.all_hands_encoder(encoder_input),
            ), dim=-1)
        elif all_hands_ctde is not None:
            raise ValueError("all_hands_ctde supplied to a CTDE-disabled model")
        if self.config.dd_table_ctde_required:
            if dd_table_ctde is None or dd_table_ctde.shape != leading + (20,):
                raise ValueError("dd_table_ctde is required with shape (...,20)")
            features = torch.cat((features, dd_table_ctde.to(features.dtype)), dim=-1)
        elif dd_table_ctde is not None:
            raise ValueError("dd_table_ctde supplied to a disabled model")
        if self.config.reference_score_ctde_required:
            if reference_score_ctde is None or reference_score_ctde.shape != leading + (1,):
                raise ValueError("reference_score_ctde is required with shape (...,1)")
            features = torch.cat((features, reference_score_ctde.to(features.dtype)), dim=-1)
        elif reference_score_ctde is not None:
            raise ValueError("reference_score_ctde supplied to a disabled model")
        if self.population_embedding is not None:
            if population_ids is None or population_ids.shape != leading:
                raise ValueError("population_ids must match the observation batch")
            features = torch.cat((features, self.population_embedding(population_ids.long())), dim=-1)
        elif population_ids is not None:
            raise ValueError("population_ids supplied to an unconditioned model")

        state = self.state_encoder(features)
        expanded_state = state.unsqueeze(-2).expand(
            *leading, self.config.num_actions, state.shape[-1]
        )
        raw_advantage = self.advantage_net(torch.cat((
            expanded_state, action_features_ctde.to(state.dtype)
        ), dim=-1)).squeeze(-1)
        legal_count = legal.sum(dim=-1, keepdim=True).clamp_min(1)
        centered_advantage = raw_advantage - (
            raw_advantage.masked_fill(~legal, 0).sum(dim=-1, keepdim=True)
            / legal_count
        )
        q_values = self.value_head(state) + centered_advantage
        if self.config.analytic_stop_baseline_required:
            if (
                analytic_stop_baseline_ctde is None
                or analytic_stop_baseline_ctde.shape
                != leading + (self.config.num_actions,)
            ):
                raise ValueError(
                    "analytic_stop_baseline_ctde is required with shape (...,38)"
                )
            if not torch.isfinite(analytic_stop_baseline_ctde).all():
                raise ValueError("analytic stop baseline must contain finite IMP values")
            q_values = q_values + analytic_stop_baseline_ctde.to(q_values.dtype)
        elif analytic_stop_baseline_ctde is not None:
            raise ValueError("analytic stop baseline supplied to a disabled model")
        return q_values.masked_fill(~legal, MASKED_Q_VALUE)


class TaskQDataset(Dataset):
    """One decision-action continuation rollout per task-only training row.

    Labels are always from terminal duplicate DDS scoring and use the acting
    partnership's IMP perspective. A pair of rows sharing a decision can be
    compared with :func:`paired_q_difference_loss`; no PPO fields exist here.
    """

    def __init__(
        self,
        observations: torch.Tensor,
        all_hands_ctde: torch.Tensor,
        legal_action_masks: torch.Tensor,
        actions: torch.Tensor,
        terminal_duplicate_dds_imp: torch.Tensor,
        population_ids: Optional[torch.Tensor] = None,
        *,
        dd_table_ctde: Optional[torch.Tensor] = None,
        reference_score_ctde: Optional[torch.Tensor] = None,
        action_features_ctde: Optional[torch.Tensor] = None,
        analytic_stop_baseline_ctde: Optional[torch.Tensor] = None,
        label_source: str = TASK_Q_LABEL_SOURCE,
    ):
        if label_source != TASK_Q_LABEL_SOURCE:
            raise ValueError(
                "TaskQDataset accepts terminal duplicate DDS-IMP labels only"
            )
        if observations.ndim != 2:
            raise ValueError("observations must have shape (samples, obs_dim)")
        if all_hands_ctde.shape not in (
            (observations.shape[0], 4, 52),
            (observations.shape[0], 4 * 52),
        ):
            raise ValueError(
                "all_hands_ctde must have shape (samples, 4, 52) or (samples, 208)"
            )
        if legal_action_masks.shape != (observations.shape[0], NUM_BIDS):
            raise ValueError("legal_action_masks must have shape (samples, 38)")
        if actions.shape != (observations.shape[0],):
            raise ValueError("actions must have shape (samples,)")
        if terminal_duplicate_dds_imp.shape != actions.shape:
            raise ValueError("terminal_duplicate_dds_imp must match actions")
        _validate_legal_actions(legal_action_masks, actions, "actions")
        if population_ids is not None and population_ids.shape != actions.shape:
            raise ValueError("population_ids must match actions")
        if dd_table_ctde is not None and dd_table_ctde.shape != (
            observations.shape[0], DD_TABLE_CTDE_WIDTH
        ):
            raise ValueError("dd_table_ctde must have shape (samples, 20)")
        if reference_score_ctde is not None and reference_score_ctde.shape != (
            observations.shape[0], 1
        ):
            raise ValueError("reference_score_ctde must have shape (samples, 1)")
        if action_features_ctde is not None and action_features_ctde.shape != (
            observations.shape[0], NUM_BIDS, STRUCTURED_ACTION_FEATURE_DIM
        ):
            raise ValueError("action_features_ctde must have shape (samples, 38, 20)")
        if analytic_stop_baseline_ctde is not None and analytic_stop_baseline_ctde.shape != (
            observations.shape[0], NUM_BIDS
        ):
            raise ValueError("analytic_stop_baseline_ctde must have shape (samples, 38)")
        self.observations = observations
        self.all_hands_ctde = all_hands_ctde
        self.legal_action_masks = legal_action_masks
        self.actions = actions
        self.terminal_duplicate_dds_imp = terminal_duplicate_dds_imp
        self.population_ids = population_ids
        self.dd_table_ctde = dd_table_ctde
        self.reference_score_ctde = reference_score_ctde
        self.action_features_ctde = action_features_ctde
        self.analytic_stop_baseline_ctde = analytic_stop_baseline_ctde

    def __len__(self) -> int:
        return self.observations.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = {
            "observations": self.observations[index],
            "all_hands_ctde": self.all_hands_ctde[index],
            "legal_action_mask": self.legal_action_masks[index],
            "actions": self.actions[index],
            "terminal_duplicate_dds_imp": self.terminal_duplicate_dds_imp[index],
        }
        if self.population_ids is not None:
            row["population_ids"] = self.population_ids[index]
        if self.dd_table_ctde is not None:
            row["dd_table_ctde"] = self.dd_table_ctde[index]
        if self.reference_score_ctde is not None:
            row["reference_score_ctde"] = self.reference_score_ctde[index]
        if self.action_features_ctde is not None:
            row["action_features_ctde"] = self.action_features_ctde[index]
        if self.analytic_stop_baseline_ctde is not None:
            row["analytic_stop_baseline_ctde"] = self.analytic_stop_baseline_ctde[index]
        return row

    def shard(self, shard_index: int, num_shards: int) -> "TaskQDataset":
        """Return a stable, disjoint strided shard for an independent run."""

        if num_shards <= 0 or not 0 <= shard_index < num_shards:
            raise ValueError("Invalid shard_index/num_shards")
        indices = torch.arange(shard_index, len(self), num_shards)
        population_ids = (
            None if self.population_ids is None else self.population_ids[indices]
        )
        return TaskQDataset(
            self.observations[indices],
            self.all_hands_ctde[indices],
            self.legal_action_masks[indices],
            self.actions[indices],
            self.terminal_duplicate_dds_imp[indices],
            population_ids,
            dd_table_ctde=(
                None if self.dd_table_ctde is None else self.dd_table_ctde[indices]
            ),
            reference_score_ctde=(
                None
                if self.reference_score_ctde is None
                else self.reference_score_ctde[indices]
            ),
            action_features_ctde=(
                None if self.action_features_ctde is None
                else self.action_features_ctde[indices]
            ),
            analytic_stop_baseline_ctde=(
                None if self.analytic_stop_baseline_ctde is None
                else self.analytic_stop_baseline_ctde[indices]
            ),
        )


class TaskQGroupedDataset(Dataset):
    """One item per ``(state, continuation population)`` action group."""

    def __init__(
        self,
        observations: torch.Tensor,
        all_hands_ctde: torch.Tensor,
        legal_action_masks: torch.Tensor,
        actions: Sequence[torch.Tensor],
        terminal_duplicate_dds_imp: Sequence[torch.Tensor],
        population_ids: Optional[torch.Tensor] = None,
        *,
        dd_table_ctde: Optional[torch.Tensor] = None,
        reference_score_ctde: Optional[torch.Tensor] = None,
        action_features_ctde: Optional[torch.Tensor] = None,
        analytic_stop_baseline_ctde: Optional[torch.Tensor] = None,
    ) -> None:
        group_count = observations.shape[0]
        if observations.ndim != 2:
            raise ValueError("observations must have shape (groups, obs_dim)")
        if all_hands_ctde.shape not in ((group_count, 4, 52), (group_count, 208)):
            raise ValueError("all_hands_ctde must have one full deal per group")
        if legal_action_masks.shape != (group_count, NUM_BIDS):
            raise ValueError("legal_action_masks must have shape (groups, 38)")
        if len(actions) != group_count or len(terminal_duplicate_dds_imp) != group_count:
            raise ValueError("actions and labels must have one tensor per group")
        normalized_actions: list[torch.Tensor] = []
        normalized_labels: list[torch.Tensor] = []
        for index, (group_actions, group_labels) in enumerate(
            zip(actions, terminal_duplicate_dds_imp, strict=True)
        ):
            if group_actions.ndim != 1 or group_actions.numel() == 0:
                raise ValueError("every group must contain at least one selected action")
            if group_labels.shape != group_actions.shape:
                raise ValueError("group labels must match selected actions")
            if torch.unique(group_actions).numel() != group_actions.numel():
                raise ValueError("selected actions must be unique within a group")
            _validate_legal_actions(
                legal_action_masks[index].expand(group_actions.shape[0], -1),
                group_actions,
                "actions",
            )
            normalized_actions.append(group_actions.long())
            normalized_labels.append(group_labels)
        if population_ids is not None and population_ids.shape != (group_count,):
            raise ValueError("population_ids must have shape (groups,)")
        if dd_table_ctde is not None and dd_table_ctde.shape != (
            group_count, DD_TABLE_CTDE_WIDTH
        ):
            raise ValueError("dd_table_ctde must have shape (groups, 20)")
        if reference_score_ctde is not None and reference_score_ctde.shape != (
            group_count, 1
        ):
            raise ValueError("reference_score_ctde must have shape (groups, 1)")
        if action_features_ctde is not None and action_features_ctde.shape != (
            group_count, NUM_BIDS, STRUCTURED_ACTION_FEATURE_DIM
        ):
            raise ValueError("action_features_ctde must have shape (groups, 38, 20)")
        if analytic_stop_baseline_ctde is not None and analytic_stop_baseline_ctde.shape != (
            group_count, NUM_BIDS
        ):
            raise ValueError("analytic_stop_baseline_ctde must have shape (groups, 38)")
        self.observations = observations
        self.all_hands_ctde = all_hands_ctde
        self.legal_action_masks = legal_action_masks
        self.actions = normalized_actions
        self.terminal_duplicate_dds_imp = normalized_labels
        self.population_ids = population_ids
        self.dd_table_ctde = dd_table_ctde
        self.reference_score_ctde = reference_score_ctde
        self.action_features_ctde = action_features_ctde
        self.analytic_stop_baseline_ctde = analytic_stop_baseline_ctde

    def __len__(self) -> int:
        return self.observations.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "observations": self.observations[index],
            "all_hands_ctde": self.all_hands_ctde[index],
            "legal_action_mask": self.legal_action_masks[index],
            "actions": self.actions[index],
            "terminal_duplicate_dds_imp": self.terminal_duplicate_dds_imp[index],
        }
        if self.population_ids is not None:
            item["population_ids"] = self.population_ids[index]
        if self.dd_table_ctde is not None:
            item["dd_table_ctde"] = self.dd_table_ctde[index]
        if self.reference_score_ctde is not None:
            item["reference_score_ctde"] = self.reference_score_ctde[index]
        if self.action_features_ctde is not None:
            item["action_features_ctde"] = self.action_features_ctde[index]
        if self.analytic_stop_baseline_ctde is not None:
            item["analytic_stop_baseline_ctde"] = self.analytic_stop_baseline_ctde[index]
        return item


def collate_task_q_groups(
    groups: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Pad selected actions while retaining an explicit validity mask."""

    if not groups:
        raise ValueError("cannot collate an empty group batch")
    max_actions = max(group["actions"].numel() for group in groups)
    actions = torch.zeros(len(groups), max_actions, dtype=torch.int64)
    labels = torch.zeros(len(groups), max_actions, dtype=torch.float32)
    selected = torch.zeros(len(groups), max_actions, dtype=torch.bool)
    for index, group in enumerate(groups):
        count = group["actions"].numel()
        actions[index, :count] = group["actions"].long()
        labels[index, :count] = group["terminal_duplicate_dds_imp"].float()
        selected[index, :count] = True
    batch = {
        "observations": torch.stack([group["observations"] for group in groups]),
        "all_hands_ctde": torch.stack([group["all_hands_ctde"] for group in groups]),
        "legal_action_mask": torch.stack(
            [group["legal_action_mask"] for group in groups]
        ),
        "actions": actions,
        "terminal_duplicate_dds_imp": labels,
        "selected_action_mask": selected,
    }
    has_population = ["population_ids" in group for group in groups]
    if any(has_population) != all(has_population):
        raise ValueError("population_ids must be present for every group or none")
    if all(has_population):
        batch["population_ids"] = torch.stack(
            [group["population_ids"] for group in groups]
        )
    has_dd_table = ["dd_table_ctde" in group for group in groups]
    if any(has_dd_table) != all(has_dd_table):
        raise ValueError("dd_table_ctde must be present for every group or none")
    if all(has_dd_table):
        batch["dd_table_ctde"] = torch.stack(
            [group["dd_table_ctde"] for group in groups]
        )
    has_reference = ["reference_score_ctde" in group for group in groups]
    if any(has_reference) != all(has_reference):
        raise ValueError(
            "reference_score_ctde must be present for every group or none"
        )
    if all(has_reference):
        batch["reference_score_ctde"] = torch.stack(
            [group["reference_score_ctde"] for group in groups]
        )
    has_action_features = ["action_features_ctde" in group for group in groups]
    if any(has_action_features) != all(has_action_features):
        raise ValueError("action_features_ctde must be present for every group or none")
    if all(has_action_features):
        batch["action_features_ctde"] = torch.stack(
            [group["action_features_ctde"] for group in groups]
        )
    has_baseline = ["analytic_stop_baseline_ctde" in group for group in groups]
    if any(has_baseline) != all(has_baseline):
        raise ValueError(
            "analytic_stop_baseline_ctde must be present for every group or none"
        )
    if all(has_baseline):
        batch["analytic_stop_baseline_ctde"] = torch.stack(
            [group["analytic_stop_baseline_ctde"] for group in groups]
        )
    return batch


def _batch_to_device(
    batch: Mapping[str, torch.Tensor], device: str | torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def forward_task_q_batch(
    model: nn.Module, batch: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """Dispatch a batch to legacy or structured Task-Q without silent fields."""
    kwargs = {
        "dd_table_ctde": batch.get("dd_table_ctde"),
        "reference_score_ctde": batch.get("reference_score_ctde"),
    }
    if isinstance(model, StructuredTaskQNetwork):
        kwargs["action_features_ctde"] = batch.get("action_features_ctde")
        if model.config.analytic_stop_baseline_required:
            kwargs["analytic_stop_baseline_ctde"] = batch.get(
                "analytic_stop_baseline_ctde"
            )
    return model(
        batch["observations"],
        batch["all_hands_ctde"],
        batch["legal_action_mask"],
        batch.get("population_ids"),
        **kwargs,
    )


def train_task_q_batch(
    model: TaskQNetwork,
    batch: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    device: str | torch.device,
    loss: str = "smooth_l1",
) -> float:
    """Train one task-only batch on an explicitly selected device."""

    if getattr(model, "_task_q_frozen", False):
        raise RuntimeError("Task-Q is frozen; optimizer steps are prohibited")
    batch_d = _batch_to_device(batch, device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    q_values = forward_task_q_batch(model, batch_d)
    objective = terminal_imp_regression_loss(
        q_values,
        batch_d["legal_action_mask"],
        batch_d["actions"],
        batch_d["terminal_duplicate_dds_imp"],
        loss=loss,
    )
    objective.backward()
    optimizer.step()
    return objective.detach().item()


class TaskQTrainer:
    """Near-on-policy lifecycle for an online Q and its target network.

    The optimizer is required to contain exactly the online Task-Q parameters.
    This rejects target-network, PPO, critic, or other foreign parameters rather
    than relying on callers to keep optimizer parameter groups separate.
    """

    def __init__(
        self,
        model: TaskQNetwork,
        optimizer: torch.optim.Optimizer,
        *,
        policy_version: int,
        snapshot_hash: str,
        label_objective: str = TASK_Q_LABEL_OBJECTIVE,
        target_model: Optional[TaskQNetwork] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.binding = TaskQTrainingBinding(
            policy_version=policy_version,
            snapshot_hash=snapshot_hash,
            label_objective=label_objective,
        )
        self._validate_optimizer_ownership()
        self.target_model = copy.deepcopy(model) if target_model is None else target_model
        if (
            self.target_model.config != model.config
            or type(self.target_model) is not type(model)
        ):
            raise ValueError("target_model config must match online Task-Q config")
        self.target_model.load_state_dict(model.state_dict())
        self.target_model.eval()
        self.target_model.requires_grad_(False)
        self.frozen = False
        setattr(self.model, "_task_q_frozen", False)

    def _validate_optimizer_ownership(self) -> None:
        expected = {id(parameter) for parameter in self.model.parameters()}
        actual_list = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ]
        actual = {id(parameter) for parameter in actual_list}
        if len(actual) != len(actual_list):
            raise ValueError("optimizer contains duplicate parameters")
        if actual != expected:
            raise ValueError(
                "optimizer must contain exactly the online Task-Q parameters; "
                "PPO, target-network, and other foreign parameters are prohibited"
            )

    def train_batch(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        device: str | torch.device,
        loss: str = "smooth_l1",
    ) -> float:
        if self.frozen:
            raise RuntimeError("Task-Q is frozen; optimizer steps are prohibited")
        self._validate_optimizer_ownership()
        return train_task_q_batch(
            self.model, batch, self.optimizer, device=device, loss=loss
        )

    def train_grouped_batch(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        device: str | torch.device,
        pairwise_loss_weight: float = 1.0,
        state_centered_loss_weight: float = 0.0,
        point_loss_weight: float = 1.0,
    ) -> float:
        if self.frozen:
            raise RuntimeError("Task-Q is frozen; optimizer steps are prohibited")
        self._validate_optimizer_ownership()
        batch_d = _batch_to_device(batch, device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        q_values = forward_task_q_batch(self.model, batch_d)
        objective = grouped_task_q_loss(
            q_values,
            batch_d["legal_action_mask"],
            batch_d["actions"],
            batch_d["terminal_duplicate_dds_imp"],
            batch_d["selected_action_mask"],
            pairwise_loss_weight=pairwise_loss_weight,
            state_centered_loss_weight=state_centered_loss_weight,
            point_loss_weight=point_loss_weight,
        )
        objective.backward()
        self.optimizer.step()
        return objective.detach().item()

    @torch.no_grad()
    def hard_update_target(self) -> None:
        """Copy the complete online state into the target network."""

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    @torch.no_grad()
    def polyak_update_target(self, tau: float) -> None:
        """Apply ``target <- (1-tau) target + tau online``."""

        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        for target, online in zip(
            self.target_model.parameters(), self.model.parameters(), strict=True
        ):
            target.lerp_(online.detach(), tau)
        for target, online in zip(
            self.target_model.buffers(), self.model.buffers(), strict=True
        ):
            if target.is_floating_point():
                target.lerp_(online.detach(), tau)
            else:
                target.copy_(online)
        self.target_model.eval()

    def freeze(self) -> None:
        """Seal the calibrated Q before it is consumed by PPO/DRI."""

        self.frozen = True
        setattr(self.model, "_task_q_frozen", True)
        self.model.eval()
        self.model.requires_grad_(False)
        self.optimizer.zero_grad(set_to_none=True)

    def checkpoint(
        self,
        *,
        population_manifest: Optional[Sequence[str]] = None,
        run_config: Optional[TaskQRunConfig] = None,
        extra_metadata: Optional[Mapping[str, object]] = None,
    ) -> dict[str, object]:
        return task_q_checkpoint(
            self.model,
            population_manifest=population_manifest,
            run_config=run_config,
            extra_metadata=extra_metadata,
            training_binding=self.binding,
            target_model=self.target_model,
            frozen=self.frozen,
        )


@torch.no_grad()
def validate_task_q_batch(
    model: TaskQNetwork,
    batch: Mapping[str, torch.Tensor],
    *,
    device: str | torch.device,
) -> dict[str, float]:
    """Validate one held-out batch on an explicitly selected device."""

    batch_d = _batch_to_device(batch, device)
    model.eval()
    q_values = forward_task_q_batch(model, batch_d)
    return task_q_validation_metrics(
        q_values,
        batch_d["legal_action_mask"],
        batch_d["actions"],
        batch_d["terminal_duplicate_dds_imp"],
    )


def _validate_legal_actions(
    legal_action_mask: torch.Tensor, actions: torch.Tensor, name: str
) -> None:
    if actions.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must contain integer action indices")
    if actions.shape != legal_action_mask.shape[:-1]:
        raise ValueError(f"{name} must match the legal-mask batch shape")
    if torch.any(actions < 0) or torch.any(actions >= legal_action_mask.shape[-1]):
        raise ValueError(f"{name} contains an out-of-range action")
    chosen_legal = legal_action_mask.to(torch.bool).gather(
        -1, actions.long().unsqueeze(-1)
    ).squeeze(-1)
    if not torch.all(chosen_legal):
        raise ValueError(f"{name} contains an illegal action")


def terminal_imp_regression_loss(
    masked_q_values: torch.Tensor,
    legal_action_mask: torch.Tensor,
    actions: torch.Tensor,
    terminal_duplicate_dds_imp: torch.Tensor,
    *,
    loss: str = "smooth_l1",
) -> torch.Tensor:
    """Regress selected legal actions on task-only terminal DDS-IMP labels."""

    _validate_legal_actions(legal_action_mask, actions, "actions")
    if terminal_duplicate_dds_imp.shape != actions.shape:
        raise ValueError("terminal_duplicate_dds_imp must match actions")
    prediction = masked_q_values.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
    target = terminal_duplicate_dds_imp.to(prediction.dtype)
    if loss == "smooth_l1":
        return F.smooth_l1_loss(prediction, target)
    if loss == "mse":
        return F.mse_loss(prediction, target)
    raise ValueError(f"Unsupported loss: {loss}")


def paired_q_difference_loss(
    masked_q_values: torch.Tensor,
    legal_action_mask: torch.Tensor,
    action_a: torch.Tensor,
    action_b: torch.Tensor,
    terminal_duplicate_dds_imp_difference: torch.Tensor,
    *,
    loss: str = "smooth_l1",
) -> torch.Tensor:
    """Fit within-decision Q(a)-Q(b) to paired terminal DDS-IMP differences."""

    _validate_legal_actions(legal_action_mask, action_a, "action_a")
    _validate_legal_actions(legal_action_mask, action_b, "action_b")
    if terminal_duplicate_dds_imp_difference.shape != action_a.shape:
        raise ValueError("terminal_duplicate_dds_imp_difference must match actions")
    q_a = masked_q_values.gather(-1, action_a.long().unsqueeze(-1)).squeeze(-1)
    q_b = masked_q_values.gather(-1, action_b.long().unsqueeze(-1)).squeeze(-1)
    prediction = q_a - q_b
    target = terminal_duplicate_dds_imp_difference.to(prediction.dtype)
    if loss == "smooth_l1":
        return F.smooth_l1_loss(prediction, target)
    if loss == "mse":
        return F.mse_loss(prediction, target)
    raise ValueError(f"Unsupported loss: {loss}")


def grouped_task_q_loss(
    masked_q_values: torch.Tensor,
    legal_action_mask: torch.Tensor,
    actions: torch.Tensor,
    terminal_duplicate_dds_imp: torch.Tensor,
    selected_action_mask: torch.Tensor,
    *,
    pairwise_loss_weight: float = 1.0,
    state_centered_loss_weight: float = 0.0,
    point_loss_weight: float = 1.0,
) -> torch.Tensor:
    """Point + all-pairs difference + optional within-state centered loss."""

    if (
        pairwise_loss_weight < 0
        or state_centered_loss_weight < 0
        or point_loss_weight < 0
    ):
        raise ValueError("grouped loss weights cannot be negative")
    if actions.ndim != 2 or terminal_duplicate_dds_imp.shape != actions.shape:
        raise ValueError("grouped actions and labels must have shape (groups, selected)")
    if selected_action_mask.shape != actions.shape:
        raise ValueError("selected_action_mask must match grouped actions")
    selected = selected_action_mask.to(torch.bool)
    if not torch.all(selected.any(dim=-1)):
        raise ValueError("every group must contain at least one selected action")
    group_indices, selected_indices = torch.where(selected)
    flat_actions = actions[group_indices, selected_indices]
    _validate_legal_actions(
        legal_action_mask[group_indices], flat_actions, "actions"
    )
    prediction = masked_q_values.gather(-1, actions.long()).masked_select(selected)
    target = terminal_duplicate_dds_imp.to(masked_q_values.dtype).masked_select(selected)
    point = F.smooth_l1_loss(prediction, target)
    pair_predictions: list[torch.Tensor] = []
    pair_targets: list[torch.Tensor] = []
    centered_predictions: list[torch.Tensor] = []
    centered_targets: list[torch.Tensor] = []
    gathered = masked_q_values.gather(-1, actions.long())
    labels = terminal_duplicate_dds_imp.to(masked_q_values.dtype)
    for group_index in range(actions.shape[0]):
        group_prediction = gathered[group_index][selected[group_index]]
        group_target = labels[group_index][selected[group_index]]
        if group_prediction.numel() > 1:
            pairs = torch.triu_indices(
                group_prediction.numel(), group_prediction.numel(), offset=1,
                device=group_prediction.device,
            )
            pair_predictions.append(
                group_prediction[pairs[0]] - group_prediction[pairs[1]]
            )
            pair_targets.append(group_target[pairs[0]] - group_target[pairs[1]])
        centered_predictions.append(group_prediction - group_prediction.mean())
        centered_targets.append(group_target - group_target.mean())
    zero = point * 0.0
    pairwise = (
        F.smooth_l1_loss(torch.cat(pair_predictions), torch.cat(pair_targets))
        if pair_predictions else zero
    )
    centered = (
        F.smooth_l1_loss(
            torch.cat(centered_predictions), torch.cat(centered_targets)
        ) if state_centered_loss_weight else zero
    )
    return (
        point_loss_weight * point
        + pairwise_loss_weight * pairwise
        + state_centered_loss_weight * centered
    )


@torch.no_grad()
def paired_q_metrics(
    masked_q_values: torch.Tensor,
    action_a: torch.Tensor,
    action_b: torch.Tensor,
    terminal_duplicate_dds_imp_difference: torch.Tensor,
) -> dict[str, float]:
    """Metrics for value calibration and legal-action ranking sensitivity."""

    q_a = masked_q_values.gather(-1, action_a.long().unsqueeze(-1)).squeeze(-1)
    q_b = masked_q_values.gather(-1, action_b.long().unsqueeze(-1)).squeeze(-1)
    prediction = q_a - q_b
    target = terminal_duplicate_dds_imp_difference.to(prediction.dtype)
    error = prediction - target
    non_ties = target != 0
    ranking_accuracy = (
        (torch.sign(prediction[non_ties]) == torch.sign(target[non_ties])).float().mean()
        if torch.any(non_ties)
        else torch.tensor(float("nan"), device=prediction.device)
    )
    return {
        "paired_difference_mae": error.abs().mean().item(),
        "paired_difference_rmse": error.square().mean().sqrt().item(),
        "paired_ranking_accuracy_non_ties": ranking_accuracy.item(),
        "paired_non_tie_count": int(non_ties.sum().item()),
    }


@torch.no_grad()
def task_q_validation_metrics(
    masked_q_values: torch.Tensor,
    legal_action_mask: torch.Tensor,
    actions: torch.Tensor,
    terminal_duplicate_dds_imp: torch.Tensor,
) -> dict[str, float]:
    """Point calibration metrics for a held-out continuation-rollout set."""

    _validate_legal_actions(legal_action_mask, actions, "actions")
    prediction = masked_q_values.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
    target = terminal_duplicate_dds_imp.to(prediction.dtype)
    if target.shape != prediction.shape:
        raise ValueError("terminal_duplicate_dds_imp must match actions")
    error = prediction - target
    return {
        "terminal_imp_mae": error.abs().mean().item(),
        "terminal_imp_rmse": error.square().mean().sqrt().item(),
        "terminal_imp_bias": error.mean().item(),
        "sample_count": int(target.numel()),
    }


def task_q_checkpoint(
    model: TaskQNetwork,
    *,
    population_manifest: Optional[Sequence[str]] = None,
    run_config: Optional[TaskQRunConfig] = None,
    extra_metadata: Optional[Mapping[str, object]] = None,
    training_binding: Optional[TaskQTrainingBinding] = None,
    target_model: Optional[TaskQNetwork] = None,
    frozen: Optional[bool] = None,
) -> dict[str, object]:
    """Build a self-describing checkpoint tied to the task-only protocol."""

    manifest = list(population_manifest or [])
    if len(manifest) != model.config.num_populations:
        raise ValueError("population_manifest must match num_populations")
    metadata: dict[str, object] = {
        "checkpoint_version": TASK_Q_CHECKPOINT_VERSION,
        "label_source": TASK_Q_LABEL_SOURCE,
        "reward_components": ["terminal_duplicate_dds_imp"],
        "excludes": ["information_bonus", "ppo_advantage", "shaped_reward"],
        "action_mapping_version": ACTION_MAPPING_VERSION,
        "config": asdict(model.config),
        "population_manifest": manifest,
        "run_config": asdict(run_config) if run_config is not None else None,
        "ctde_full_deal_required": model.config.ctde_full_deal_required,
        "dd_table_ctde_required": model.config.dd_table_ctde_required,
        "dd_table_normalization": model.config.dd_table_normalization,
        "reference_score_ctde_required": model.config.reference_score_ctde_required,
        "reference_score_normalization": model.config.reference_score_normalization,
        "evaluator_kind": (
            "shared_scalar_action_v1"
            if isinstance(model, StructuredTaskQNetwork)
            else "independent_action_heads_v1"
        ),
        "q_input_contract": (
            "Q(s,a)=V(s)+center_legal(A(h(s),phi(a,auction,payoff)))"
            if isinstance(model, StructuredTaskQNetwork)
            else
            "Q(o_public+s_private_actor, s_private_all_hands_ctde, "
            "dd_table_ctde_20_tricks_div_13, "
            "reference_score_ns_ctde_1_div_7600, a, e(Pi))"
            if model.config.ctde_full_deal_required
            else "NON_FORMAL_OPT_OUT_Q(o_actor, a, e(Pi))"
        ),
        "training_binding": (
            asdict(training_binding) if training_binding is not None else None
        ),
        "frozen": frozen,
        "has_target_network": target_model is not None,
    }
    if target_model is not None and (
        target_model.config != model.config or type(target_model) is not type(model)
    ):
        raise ValueError("target_model config must match online Task-Q config")
    if extra_metadata:
        protected = set(metadata)
        overlap = protected.intersection(extra_metadata)
        if overlap:
            raise ValueError(f"Cannot override protected metadata: {sorted(overlap)}")
        metadata.update(extra_metadata)
    checkpoint: dict[str, object] = {
        "task_q": model.state_dict(),
        "metadata": metadata,
    }
    if target_model is not None:
        checkpoint["target_task_q"] = target_model.state_dict()
    return checkpoint


def load_task_q_checkpoint(
    checkpoint: Mapping[str, object] | str | Path,
    *,
    map_location: str | torch.device = "cpu",
    allow_legacy_without_dd_table: bool = False,
) -> tuple[TaskQNetwork, dict[str, object]]:
    """Load and validate a task-only Q checkpoint."""

    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location=map_location, weights_only=False)
    metadata = dict(checkpoint["metadata"])  # type: ignore[arg-type]
    version = metadata.get("checkpoint_version")
    if version not in {TASK_Q_CHECKPOINT_VERSION, LEGACY_TASK_Q_CHECKPOINT_VERSION}:
        raise ValueError("Unsupported Task-Q checkpoint version")
    if version == LEGACY_TASK_Q_CHECKPOINT_VERSION:
        if not allow_legacy_without_dd_table:
            raise ValueError(
                "legacy Task-Q lacks dd_table CTDE payoff features; pass "
                "allow_legacy_without_dd_table=True for explicit compatibility"
            )
        legacy_config = dict(metadata["config"])  # type: ignore[arg-type]
        legacy_config["dd_table_ctde_required"] = False
        legacy_config["dd_table_normalization"] = DD_TABLE_CTDE_NORMALIZATION
        legacy_config["reference_score_ctde_required"] = False
        legacy_config["reference_score_normalization"] = (
            REFERENCE_SCORE_CTDE_NORMALIZATION
        )
        metadata["config"] = legacy_config
        metadata["dd_table_ctde_required"] = False
        metadata["dd_table_normalization"] = DD_TABLE_CTDE_NORMALIZATION
        metadata["reference_score_ctde_required"] = False
        metadata["reference_score_normalization"] = (
            REFERENCE_SCORE_CTDE_NORMALIZATION
        )
        metadata["legacy_dd_table_compatibility"] = True
    if metadata.get("label_source") != TASK_Q_LABEL_SOURCE:
        raise ValueError("Task-Q checkpoint is not terminal duplicate DDS-IMP only")
    if metadata.get("action_mapping_version") != ACTION_MAPPING_VERSION:
        raise ValueError("Task-Q action mapping does not match this codebase")
    config = TaskQConfig(**metadata["config"])  # type: ignore[arg-type]
    if metadata.get("ctde_full_deal_required") != config.ctde_full_deal_required:
        raise ValueError("Task-Q CTDE metadata/config mismatch")
    if metadata.get("dd_table_ctde_required") != config.dd_table_ctde_required:
        raise ValueError("Task-Q dd_table CTDE metadata/config mismatch")
    if metadata.get("dd_table_normalization") != config.dd_table_normalization:
        raise ValueError("Task-Q dd_table normalization metadata/config mismatch")
    if (
        metadata.get("reference_score_ctde_required")
        != config.reference_score_ctde_required
    ):
        raise ValueError("Task-Q reference score CTDE metadata/config mismatch")
    if (
        metadata.get("reference_score_normalization")
        != config.reference_score_normalization
    ):
        raise ValueError("Task-Q reference score normalization metadata/config mismatch")
    binding_data = metadata.get("training_binding")
    if binding_data is not None:
        TaskQTrainingBinding(**binding_data)  # type: ignore[arg-type]
    if metadata.get("has_target_network") and "target_task_q" not in checkpoint:
        raise ValueError("Task-Q checkpoint is missing its declared target network")
    evaluator_kind = metadata.get(
        "evaluator_kind", "independent_action_heads_v1"
    )
    if evaluator_kind == "shared_scalar_action_v1":
        model = StructuredTaskQNetwork(config)
    elif evaluator_kind == "independent_action_heads_v1":
        model = TaskQNetwork(config)
    else:
        raise ValueError("Unsupported Task-Q evaluator kind")
    model.load_state_dict(checkpoint["task_q"])  # type: ignore[arg-type]
    if metadata.get("frozen"):
        setattr(model, "_task_q_frozen", True)
        model.eval()
        model.requires_grad_(False)
    return model, metadata
