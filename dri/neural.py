"""Safe batched neural evaluation of heard/deaf decision regret reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from networks.policy_net import BELIEF_FEAT_DIM, MLPPolicyNetwork
from networks.task_q import TaskQNetwork, StructuredTaskQNetwork


@dataclass(frozen=True)
class NeuralDecisionRegretResult:
    q_task: torch.Tensor
    heard_action_distribution: torch.Tensor
    deaf_action_distribution: torch.Tensor
    heard_regret: torch.Tensor
    deaf_regret: torch.Tensor
    decision_regret_reduction: torch.Tensor


def _require_frozen_task_q(task_q: TaskQNetwork) -> None:
    if not getattr(task_q, "_task_q_frozen", False):
        raise RuntimeError("Task-Q must be lifecycle-frozen before DRI inference")
    if task_q.training or any(parameter.requires_grad for parameter in task_q.parameters()):
        raise RuntimeError("frozen Task-Q must be in eval mode with gradients disabled")


@torch.no_grad()
def neural_decision_regret_reduction(
    actor: MLPPolicyNetwork,
    task_q: TaskQNetwork,
    observations: torch.Tensor,
    all_hands_ctde: Optional[torch.Tensor],
    legal_action_mask: torch.Tensor,
    heard_belief_features: torch.Tensor,
    deaf_belief_features: torch.Tensor,
    population_ids: Optional[torch.Tensor] = None,
    dd_table_ctde: Optional[torch.Tensor] = None,
    reference_score_ctde: Optional[torch.Tensor] = None,
    action_features_ctde: Optional[torch.Tensor] = None,
) -> NeuralDecisionRegretResult:
    """Evaluate both belief arms on one identical mechanical decision state.

    A single observation, full-deal tensor, and legal mask are deliberately
    shared by both actor calls and the Task-Q call.  Only the 96-dimensional
    belief activation differs between heard and deaf.
    """

    _require_frozen_task_q(task_q)
    if not actor.belief_conditioned:
        raise ValueError("heard/deaf override requires a belief-conditioned actor")
    expected_belief_shape = observations.shape[:-1] + (BELIEF_FEAT_DIM,)
    if heard_belief_features.shape != expected_belief_shape:
        raise ValueError("heard_belief_features has the wrong shape")
    if deaf_belief_features.shape != expected_belief_shape:
        raise ValueError("deaf_belief_features has the wrong shape")
    expected_mask_shape = observations.shape[:-1] + (actor.num_actions,)
    if legal_action_mask.shape != expected_mask_shape:
        raise ValueError("legal_action_mask has the wrong shape")
    if not torch.all((legal_action_mask == 0) | (legal_action_mask == 1)):
        raise ValueError("legal_action_mask must contain only 0/1 values")
    if not torch.all(legal_action_mask.to(torch.bool).any(dim=-1)):
        raise ValueError("every decision must have at least one legal action")
    # One normalized tensor is the sole mechanical legality source for both
    # intervention arms and Task-Q (and supports callers supplying bool masks).
    mechanical_legal_mask = legal_action_mask.to(
        device=observations.device, dtype=observations.dtype
    )

    heard_logits = actor.forward_with_belief_features(
        observations, mechanical_legal_mask, heard_belief_features
    )
    deaf_logits = actor.forward_with_belief_features(
        observations, mechanical_legal_mask, deaf_belief_features
    )
    heard_policy = F.softmax(heard_logits, dim=-1)
    deaf_policy = F.softmax(deaf_logits, dim=-1)
    task_q_kwargs = {}
    if isinstance(task_q, StructuredTaskQNetwork):
        if action_features_ctde is None:
            raise ValueError("structured Task-Q requires action_features_ctde")
        task_q_kwargs["action_features_ctde"] = action_features_ctde
    q_values = task_q(
        observations,
        all_hands_ctde,
        mechanical_legal_mask,
        population_ids,
        dd_table_ctde=dd_table_ctde,
        reference_score_ctde=reference_score_ctde,
        **task_q_kwargs,
    )
    legal = mechanical_legal_mask.to(dtype=torch.bool)
    legal_q = q_values.masked_fill(~legal, float("-inf"))
    best_q = legal_q.max(dim=-1).values
    expectation_q = q_values.masked_fill(~legal, 0.0)
    heard_regret = best_q - (heard_policy * expectation_q).sum(dim=-1)
    deaf_regret = best_q - (deaf_policy * expectation_q).sum(dim=-1)
    dri = deaf_regret - heard_regret
    return NeuralDecisionRegretResult(
        q_task=q_values,
        heard_action_distribution=heard_policy,
        deaf_action_distribution=deaf_policy,
        heard_regret=heard_regret,
        deaf_regret=deaf_regret,
        decision_regret_reduction=dri,
    )
