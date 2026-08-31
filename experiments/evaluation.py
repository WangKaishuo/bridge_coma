"""Black-box evaluation for bridge bidding agents.

The evaluator only exposes legal game information to a policy.  It never accepts
or forwards an opponent model, belief network, critic, DDS table, or hidden hand.
This boundary is the deployment contract used by the formal experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np
import torch

from env import BID_1C, NUM_BIDS
from networks.policy_net import (
    BELIEF_OBS_DIM,
    append_belief_features,
    convert_hands_suit_to_rank,
    get_openspiel_obs,
    hands_to_openspiel_state,
    openspiel_raw_to_ours,
    ours_to_openspiel_raw,
    physical_to_openspiel_player,
    make_belief_features_prior,
)
from utils.imp import score_to_imp


class BiddingPolicy(Protocol):
    """A policy bound to one deal.

    Implementations may retain their own private hand representation, but the
    evaluator provides only public auction state through ``observation`` and
    ``history`` at action time.
    """

    def act(self, observation: dict, player: int, history: list[int]) -> int:
        ...


PolicyFactory = Callable[[np.ndarray, int, tuple[bool, bool]], BiddingPolicy]


@dataclass(frozen=True)
class EvaluationDeal:
    hands: np.ndarray
    dd_table: np.ndarray
    dealer: int
    vulnerability: tuple[bool, bool]


@dataclass(frozen=True)
class MatchResult:
    mean_imp: float
    std_imp: float
    standard_error: float
    ci_low: float
    ci_high: float
    wins: int
    losses: int
    ties: int
    imps: np.ndarray


@dataclass(frozen=True)
class StratifiedMatchResult:
    """Overall duplicate result plus auction-competition strata.

    A paired deal is ``competitive`` or ``non_competitive`` only when both
    cross-table auctions agree on that classification.  Deals whose treatment
    changes whether the opponents enter the auction are reported as ``mixed``
    rather than being assigned post-treatment to one side.
    """

    overall: MatchResult
    competitive: MatchResult
    non_competitive: MatchResult
    mixed: MatchResult


@dataclass(frozen=True)
class ExecutionAblationResult:
    """Paired comparison of two execution modes for the same trained agent."""

    mean_imp_delta: float
    std_imp_delta: float
    action_disagreement_rate: float
    auction_disagreement_rate: float
    contract_disagreement_rate: float
    score_disagreement_rate: float
    paired_imp_deltas: np.ndarray


class MAPPOPolicy:
    """Deployment wrapper for a standard 571-dimensional MAPPO policy."""

    def __init__(
        self,
        agent,
        hands: np.ndarray,
        dealer: int,
        vulnerability: tuple[bool, bool],
        deterministic: bool = True,
    ):
        self.agent = agent
        self.hands_rm = convert_hands_suit_to_rank(hands)
        self.dealer = dealer
        self.vulnerability = vulnerability
        self.deterministic = deterministic

    def act(self, observation: dict, player: int, history: list[int]) -> int:
        state = hands_to_openspiel_state(
            self.hands_rm, self.dealer, vulnerability=self.vulnerability
        )
        for action in history:
            raw_action = ours_to_openspiel_raw(action)
            if raw_action in state.legal_actions():
                state.apply_action(raw_action)

        observer = physical_to_openspiel_player(player, self.dealer)
        flat_obs = get_openspiel_obs(state, observer)
        legal_mask = np.zeros(NUM_BIDS, dtype=np.float32)
        for raw_action in state.legal_actions():
            action = openspiel_raw_to_ours(raw_action)
            if 0 <= action < NUM_BIDS:
                legal_mask[action] = 1.0

        obs_t = torch.as_tensor(flat_obs, dtype=torch.float32, device=self.agent.device)
        legal_t = torch.as_tensor(
            legal_mask, dtype=torch.float32, device=self.agent.device
        )
        actor = self.agent.get_actor(player)
        with torch.no_grad():
            action, _, _ = actor.get_action(
                obs_t.unsqueeze(0), legal_t.unsqueeze(0), self.deterministic
            )
        return int(action.item())

    def action_probability(
        self, observation: dict, player: int, history: list[int], action: int
    ) -> float:
        """Return the audited behavior probability for stochastic B_cf traces."""
        if self.deterministic:
            return 1.0
        state = hands_to_openspiel_state(
            self.hands_rm, self.dealer, vulnerability=self.vulnerability
        )
        for previous in history:
            raw_previous = ours_to_openspiel_raw(previous)
            if raw_previous in state.legal_actions():
                state.apply_action(raw_previous)
        raw_action = ours_to_openspiel_raw(int(action))
        if raw_action not in state.legal_actions():
            raise ValueError("requested behavior action is illegal")
        observer = physical_to_openspiel_player(player, self.dealer)
        flat_obs = get_openspiel_obs(state, observer)
        legal_mask = np.zeros(NUM_BIDS, dtype=np.float32)
        for raw_legal in state.legal_actions():
            legal_action = openspiel_raw_to_ours(raw_legal)
            if 0 <= legal_action < NUM_BIDS:
                legal_mask[legal_action] = 1.0
        obs_t = torch.as_tensor(
            flat_obs, dtype=torch.float32, device=self.agent.device
        ).unsqueeze(0)
        legal_t = torch.as_tensor(
            legal_mask, dtype=torch.float32, device=self.agent.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits = self.agent.get_actor(player)(obs_t, legal_t)
            probability = torch.softmax(logits, dim=-1)[0, int(action)]
        return float(probability.item())


def make_mappo_factory(agent, deterministic: bool = True) -> PolicyFactory:
    """Return a deal-bound factory for a standard deployment policy."""

    def factory(hands, dealer, vulnerability):
        return MAPPOPolicy(agent, hands, dealer, vulnerability, deterministic)

    return factory


class BeliefConditionedPolicy(MAPPOPolicy):
    """Legacy 667-dimensional policy used only for execution ablation.

    Both partner and RHO predictions come from the agent's own BeliefNet.  The
    policy never reads an opponent model.  With ``use_prior=True``, the same
    checkpoint receives uninformative prior features.
    """

    def __init__(self, *args, belief_net, use_prior: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.belief_net = belief_net
        self.use_prior = use_prior

    def act(self, observation: dict, player: int, history: list[int]) -> int:
        state = hands_to_openspiel_state(
            self.hands_rm, self.dealer, vulnerability=self.vulnerability
        )
        for action in history:
            raw_action = ours_to_openspiel_raw(action)
            if raw_action in state.legal_actions():
                state.apply_action(raw_action)

        observer = physical_to_openspiel_player(player, self.dealer)
        base_obs = get_openspiel_obs(state, observer)
        if self.use_prior:
            belief_features = make_belief_features_prior()
        else:
            obs_t = torch.as_tensor(
                base_obs, dtype=torch.float32, device=self.agent.device
            ).unsqueeze(0)
            partner = torch.tensor(
                [physical_to_openspiel_player((player + 2) % 4, self.dealer)],
                dtype=torch.long,
                device=self.agent.device,
            )
            rho = torch.tensor(
                [physical_to_openspiel_player((player - 1) % 4, self.dealer)],
                dtype=torch.long,
                device=self.agent.device,
            )
            with torch.no_grad():
                partner_probs = self.belief_net.get_probs(obs_t, partner)
                rho_probs = self.belief_net.get_probs(obs_t, rho)
            belief_features = torch.cat(
                [partner_probs, rho_probs], dim=-1
            ).squeeze(0).cpu().numpy()
        flat_obs = append_belief_features(base_obs, belief_features)
        if flat_obs.shape != (BELIEF_OBS_DIM,):
            raise RuntimeError(f"Expected {BELIEF_OBS_DIM} features, got {flat_obs.shape}")

        legal_mask = np.zeros(NUM_BIDS, dtype=np.float32)
        for raw_action in state.legal_actions():
            action = openspiel_raw_to_ours(raw_action)
            if 0 <= action < NUM_BIDS:
                legal_mask[action] = 1.0
        obs_t = torch.as_tensor(flat_obs, dtype=torch.float32, device=self.agent.device)
        legal_t = torch.as_tensor(
            legal_mask, dtype=torch.float32, device=self.agent.device
        )
        actor = self.agent.get_actor(player)
        with torch.no_grad():
            action, _, _ = actor.get_action(
                obs_t.unsqueeze(0), legal_t.unsqueeze(0), self.deterministic
            )
        return int(action.item())


def make_belief_conditioned_factory(
    agent,
    belief_net,
    use_prior: bool,
    deterministic: bool = True,
) -> PolicyFactory:
    """Create a legacy BCA policy factory for paired execution ablation."""

    def factory(hands, dealer, vulnerability):
        return BeliefConditionedPolicy(
            agent,
            hands,
            dealer,
            vulnerability,
            deterministic,
            belief_net=belief_net,
            use_prior=use_prior,
        )

    return factory


def sample_evaluation_deals(env, count: int, seed: int) -> list[EvaluationDeal]:
    """Sample a reproducible deal set, including dealer and vulnerability."""
    rng_state = np.random.get_state()
    np.random.seed(seed)
    deals = []
    try:
        vulnerabilities = (
            (False, False), (True, False), (False, True), (True, True)
        )
        for _ in range(count):
            hands, dd_table = env.generate_deal()
            deals.append(EvaluationDeal(
                hands=hands.copy(),
                dd_table=dd_table.copy(),
                dealer=int(env._sampled_dealer),
                vulnerability=vulnerabilities[np.random.randint(4)],
            ))
    finally:
        np.random.set_state(rng_state)
    return deals


def _play_table(env, deal, opener_factory, overcaller_factory):
    opener = opener_factory(deal.hands, deal.dealer, deal.vulnerability)
    overcaller = overcaller_factory(deal.hands, deal.dealer, deal.vulnerability)
    return env.play_mixed(
        deal.hands,
        deal.dd_table,
        opener_policy=opener.act,
        overcaller_policy=overcaller.act,
        vulnerability=deal.vulnerability,
        dealer=deal.dealer,
    )


def _summarize_imps(imps) -> MatchResult:
    """Build a confidence-interval summary from paired IMP observations."""
    values = np.asarray(imps, dtype=np.float64)
    n = max(len(values), 1)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    se = std / np.sqrt(n)
    mean = float(values.mean()) if len(values) else 0.0
    return MatchResult(
        mean_imp=mean,
        std_imp=std,
        standard_error=se,
        ci_low=mean - 1.96 * se if len(values) else 0.0,
        ci_high=mean + 1.96 * se if len(values) else 0.0,
        wins=int((values > 0).sum()),
        losses=int((values < 0).sum()),
        ties=int((values == 0).sum()),
        imps=values,
    )


def auction_is_competitive(history: Sequence[int], dealer: int) -> bool:
    """Whether both partnerships made at least one contract bid."""
    bidding_sides = {
        (dealer + index) % 2
        for index, action in enumerate(history)
        if action >= BID_1C
    }
    return len(bidding_sides) == 2


def evaluate_match_stratified(
    env, deals: Sequence[EvaluationDeal], agent_a, agent_b
) -> StratifiedMatchResult:
    """Evaluate paired duplicate IMP and stratify by auction competition."""
    strata = {"competitive": [], "non_competitive": [], "mixed": []}
    imps = []
    for deal in deals:
        _, score_ab, history_ab = _play_table(env, deal, agent_a, agent_b)
        _, score_ba, history_ba = _play_table(env, deal, agent_b, agent_a)
        score_difference = score_ab - score_ba
        if deal.dealer % 2 == 1:
            score_difference = -score_difference
        imp = float(score_to_imp(score_difference))
        imps.append(imp)

        ab_comp = auction_is_competitive(history_ab, deal.dealer)
        ba_comp = auction_is_competitive(history_ba, deal.dealer)
        if ab_comp and ba_comp:
            stratum = "competitive"
        elif not ab_comp and not ba_comp:
            stratum = "non_competitive"
        else:
            stratum = "mixed"
        strata[stratum].append(imp)

    return StratifiedMatchResult(
        overall=_summarize_imps(imps),
        competitive=_summarize_imps(strata["competitive"]),
        non_competitive=_summarize_imps(strata["non_competitive"]),
        mixed=_summarize_imps(strata["mixed"]),
    )


def evaluate_match(env, deals: Sequence[EvaluationDeal], agent_a, agent_b) -> MatchResult:
    """Evaluate two black-box agents using duplicate cross-table IMP scoring."""
    return evaluate_match_stratified(env, deals, agent_a, agent_b).overall


class _TracingPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.actions: list[int] = []

    def act(self, observation, player, history):
        action = self.policy.act(observation, player, history)
        self.actions.append(action)
        return action


def evaluate_execution_ablation(
    env,
    deals: Sequence[EvaluationDeal],
    reference_factory: PolicyFactory,
    ablated_factory: PolicyFactory,
    opponent_factory: PolicyFactory,
) -> ExecutionAblationResult:
    """Measure direct behavioural effects of removing an execution-time feature.

    Both modes play the same deals in the same role against the same opponent.
    The result reports paired IMP changes and action/auction/contract differences,
    which are more sensitive than aggregate IMP alone.
    """
    imp_deltas = []
    action_differences = 0
    compared_actions = 0
    auction_differences = 0
    contract_differences = 0
    score_differences = 0

    for deal in deals:
        reference = _TracingPolicy(reference_factory(
            deal.hands, deal.dealer, deal.vulnerability
        ))
        ablated = _TracingPolicy(ablated_factory(
            deal.hands, deal.dealer, deal.vulnerability
        ))
        opponent_1 = opponent_factory(deal.hands, deal.dealer, deal.vulnerability)
        opponent_2 = opponent_factory(deal.hands, deal.dealer, deal.vulnerability)

        contract_ref, score_ref, history_ref = env.play_mixed(
            deal.hands, deal.dd_table, reference.act, opponent_1.act,
            vulnerability=deal.vulnerability, dealer=deal.dealer,
        )
        contract_abl, score_abl, history_abl = env.play_mixed(
            deal.hands, deal.dd_table, ablated.act, opponent_2.act,
            vulnerability=deal.vulnerability, dealer=deal.dealer,
        )

        common = min(len(reference.actions), len(ablated.actions))
        compared_actions += max(len(reference.actions), len(ablated.actions))
        action_differences += sum(
            a != b for a, b in zip(reference.actions[:common], ablated.actions[:common])
        ) + abs(len(reference.actions) - len(ablated.actions))
        auction_differences += int(history_ref != history_abl)
        contract_differences += int(contract_ref != contract_abl)
        score_differences += int(score_ref != score_abl)

        sign = -1 if deal.dealer % 2 == 1 else 1
        imp_deltas.append(float(score_to_imp(sign * (score_ref - score_abl))))

    deltas = np.asarray(imp_deltas, dtype=np.float64)
    n = max(len(deals), 1)
    return ExecutionAblationResult(
        mean_imp_delta=float(deltas.mean()) if len(deltas) else 0.0,
        std_imp_delta=float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
        action_disagreement_rate=action_differences / max(compared_actions, 1),
        auction_disagreement_rate=auction_differences / n,
        contract_disagreement_rate=contract_differences / n,
        score_disagreement_rate=score_differences / n,
        paired_imp_deltas=deltas,
    )
