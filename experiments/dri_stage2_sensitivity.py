"""Stage 2A continuation-population sensitivity analysis.

This module deliberately starts at the empirical-Q boundary.  Rollout workers
can run locally or on a server as long as they emit the JSON schema documented
by :data:`SCHEMA_DESCRIPTION`.  No actor, critic, DDS, or Bridge environment is
needed to reproduce the sensitivity report.

Run with::

    python -m experiments.dri_stage2_sensitivity input.json output.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from utils.dds_reference import (
    ACTING_PARTNERSHIP_UTILITY,
    DDS_PAR_REFERENCE_KIND,
    actor_duplicate_imp,
)


SCHEMA_VERSION = "dri-stage2-q-samples-v1"
PROBE_SCHEMA_VERSION = "dri-stage2-q-samples-v2"
ACTOR_DDS_PAR_LABEL = (
    "actor_oriented_duplicate_dds_imp_against_dds_dealer_par_v1"
)
STRATUM_FIELDS = (
    "auction_depth",
    "competitive_status",
    "contract_level",
    "partnership",
    "next_receiver_opportunity",
)

SCHEMA_DESCRIPTION = """Top-level required fields:
  schema_version: "dri-stage2-q-samples-v1"
  study_id: stable identifier for this frozen analysis input
  sampling_metadata: {state_seed, deal_split_id, trajectory_sources,
    state_sampling_method, common_random_numbers, ...}
  populations: list of {population_id, description, seats, mixture,
    sampling_metadata}; seats maps each controlled seat to a checkpoint ID and
    mixture is a list of {member_id, weight, checkpoint_ids_by_seat}.
  states: list of state records.  A state record contains state_id, deal_id,
    vulnerability, dealer, public_history, acting_seat, legal_actions,
    behavior (population_id, checkpoint_id, action, sampling_probability),
    sampling_probability, strata, and actions.  Private CTDE labels such as
    private_hands_ctde may be retained and are passed through but not analyzed.
  state.actions: list of {action, inclusion_probability, populations}.
    populations maps population_id to {q_samples, rollout_seeds, sample_ids,
    sample_weights, sampling_metadata}.  q_samples are terminal duplicate
    DDS-IMP returns.  Optional sample_weights must align with q_samples and are
    normalized when computing q_mean; omission means equal weights.
    rollout_seeds/sample_ids should align across populations/actions whenever
    common random numbers are used.  A pre-aggregated q_mean may replace
    q_samples, but raw samples are preferred for auditability.
"""


class SchemaError(ValueError):
    """Raised when an input cannot support an auditable paired analysis."""


def _seat_index(value: Any) -> int:
    if isinstance(value, str) and value.upper() in "NESW":
        return "NESW".index(value.upper())
    seat = int(value)
    if not 0 <= seat < 4:
        raise SchemaError("acting_seat must lie in [0, 4) or be N/E/S/W")
    return seat


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise SchemaError(f"{context} is missing required field {key!r}")
    return mapping[key]


def validate_input(payload: Mapping[str, Any]) -> None:
    """Validate identifiers, metadata, legal-action coverage, and Q samples."""
    if payload.get("schema_version") not in (SCHEMA_VERSION, PROBE_SCHEMA_VERSION):
        raise SchemaError(
            f"schema_version must be {SCHEMA_VERSION!r} or "
            f"{PROBE_SCHEMA_VERSION!r}, got "
            f"{payload.get('schema_version')!r}"
        )
    _require(payload, "study_id", "input")
    sampling = _require(payload, "sampling_metadata", "input")
    for key in ("state_seed", "deal_split_id", "trajectory_sources"):
        _require(sampling, key, "sampling_metadata")

    populations = _require(payload, "populations", "input")
    if len(populations) < 2:
        raise SchemaError("at least two continuation populations are required")
    population_ids: list[str] = []
    for index, population in enumerate(populations):
        context = f"populations[{index}]"
        population_id = str(_require(population, "population_id", context))
        population_ids.append(population_id)
        seats = _require(population, "seats", context)
        mixture = _require(population, "mixture", context)
        _require(population, "sampling_metadata", context)
        if not seats and not mixture:
            raise SchemaError(f"{context} must declare seat checkpoints or a mixture")
        for seat, checkpoint_id in seats.items():
            if not str(checkpoint_id):
                raise SchemaError(f"{context}.seats[{seat!r}] has an empty checkpoint ID")
        if mixture:
            weights = [float(_require(member, "weight", context)) for member in mixture]
            for member in mixture:
                _require(member, "member_id", context)
                _require(member, "checkpoint_ids_by_seat", context)
            if any(weight < 0 for weight in weights) or not math.isclose(
                sum(weights), 1.0, rel_tol=1e-6, abs_tol=1e-6
            ):
                raise SchemaError(f"{context} mixture weights must be nonnegative and sum to 1")
    if len(set(population_ids)) != len(population_ids):
        raise SchemaError("population_id values must be unique")
    known_populations = set(population_ids)

    states = _require(payload, "states", "input")
    seen_states: set[str] = set()
    for index, state in enumerate(states):
        context = f"states[{index}]"
        for key in (
            "state_id", "deal_id", "vulnerability", "dealer", "public_history",
            "acting_seat", "legal_actions", "behavior", "sampling_probability",
            "strata", "actions",
        ):
            _require(state, key, context)
        state_id = str(state["state_id"])
        if state_id in seen_states:
            raise SchemaError(f"duplicate state_id {state_id!r}")
        seen_states.add(state_id)
        if float(state["sampling_probability"]) <= 0:
            raise SchemaError(f"{context}.sampling_probability must be positive")
        behavior = state["behavior"]
        for key in ("population_id", "checkpoint_id", "action", "sampling_probability"):
            _require(behavior, key, f"{context}.behavior")
        if behavior["population_id"] not in known_populations:
            raise SchemaError(f"{context}.behavior references an unknown population")
        for field in STRATUM_FIELDS:
            _require(state["strata"], field, f"{context}.strata")

        legal_actions = [int(action) for action in state["legal_actions"]]
        if len(set(legal_actions)) != len(legal_actions):
            raise SchemaError(f"{context}.legal_actions contains duplicates")
        action_records = {int(record["action"]): record for record in state["actions"]}
        if not set(action_records).issubset(legal_actions):
            raise SchemaError(f"{context}.actions contains an illegal action")
        if int(behavior["action"]) not in action_records:
            raise SchemaError(f"{context} must include the behavior action")
        crn_signature: tuple[tuple[Any, ...], tuple[Any, ...]] | None = None
        declared_reference: tuple[Any, Any, Any, int] | None = None
        for action, record in action_records.items():
            if float(_require(record, "inclusion_probability", context)) <= 0:
                raise SchemaError(f"{context} action {action} has invalid inclusion probability")
            estimates = _require(record, "populations", context)
            unknown = set(estimates) - known_populations
            if unknown:
                raise SchemaError(f"{context} action {action} references {sorted(unknown)}")
            for population_id, estimate in estimates.items():
                samples = estimate.get("q_samples")
                q_mean = estimate.get("q_mean")
                if samples is None and q_mean is None:
                    raise SchemaError(
                        f"{context} action {action}/{population_id} needs q_samples or q_mean"
                    )
                if samples is None and "sample_weights" in estimate:
                    raise SchemaError(
                        f"{context} action {action}/{population_id} sample_weights "
                        "require q_samples"
                    )
                metadata = estimate.get("sampling_metadata", {})
                estimator = metadata.get("estimator")
                if estimator not in (
                    None,
                    "monte_carlo_member_draw",
                    "exact_finite_mixture",
                    "unbiased_stratified_finite_mixture",
                    "unbiased_nested_stratified_finite_mixture",
                ):
                    raise SchemaError(
                        f"{context} action {action}/{population_id} has unknown estimator"
                    )
                if estimator in {
                    "exact_finite_mixture",
                    "unbiased_stratified_finite_mixture",
                    "unbiased_nested_stratified_finite_mixture",
                } and samples is None:
                    raise SchemaError(
                        f"{context} action {action}/{population_id} exact_finite_mixture "
                        "requires q_samples and sample_weights"
                    )
                if samples is not None:
                    if not samples or not all(math.isfinite(float(x)) for x in samples):
                        raise SchemaError(
                            f"{context} action {action}/{population_id} has invalid q_samples"
                        )
                    for aligned_key in (
                        "rollout_seeds", "sample_ids", "realized_member_ids",
                        "checkpoint_ids_by_seat", "sampling_probabilities",
                        "sample_weights", "terminal_outcomes",
                    ):
                        aligned = estimate.get(aligned_key)
                        if aligned is not None and len(aligned) != len(samples):
                            raise SchemaError(
                                f"{context} action {action}/{population_id} {aligned_key} "
                                "must align with q_samples"
                            )
                    probabilities = estimate.get("sampling_probabilities")
                    if probabilities is not None and any(
                        not 0 < float(probability) <= 1 for probability in probabilities
                    ):
                        raise SchemaError(
                            f"{context} action {action}/{population_id} has invalid "
                            "sampling_probabilities"
                        )
                    weights = estimate.get("sample_weights")
                    if weights is not None:
                        try:
                            normalized_weights = [float(weight) for weight in weights]
                        except (TypeError, ValueError) as error:
                            raise SchemaError(
                                f"{context} action {action}/{population_id} sample_weights "
                                "must be a numeric sequence"
                            ) from error
                        if any(
                            not math.isfinite(weight) or weight < 0
                            for weight in normalized_weights
                        ) or not sum(normalized_weights) > 0:
                            raise SchemaError(
                                f"{context} action {action}/{population_id} sample_weights "
                                "must be finite, nonnegative, and have positive total weight"
                            )
                    if estimator in {
                        "exact_finite_mixture",
                        "unbiased_stratified_finite_mixture",
                    } and weights is None:
                        raise SchemaError(
                            f"{context} action {action}/{population_id} exact_finite_mixture "
                            "requires sample_weights"
                        )
                    if metadata.get("common_random_numbers") is True:
                        seeds = estimate.get("rollout_seeds")
                        sample_ids = estimate.get("sample_ids")
                        if seeds is None or sample_ids is None:
                            raise SchemaError(
                                f"{context} action {action}/{population_id} declares CRN "
                                "without rollout_seeds and sample_ids"
                            )
                        if len(set(sample_ids)) != len(sample_ids):
                            raise SchemaError(
                                f"{context} action {action}/{population_id} has duplicate sample_ids"
                            )
                        signature = (tuple(seeds), tuple(sample_ids))
                        if crn_signature is None:
                            crn_signature = signature
                        elif signature != crn_signature:
                            raise SchemaError(
                                f"{context} CRN rollout_seeds/sample_ids are not aligned "
                                "across actions and populations"
                            )
                    if metadata.get("label_definition") is not None:
                        label_definition = metadata["label_definition"]
                        if label_definition not in (
                            "duplicate_dds_imp_against_reference_table",
                            ACTOR_DDS_PAR_LABEL,
                        ):
                            raise SchemaError(
                                f"{context} action {action}/{population_id} has a non-duplicate label"
                            )
                        reference_table = metadata.get("reference_table")
                        if not isinstance(reference_table, Mapping):
                            raise SchemaError(
                                f"{context} action {action}/{population_id} is missing reference_table"
                            )
                        for reference_key in (
                            "reference_table_id", "reference_policy_id", "reference_score_ns"
                        ):
                            _require(
                                reference_table,
                                reference_key,
                                f"{context} action {action}/{population_id}.reference_table",
                            )
                        reference_signature = (
                            reference_table["reference_table_id"],
                            reference_table["reference_policy_id"],
                            reference_table.get("reference_kind"),
                            int(reference_table["reference_score_ns"]),
                        )
                        if declared_reference is None:
                            declared_reference = reference_signature
                        elif reference_signature != declared_reference:
                            raise SchemaError(
                                f"{context} mixes reference-table results across Q samples"
                            )
                        if label_definition == ACTOR_DDS_PAR_LABEL:
                            if payload.get("schema_version") != PROBE_SCHEMA_VERSION:
                                raise SchemaError(
                                    "actor-oriented DDS-par labels require schema v2"
                                )
                            if reference_table.get("reference_kind") != DDS_PAR_REFERENCE_KIND:
                                raise SchemaError(
                                    f"{context} action {action}/{population_id} must declare "
                                    f"reference_kind={DDS_PAR_REFERENCE_KIND!r}"
                                )
                            if metadata.get("utility_perspective") != ACTING_PARTNERSHIP_UTILITY:
                                raise SchemaError(
                                    f"{context} action {action}/{population_id} must declare "
                                    "acting-partnership utility"
                                )
                            outcomes = estimate.get("terminal_outcomes")
                            if outcomes is None or len(outcomes) != len(samples):
                                raise SchemaError(
                                    f"{context} action {action}/{population_id} requires "
                                    "raw terminal_outcomes aligned with q_samples"
                                )
                            for sample_index, (sample, outcome) in enumerate(
                                zip(samples, outcomes)
                            ):
                                outcome_context = (
                                    f"{context} action {action}/{population_id} "
                                    f"terminal_outcomes[{sample_index}]"
                                )
                                for outcome_key in (
                                    "terminal_score_ns", "terminal_contract",
                                    "terminal_history", "reference_score_ns",
                                    "raw_score_difference_ns", "duplicate_imp_ns",
                                    "acting_seat", "acting_partnership_sign",
                                    "duplicate_imp_actor", "utility_perspective",
                                    "rollout_seed",
                                ):
                                    _require(outcome, outcome_key, outcome_context)
                                acting_seat = _seat_index(outcome["acting_seat"])
                                if acting_seat != _seat_index(state["acting_seat"]):
                                    raise SchemaError(
                                        f"{outcome_context} does not use the state's acting seat"
                                    )
                                expected = actor_duplicate_imp(
                                    int(outcome["terminal_score_ns"]),
                                    int(reference_table["reference_score_ns"]),
                                    acting_seat,
                                )
                                if int(outcome["reference_score_ns"]) != int(
                                    reference_table["reference_score_ns"]
                                ):
                                    raise SchemaError(
                                        f"{outcome_context} reference score mismatch"
                                    )
                                if int(outcome["raw_score_difference_ns"]) != (
                                    int(outcome["terminal_score_ns"])
                                    - int(outcome["reference_score_ns"])
                                ):
                                    raise SchemaError(
                                        f"{outcome_context} raw score difference mismatch"
                                    )
                                if (
                                    int(outcome["duplicate_imp_actor"]) != expected
                                    or not math.isclose(float(sample), expected)
                                    or int(outcome["acting_partnership_sign"])
                                    != (1 if acting_seat % 2 == 0 else -1)
                                    or outcome["utility_perspective"]
                                    != ACTING_PARTNERSHIP_UTILITY
                                ):
                                    raise SchemaError(
                                        f"{outcome_context} actor utility mismatch"
                                    )


def _q_values(state: Mapping[str, Any], population_id: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for record in state["actions"]:
        estimate = record["populations"].get(population_id)
        if estimate is None:
            continue
        if "q_samples" in estimate:
            samples = [float(x) for x in estimate["q_samples"]]
            weights = estimate.get("sample_weights")
            if weights is None:
                value = mean(samples)
            else:
                normalized_weights = [float(weight) for weight in weights]
                total_weight = sum(normalized_weights)
                value = sum(
                    sample * weight
                    for sample, weight in zip(samples, normalized_weights)
                ) / total_weight
        else:
            value = float(estimate["q_mean"])
        values[int(record["action"])] = value
    return values


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2:
        return None
    x_mean, y_mean = mean(x), mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x) * sum((b - y_mean) ** 2 for b in y)
    )
    return numerator / denominator if denominator else None


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    return _pearson(_average_ranks(x), _average_ranks(y))


def _kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float | None:
    concordant = discordant = ties_x = ties_y = 0
    for left in range(len(x)):
        for right in range(left + 1, len(x)):
            dx = (x[left] > x[right]) - (x[left] < x[right])
            dy = (y[left] > y[right]) - (y[left] < y[right])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_x)
        * (concordant + discordant + ties_y)
    )
    return (concordant - discordant) / denominator if denominator else None


def _argmax(values: Mapping[int, float]) -> int:
    """Deterministic tie-break: the numerically smallest action wins."""
    return min(values, key=lambda action: (-values[action], action))


def _top_k(values: Mapping[int, float], k: int) -> set[int]:
    return set(sorted(values, key=lambda action: (-values[action], action))[:k])


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def _state_metrics(
    state: Mapping[str, Any],
    population_a: str,
    population_b: str,
    top_k: int,
    advantage_threshold: float,
) -> dict[str, Any] | None:
    q_a = _q_values(state, population_a)
    q_b = _q_values(state, population_b)
    common = sorted(set(q_a) & set(q_b))
    if not common:
        return None
    q_a = {action: q_a[action] for action in common}
    q_b = {action: q_b[action] for action in common}
    a_values = [q_a[action] for action in common]
    b_values = [q_b[action] for action in common]
    best_a, best_b = _argmax(q_a), _argmax(q_b)
    k = min(top_k, len(common))
    top_a, top_b = _top_k(q_a, k), _top_k(q_b, k)

    sign_matches = 0
    separated_pairs = 0
    difference_errors: list[float] = []
    for left_index, left in enumerate(common):
        for right in common[left_index + 1:]:
            delta_a = q_a[left] - q_a[right]
            delta_b = q_b[left] - q_b[right]
            difference_errors.append(delta_a - delta_b)
            # Symmetric thresholding avoids declaring an ambiguous value a sign error.
            if min(abs(delta_a), abs(delta_b)) >= advantage_threshold:
                separated_pairs += 1
                sign_matches += int((delta_a > 0) == (delta_b > 0))

    return {
        "state_id": state["state_id"],
        "common_action_count": len(common),
        "spearman": _spearman(a_values, b_values),
        "kendall_tau_b": _kendall_tau_b(a_values, b_values),
        "top1_agreement": float(best_a == best_b),
        "topk_overlap_fraction": len(top_a & top_b) / k,
        "a_regret_selecting_b": max(q_a.values()) - q_a.get(best_b, max(q_a.values())),
        "b_regret_selecting_a": max(q_b.values()) - q_b.get(best_a, max(q_b.values())),
        "separated_pair_count": separated_pairs,
        "sign_match_count": sign_matches,
        "paired_difference_errors": difference_errors,
    }


def _finite_mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return mean(finite) if finite else None


def _aggregate(rows: Sequence[Mapping[str, Any]], regret_quantile: float) -> dict[str, Any]:
    errors = [error for row in rows for error in row["paired_difference_errors"]]
    regrets_a = [float(row["a_regret_selecting_b"]) for row in rows]
    regrets_b = [float(row["b_regret_selecting_a"]) for row in rows]
    separated = sum(int(row["separated_pair_count"]) for row in rows)
    matches = sum(int(row["sign_match_count"]) for row in rows)
    return {
        "state_count": len(rows),
        "mean_common_action_count": _finite_mean(row["common_action_count"] for row in rows),
        "mean_spearman": _finite_mean(row["spearman"] for row in rows),
        "mean_kendall_tau_b": _finite_mean(row["kendall_tau_b"] for row in rows),
        "top1_agreement_rate": _finite_mean(row["top1_agreement"] for row in rows),
        "mean_topk_overlap_fraction": _finite_mean(
            row["topk_overlap_fraction"] for row in rows
        ),
        "separated_pair_count": separated,
        "separated_advantage_sign_agreement_rate": matches / separated if separated else None,
        "cross_policy_selection_regret": {
            "a_evaluates_b_selection": {
                "mean": _finite_mean(regrets_a),
                "high_quantile": _quantile(regrets_a, regret_quantile),
            },
            "b_evaluates_a_selection": {
                "mean": _finite_mean(regrets_b),
                "high_quantile": _quantile(regrets_b, regret_quantile),
            },
        },
        "paired_q_difference_error": {
            "pair_count": len(errors),
            "mean_error": _finite_mean(errors),
            "mae": _finite_mean(abs(error) for error in errors),
            "rmse": math.sqrt(mean(error * error for error in errors)) if errors else None,
            "high_quantile_absolute_error": _quantile(
                [abs(error) for error in errors], regret_quantile
            ),
        },
    }


def _stratum_keys(state: Mapping[str, Any]) -> list[str]:
    strata = state["strata"]
    keys = ["overall"]
    keys.extend(f"{field}={str(strata[field]).lower()}" for field in STRATUM_FIELDS)
    full = "|".join(f"{field}={str(strata[field]).lower()}" for field in STRATUM_FIELDS)
    keys.append(f"joint:{full}")
    return keys


def analyze_sensitivity(
    payload: Mapping[str, Any],
    *,
    top_k: int = 3,
    advantage_threshold: float = 0.5,
    regret_quantile: float = 0.9,
) -> dict[str, Any]:
    """Compute all pairwise continuation-population metrics and strata."""
    validate_input(payload)
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if advantage_threshold < 0:
        raise ValueError("advantage_threshold must be nonnegative")
    if not 0 < regret_quantile < 1:
        raise ValueError("regret_quantile must be between 0 and 1")

    population_ids = [population["population_id"] for population in payload["populations"]]
    pairs: dict[str, Any] = {}
    for left_index, population_a in enumerate(population_ids):
        for population_b in population_ids[left_index + 1:]:
            by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
            skipped = 0
            for state in payload["states"]:
                row = _state_metrics(
                    state, population_a, population_b, top_k, advantage_threshold
                )
                if row is None:
                    skipped += 1
                    continue
                for key in _stratum_keys(state):
                    by_stratum[key].append(row)
            pair_id = f"{population_a}__vs__{population_b}"
            pairs[pair_id] = {
                "population_a": population_a,
                "population_b": population_b,
                "states_without_common_q": skipped,
                "by_stratum": {
                    key: _aggregate(rows, regret_quantile)
                    for key, rows in sorted(by_stratum.items())
                },
            }

    return {
        "schema_version": "dri-stage2-sensitivity-report-v1",
        "source_schema_version": payload["schema_version"],
        "study_id": payload["study_id"],
        "analysis_config": {
            "top_k": top_k,
            "topk_definition": "intersection size divided by min(top_k, common actions)",
            "advantage_threshold": advantage_threshold,
            "advantage_threshold_definition": (
                "both populations must have absolute pairwise advantage at least threshold"
            ),
            "regret_quantile": regret_quantile,
            "tie_break": "lowest numeric action ID",
        },
        "sampling_metadata": payload["sampling_metadata"],
        "population_manifest": payload["populations"],
        "population_pairs": pairs,
        "uncertainty_policy": {
            "independent_unit": "deal",
            "sample_weights_affect": "q_point_estimates_only",
            "exact_finite_mixture": (
                "member enumeration has no member-sampling error; it does not "
                "create a deal-level confidence interval"
            ),
            "state_rows_are_not_independent_replicates": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=SCHEMA_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Q-sample JSON input")
    parser.add_argument("output", type=Path, help="sensitivity JSON output")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--advantage-threshold", type=float, default=0.5)
    parser.add_argument("--regret-quantile", type=float, default=0.9)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = analyze_sensitivity(
        payload,
        top_k=args.top_k,
        advantage_threshold=args.advantage_threshold,
        regret_quantile=args.regret_quantile,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "study_id": report["study_id"],
        "population_pairs": len(report["population_pairs"]),
    }))


if __name__ == "__main__":
    main()
