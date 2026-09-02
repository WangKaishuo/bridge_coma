# Checkpoint-aligned frozen-Judge diagnostic protocol

Frozen: 2026-08-28, before any formal output from this diagnostic was generated.

## Purpose

This read-only experiment tests whether the proxy-mechanism findings align with the same policy checkpoints used for the thesis task-return claims. It neither trains nor updates a policy or Judge. It does not select checkpoints and cannot turn the three training seeds into a population-level significance claim.

## Frozen objects

- Environment data: `data/pgx_eval_500k`
- Frozen Judge: `results/sl_base_bca.pt`
- Training seeds: 42, 43, 44
- Endpoints: rounds 100 and 120
- Reference treatment: TASK (historical arm `a`)
- Comparisons: TEAM (`b`), DUAL (`legacy_c`), and DAPS (`strict_c`)
- Seed-42 round-100 actors: `results/seed42_actor_timeline_v1/actor_only/seed42/{arm}/seed42_{arm}_round100_actor_only.pt`
- Seed-42 round-120 actors: `results/pure_models_archive/seed42/round120/{arm}_seed42_round120_actor_only.pt`
- Seed-43/44 actors: `results/seed43_44_round120_actor_timeline_v1/actor_only/seed{seed}/{arm}/seed{seed}_{arm}_round{round}_actor_only.pt`
- Every run uses CPU deployment mode. No parameter is updated.

## Fixed sampling plan

- Exactly 1,000 duplicate paired deals per treatment-versus-TASK comparison.
- Evaluation seed `20260714` for all 18 comparisons.
- The same deterministic evaluation support is reused across treatments, seeds, and endpoints.
- At most 20 deals may be used for implementation smoke tests. Smoke-test output is excluded from analysis.

## Estimands

Primary, computed as paired comparison-minus-TASK differences in per-deal totals:

1. partner frozen-Judge loss reduction;
2. opponent frozen-Judge loss reduction;
3. secrecy difference, partner minus opponent loss reduction.

The fixed-Judge match score is only a sanity check. Competitive/non-competitive/mixed strata are exploratory.

Judge-quality diagnostics, reported descriptively by policy and receiver:

- mean frozen-Judge cross-entropy before and after each observed call;
- mean Brier score before and after each call;
- 10-bin equal-width expected calibration error (ECE), pooling the Judge's 16 Bernoulli honour outputs and 32 one-hot suit-length outputs. Because these outcomes are structurally dependent, ECE is a descriptive calibration index, not an inferential statistic;
- reliability-bin counts, mean confidence, and empirical frequency.

Cross-entropy uses the Judge's training-compatible equal weighting of mean honour BCE and mean suit-length categorical CE. Brier score analogously averages honour Bernoulli squared error and suit-length multiclass squared error.

## Reporting rules

- Per-comparison 95% intervals are paired normal-approximation intervals over 1,000 deals, matching the earlier diagnostic. They describe fixed-checkpoint, fixed-support contrasts and are not independent-seed inference.
- The three seed point estimates at each endpoint are summarized by their mean, full range, and direction count. A one-sided exact sign test is reported only when all three directions agree; its smallest possible p-value is 1/8. No random-effects interval is used.
- A mechanism direction is considered replicated only if it is directionally consistent across seeds and not contradicted by the corresponding endpoint results.
- Null, sign-reversed, or checkpoint-dependent results are retained.
- Judge calibration differences are used to assess measurement comparability, not to post-select or discard a treatment result.

## Planned outputs

- 18 JSON files named `seed{seed}_round{round}_task_vs_{treatment}_1000.json`
- `aligned_diagnostic_summary.json`
- `aligned_diagnostic_summary.csv`
- `judge_reliability.csv`
- `REPORT.md`
