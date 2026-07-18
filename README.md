# Bridge-COMA

Bridge-COMA studies communication credit assignment in competitive contract
bridge bidding.  A public bid can improve a partner's hidden-hand inference
while also leaking information to an opponent.  The project adds a dense,
belief-based communication reward to MAPPO with fictitious self-play (FSP).

## Current handoff

- The unrestricted main environment, complete-auction training path, memmapped
  10M training pool, black-box evaluator, pilot, memory-safe single-round
  profile, and formal seed-42 A/B run are complete.
- Formal A and B both completed 60 rounds without a crash or numerical failure.
  The round-60 5000-deal match was inconclusive: B vs A was
  `+0.0132 IMP/deal`, 95% CI `[-0.1050, +0.1314]`.
- Post-run diagnostics show that the RL pipeline did learn: A vs SL was
  `+0.4398 IMP/deal` and B vs SL was `+0.4378`, both with positive 95% CIs.
- Frozen-Judge decomposition also shows that B increased partner information,
  but 87.6% of that increase was accompanied by increased opponent information.
  B therefore passed the information-manipulation check even though it did not
  improve IMP over A.
- Agent C has deliberately **not** been started.  The old "B must beat A" gate
  is no longer the right prerequisite for C; the next decision is to specify a
  scientifically meaningful leakage weight.  The configured `beta=0.05`
  subtracts only 5% of an opponent term that is the same scale as the partner
  term, so an unchanged C run would be a weak secrecy test.
- The full operational and statistical record is in **Formal unrestricted A/B
  experiment record (2026-07-15 to 2026-07-16)** immediately below.

## Formal unrestricted A/B experiment record (2026-07-15 to 2026-07-16)

This is the authoritative record of the first full-scale unrestricted main
experiment.  It supersedes the planning estimates later in this README.  All
reported IMP values in A/B matches use B's perspective unless stated otherwise.

### Question, arms, and stopping rule

The run tested whether the partner-information reward improves a task-only
MAPPO/FSP bidder when architecture, initialization, data, optimizer, and random
seed are held fixed:

| Arm | Task reward | Partner information reward | Opponent leakage penalty |
|---|---:|---:|---:|
| A | yes | no | no |
| B | yes | yes | no |

The agreed staged plan was:

1. run A and B first for seed 42;
2. pause both after round 30 and play a fixed 5000-deal B-vs-A match;
3. continue to round 60 after review if the intermediate result was promising;
4. run C only after B had demonstrated a useful improvement over A;
5. decide whether more than 60 rounds were justified from convergence and
   head-to-head results rather than extending automatically.

The round-30 evaluator labelled the result `INCONCLUSIVE`.  Training was held,
the result was reviewed by the user, and the user explicitly authorized
continuation to round 60.  The round-60 result was also `INCONCLUSIVE`; C remains
unstarted.

### Machine, datasets, and reproducibility

- Server: 16 physical / 32 logical CPU cores, 128 GiB RAM, and two 24 GiB
  NVIDIA A10 GPUs.
- Training data: `data/pgx_train_10m_memmap`, a 10M-deal memory-mapped pool.
- Evaluation data: `data/pgx_eval_500k_memmap`, a disjoint 500k-deal
  memory-mapped pool.
- Formal seed: 42.  Evaluation seed: 20260714.
- Every formal match used 5000 paired deals with duplicate cross-table role
  swapping, identical dealer and vulnerability, and black-box policy access.
- The round-30 and round-60 comparisons reused the same evaluation seed and
  deal source, making their change directly interpretable without a changed
  test set.
- There was no dataset sharding.  Memory mapping prevented the 10M pool from
  being copied into each process, and `rollout_chunk_deals=8192` bounded
  intermediate rollout construction without changing the sampled training
  volume.

### Formal hyperparameters and training volume

| Setting | Value |
|---|---:|
| rounds | 60 |
| phases per round | 2 (NS and EW) |
| steps per phase | 256 |
| deals per step | 512 |
| deals per phase | 131,072 |
| PPO batch size | 512 |
| PPO epochs | 4 |
| learning rate | 3e-6 |
| entropy coefficient | 0.01 |
| FSP pool cap | 10 |
| FSP add interval | 1 round |
| information weight | 0.05 |
| leakage beta | 0.05 |
| actor belief auxiliary coefficient | 0.1 |
| information calibration deals | 2048 |
| checkpoint interval | every round |

Each arm therefore collected
`60 * 2 * 256 * 512 = 15,728,640` deal episodes.  A and B together collected
31,457,280 deal episodes.  With the observed auction length near 10.5 calls,
this is approximately 165 million environment action steps per arm.  A/B used
the same `sl_base.pt` policy initializer; B's frozen communication Judge came
from `sl_base_bca.pt` and was not exposed at evaluation time.

### Pilot and full-round profiling

Before the 60-round launch, one complete unrestricted pilot round was run for
A, B, and C.  All three reached their final checkpoint.  Their initial task
rollout was identical (`-0.079 +/- 7.559 IMP`), and their common initial auction
health was: mean length 10.46, p95 17, 52.8% competitive, 0.4% all-pass,
3.65% doubles, and 0.12% redoubles.  This established wiring and parity; it was
not used as a performance result.

A separate concurrent A/B full-round profile used the exact formal sampling
and PPO settings:

| Component | A | B |
|---|---:|---:|
| NS environment collection | 225.3 s | 293.7 s |
| NS information reward | 0.0 s | 4.5 s |
| NS packing | 43.9 s | 40.6 s |
| NS PPO | 83.9 s | 85.8 s |
| EW environment collection | 228.3 s | 296.5 s |
| EW information reward | 0.0 s | 4.6 s |
| EW packing | 42.1 s | 44.3 s |
| EW PPO | 83.2 s | 82.2 s |
| complete round | 718.1 s | 868.2 s |
| complete process | 732.7 s | 888.0 s |

The actual bottleneck was sequential environment collection, especially B's
belief/Judge-conditioned rollout path; PPO was not the dominant cost.  The
profile peak resident memory was 5.61 GiB for A and 6.11 GiB for B, minimum
system `MemAvailable` was 115.1 GiB, and peak GPU allocations were 1206 MiB and
1508 MiB respectively.  This left a large safety margin for the formal run.

### Memory and failure safeguards

The formal processes ran independently, A on GPU 0 and B on GPU 1, under a
supervisor that:

- used memory-mapped train and evaluation data and refused to start if either
  data file was missing;
- refused to overwrite an existing result directory;
- required at least 40 GiB of free disk before launch;
- set `MALLOC_ARENA_MAX=2` and PyTorch expandable CUDA allocation segments;
- capped FSP at 10 and wrote a resume checkpoint every round;
- sampled process RSS, system `MemAvailable`, and GPU allocation every five
  minutes;
- terminated and marked the run blocked if available RAM fell below 16 GiB;
- scanned for Traceback, CUDA OOM, killed processes, and NaN;
- treated any process exit before its final checkpoint as a failure.

No guard triggered.  In the formal run, peak RSS was 7.03 GiB for A and
7.79 GiB for B, minimum `MemAvailable` was 111.2 GiB, and peak GPU allocation
was 1316 MiB for A and 2142 MiB for B.  After completion the filesystem had
426 GiB free.  The final deployment checkpoint for each arm was about 479 MB;
the round-resumable checkpoint, including optimizer/FSP state, was about
1.38 GB.

### Runtime and late-round stability

The supervisor launched A and B at 2026-07-15 14:48 UTC and recorded formal
completion at 2026-07-16 05:03 UTC.

| Arm | Process time | Completion |
|---|---:|---|
| A | 46,207.2 s (12 h 50 min) | round 60 checkpoint saved |
| B | 51,107.6 s (14 h 12 min) | round 60 checkpoint saved |

Across rounds 46-60, neither policy showed numerical or auction collapse:

- A competitive-auction rate stayed between 51.5% and 52.7%; B stayed between
  54.0% and 54.8%.
- All-pass stayed at 0.3%-0.4%, doubles near 4.5%-5.0%, and redoubles near
  0.12%-0.14%.
- Mean auction length stayed near 10.2-10.5 calls with p95 16-17.
- Actor belief loss declined from 1.8301 to 1.8214 for A and from 1.8241 to
  1.8139 for B.
- B's recorded information-reward statistic `step_ir` remained stable near
  0.136-0.137.

These observations establish stable execution and a persistent training
signal.  They do **not** establish task improvement, because self-play rollout
statistics are not a substitute for the held-out cross-agent match.

### Round-30 gate

The watcher paused both processes immediately after their round-30 resume
checkpoints, copied immutable gate snapshots, and evaluated B against A on 5000
paired deals:

| Stratum | Mean IMP/deal | 95% CI | B wins / A wins / ties |
|---|---:|---:|---:|
| Overall | +0.0444 | [-0.0577, +0.1465] | 554 / 522 / 3924 |
| Competitive | +0.1412 | [-0.0047, +0.2871] | 304 / 257 / 1904 |
| Non-competitive | +0.0200 | [-0.1057, +0.1457] | 142 / 139 / 1923 |
| Mixed classification | -0.5136 | [-1.2169, +0.1897] | 108 / 126 / 97 |

The competitive subset was close to a positive significance boundary, but the
overall confidence interval crossed zero.  The automatic decision was
`INCONCLUSIVE`, not `WIN`.  Training resumed only after user review and explicit
approval.

### Round-60 final B-vs-A evaluation

After both final checkpoints were complete, an eval-only run used the same
5000-deal source and seed:

| Stratum | Mean IMP/deal | 95% CI | B wins / A wins / ties |
|---|---:|---:|---:|
| Overall | +0.0132 | [-0.1050, +0.1314] | 707 / 693 / 3600 |
| Competitive | +0.0184 | [-0.1604, +0.1973] | 369 / 356 / 1660 |
| Non-competitive | -0.0707 | [-0.2055, +0.0642] | 145 / 156 / 1807 |
| Mixed classification | +0.3373 | [-0.2419, +0.9165] | 193 / 181 / 133 |

The evaluator again returned `INCONCLUSIVE`.  B's point estimate remained very
slightly positive overall, so the result is not evidence that B is worse than
A.  It failed the original B-must-beat-A gate: the experiment did not show that
B is better, and the promising round-30 competitive point estimate did not grow
with another 30 rounds.  The post-run diagnostics below show that this gate was
too strict as a prerequisite for the wiretap arm: B manipulated the intended
information quantity even though the manipulation did not improve IMP.

An additional human-readable trace was produced for the first 100 deals of the
same seeded evaluation stream.  It contains all four hands, HCP and shape,
dealer/vulnerability, both cross-table auctions, controllers, contracts, DDS
tricks, scores, and duplicate IMP.  That 100-deal diagnostic subset gave B
`-0.160 IMP/deal` with 11 B wins, 15 A wins, and 74 ties; it is for qualitative
auction inspection only and must not replace the 5000-deal result.

### Post-run A/B-vs-SL and frozen-Judge diagnostics

Two eval-only diagnostics were then run on the same 5000-deal source and seed.
They did not modify a checkpoint or expose the Judge to a playing policy.

First, each trained arm played the original `sl_base.pt` policy in black-box
duplicate cross-table evaluation:

| Match and stratum | Mean IMP/deal | 95% CI | Trained wins / SL wins / ties |
|---|---:|---:|---:|
| A vs SL, overall | +0.4398 | [+0.2859, +0.5937] | 1307 / 982 / 2711 |
| A vs SL, competitive | +0.4039 | [+0.1646, +0.6431] | 658 / 509 / 1064 |
| A vs SL, non-competitive | +0.2492 | [+0.0507, +0.4476] | 330 / 250 / 1487 |
| A vs SL, mixed | +1.1154 | [+0.5867, +1.6441] | 319 / 223 / 160 |
| B vs SL, overall | +0.4378 | [+0.2816, +0.5940] | 1300 / 997 / 2703 |
| B vs SL, competitive | +0.4738 | [+0.2380, +0.7097] | 687 / 547 / 1117 |
| B vs SL, non-competitive | +0.1142 | [-0.0948, +0.3233] | 320 / 253 / 1423 |
| B vs SL, mixed | +1.2971 | [+0.7537, +1.8405] | 293 / 197 / 163 |

This decisively rejects the cheap explanation that A and B stayed at the SL
initializer.  Both learned task-improving policies of almost identical overall
strength.  The A/B null result is therefore not caused by a completely inactive
RL pipeline.

Second, the frozen training Judge measured the raw signed loss reduction caused
by every actual A and B call in duplicate B-vs-A cross-play.  Partner and next-
opponent receiver observations, bidder-hand targets, and immediate before/after
states exactly match the training definition.  Confidence intervals treat each
paired deal, not each call, as an independent unit.  The replayed match exactly
reproduced the formal B-vs-A result (`+0.0132 IMP/deal`, 707/693/3600), providing
an end-to-end deal-order and scoring check.

| Model | Calls | Partner gain/call | Opponent gain/call | Partner total/deal | Opponent total/deal | Partner - opponent/deal |
|---|---:|---:|---:|---:|---:|---:|
| A | 51,614 | +0.08628 | +0.08800 | +0.89063 | +0.90843 | -0.01780 |
| B | 51,491 | +0.09203 | +0.09307 | +0.94772 | +0.95844 | -0.01072 |

The paired B-minus-A changes were:

| Quantity, per deal | B - A | 95% CI |
|---|---:|---:|
| Partner information gain | +0.05709 | [+0.04853, +0.06564] |
| Opponent information gain | +0.05001 | [+0.04195, +0.05808] |
| Raw secrecy difference (`partner - opponent`) | +0.00708 | [+0.00084, +0.01331] |
| Configured C expression (`partner - 0.05 * opponent`) | +0.05459 | [+0.04632, +0.06286] |

B therefore did what its shaping reward asked: it increased frozen-Judge
partner information.  However, `0.05001 / 0.05709 = 87.6%` of that increment
was accompanied by increased opponent information.  Only a small residual
improved the partner-minus-opponent balance, and none of these information
changes produced a measurable B-vs-A IMP gain.

This resolves the immediate decision tree:

- `A ~= SL` is false; do not treat the result as a dead RL pipeline.
- `B did not change partner information` is false; the reward reached behaviour.
- The data support substantial public-channel leakage, although not exact
  one-for-one washout: B's partner gain rose slightly more than opponent gain.
- B no longer needs to beat A as a prerequisite for testing the leakage arm.
  The information manipulation itself is established.
- Before launching C, revisit `beta`.  With partner and opponent terms on the
  same empirical scale, `beta=0.05` is dominated by the partner term and is not
  a strong test of the wiretap difference.  Any C variants and selection rule
  should be declared before inspecting their IMP results.

### Artifacts and current decision

Server artifacts are under:

```text
results/main_unrestricted_pilot_seed42/
results/profile_full_round_memory_safe_seed42/
results/main_unrestricted_formal_memory_safe_seed42/A/
results/main_unrestricted_formal_memory_safe_seed42/B/
results/main_unrestricted_formal_memory_safe_seed42/resources.tsv
results/main_unrestricted_formal_memory_safe_seed42/round30_gate/
results/main_unrestricted_formal_memory_safe_seed42/round60_eval/result.json
results/main_unrestricted_formal_memory_safe_seed42/round60_eval/B_vs_A_100_auctions.txt
results/main_unrestricted_formal_memory_safe_seed42/round60_diagnostics/agents_vs_sl.json
results/main_unrestricted_formal_memory_safe_seed42/round60_diagnostics/judge_information.json
```

The engineering outcome is successful: the unrestricted pipeline is stable,
memory-safe, reproducible, learns substantially beyond SL, and produces
bridge-plausible auctions at the full planned scale.  The scientific A-vs-B
outcome is also sharper: partner-only shaping measurably changed the intended
Judge information quantity, but most of the extra information was public
leakage and there was no reliable duplicate-IMP gain.  Therefore:

- preserve all A/B checkpoints and evaluation artifacts;
- do not relabel the result as either a B win or proof that B is harmful;
- do not run the original `beta=0.05` C or a three-seed sweep blindly;
- specify the C leakage weights and selection rule from the A/B information
  decomposition, then pilot the declared C variants before full training.

## Research design

Both the controlled validation and main experiment use three agents with an
identical 571-dimensional external interface and identical internal
partner/RHO belief heads:

| Agent | Task reward | Partner information | Opponent leakage penalty |
|---|---:|---:|---:|
| A | yes | no | no |
| B | yes | yes | no |
| C | yes | yes | yes |

The frozen Judge BeliefNet is a **training-only communication critic** for
Agents B and C.  It is separate from the trainable belief head deployed inside
every A/B/C actor.  For a bid by player `i`, both Judge receivers predict the
same target, `i`'s hidden hand.
The partner prediction conditions on the partner's private hand; the opponent
prediction conditions on the next opponent's private hand.  Each information
gain is measured immediately before and after that single bid.  At execution an
agent computes its own partner/RHO belief activation from its legal 571-dim
observation and never reads an opponent's network.

The information reward is

\[
r_{info}=\Delta \ell_{partner}-\beta\Delta \ell_{opponent},
\]

where each gain is the **signed** reduction in calibrated hand-prediction loss
caused by the current bid.  Before training, partner-only episode totals are
pre-sampled once to define a B/C-shared scale (default target: 5% of IMP reward
standard deviation).  That scale is frozen for every subsequent round.

## Evaluation contract

Formal evaluation is black-box.  A policy receives only:

- its legal game observation;
- the public auction history;
- the current seat and legal actions.

The evaluator never provides an opponent checkpoint, BeliefNet, critic, hidden
hand, or DDS table.  Duplicate cross-table scoring rotates the two agents through
opener and overcaller roles on the same deals.

`experiments/evaluation.py` also provides a paired execution ablation reporting:

- action disagreement rate;
- auction disagreement rate;
- contract and score disagreement rates;
- paired IMP differences.

These metrics are more sensitive than comparing aggregate IMP alone.

Legacy BCA execution ablation:

```bash
python experiments/execution_ablation.py \
  --checkpoint results/agent_a.pt \
  --data data/competitive_500k.npz \
  --deals 5000
```

## Main files

```text
env/                         Bridge bidding and duplicate-table rules
subgames/competitive_env.py  Fixed-prefix 1H-1S competitive environment
subgames/subgame_trainer.py  MAPPO/FSP rollout and training loop
networks/policy_net.py       571 API, internal belief actor, centralized critic
networks/belief_net.py       Hidden-hand semantic model and information reward
experiments/evaluation.py    Black-box evaluation and execution ablations
experiments/subgame_validation.py  A/B/C validation entry point
experiments/benchmark_training.py   End-to-end training throughput benchmark
scripts/cloud_benchmark_quad.sh     Multi-process/two-GPU throughput sweep
utils/sl_pretrain.py         SAYC supervised policy pretraining
tests/test_all.py            Core environment, scoring, and network tests
```

## Setup

```bash
pip install -r requirements.txt
```

OpenSpiel is required for policy training and evaluation.  The DDS datasets and
pretrained checkpoints are stored under `data/` and `results/` respectively.

## Validation experiment

Quick wiring check:

```bash
python experiments/subgame_validation.py --quick --rounds 1 --eval-deals 100
```

Single-seed subgame validation:

```bash
python experiments/subgame_validation.py \
  --data data/competitive_500k.npz \
  --sl-checkpoint results/sl_base.pt \
  --belief-checkpoint results/sl_base_bca.pt \
  --rounds 25 \
  --eval-deals 5000 \
  --seed 42
```

### Split Colab sessions

Agents can be trained independently to fit Colab runtime limits. Use the same
seed, SL checkpoint, BeliefNet checkpoint, data, and hyperparameters in every
session:

```bash
# Session 1
python experiments/subgame_validation.py --train-agents A --seed 42 \
  --data data/competitive_500k.npz --output-dir results/competitive_v2

# Session 2
python experiments/subgame_validation.py --train-agents B --seed 42 \
  --data data/competitive_500k.npz --output-dir results/competitive_v2

# Session 3
python experiments/subgame_validation.py --train-agents C --seed 42 \
  --data data/competitive_500k.npz --output-dir results/competitive_v2
```

After copying all three checkpoints into the same output directory, evaluate
without training:

```bash
python experiments/subgame_validation.py --eval-only --seed 42 \
  --data data/competitive_500k.npz \
  --output-dir results/competitive_v2 --eval-deals 5000
```

`--agent-a`, `--agent-b`, and `--agent-c` may be used when the checkpoints are
stored in different directories.

Outputs are written to `results/competitive_v2/`.  Every policy checkpoint is
self-contained for execution, including its internal belief head.  The frozen
Judge may also be stored for research diagnostics, but is not loaded by the
evaluator.

## Correctness invariants

- Policy input is always the standard 571-dimensional OpenSpiel observation;
  the 96 belief features are an internal activation only.
- Card conversion is explicit: datasets use suit-major order and OpenSpiel uses
  rank-major order.
- Bidding actions use OpenSpiel's native offset ordering: Pass/Dbl/RDbl/1C..7NT
  are raw actions 52/53/54/55..89 and map to policy outputs 0..37.
- Deals are interleaved when reconstructed in OpenSpiel.
- Dealer rotation and vulnerability are applied to both training and evaluation.
- In the controlled subexperiment, the constrained opener hand, dealer, DDS
  declarer axis, and fixed `1H-1S` callers rotate together.
- Actor selection and task-reward attribution use physical N/E/S/W seats.
- Information gain uses the state immediately after the attributed bid; it must
  not include intervening actions from other players.
- Partner and opponent gains use receiver-specific observations but the same
  bidder position and bidder-hand target.
- Judge gains retain their sign; no ReLU or positive-only clipping is allowed.
- B and C share one frozen partner-only scale calibrated before training.
- A scaled information bonus remains attached to the bid that produced it; it
  is not moved to the terminal transition.
- Information reward is computed for the training side, independent of physical
  seat or opener/overcaller role.
- Evaluation uses identical deal, dealer, and vulnerability sets for each paired
  comparison.

## Tests

```bash
python tests/test_all.py
python -m unittest discover -s tests -v
```

## Server performance and formal-run handoff (2026-07-15)

### July 15 conversation handoff: research interpretation and decisions

This subsection preserves the scientific and bridge-domain conclusions from
the July 15 planning conversation, not just the engineering benchmark.

#### How to interpret the current A/B/C result

- In bridge, an advantage of **+0.1 IMP/deal is already important**.  Do not use
  effect-size intuitions from games where only multi-IMP changes matter.
- The current question is not whether Agent A is good in isolation.  It is why
  B and C did not beat A in the completed controlled experiment, despite the
  intended communication reward.
- The corrected action mask removed the earlier pathological repeated
  Double/Redouble behaviour.  The user's expert bridge inspection finds the
  new bidding dramatically more coherent than the pre-fix models.
- The user manually inspected 100 A-vs-B deals.  Their bridge judgement is that
  A and B currently resemble players with roughly one or two years of amateur
  experience: much better than before, but still far below expert level.  B is
  slightly more human-like and handles some auctions somewhat better than A,
  although the difference is small and 100 deals cannot establish strength.
- Therefore the negative aggregate B-vs-A result does not by itself prove that
  information reward is conceptually harmful.  Two live explanations remain:
  (1) both models are undertrained and B's advantage may emerge only at much
  larger scale; or (2) the expert intuition that the more human-like treatment
  should be stronger is misleading, and task-only PPO is genuinely better.

The earlier interpretation that B increased opponent gain more than partner
gain is also not sufficient.  The controlled dataset forces a competitive
`1H-1S` start, while ordinary bridge contains a substantial non-competitive
auction region (estimated in the conversation at roughly 30-40%).  Information
shaping may be more useful when partners have room to exchange constructive
information without immediate interference.  The main experiment must cover
that region before concluding that `r_info` has no value.

#### Experimental priorities

1. Preserve the corrected bidding mask, dealer/seat mapping, reward attribution,
   GAE, value clipping, and FSP snapshot fixes.  The old apparent `+3.8 IMP`
   result passed through known bugs and must not be treated as a trustworthy
   target.
2. Do not spend the main budget merely repeating the forced-competition
   A/B/C experiment.  Implement and validate unrestricted random deals and
   complete auctions, including both competitive and non-competitive cases.
3. Run a short end-to-end pilot on the actual main-experiment entry point to
   measure auction length, throughput, reward statistics, action-mask health,
   checkpoint/resume, and GPU allocation before the long run.
4. Train A/B/C from the same `sl_base.pt`, Judge checkpoint, seed schedule, deal
   distribution, architecture, FSP settings, optimizer settings, and evaluation
   deals.  Only the intended information-reward terms may differ.
5. Evaluate on large paired duplicate samples.  Because +0.1 IMP is meaningful,
   confidence intervals and multiple seeds matter; 100 manually inspected
   deals are qualitative diagnostics, not the strength result.
6. Alongside IMP, stratify evaluation by competitive versus non-competitive
   auctions.  Aggregate IMP alone could hide the region in which information
   reward helps or hurts.

The initial scale discussed for the main run was inspired by prior bridge-RL
work but remains a compute hypothesis, not a fixed law.  `deals_per_step=4096`
was only an early assumption.  Actual rollout and PPO batch sizes must follow
measured throughput while preserving the intended total number of deals and
optimization semantics.

#### Compute, budget, and operational decision

- Budget discussed: approximately CNY 1,000-2,000, with no more than two weeks
  of server time and a preference for leaving room for a failed experiment.
- DDS generation is CPU-heavy and should not consume expensive GPU rental time.
  Ten million precomputed PGX/DDS training deals and a separate 500k evaluation
  set are already present locally; verify the selected main entry point uses the
  intended files and does not accidentally reuse the constrained 500k set.
- The rented test server costs about CNY 60/day and provides two A10 24 GiB GPUs,
  16 physical/32 logical CPU cores, and 128 GiB RAM.  A single process initially
  looked no better than the local 40-series GPU, but optimization plus concurrent
  independent runs made the whole machine cost-effective.
- Rent one week for the prepared run and retain the ability to extend.  Reserve
  two weeks up front only if seller availability/discount makes extension risky
  or if both a complete failed full-scale run and a complete rerun must fit.

Long jobs should run detached from the SSH session, write one log per agent,
save a resume checkpoint every round, and emit small structured health summaries.
Monitoring should be event-driven or every few hours (process alive, round
progress, NaN/Inf, GPU idle, memory/disk, checkpoint freshness), rather than
continuous conversational polling.  This substantially reduces assistant-token
usage; deeper analysis is needed only on an alert or phase boundary.

#### Product-oriented model after the controlled study

The scientific A/B/C experiment is not the final performance ceiling.  Once the
best treatment is identified, train a separate strongest model with more data,
more rounds, selected hyperparameters, multiple seeds/checkpoint selection or
ensembling, and possibly greater capacity.  The product goal is a useful bridge
bidding bot, not merely a statistically clean ablation.

Monte Carlo search should sample hidden deals from a belief conditioned on the
player's hand and public auction, then use the learned policy as a prior or
rollout policy.  Naive perfect-information Monte Carlo can suffer strategy
fusion and should not be presented as a correct information-set solution.  The
base policy must first be strong enough that search refines good bidding rather
than averaging weak continuations.

**Launch gate:** all throughput measurements below used
`CompetitiveSubgameEnv` and the constrained `1H-1S` data.  They validate the
training engine and hardware, not the unrestricted main-experiment
distribution.  Do not label the next run as the main experiment until its
entry point uses random deals, complete auctions from the opening call, and the
intended competitive/non-competitive mixture.  Benchmark that entry point
briefly before extrapolating the times below.

The rented benchmark machine has 16 physical/32 logical CPU cores, 128 GiB RAM,
and two 24 GiB NVIDIA A10 GPUs.  The original rollout path was CPU/OpenSpiel
bound: one Agent-B process reached only about 192 deals/s and averaged less than
10% GPU utilization.  Increasing `deals_per_step` from 512 to 4096 made it
slower (about 174 deals/s), and Python threads also reduced throughput.  Keep
`collector_workers=1`.

`encode_openspiel_auction_observation` now constructs the exact 571-dimensional
OpenSpiel auction tensor directly instead of dealing 52 cards and replaying the
auction into a new OpenSpiel state for every observation.  It includes the
auction/opening-lead phase transition after the final three passes.  Random
deals, vulnerabilities, dealers, histories, all four receiver seats, and
terminal states are tested element-for-element against OpenSpiel in
`tests/test_fast_observation.py`.  The optimized path is enabled by
`fast_observation_encoding=True` and reproduces the old actions, task IMP,
calibration scale, PPO losses, and information reward while raising one-process
throughput to about 335 deals/s.

Measured end-to-end multi-process throughput with four PPO epochs was:

| Concurrent processes | PPO batch | Aggregate deals/s | Per-process deals/s |
|---:|---:|---:|---:|
| 3 | 512 | 878 | 320-339 |
| 4 | 512 | 1092 | 300-303 |
| 4 | 1024 | 1192 | 332-334 |

The 1024 PPO batch is faster but changes the number and noise of optimizer
updates.  Use 512 for the controlled formal comparison unless a learning-quality
pilot justifies 1024.  The server has enough CPU capacity for four independent
processes, but the formal A/B/C run needs three.  Expected runtime is
`C > B > A` because C computes both information terms, B computes partner gain,
and A has no information reward.  Balance the GPUs as follows:

```text
GPU 0: Agent A + Agent B
GPU 1: Agent C
```

Run A/B/C as independent processes with the same seed, data, SL checkpoint,
Judge checkpoint, and hyperparameters.  Do not run the built-in sequential
`--train-agents A B C` path when wall-clock time matters.  Each process should
use one label (`--train-agents A`, etc.), write its own log, and rely on the
per-round resume checkpoint.

Recommended controlled-run performance settings:

```text
deals_per_step=512
steps_per_phase=256
batch_size=512
num_epochs=4
collector_workers=1
fast_observation_encoding=True
```

`steps_per_phase * deals_per_step` is the number of deals collected per side in
each round.  Therefore `256 * 512` preserves exactly the same 131,072 deals as
the earlier hypothetical `32 * 4096`, while using the substantially faster
rollout batch size.  With three concurrent agents, the 60-round version contains
47.2 million deals in total and is estimated at about 15 hours in the current
competitive environment.  Unrestricted full auctions should be budgeted at
roughly 1.5-2 times that duration until measured.  A 100-round configuration at
the earlier 64-by-4096 scale is about 157.3 million total deals: approximately
50 hours in the current environment and conservatively 3-4 days with full
auctions.

One server week is sufficient for one successful formal run at these scales.
If a complete failed full-scale run and a complete rerun must both fit without
extension risk, reserve the second week; otherwise rent one week and extend only
if the learning curves or infrastructure require it.

The scientific A/B/C comparison should keep architecture, data, and optimizer
settings controlled.  After selecting the best treatment, a separate
performance-oriented model may use more deals, larger capacity, multiple seeds
or checkpoint selection, and belief-conditioned information-set Monte Carlo
search; it need not be constrained to the ablation's compute budget.

The preliminary report is available as `paper.pdf`.  It documents the earlier
Full Disclosure/BCA formulation; the formal experiment described here replaces
execution-time convention sharing with training-only semantic modelling.

## Migration note

Pre-refactor subgame results must be rerun.  The old implementation used the
bidder's later turn as the post-bid state (including intervening actions), used
different hidden-hand targets for partner and opponent terms, restricted the
information reward to opener-side actions, and moved all per-bid bonuses to the
terminal transition.  The current implementation fixes all four attribution
errors and tests the receiver/target contract directly.

## Plain-571 control experiment

The July 2026 A/B/C run did not establish that any RL agent improved over the
plain supervised policy. Before changing the information reward again, run the
control that removes the belief path shared by all three agents:

```bash
python -u experiments/plain_571_control.py \
  --data data/competitive_500k.npz \
  --sl-checkpoint results/sl_base.pt \
  --output-dir results/plain_571_control_seed42 \
  --rounds 25 --eval-deals 5000 --seed 42 --eval-seed 20260714
```

This is a plain 571-dimensional Agent A: no actor belief head, no Judge, no
belief auxiliary loss, and no information reward. It uses the corrected current
action mapping, seat handling, GAE, value clipping, FSP snapshot cache, and
black-box duplicate evaluation. Its defaults match the key observed settings of
the old successful 571 run: entropy coefficient `0.001` and an FSP pool capped
at 10. It writes the model, summary, and run manifest to the output directory.

## July 2026 investigation handoff

This section records the decisions and evidence from the long debugging
conversation so that a new conversation can continue without reconstructing
the history.

### Checkpoint meanings

- `results/sl_base.pt` is the pure 571-dimensional SAYC supervised policy. It
  contains no BeliefNet or belief feature path. It is the correct initializer
  for every RL agent and the permanent SL anchor in the FSP pool.
- `results/sl_base_bca.pt` is the Stage-A SAYC BeliefNet supplier. It is used to
  initialize a Judge or decoder, not as an actor checkpoint. Because it learned
  SAYC semantics, it cannot automatically interpret bids after a policy drifts
  to a different protocol.
- `results/sl_base_bca_stageB.pt` is the deprecated legacy 667-dimensional
  actor. It fine-tuned the entire actor, caused negative transfer, and mixed
  bidding-pattern change with convention-card effects.
- `results/sl_base_bca_refine.pt` freezes the 571-dimensional SL actor and trains
  a small residual belief adapter whose gate starts at zero. It was designed as
  a convention-aware SL opponent without changing the base bidding pattern. It
  is not the initializer for A/B/C.

The completed July run correctly used `sl_base.pt` for A/B/C initialization and
`sl_base_bca.pt` for the frozen Judge/internal-decoder warm start. Therefore the
failure cannot be explained by selecting the wrong SL checkpoint.

### Confirmed engineering repairs

The following bugs were found in the inherited experiment and repaired before
the latest run:

- OpenSpiel bidding actions were mapped with the wrong raw-action offset. The
  corrected mapping is Pass/Dbl/RDbl/1C..7NT = 52/53/54/55..89.
- Dealer rotation did not consistently rotate observation ownership, actor
  seats, constrained hands, fixed-prefix callers, and DDS declarer axes.
- GAE terminal boundaries were displaced, allowing returns to cross episode
  boundaries.
- PPO value clipping compared the new value against itself and therefore did
  not clip updates correctly.
- The FSP actor cache could combine role networks from different historical
  snapshots.
- Information reward formerly used a later turn by the same bidder as the
  post-bid state, thereby including intervening bids; it also used inconsistent
  hidden-hand targets, covered only opener-side actions, and moved the summed
  reward to a terminal transition.

Relevant commits on `main` are:

- `c19a861` — correct OpenSpiel action mapping;
- `2e52499` — synchronize dealer rotation;
- `89c1968` — repair reward/GAE/value/FSP training infrastructure;
- `96d08ef` — keep the requested single-table/random-seat training design;
- `da29f5d` — signed, frozen-scale information shaping and internal actor belief.

### Current formal design

A/B/C expose the same 571-dimensional black-box interface. Internally, the July
design gives all three a trainable partner/RHO belief decoder. The separate
Judge is frozen, used only while training to measure communication reward, and
is never available to the evaluator. A uses task reward only; B adds partner
information gain; C additionally subtracts opponent leakage.

Information gain was changed from positive-only `ReLU(delta)` to signed gain.
Its scale is calibrated once before training from a fixed sample and then
frozen. The latest run used information weight `0.05`, calibration size 2048,
and actor belief auxiliary coefficient `0.1`. A/B/C had identical actor
architectures and differed only in the intended reward terms.

The proposed action-space projection/residualization of information gain was
discussed and rejected for now. Its claimed covariance guarantee did not map
cleanly to the actual PPO/Adam gradient, and projecting only on bid height could
remove useful signal or leave other behavioural distortions. The accepted
minimal changes were signed gain, frozen scaling, a smaller weight, separated
Judge/Actor belief, and common A/B/C architecture.

### Completed July experiment

The completed Colab experiment used commit `da29f5d`, seed 42, 25 rounds, and
5000 paired evaluation deals. Artifacts are in Google Drive under
`bridge_coma/competitive_v4_signed_internal/`:

- `agent_a_seed42.pt` — 479,437,787 bytes;
- `agent_b_seed42.pt` — 479,437,979 bytes;
- `agent_c_seed42.pt` — 479,437,979 bytes;
- `full_run.log`, `run_manifest.json`, and `summary_seed42.json`.

Pairwise results were:

| Match | Mean IMP | 95% CI | W/L/T |
|---|---:|---:|---:|
| B vs A | -0.079 | [-0.183, +0.025] | 581/619/3800 |
| C vs A | -0.169 | [-0.275, -0.063] | 564/646/3790 |
| C vs B | -0.047 | [-0.150, +0.055] | 562/569/3869 |

These results show that B/C did not improve on A, but they do not show that A
learned successfully. The user's inspection indicates that all three are close
to `sl_base.pt`. That makes a shared training-path failure more plausible than
a B/C-only shaping failure.

### Why the old result is not yet decisive

The old `results/571agentA.txt` run disabled belief conditioning, used entropy
coefficient `0.001`, capped the FSP pool at 10, and reported Agent A vs SL at
`+3.784 IMP`. Its training metric rose from approximately zero to about +1.6 by
round 25. The current experiment instead enabled the internal actor belief head
and auxiliary loss for every arm, used entropy coefficient `0.01`, and allowed
the FSP pool to grow to 26 because the quality gate was not enabled.

However, the old +3.784 result passed through several now-confirmed action,
seat, reward, and evaluation bugs. It is evidence that the old program produced
a strong-looking result, not proof that a correct implementation must reproduce
that number.

### Working diagnosis and next decision

The first diagnostic is now `experiments/plain_571_control.py`. It isolates the
corrected infrastructure from the new shared belief architecture.

- If the control clearly beats `sl_base.pt`, the common internal belief path,
  belief auxiliary loss, or related optimizer interaction is the leading cause
  of the July failure. Follow-up ablations should add the belief head and FSP
  changes one at a time.
- If the control also remains near SL, do not tune `r_info`. First reconcile the
  corrected task reward and evaluation with the old pipeline and test whether
  the historical +3.784 improvement was created by the repaired bugs.

Multi-seed experiments remain required for the final study, and the existing
multi-seed support should be retained, but no multi-seed sweep should begin
until this single-seed control establishes that task-only RL can learn at all.
