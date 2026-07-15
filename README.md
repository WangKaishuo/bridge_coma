# Bridge-COMA

Bridge-COMA studies communication credit assignment in competitive contract
bridge bidding.  A public bid can improve a partner's hidden-hand inference
while also leaking information to an opponent.  The project adds a dense,
belief-based communication reward to MAPPO with fictitious self-play (FSP).

## Current handoff

- `experiments/subgame_validation.py` is a controlled **subexperiment** on
  constrained deals with the relative prefix `dealer:1H, dealer+1:1S`.
- The planned **main experiment** uses unrestricted random deals and complete
  auctions; the fixed prefix must not shape its environment or interfaces.
- Policies, critics, buffers, and FSP checkpoints use physical N/E/S/W seats.
  Random dealer rotation exposes every seat to different auction roles.
- Agent A/B/C may be trained in separate Colab sessions.  The July Agent A run
  logged in `agentA.txt` predates the synchronized-dealer fix and must be rerun.
- After the controlled A/B/C rerun, the next task is the unrestricted main
  environment and black-box published-agent baseline adapters.

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
python -m unittest tests.test_information_reward tests.test_receiver_rollout
```

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
