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
