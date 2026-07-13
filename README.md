# Bridge-COMA

Bridge-COMA studies communication credit assignment in competitive contract
bridge bidding.  A public bid can improve a partner's hidden-hand inference
while also leaking information to an opponent.  The project adds a dense,
belief-based communication reward to MAPPO with fictitious self-play (FSP).

## Research design

The formal experiment uses three agents with identical 571-dimensional
OpenSpiel policies:

| Agent | Task reward | Partner information | Opponent leakage penalty |
|---|---:|---:|---:|
| A | yes | no | no |
| B | yes | yes | no |
| C | yes | yes | yes |

BeliefNet is a **training-only communication critic** for Agents B and C.  For a
bid by player `i`, both receivers predict the same target, `i`'s hidden hand.
The partner prediction conditions on the partner's private hand; the opponent
prediction conditions on the next opponent's private hand.  Each information
gain is measured immediately before and after that single bid.  The policy never
consumes belief features at execution time, and an agent never reads an
opponent's network.

The information reward is

\[
r_{info}=\Delta \ell_{partner}-\beta\Delta \ell_{opponent},
\]

where each gain is the reduction in calibrated hand-prediction loss caused by
the current bid.  The reward is scaled relative to observed IMP variance before
being added to the terminal task reward.

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
networks/policy_net.py       571-dimensional policy and centralized critic
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

Outputs are written to `results/competitive_v2/`.  Every policy checkpoint is
self-contained for execution; a BeliefNet state may also be stored for research
diagnostics, but it is not loaded by the evaluator.

## Correctness invariants

- Policy input is always the standard 571-dimensional OpenSpiel observation.
- Card conversion is explicit: datasets use suit-major order and OpenSpiel uses
  rank-major order.
- Deals are interleaved when reconstructed in OpenSpiel.
- Dealer rotation and vulnerability are applied to both training and evaluation.
- Information gain uses the state immediately after the attributed bid; it must
  not include intervening actions from other players.
- Partner and opponent gains use receiver-specific observations but the same
  bidder position and bidder-hand target.
- A scaled information bonus remains attached to the bid that produced it; it
  is not moved to the terminal transition.
- Information reward is computed for the training side, independent of physical
  seat or opener/overcaller role.
- Evaluation uses identical deal, dealer, and vulnerability sets for each paired
  comparison.

## Tests

```bash
python tests/test_all.py
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
