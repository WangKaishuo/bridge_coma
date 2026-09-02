# MARL Dual Audience

This repository provides code and pretrained models for multi-agent
reinforcement learning in bridge bidding, studying how bids convey information
to both teammates and opponents. It includes training and evaluation tools for
comparing task-only rewards with teammate information gains and opponent
information penalties.

It accompanies the MSc dissertation *Auditing Dual-Audience Reward Shaping:
Reward Semantics, Proxy Movement, and Task Performance in Multi-Agent
Reinforcement Learning* (K. Wang, University of Bristol, 2026), which reports
the results these artefacts come from and states what they do and do not
establish. This README covers how to run the code; the dissertation covers what
the numbers mean.

## What is included

```text
algorithms/     PPO, MAPPO, and behavioral-cloning components
data/           512-deal DDS sample and the two supervised checkpoints
env/            bridge auction state and legal-action rules
experiments/    training, evaluation, model export, and inspection entry points
networks/       policy, value, belief, and archived Task-Q networks
subgames/       complete-auction environment and self-play trainer
utils/          scoring, DDS loading, features, statistics, and FSP utilities
models/         twelve inference-only models and their manifest
results/        workbook, formal matrices, and supplementary diagnostics
```

Development tests, debugging scripts, server controllers, training logs,
optimizer states, and experiment chronology are intentionally excluded.

## Methods and historical labels

| Thesis name | Code/display alias | Historical label | Reward definition |
|---|---|---|---|
| TASK | Task-only | A | terminal duplicate-IMP task reward |
| TEAM | Team-BG | B | task reward plus teammate belief gain |
| DUAL | DUAL-local | Legacy-C | local teammate gain and opponent leakage penalty |
| DAPS | DAPS | Strict-C | actor-time potential-difference formulation |

We use **run** for a training replicate and **seed** for random-number
initialisation. Historical filenames and raw JSON field names are unchanged.

## Installation

Python 3.10 or later is recommended. Create a virtual environment, activate it,
and install the dependencies:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

OpenSpiel supplies the 571-dimensional bridge auction observation. The bundled
sample does not require DDS generation; `endplay` is needed only when creating
a new dataset.

## Verify the models

All twelve `.pt` files are stored directly in `models/`.

Each file contains only four actor state dictionaries and the architecture
metadata needed for inference. It does not contain a critic, optimizer,
BeliefNet, FSP pool, random-number state, or training-resume state. Verify file
integrity against `models/MANIFEST.csv`.

Inspect one model without running an evaluation:

```bash
python -m experiments.inspect_model \
  models/seed42_a_round100_actor_only.pt
```

## Evaluation data

A 512-deal sample is included at `data/sample_dds/dds_0000.npz`. It is large
enough to run the commands below immediately after installation. It is a smoke
test for the complete loading and evaluation path, not a replacement for the
20,000-deal published comparisons.

Evaluation accepts a DDS dataset in either of these formats:

- a compact `.npz` file containing `decks` with shape `(N, 52)` and `tricks`
  with shape `(N, 5, 4)`; or
- a memmap dataset directory accepted by `utils.dds_data.create_loader`.

The `decks` array stores the owner of every suit-major card as an integer from
0 to 3. The `tricks` array stores double-dummy tricks by strain and declarer.
The full archived training and evaluation splits are not included because they
are substantially larger than this source release. A new DDS dataset can be
generated with, for example:

```bash
python -m utils.dds_data \
  --num_samples 10000 \
  --batch_size 10000 \
  --num_workers 8 \
  --output_dir data/generated \
  --seed_offset 0
```

Generation time depends strongly on CPU count. The command prints progress and
the measured elapsed time when it finishes.

## Run a paired comparison

The following command compares Team-BG against Task-only on the same sampled
deals. The reported orientation is the second model minus the first model.

```bash
python -m experiments.evaluate_resume_ab \
  --agent-a models/seed42_a_round100_actor_only.pt \
  --agent-b models/seed42_b_round100_actor_only.pt \
  --data data/sample_dds \
  --deals 200 \
  --seed 2026072401 \
  --output local_evaluation.json \
  --cpu
```

Remove `--cpu` to use CUDA when available. Increase `--deals` only after the
small command completes successfully.

## Run a four-policy matrix

```bash
python -m experiments.evaluate_joint_round_robin \
  --agent Task-only=models/seed42_a_round100_actor_only.pt \
  --agent Team-BG=models/seed42_b_round100_actor_only.pt \
  --agent DUAL-local=models/seed42_legacy_c_round100_actor_only.pt \
  --agent DAPS=models/seed42_strict_c_round100_actor_only.pt \
  --data data/sample_dds \
  --deals 200 \
  --seed 2026072401 \
  --output-dir local_matrix \
  --cpu
```

Every comparison is evaluated on common sampled support. The output records the
comparison orientation explicitly.


## Training code

`experiments.main_experiment` trains on complete auctions. To run one small
training round with the bundled DDS sample and supervised checkpoints:

```bash
python -m experiments.main_experiment \
  --data data/sample_dds \
  --eval-data data/sample_dds \
  --quick --rounds 1 --eval-deals 200 --beta 1 \
  --output-dir local_training \
  --cpu
```

This runs TASK (A), TEAM (B), and DUAL (C). For DAPS, add
`--train-agents C --info-potential-shaping` and use a separate output directory.
The full 10M/500k training and evaluation splits are not included. Use
`python -m experiments.main_experiment --help` for all training options.

## Supervised checkpoints

Two supervised artefacts are included under `data/`, alongside the DDS sample,
because both are inputs to training rather than outputs of it. Sizes and SHA-256
hashes are in `data/MANIFEST.csv`:

- `sl_base.pt` — the behavioural-cloning policy used to initialise every arm and
  retained as the permanent member of each fictitious-self-play pool. It is also
  the SL partner and opponent in the cross-play comparisons. Default for
  `--sl-checkpoint`.
- `sl_base_bca.pt` — the belief network, frozen throughout the formal experiment
  and used as the judge that scores receiver predictions. Default for
  `--belief-checkpoint`.

Releasing the judge matters for checking the proxy results. The diagnostic
records under `results/aligned_proxy_diagnostics/` retain aggregate estimates
but not the 1,000 per-deal values behind them; with this checkpoint, the
released actors and a DDS source, those belief-gain scores can be recomputed
rather than taken on trust.

## Results

Open `results/results.xlsx`. It contains:

- `Round100 Primary`: nine treatment-minus-TASK estimates;
- `Round100 Pairwise`: all 18 direct within-run contrasts;
- `Round120 Continuation`: 18 continuation contrasts;
- `External Crossplay` and `Shared Partner Crossplay`: supplementary
  interoperability comparisons, including the direct SL control;
- `Semantic Drift`: 36 descriptive treatment-minus-TASK differences;
- `Gradient Audit`: eight seat-specific measurements from the round-100 audit;
- `Aligned Proxy Diagnostics`: 54 endpoint-aligned estimates and intervals;
- `Evaluation Protocol`: the common formal task-evaluation contract; and
- `Model Manifest`: model file sizes and SHA-256 hashes.

The six formal matrices use the same 20,000 unique deals sampled without
replacement, with evaluation seed `1611583527`. Their source summaries,
sampled indices and compressed per-deal IMP vectors are included under
`results/unified_joint_evaluation_v1/`. Positive values favour the first named
policy in a workbook comparison. Confidence intervals use paired-deal standard
errors and a normal 1.96 multiplier. They are conditional on the fixed policies
being compared: they are not across-run intervals, and they are not
multiplicity-adjusted. An interval containing zero is inconclusive, not
evidence that two policies are equivalent.

## Reproducibility notes

- Input observations have 571 dimensions and use the OpenSpiel-compatible
  bridge representation.
- The public action space has 38 outputs: Pass, Double, Redouble, and 35 ordered
  contract calls.
- Model files use action mapping `openspiel_native_52_89_v1`.
- Round-100 models cover runs 42, 43, and 44 for all four methods.
- Published model files are inference artifacts, not resumable training files.
