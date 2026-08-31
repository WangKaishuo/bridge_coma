# Bridge-COMA

Bridge-COMA studies whether auxiliary information rewards can improve
communication learning in competitive contract-bridge bidding. This public
release contains the source code used by the thesis experiments, one compact
results workbook, and twelve inference-only round-100 models distributed as
GitHub Release assets.

## What is included

```text
algorithms/     PPO, MAPPO, and behavioral-cloning components
dri/            receiver-information reward components used by the trainer
env/            bridge auction state and legal-action rules
experiments/    training, evaluation, model export, and inspection entry points
networks/       policy, value, belief, and Task-Q networks
subgames/       complete-auction environment and self-play trainer
utils/          scoring, DDS loading, features, statistics, and FSP utilities
models/         model manifest and download instructions
results/        bridge_coma_results.xlsx
```

Development tests, debugging scripts, server controllers, training logs,
optimizer states, and experiment chronology are intentionally excluded.

## Methods and historical labels

| Public name | Historical label | Reward definition |
|---|---|---|
| Task-only | A | terminal duplicate-IMP task reward |
| Team-BG | B | task reward plus teammate belief gain |
| DUAL-local | Legacy-C | local teammate gain and opponent leakage penalty |
| DAPS | Strict-C | actor-time potential-difference formulation |

## Installation

Python 3.10 or later is recommended. Create a virtual environment, activate it,
and install the dependencies:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

OpenSpiel supplies the 571-dimensional bridge auction observation. The
`endplay` package is needed only to generate new DDS data.

## Download and verify the models

Download these three assets from the repository's Releases page:

```text
round100_models_seed42.zip
round100_models_seed43.zip
round100_models_seed44.zip
```

Extract them into the following directories:

```text
models/round100/seed42/
models/round100/seed43/
models/round100/seed44/
```

Each file contains only four actor state dictionaries and the architecture
metadata needed for inference. It does not contain a critic, optimizer,
BeliefNet, FSP pool, random-number state, or training-resume state. Verify file
integrity against `models/MANIFEST.csv`.

Inspect one model without running an evaluation:

```bash
python -m experiments.inspect_model \
  models/round100/seed42/seed42_a_round100_actor_only.pt
```

## Evaluation data

Evaluation requires a DDS dataset in either of these formats:

- a compact `.npz` file containing `decks` with shape `(N, 52)` and `tricks`
  with shape `(N, 5, 4)`; or
- a memmap dataset directory accepted by `utils.dds_data.create_loader`.

The `decks` array stores the owner of every suit-major card as an integer from
0 to 3. The `tricks` array stores double-dummy tricks by strain and declarer.
Datasets are not included because they can be regenerated and are substantially
larger than the source release.

## Run a paired comparison

The following command compares Team-BG against Task-only on the same sampled
deals. The reported orientation is the second model minus the first model.

```bash
python -m experiments.evaluate_resume_ab \
  --agent-a models/round100/seed42/seed42_a_round100_actor_only.pt \
  --agent-b models/round100/seed42/seed42_b_round100_actor_only.pt \
  --data data/evaluation \
  --deals 1000 \
  --seed 2026072401 \
  --output local_evaluation.json \
  --cpu
```

Remove `--cpu` to use CUDA when available. Increase `--deals` only after the
small command completes successfully.

## Run a four-policy matrix

```bash
python -m experiments.evaluate_joint_round_robin \
  --agent Task-only=models/round100/seed42/seed42_a_round100_actor_only.pt \
  --agent Team-BG=models/round100/seed42/seed42_b_round100_actor_only.pt \
  --agent DUAL-local=models/round100/seed42/seed42_legacy_c_round100_actor_only.pt \
  --agent DAPS=models/round100/seed42/seed42_strict_c_round100_actor_only.pt \
  --data data/evaluation \
  --deals 1000 \
  --seed 2026072401 \
  --output-dir local_matrix \
  --cpu
```

Every comparison is evaluated on common sampled support. The output records the
comparison orientation explicitly.

## Read the published results

Open `results/bridge_coma_results.xlsx`. It contains:

- the reported three-seed round-100 comparisons;
- formula-driven round-120 endpoint matrices and supervised-anchor results;
- the underlying paired round-120 IMP vectors; and
- the model file sizes and SHA-256 hashes.

A confidence interval containing zero is inconclusive. It is not evidence that
two policies are equivalent. Comparisons should normally use policies from the
same training round and the same evaluation support.

## Training code

`experiments.main_experiment` is the unrestricted complete-auction training
entry point. A full training run additionally requires training/evaluation DDS
data and supervised actor and belief bootstrap checkpoints. Those large
bootstrap artifacts are not part of this compact thesis release. Run

```bash
python -m experiments.main_experiment --help
```

to inspect the complete configuration interface.

## Reproducibility notes

- Input observations have 571 dimensions and use the OpenSpiel-compatible
  bridge representation.
- The public action space has 38 outputs: Pass, Double, Redouble, and 35 ordered
  contract calls.
- Model files use action mapping `openspiel_native_52_89_v1`.
- Round-100 models cover seeds 42, 43, and 44 for all four methods.
- Published model files are inference artifacts, not resumable training files.

