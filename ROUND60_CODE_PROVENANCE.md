# Round-60 code archive provenance

This branch preserves the source tree used for the formal learning-rate
`1e-5`, seed-42 run that produced the round-60 checkpoints under
`results/main_lr1e-5_seed42` on 2026-07-16/17.

The training server did not contain Git.  The runnable snapshot was therefore
reconstructed from the exact deployment artifacts retained on both the local
workstation and the training server, in their original application order:

1. `cloud_benchmark_bundle.tar.gz`
2. `.codex_safe_patch.tar.gz`
3. `.codex_profile_patch.tar.gz`
4. `.codex_optimization_patch.tar.gz`
5. `.codex_final_optimization.tar.gz`
6. `.codex_round30_gate.tar.gz`
7. Source and launch scripts created before the DRI work began (cutoff:
   2026-07-17 06:30 UTC / 07:30 Europe/London)

The server-side `README.md` retained from the completed run is included.  DRI,
Task-Q, direct-counterfactual, and later auxiliary-update code is intentionally
absent: it was created only after this run and is not part of the archived
training implementation.

Important run entry points are:

- `experiments/main_experiment.py`
- `scripts/run_a100_b60_lr1e5_seed42.sh`
- `scripts/run_main_ab_memory_safe.sh`
- `scripts/gate_main_ab_round30.sh`

Model checkpoints and large datasets are not stored in Git.  This branch is a
source-code rollback point; the round-60 checkpoints remain in the experiment
artifact storage.
