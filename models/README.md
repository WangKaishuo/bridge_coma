# Round-100 models

The twelve inference-only round-100 models are stored directly in this
directory. Verify their sizes and SHA-256 hashes against `MANIFEST.csv`.

Each `.pt` file contains these keys:

```text
actor_n, actor_e, actor_s, actor_w
obs_dim, hidden_dim
actor_belief_conditioned, actor_belief_hidden_dim
action_mapping_version
```

The files intentionally omit all training-resume state.
