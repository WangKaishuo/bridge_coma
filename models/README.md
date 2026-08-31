# Round-100 models

The twelve inference-only models are distributed in three GitHub Release
archives, one per training seed. Extract each archive into
`models/round100/seed<seed>/` and verify the files against `MANIFEST.csv`.
The archive-level sizes and hashes are recorded in `ASSETS.csv`.

Each `.pt` file contains these keys:

```text
actor_n, actor_e, actor_s, actor_w
obs_dim, hidden_dim
actor_belief_conditioned, actor_belief_hidden_dim
action_mapping_version
```

The files intentionally omit all training-resume state.
