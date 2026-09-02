# Checkpoint-aligned proxy-diagnostic data

This directory contains the retained data behind the thesis checkpoint-aligned
proxy-diagnostic figure (`fig:aligned-proxy-v2`).

The plotted data are in `aligned_diagnostic_summary.csv` and
`aligned_diagnostic_summary.json`. They contain 54 records:

- three training runs (42--44);
- rounds 100 and 120;
- Team-BG, DUAL-local, and DAPS versus Task-only; and
- teammate gain, opponent gain, and secrecy difference.

The display term `teammate_gain` maps to `partner_gain` in the unchanged raw
JSON/CSV schema. Historical `seed` fields identify training runs; the separate
evaluation seed identifies the diagnostic sampling.

Two naming notes for anyone reading these records against the dissertation.
The quantity stored as `secrecy_difference` is called the *audience difference*
there; the schema name is historical and is retained unchanged, but nothing in
this setting is hidden from the opponent, so it should not be read as a secrecy
measure. And `opponent_gain` refers to one seat: the sender's left-hand
opponent, `(sender + 1) % 4`, who acts next. The right-hand opponent is never
scored and no field aggregates both, so these values bound exposure to one
specified observer rather than total opponent exposure. Note also that this is
not the opponent the deployed actor models internally, whose belief head is
supervised on the partner and the right-hand opponent.

Each record reports the paired comparison-minus-Task-only mean, standard
deviation, standard error, and normal-approximation 95% interval over 1,000
common evaluation instances. The evaluation seed is `20260714`.

The 18 `seed*_1000.json` files retain the full aggregate diagnostic output for
each comparison, including stratum summaries, frozen-judge quality statistics,
and calibration summaries. `judge_reliability.csv` provides the extracted
reliability-bin data. `PROTOCOL.md` was frozen before the formal diagnostic
outputs were inspected. The interpretation of these estimates is given in the
dissertation rather than here, so that a single corrected account governs and
this directory cannot drift away from it.

`PROTOCOL.md` names the original experimental inputs by their paths in the
working project, not by their location in this release, so those paths will not
resolve here. One of the objects it names is nevertheless included: the frozen
judge it calls `results/sl_base_bca.pt` is released as `data/sl_base_bca.pt`,
having been placed with the other training inputs. Together with the released
round-100 actors and a DDS source, that checkpoint allows the belief-gain
scores summarised in this directory to be recomputed rather than taken on
trust.

The experiment JSON files do not contain the 1,000 individual per-deal vectors.
The public release therefore provides the exact stored estimates used in the
figure without claiming or reconstructing unavailable raw observations.
