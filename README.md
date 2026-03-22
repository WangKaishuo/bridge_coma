# Bridge-COMA

**Dual-Information Credit Assignment for Cooperative-Competitive Multi-Agent Coordination**
MSc Research Project — Kaishuo Wang, 2026

$$r_{\text{info}} = I(\text{bid};\,\text{hand} \mid \text{partner}) - \beta \cdot I(\text{bid};\,\text{hand} \mid \text{opponent})$$

---

## ⚠️ CRITICAL WORKFLOW NOTES (read at start of every session)

1. **Claude has NO cross-session memory.** Paste this README at the start of each new conversation.
2. **NEVER use `/mnt/project/` as base for edits.** That directory is the version last manually uploaded by Titus and may be several patches behind. Always base edits on the most recent file in `/home/claude/` or `/mnt/user-data/outputs/`.
3. **Config lives in `subgame_validation.py`, not `subgame_trainer.py`.** SubgameConfig kwargs in `subgame_validation.py` override all defaults. Always edit `subgame_validation.py` for hyperparameter changes.
4. **Five files to keep in sync:** `subgame_trainer.py`, `subgame_validation.py`, `competitive_env.py`, `belief_net.py`, `fsp_pool.py`.
5. **belief_net.py and hand_features.py were rewritten in P86.** Do NOT reference old versions with pos_weight=3.0/7.0 or unified BCE loss.

---

## Project Progress

### Phase 1: Environment & Infrastructure ✅ Complete

### Phase 2: Subgame Validation 🔄 In Progress

#### Stayman ⏸ Deferred — null result structurally expected

#### Competitive Subgame 🔄 Active

| Item | Status |
|------|--------|
| Env + DDS data (500k deals) | ✅ |
| P54–P86: Core infrastructure + Belief Net rewrite | ✅ |
| P87b: r_info weight 0.02→0.2 | ✅ |
| P88: KL anneal 0.5→0.0, first B>A result | ✅ |
| **P93: Dealer eval bug fix** | ✅ |
| **P93: Belief Net on-policy update (replaces replay buffer)** | ✅ |
| **P93: Corrected eval (eval_paired.py)** | ✅ |
| P93 experiment run (P88 config + on-policy belief) | 🔄 In progress |
| Multi-seed validation (3 seeds) | ⏳ After P93 confirmed |

---

## ⚠️ P93 CRITICAL BUG FIX: Dealer Encoding in Eval

### The Bug
`play_mixed()` and `dds_oracle_evaluate()` in `competitive_env.py` never updated `env.dealer` (i.e. `self.dealer`). Policy closures read `env.dealer` for `encode_obs_flat()`, which uses `dealer` to compute `caller = (dealer + step_idx) % 4` — determining which player made each bid in the `who_called` matrix.

**Training was NOT affected** — `_collect_episodes_batch` uses independent `envs[i]` with correct per-slot `slot_dealer[i]`.

**All eval H2H numbers from P82–P92 were wrong** — `cross_evaluate()`, `evaluate_head_to_head()`, and `dds_oracle_evaluate()` all had incorrect dealer encoding, systematically underestimating agent strength.

### The Fix (competitive_env.py)
```python
# play_mixed(): line 542
self.dealer = dealer  # P93 fix: policy closures read env.dealer

# dds_oracle_evaluate(): line 667
env.dealer = dealer   # P93 fix
```

### Impact
- vs SL IMP jumped from ~+1 to ~+7–8 (agents were always stronger than we thought)
- A vs B **direction may have been wrong** in some experiments
- All historical eval numbers must be re-verified with `eval_paired.py`

---

## Experimental Results History (Corrected)

### P88 — Re-evaluated with dealer fix (eval_paired.py, 5000 deals)

**Config**: 20 rounds, λ: 0.5→0.0 over 50%, pool=10, interval=2

| Matchup | IMP | p-value | |
|---------|-----|---------|--|
| A vs SL | **+6.176** | 0.000 ✅ | |
| B vs SL | **+8.065** | 0.000 ✅ | |
| A vs B  | **-3.181** | 0.000 ✅ | **B wins** |
| Paired (A_SL - B_SL) | **-1.889** | 0.000 ✅ | **B stronger vs SL** |

**This is the strongest positive result.** r_info produces both internal superiority (B>>A) and external generalization (B vs SL > A vs SL by 1.9 IMP).

### P92 (self-play, λ=0 全程) — Re-evaluated with dealer fix

**Config**: 30 rounds, λ=0.0 throughout, no FSP (self-play only)

| Matchup | IMP | p-value | |
|---------|-----|---------|--|
| A vs SL | +7.071 | 0.000 ✅ | |
| B vs SL | +7.388 | 0.000 ✅ | |
| A vs B  | +0.075 | 0.173 (ns) | No difference |
| Paired (A_SL - B_SL) | -0.317 | 0.010 | Marginal |

**r_info had no meaningful effect.** Cause identified: Belief Net degraded catastrophically during training — length accuracy fell from 0.488 (pretrain) to 0.230 (final), making r_info essentially random noise.

### Key Comparison: Why P88 > P92

| Factor | P88 (B>A ✅) | P92 (B≈A ❌) |
|--------|-------------|-------------|
| KL schedule | 0.5→0.0 over 50% | 0.0 throughout |
| Opponent | FSP pool (10 checkpoints) | Self-play only |
| Rounds | 20 | 30 |
| Final NS entropy | ~0.52 | ~0.37 |
| Belief Net health | Maintained (slow drift) | Collapsed (length 0.23) |
| Diagnosis | KL stabilizes early training → belief net learns good foundation → r_info effective in later rounds | No KL → immediate drift → belief net can't track → r_info = noise |

### Pre-P93 eval numbers (INVALID — dealer bug)

All H2H numbers in previous README versions (P88: A vs SL = +2.4, etc.) were **systematically wrong** due to the dealer encoding bug. Only the corrected numbers above should be used.

---

## P93: Belief Net On-Policy Update

### Problem
Belief Net was trained via JIT burn-in using a 50k FIFO replay buffer. As policy drifted (especially with low/no KL), buffer contained stale data from earlier strategies. This poisoned belief estimates — analogous to training a Critic on off-policy returns.

### Solution
Treat Belief Net like a Critic: train on current round's on-policy rollout data only, continual learning on previous weights.

- **After** both tables' PPO updates, extract belief data from `ns_eps + ew_eps`
- Train 8 epochs with early stopping (patience=2), 90/10 train/val split
- LR = 5e-5 (lower than pretrain's 1e-4 for smooth continual learning)
- No replay buffer — data from current round only, discarded after use
- Network weights carry forward (not reset) — low-frequency knowledge preserved in weights

### Why Not Other Approaches
- **Replay buffer (old method)**: Stale data poisons belief when policy drifts
- **Joint training (Rong et al.)**: Gradient interference between belief loss and PPO loss
- **SL anchor buffer**: Only safe when KL keeps policy near SL; contradictory when λ→0
- **On-policy (chosen)**: Matches Critic training paradigm; no distribution shift by construction

---

## Architecture

### Actor: `MLPPolicyNetwork` (301 dims → 4×1024 MLP → 38 logits)
```
hand(52) + history_flat(152, who-made-it) + position(4) + vulnerability(2) + dealer(1) = 301
```
Note: `encode_obs_flat(obs, dealer, history_int)` uses `dealer` to compute
`caller = (dealer + step_idx) % 4` for the `who_called` matrix. **Getting dealer wrong
causes catastrophic input corruption** (P93 bug).

### Critic: `MLPValueNetwork` (CTDE, 509 dims)
```
flat_obs(301) + all_hands(4×52=208) = 509
```

### Belief Network (P86, 268 dims → shared 2×512 trunk → dual head)
```
Input: observer_hand(52) + history_flat(152) + pos_embed×2(64) = 268

Honor head: trunk → Linear(512, 16) → sigmoid → calibrated P(honor)
  Loss: BCEWithLogitsLoss (NO pos_weight)

Length head: trunk → Linear(512, 32) → softmax(per suit, 8 bins) → P(length)
  Loss: CrossEntropyLoss (8-class per suit)
```

### HAPPO: 8 independent networks — `actor_n/s/e/w` + `critic_n/s/e/w`

---

## Training Pipeline (P93, current)

### Stage 1: SL Initialization
Load `results/sl_base.pt` (9.9M SAYC deals, 4 actors with identical weights).

### Stage 1.5: Belief Net Pretrain (Agent B only)
10k deals → train to convergence (~300 epochs). No replay buffer seeding (P93: removed).

### Stage 2: RL Fine-tuning (20 rounds)

Each round:
1. FSP pool: sample checkpoint as opponent (SL is permanent member)
2. **Table 1 (NS)**: collect 32768 deals, agent=NS, FSP=EW → r_info bonus → PPO update N+S
3. **Table 2 (EW)**: collect 32768 deals, agent=EW, FSP=NS → r_info bonus → PPO update E+W
4. **On-policy Belief Update (P93)**: train belief net on this round's rollout data (ns_eps + ew_eps), 8 epochs, early stopping
5. **Mini eval**: vs SL H2H (1000 deals)

**Reward**: `score_to_imp(score - dds_optimal)`, terminal only.

**r_info (P87b)**: Dynamic normalization. With w=0.2, step_ir ≈ 0.9–1.4 IMP.

**KL schedule**: λ linearly 0.5→0.0 over first 50% of rounds (P88 config).

### Stage 3: Evaluation
- A vs SL, B vs SL, A vs B (1000 deals each, Wilcoxon)
- **Paired eval** (`eval_paired.py`): same 5000 deals for all 3 matchups, paired Wilcoxon on (A_SL - B_SL)
- Belief Net quality (honor/length accuracy)

### Hyperparameters (P93, set in `subgame_validation.py`)

| Param | Value | Note |
|-------|-------|------|
| lr | 3e-6 | Kita et al. |
| kl_lambda_start / end | 0.5 / 0.0 | P93: restore P88 anneal |
| kl_anneal_frac | 0.5 | Over first 50% of rounds |
| deals_per_step | 512 | |
| steps_per_phase | 64 | → 32768 deals/table/round |
| num_rounds | 20 | P93: restore P88 |
| beta (internal) | 0.05 | I(partner) - β·I(opponent) |
| info_reward_weight | 0.2 | P87b |
| fsp_pool_size | 10 | 1 permanent SL + 9 FIFO |
| fsp_add_interval | 2 | |
| belief_update_epochs | 8 | P93: on-policy, early stop patience=2 |
| belief_update_lr | 5e-5 | P93: lower than pretrain 1e-4 |

---

## Running the Experiment

```bash
# P93: Train Agent B only (load pre-trained Agent A)
python experiments/subgame_validation.py \
    --data data/competitive_500k.npz \
    --sl_checkpoint results/sl_base.pt \
    --load_agent_a results/competitive/agent_a_seed42.pt \
    --seed 42

# Paired eval (corrected, same deals for all matchups)
python eval_paired.py \
    --data data/competitive_500k.npz \
    --sl_checkpoint results/sl_base.pt \
    --agent_a results/competitive/agent_a_seed42.pt \
    --agent_b results/competitive/agent_b_seed42.pt \
    --num_deals 5000
```

### Key log indicators
```
[Round N] regret=+3.80±6.65  (NS=+3.99 EW=+3.65)  fsp=10
  regret: mean DDS IMP regret vs FSP pool (relative, not absolute)

NS │ N: pl=-0.015 vl=6.8 ent=0.66 │ S: ... kl=0.97(λ=0.250)
  ent: healthy range 0.5–0.8 for competitive bidding
  kl: with λ annealing, expected to rise as λ decreases

r_info │ step_ir=0.93  belief_loss=1.79
  step_ir: ~0.9 IMP is the target range

[Belief Update] 45000 samples, 5 epochs, val_loss=1.8200
  P93: on-policy update. Watch val_loss — should track, not diverge.

[Head-to-Head] agent vs SL  n=1000  p=0.003
  PRIMARY convergence metric (now with correct dealer encoding)
```

---

## Full Bug Fix History

### Phase 1 (P0–P6)
P0 package · P1 termination · P2 reward · P3 eval · P4 scoring · P5 vulnerability · P6 dealer

### Phase 2 (P7–P93)

| # | Problem | Fix |
|---|---------|-----|
| P7–P53 | Various early issues | See previous README versions |
| P54–P77 | Competitive env infrastructure | Dealer rotation, dual-table, FSP, batch rollout |
| P82 | NS/EW asymmetry | Dual-table symmetric training |
| P83 | r_info drowns base reward | Separate beta vs info_reward_weight |
| P84 | Belief Net catastrophic forgetting | FIFO Replay Buffer (50k) — **superseded by P93** |
| P86 | Belief Net miscalibration | No pos_weight, CE for length, prior bias init |
| P87b | r_info signal invisible to PPO | info_reward_weight 0.02→0.2 |
| P88 | KL=1.5 locked agent at SL | KL anneal: 0.5→0.0 over 50% rounds |
| P89 | Pool=3 too homogeneous | Reverted to pool=10 |
| P90 | FSP evicts SL | SL as permanent FSP member |
| P91 | No absolute training metric | Mini DDS Oracle + vs SL eval every round |
| **P93** | **`play_mixed` / `cross_evaluate` / `dds_oracle_evaluate` used wrong `env.dealer`** | **`self.dealer = dealer` in `play_mixed`; `env.dealer = dealer` in oracle eval** |
| **P93** | **Belief Net trained on stale replay buffer data → length acc 0.49→0.23** | **On-policy update: train on current round's rollout only, no replay buffer** |
| **P93** | **Eval used different deals per matchup → no paired comparison** | **`eval_paired.py`: pre-sample deals, all matchups on same deals, paired Wilcoxon** |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| MLP+flat, no LSTM | Kita 2024: no benefit ≤12 tokens |
| FSP pool=10 + SL permanent | Opponent diversity prevents co-adaptation; SL anchor prevents forgetting |
| KL anneal 0.5→0.0 over 50% | Early stability for belief learning; late freedom for specialization |
| On-policy belief update (P93) | Belief Net = Information Critic; must track current policy like Value Critic |
| Dynamic r_info normalization | `(imp_std/rinfo_std) * w` auto-scales to IMP magnitude |
| eval_paired.py for H2H | Same deals across matchups enables paired statistical test |

---

## Key Lessons Learned

1. **Dealer encoding is catastrophic when wrong.** `encode_obs_flat` uses dealer to assign `who_called` — wrong dealer = wrong input for every bid in history. Always verify `env.dealer` matches the actual dealer in eval paths.
2. **Training data was correct all along.** `_collect_episodes_batch` uses independent envs with correct dealer. The bug was eval-only, meaning trained agents were always stronger than eval indicated.
3. **FSP pool diversity > self-play purity.** P88 (FSP, 20 rounds) produced B>>A; P92 (self-play, 30 rounds) produced B≈A. Multiple diverse opponents prevent co-adaptation and slow entropy collapse.
4. **KL anneal is not "constraining r_info" — it's stabilizing belief learning.** The first 10 rounds with λ=0.5→0.25 keep policy drift slow enough for belief net to build a robust foundation. r_info becomes effective in the second half when λ→0.
5. **Belief Net must be on-policy.** Stale replay data from earlier strategies has different bid→hand mappings. This poisons belief estimates, making r_info = noise. Treat belief net like a Critic.
6. **Paired eval is essential.** Unpaired eval (different deals per matchup) cannot detect 0.3 IMP differences. `eval_paired.py` with 5000 shared deals detected B > A at p=0.000 for P88.

---

## Project Structure

```
bridge-coma/
├── subgames/
│   ├── competitive_env.py      # P93: dealer fix in play_mixed + dds_oracle_evaluate
│   ├── subgame_trainer.py      # P93: on-policy belief update, no replay buffer
│   ├── subgame_validation.py   # P93: KL 0.5→0.0, 20 rounds, load_agent_a
│   └── action_mask.py
├── networks/
│   ├── policy_net.py           # 301-dim MLP, encode_obs_flat
│   └── belief_net.py           # P86: dual-head, no pos_weight, prior bias init
├── algorithms/
│   ├── mappo.py                # HAPPO: actor/critic ×4
│   └── ippo.py
├── utils/
│   ├── hand_features.py        # P86: Brier/NLL metrics
│   ├── fsp_pool.py             # P90: permanent member support
│   ├── sl_pretrain.py / scoring.py / imp.py
│   ├── dds_data.py / running_stats.py
│   └── generate_subgame_data.py
├── eval_paired.py              # P93: paired eval (same deals, paired Wilcoxon)
├── eval_vs_sl.py               # OLD standalone eval (unpaired, deprecated)
├── belief_diagnostic.py
├── results/competitive/
│   ├── agent_a_seed42.pt       # P88 Agent A (can be reused with --load_agent_a)
│   └── agent_b_seed42.pt
└── data/competitive_500k.npz
```

---

## Pending / Next Steps

1. **P93 experiment running**: P88 config + on-policy belief update. Only Agent B trains (Agent A loaded from P88 checkpoint). Watch belief_loss and length accuracy — should stay above 0.40.
2. **If B > A confirmed with healthy belief**: Multi-seed validation (seeds 42, 123, 456).
3. **If belief still degrades**: Consider lower belief_update_lr (1e-5) or fewer epochs.
4. **Paper narrative**: P88 corrected results are the centerpiece — r_info produces +1.9 IMP generalization advantage (paired, p=0.000). KL anneal + FSP diversity are necessary enabling conditions.

---

## Compute Budget (T4 GPU)

| Stage | Time |
|-------|------|
| SL pretrain (10 epochs) | ~20 min |
| Belief pretrain (300 epochs) | ~15 min |
| Agent A (20 rounds, if not loaded) | ~80 min |
| Agent B (20 rounds + on-policy belief) | ~90 min |
| Stage 3 eval (3×1000 deals) | ~5 min |
| eval_paired.py (5000 deals ×3) | ~10 min |

---

*README version: P93*
*Last updated: 2026-03-22*
