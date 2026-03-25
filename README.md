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
6. **Two SL pretrain files now exist:** `sl_pretrain.py` (original 301-dim SAYC, outputs `sl_base.pt`) and `sl_pretrain_bca.py` (P101: 397-dim BCA with `--init_from`, outputs `sl_base_bca.pt`). Do NOT conflate them.
7. **Paired eval is now built into `subgame_validation.py`** via `--eval-only` or `--skip_training`. New: `--eval_deals` controls Stage 3 deal count.
8. **BCA architecture uses `sl_base.pt` as foundation.** The 397-dim actor is created by 301→397 zero-init from `sl_base.pt`, then finetuned in `sl_pretrain_bca.py` with belief columns only trainable. Do NOT train 397-dim from scratch.
9. **P100: BCA is the standard baseline for ALL agents.** The only experimental variable is r_info configuration. Never compare BCA agents against non-BCA agents.
10. **P101: Actor input is 397-dim** = 301 (base obs) + 48 (partner belief) + 48 (RHO belief). Old 349-dim (partner-only) is superseded. All dimension references should say 397.

---

## Project Progress

### Phase 1: Environment & Infrastructure ✅ Complete

### Phase 2: Subgame Validation — In Progress

#### Stayman ⏸ Deferred — null result structurally expected

#### Competitive Subgame — P97d (301-dim) Complete, P101 (BCA 397-dim) In Progress

| Item | Status |
|------|--------|
| P54–P86: Core infrastructure + Belief Net rewrite | ✅ |
| P87b–P97d: 301-dim experiments (convention drift, r_info diagnostic) | ✅ |
| P98: BCA architecture design (349-dim, partner-only) | ✅ |
| P99: BCA debugging + SL pretrain fix | ✅ |
| **P100: BCA as standard baseline + 3-way fair test design** | ✅ |
| **P101: 397-dim actor (partner + RHO belief)** | ✅ Code complete |
| Retrain `sl_base_bca_v2.pt` (397-dim) | ⏳ Next |
| Convention drift sweep (5 λ × 5 seeds) | ⏳ After retrain |
| A vs B vs C experiment (5 seeds) | ⏳ After drift sweep |

---

## Current Architecture: P101 BCA (397-dim)

### Actor Input Structure

```
Actor input (397-dim) = base obs (301) + partner belief (48) + RHO belief (48)

[0:4]     vulnerability (one-hot, 4 vul combinations)
[4:56]    current player's hand (52 cards)
[56:196]  who_called (35 bids × 4 seats = 140)
[196:301] double_state (35 bids × 3 states = 105)
─── base obs boundary (301) ───
[301:317] partner honor probs (16: AKQJ × 4 suits, sigmoid)
[317:349] partner length probs (32: 8 bins × 4 suits, softmax)
─── partner belief boundary (349) ───
[349:365] RHO honor probs (16: AKQJ × 4 suits, sigmoid)
[365:397] RHO length probs (32: 8 bins × 4 suits, softmax)
─── total (397) ───
```

### Why Partner + RHO (not LHO)

- **RHO bid immediately before you** → maximum information relevance for decision-making
- **Information-theoretically complete**: Given self (13 cards known) + partner belief + RHO belief → LHO is fully determined: `P(card ∈ LHO) = 1 - P(self) - P(partner) - P(RHO)`
- **Degrees of freedom**: 4 players' hands have 2 DoF given your own hand. Partner + RHO exhaust all 2 DoF. LHO belief is mathematically redundant.
- **Matches human cognition**: Bridge players always process RHO's bid before deciding

### Key Design Principle

> "The extra weights are zero-initialised to ensure the 397-dim actor is initially equivalent to the 301-dim SL actor."

- Start from high-quality `sl_base.pt` (301-dim, 9.9M SAYC deals)
- Extend to 397-dim with belief columns zero-init
- Finetune with only belief columns trainable → actor learns to USE belief features
- RL stage then further optimises the full 397-dim policy

---

## P100: Experiment Design (3-Way Fair Test)

### Core Principle

BCA is **not** an experimental treatment — it is the **minimum capability** for any bridge agent. ALL agents use BCA. The only experimental variable is r_info.

### Agent Configuration

| Agent | BCA (397-dim) | r_info | β | Role |
|-------|---------------|--------|---|------|
| SL baseline | ✓ | ✗ | — | Reference convention (zero-drift anchor) |
| A: MAPPO+BCA | ✓ | ✗ | — | Does RL improve over SL with proper understanding? |
| B: MAPPO+BCA+r_info | ✓ | ✓ | 0.0 | Partner information incentive alone |
| C: MAPPO+BCA+r_info | ✓ | ✓ | 0.05 | Full Dual-Information (partner + opponent penalty) |

### Question Chain

| Matchup | Question |
|---------|----------|
| A vs SL | Does RL improve over SL when agents understand bids? |
| B vs A | Does partner information shaping help? |
| C vs A | Does full dual-information help? |
| C vs B | Does the β opponent penalty add value over partner-only? |
| B ≈ A | BCA alone captures what r_info teaches |

### Publishable Outcomes (Both Routes)

**Route 1 (B or C > A):** Three-act story: discover problem (drift) → diagnose (communication-outcome disconnect) → solve (BCA + r_info). Target: NeurIPS/ICML.

**Route 2 (B ≈ C ≈ A, all > SL):** Methodological contribution: convention drift quantification + BCA as standard baseline + negative result on r_info. Target: AAMAS/IEEE CoG.

---

## P97d Results (301-dim baseline, for comparison)

### Final results (λ=0.3 fixed, 10 rounds, 1000 deals eval)

| Matchup | IMP | p-value | |
|---------|-----|---------|--|
| A vs SL | +3.520 | 0.000 ✅ | |
| B vs SL | +3.846 | 0.000 ✅ | |
| A vs B (H2H) | +0.186 | 0.101 (ns) | No significant difference |

### P97d partner info gain (500 deals)

| Position | Agent A | Agent B | Δ |
|----------|---------|---------|---|
| N (opener) | 0.0087 | 0.0086 | -0.1% |
| S (responder) | 0.1335 | 0.1470 | +10.1% |
| Overall | 0.0844 | 0.0934 | **+10.8%** |

**Key finding**: r_info changes communication (+10.8%) but not outcomes (A≈B in IMP). This is the **communication-outcome disconnect** that motivates BCA.

---

## P99 Preliminary Results (BCA 349-dim, SL not yet using belief net)

### λ=0, 10 rounds, 1000 deals eval

| Matchup | IMP | p-value | |
|---------|-----|---------|--|
| A vs SL | +5.215 | 0.000 ✅ | |
| B vs SL | +5.563 | 0.000 ✅ | |
| A vs B (H2H) | +0.137 | 0.341 (ns) | No significant difference |

⚠️ **Caution**: SL baseline's belief columns ≈ zero (not genuinely using belief features). The +5 IMP vs-SL advantage likely includes architectural asymmetry. A vs B comparison is clean (both have same architecture).

---

## Current Training Pipeline (P101 BCA)

### Stage 0: BCA SL Pretrain (`sl_pretrain_bca.py`)
```bash
python sl_pretrain_bca.py \
    --train data/sayc_train.txt \
    --valid data/sayc_valid.txt \
    --out results/sl_base_bca_v2.pt \
    --init_from results/sl_base.pt \
    --epochs 30 --belief_epochs 30 --device cuda
```
Output: `sl_base_bca_v2.pt` — 397-dim actor (belief columns trained) + belief net (partner + opponent targets).

### Stage 1: Load BCA SL Checkpoint
Load `sl_base_bca_v2.pt` for A, B, C, and SL baseline. All start identical.

### Stage 1.5: Belief Net Pretrain (A, B, C independently)
100k competitive subgame deals, 50 epochs each. SL keeps its belief net from Stage 0.

### Stage 2: RL Fine-tuning
A, B, C trained sequentially per round. A: pure MAPPO. B: MAPPO + r_info (β=0). C: MAPPO + r_info (β=0.05).

### Stage 3: Full 6-way evaluation
A vs SL, B vs SL, C vs SL (vs reference). A vs B, A vs C, B vs C (pairwise). Partner info gain diagnostic for all agents.

### Commands

**Full 3-way experiment:**
```bash
python subgame_validation.py \
    --data data/competitive_500k.npz \
    --sl_checkpoint results/sl_base_bca_v2.pt \
    --rounds 15 --seed 42 \
    --beta 0.0 --beta_c 0.05
```

**Quick smoke test (A and B only):**
```bash
python subgame_validation.py \
    --data data/competitive_500k.npz \
    --sl_checkpoint results/sl_base_bca_v2.pt \
    --rounds 3 --seed 42 --quick --no_agent_c
```

**Legacy 301-dim mode:**
```bash
python subgame_validation.py \
    --no_belief_conditioned \
    --sl_checkpoint results/sl_base.pt \
    --data data/competitive_500k.npz \
    --seed 42
```

---

## Convention Drift Sweep (Independent Contribution)

Agent A config only, no r_info, pure MAPPO+BCA:

| λ_KL | Purpose |
|------|---------|
| 0.0 | Maximum drift (unconstrained RL) |
| 0.1 | Light constraint |
| 0.3 | Moderate constraint |
| 0.5 | Strong constraint |
| 1.0 | Near-SL behavior |

For each λ, measure DDS regret (opponent-independent), vs-SL IMP (opponent-dependent), and drift advantage = (vs-SL IMP) - (DDS regret). 5 seeds each. Produces the "convention drift Pareto frontier" figure.

---

## Key Lessons Learned (P94–P101)

1. **Convention drift is real and large.** ~2 IMP advantage from drift alone (P94b). Prior bridge AI papers do not account for this confound.

2. **KL anchor is not a training trick — it's protocol compliance.** Without it (301-dim), agents develop private conventions that violate Full Disclosure. With BCA, protocol compliance is enforced structurally through the Belief Net convention card.

3. **Belief Net pretrain overfitting was the hidden killer.** 10k deals × 300 epochs → severe overfitting. Fix: 100k deals × 50 epochs.

4. **r_info changes communication behavior** (partner_gain +10.8% at λ=0.3, 301-dim) **but does not improve IMP outcomes**. This is the **communication-outcome disconnect** motivating BCA.

5. **BCA SL pretrain must preserve base weights.** Training 397-dim from scratch produces actors far inferior to `sl_base.pt`. Correct approach: freeze base columns, only train belief columns with gradient masking.

6. **SL baseline must genuinely use belief features** for fair vs-SL evaluation. Zero-init belief columns = SL can't read belief net = architectural advantage for RL agents.

7. **`belief_warmup_rounds` must be 0 when belief net is already trained.** Using uninformative prior in early rounds creates OOD input distribution → entropy collapse.

8. **`_pretrain_replay` must be seeded** when loading belief net from checkpoint. Without replay, on-policy belief update causes catastrophic forgetting.

9. **Agents must understand opponent bids** (P101). Partner-only belief (349-dim) leaves agents unable to process opponent bidding meaning. RHO belief (48-dim) provides opponent understanding with zero redundancy (partner + RHO = 2 DoF, information-theoretically complete).

10. **BCA is infrastructure, not contribution.** All agents use BCA. The experimental variable is r_info only. This eliminates the architectural confound in P98-P99.

---

## Bugs Found and Fixed (P98–P101)

| # | Problem | Fix |
|---|---------|-----|
| P98 | BCA architecture design | 349-dim actor, belief features as input |
| P99a | `belief_warmup_rounds=2` default | Set to 0 when belief net already trained |
| P99b | Missing `_pretrain_replay` from BCA checkpoint | Seed shared replay from SL-policy rollouts |
| P99c | `sl_base_bca.pt` trained from scratch — bad quality | New `sl_pretrain_bca.py` with `--init_from sl_base.pt`, freeze base columns |
| P99d | SL baseline not using belief features | `sl_pretrain_bca.py` trains belief columns with gradient masking |
| P99e | KL anneal 0.3→0.0 in BCA mode | Use explicit `--kl_lambda 0.0` (justified by convention card argument) |
| P99f | Stage B lr=1e-4 destroys base weights | Freeze all except belief columns, use lr=3e-4 with gradient mask |
| P100 | A vs B confounded BCA with r_info | All agents use BCA; only r_info varies |
| P101 | Agents cannot understand opponent bids | 397-dim: add 48-dim RHO belief features |

---

## Scientific Narrative for Thesis

### Three contributions

1. **Convention drift quantification** (novel): First quantitative evidence that RL self-play in bridge produces ~2 IMP illegitimate advantage through private conventions. Prior work (JPS NeurIPS 2020, Kita CoG 2024, Qiu 2024) did not quantify this confound.

2. **Protocol compliance as constrained optimization** (novel framing): Full Disclosure formalized as `max J(π) s.t. D_KL(π‖π_SL) ≤ ε`. In 301-dim: KL anchor as Lagrange multiplier. In BCA: Belief Net as structural convention card, enabling λ=0.

3. **Communication-outcome disconnect → BCA + r_info test** (core thesis claim): r_info modifies communication (+10.8%) but not outcomes in 301-dim. BCA closes the perception-action loop. Whether r_info provides incremental value under BCA is the key experimental question.

### Open questions

1. Does BCA + r_info produce B>A in IMP? (Core hypothesis)
2. Does the β opponent penalty (Agent C) add value over partner-only (Agent B)?
3. Does BCA reduce convention drift compared to 301-dim at same λ?
4. Does the Belief Net convention card provide genuine Full Disclosure compliance?
5. FSP opponents don't use belief net — does this limit training quality?

---

## Project Structure

```
bridge-coma/
├── env/
│   ├── bridge_bidding_env.py
│   └── dual_table_env.py
├── networks/
│   ├── policy_net.py               # MLPPolicyNetwork (301/397-dim), encode_obs_flat
│   └── belief_net.py               # P86: dual-head; P97: optional EWC
├── algorithms/
│   ├── mappo.py                    # HAPPO: actor/critic ×4
│   ├── ippo.py
│   └── behavioral_cloning.py
├── utils/
│   ├── scoring.py, imp.py, dds_data.py, running_stats.py
│   ├── hand_features.py, fsp_pool.py
│   ├── sl_pretrain.py              # Original 301-dim SAYC SL → sl_base.pt
│   ├── sl_pretrain_bca.py          # P101: 397-dim BCA SL → sl_base_bca_v2.pt
│   └── generate_subgame_data.py
├── subgames/
│   ├── stayman_env.py, competitive_env.py, action_mask.py
│   ├── subgame_trainer.py          # P101: dual-query belief (partner + RHO)
│   └── subgame_validation.py       # P100: 3-way test; P101: 397-dim
├── experiments/
│   └── train.py
├── tests/
│   └── test_all.py
├── results/
│   ├── sl_base.pt                  # 301-dim SAYC baseline (9.9M deals)
│   └── sl_base_bca_v2.pt           # 397-dim BCA baseline (partner + RHO belief)
└── data/
    ├── competitive_500k.npz
    ├── sayc_train.txt              # OpenSpiel SAYC dataset (1.16M games)
    └── sayc_valid.txt              # OpenSpiel SAYC test set (10k games)
```

---

## Compute Budget (T4 GPU)

| Stage | Time |
|-------|------|
| `sl_pretrain_bca.py` Stage A: Belief Net (200k games, 30 epochs, 3× data with opponent targets) | ~40 min |
| `sl_pretrain_bca.py` Stage B: 397-dim Actor finetune (2× belief inference for partner+RHO) | ~35 min |
| Belief pretrain per agent (100k deals, 50 epochs) | ~30 min |
| SL baseline belief pretrain (100k deals, 50 epochs) | ~30 min |
| Agent A (10 rounds) | ~40 min |
| Agent B (10 rounds, belief update + r_info) | ~45 min |
| Agent C (10 rounds, belief update + r_info + β) | ~45 min |
| Stage 3 eval (6-way × 5000 deals) | ~40 min |
| **Convention drift sweep (5 λ × 5 seeds × 10 rounds)** | **~24 hours** |
| **Full 3-way experiment (3 configs × 5 seeds × 15 rounds)** | **~18 hours** |

---

## Dimension Quick Reference

| Mode | Actor dim | Belief features | Checkpoint |
|------|-----------|----------------|------------|
| Legacy (no BCA) | 301 | None | `sl_base.pt` |
| P98-P99 (partner only) | 349 | 48 (partner) | `sl_base_bca.pt` (deprecated) |
| **P101 (partner + RHO)** | **397** | **96 (48 partner + 48 RHO)** | **`sl_base_bca_v2.pt`** |

---

*README version: P101 (397-dim partner + RHO belief)*
*Last updated: 2026-03-25*
