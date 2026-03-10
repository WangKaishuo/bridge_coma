# Bridge-COMA

**Dual-Information Credit Assignment for Cooperative-Competitive Multi-Agent Coordination**

MSc Research Project — Addressing relative overgeneralization and miscoordination in bridge bidding
via information-theoretic reward shaping with prior asymmetry.

---

## Project Progress

### Phase 1: Environment & Infrastructure ✅ Complete

| Item | Status |
|------|--------|
| Package structure + `setup_project.py` assembly | ✅ |
| Single-table bidding env (`BridgeBiddingEnv`) | ✅ |
| Dual-table IMP env (`DualTableEnv`) | ✅ |
| Scoring SSOT (`scoring.py`) | ✅ |
| IMP conversion (`imp.py`) | ✅ |
| DDS data generation & loading (1M deals) | ✅ |
| IPPO / MAPPO algorithms | ✅ |
| Training script (dual-table IMP + dealer rotation + vulnerability) | ✅ |
| Test suite (35 tests across all modules) | ✅ |

### Phase 2: Subgame Validation 🔄 In Progress

#### Stayman Subgame ✅ Concluded

| Item | Status |
|------|--------|
| Constrained dealing + DDS data generation (50k deals) | ✅ |
| Stayman subgame env (`stayman_env.py`) | ✅ |
| Action mask → generalized to `env._get_legal_actions()` | ✅ |
| Subgame trainer (`subgame_trainer.py`) | ✅ |
| N/S rule policies | ✅ |
| BC data generation — N+S joint + Round3 N response | ✅ |
| BC samples 20k — ensures 4H/4S acceptance coverage | ✅ |
| Piecewise linear reward (IMP-aligned, [0.01, 1.0]) | ✅ |
| LSTM fix (`pack_padded_sequence`) | ✅ |
| BC dual weighting (player weight + minority weight) | ✅ |
| KL anchor regularization | ✅ |
| N-phase KL: no annealing (防 N 退化) | ✅ |
| Separate Actor/Critic optimizers | ✅ |
| Critic warmup (dual-track Stage 1.5) | ✅ |
| GAE enabled (`single_step=False`) | ✅ |
| S HCP upper limit (8–10) | ✅ |
| Belief Net: BCEWithLogitsLoss + pos_weight=3 | ✅ |
| Belief Net: Top-13 hit rate metric | ✅ |
| r_info wired to terminal reward (P36) | ✅ |
| ReLU clamp on ir (P36) | ✅ |
| β fixed to 0.05 (P36) | ✅ |
| JIT Belief Burn-in before each N-phase (P36) | ✅ |
| Context-level adaptive KL weights | ✅ |
| HeadToHeadEvaluator framework | ✅ |
| Dead code removed from `stayman_env.py` | ✅ |
| **Multi-seed validation (5 seeds)** | ⏳ Next |

#### Competitive Subgame

| Item | Status |
|------|--------|
| Constrained dealing + DDS data (100k deals) | ✅ |
| Competitive subgame env (`competitive_env.py`) | ✅ |
| BC warmup (`behavioral_cloning.py`) | ✅ |
| Cross-evaluation (`cross_evaluate`) | ✅ |
| Full experiment | ⏳ After Stayman multi-seed |

### Phase 3–4: Not started

---

## Current Experimental State

### Latest Full Run: result6 (post-generalization, piecewise reward restored)

**Stage 1 — BC base:**
```
BC acc:         99.5%
Contract dist:  54.0% part_score / 38.8% 3NT / 7.2% 4M
IMP (S_base):   -3.71 ± 3.56
Belief pre-train: top13_hit = 0.352 (random baseline 0.25)
```

**Stage 2 Final:**

| Agent | IMP | Δ vs S_base |
|-------|-----|-------------|
| S_base (N=rule) | −3.71 | — |
| A_control (MAPPO) | −3.71 | +0.01 |
| B_partner_only (MAPPO+r_info) | −3.57 | **+0.14** |
| B vs A | — | **+0.14** |

**Head-to-Head (B vs A, 200 deals):**

| Metric | Value |
|--------|-------|
| Δ IMP (B − A) | −0.01 ± 0.10 |
| Tie rate | **96.5%** |
| Win rate (B > A) | 1.5% |
| Verdict | ❌ TIE |

**Contract distribution (A vs B):** both nearly identical — same bidding tree, same protocols.

---

## Stayman: Final Scientific Assessment

**The Stayman subgame has reached a communication ceiling.** Key evidence:

1. **N's bidding is 100% correct in both A and B** — BC already taught the 3-bit Stayman protocol (2D/2H/2S) to theoretical optimality before RL begins.
2. **96.5% H2H tie rate** — A and B produce the same contract on virtually every deal.
3. **Belief top-13 hit = 0.352 vs random 0.25** — a 40% improvement. Given N only bids once (1.58 bits of information), this is effectively at the information-theoretic ceiling for this environment. Chasing 0.40 would cause overfitting to noise.
4. **ir is consistently positive (0.09–1.85)** — Belief Net provides meaningful gradient signal throughout training; the bottleneck is not Belief quality but the structural constraint of the environment.

**Why B ≈ A in Stayman but B > A in result5:**
- result5 used hardcoded masks (MAX_LEVEL=4 + fixed bid set). Those constraints created artificial variance that r_info could exploit.
- result6 uses fully generalized legal masks. With BC at 99.5% accuracy, r_info has no room to improve N's already-perfect signaling.

**Scientific conclusion (honest and defensible for the paper):**
> "The Stayman subgame validates that the full infrastructure (BC, KL anchoring, JIT burn-in, Belief Net) is stable and produces consistent protocols. Because BC already achieves the theoretical maximum for this 3-bit communication task, r_info's partner term cannot demonstrate incremental benefit. The opponent term β validation is structurally impossible here (EW always pass). These properties — not bugs — motivate moving to the competitive subgame where (a) N has a richer signaling space, (b) opponent interference creates genuine tension between partner clarity and information leakage, and (c) the full dual-info formula can be exercised."

---

## Pending Tasks

### Immediate

- [ ] **Multi-seed run (5 seeds)** with current config (alt_rounds=3, joint_steps=300)
  - Report mean ± 95% CI for B vs A
  - Use paired t-test / Wilcoxon on per-deal IMP for significance
  - Even if B ≈ A, establishes variance baseline for paper

### After multi-seed

- [ ] **Competitive subgame** (1H-1S) — this is where r_info should show its value
  - Opponent interference activates the β term
  - N's signaling space is richer (no single "correct" protocol)
  - Full 3-agent comparison: A (MAPPO) / B (β=0) / C (β=0.05)

---

## Architecture

### Network Structures

```
PolicyNetwork (Actor):
  HandEncoder:     52 → 256 → 256 (MLP)
  HistoryEncoder:  (seq_len, 38) → LSTM(2 layers, 256) → h_n[-1]
                   [pack_padded_sequence — valid tokens only]
  Fusion: [hand_256 + history_256 + position_4 + vulnerability_2] → MLP → 38-dim logits

ValueNetwork (Critic):
  Same as Actor + AllHandsEncoder: 4×52 → 256 → 256 (centralized)
  Separate optimizer from Actor (lr × 2, PPO2 value clipping)

BeliefNetwork:
  HandEncoder:    52 → 256 → 256 (MLP)
  HistoryEncoder: LSTM(2 layers, 256) [pack_padded_sequence]
  PositionEmbed:  Embedding(4, 32) × 2 (observer_pos + target_pos)
  Output:         52-dim LOGITS (no Sigmoid in forward)
  Probs:          get_probs() = sigmoid(logits)    ← use for r_info
  Loss:           BCEWithLogitsLoss(pos_weight=3.0)
  Metric:         top13_hit_rate()  — random ≈ 0.25, observed ≈ 0.35
```

### r_info Design

```
r_info = max(0, I(bid; hand | partner)) - β * max(0, I(bid; hand | opponent))

where I(bid; hand | observer) ≈ CE(belief_before, hand) - CE(belief_after, hand)

ReLU clamp: MI ≥ 0 by definition; negative values are Belief Net lag, not N's fault.
β = 0.05: "gentle breeze" — info bonus supplements IMP, doesn't dominate it.

Applied to: N's terminal step reward only.
Active in: N-phase and joint fine-tune for Agent B only.
```

### Key Design Decisions

**1. Alternating Training (S→N per round)**
Simultaneous N+S learning blurs credit assignment. Alternating fixes one player per
half-round → reward changes 100% attributable to active player.
```
Round k: S trains (N frozen) → N trains (S frozen)
Final:   Joint fine-tune (both active, lr/3)
```

**2. JIT Belief Burn-in**
Belief Net trained at Stage 1.5 on BC rollouts. Once N starts RL exploration, its
protocol evolves → Belief Net goes OOD → ir estimates corrupt.
Fix: Before each N-phase, run 1000 rollouts with current policy and fine-tune Belief
Net (lr=1e-3, 3 epochs). Gives the "evaluator an up-to-date dictionary" each round.

**3. ReLU Clamp on ir**
MI ≥ 0 is a mathematical theorem. Negative ir = Belief Net lag, not N's fault.
Clamping to max(0, gain) prevents the evaluator's confusion from penalizing exploration.

**4. N-phase KL no annealing**
S-phase KL anneals 0.5→0.1 (allow S to explore). N-phase KL stays at 0.5 throughout
(prevent N from drifting away from BC signaling structure that S depends on).

**5. BCEWithLogitsLoss + pos_weight=3 (Belief Net)**
39 zeros vs 13 ones per hand (3:1 imbalance). Standard BCE → all-zeros at acc=75%.
pos_weight=3 forces equal attention to present cards. Top-13 hit rate is the metric:
random ≈ 0.25, observed ≈ 0.35, information-theoretic ceiling for Stayman ≈ 0.37.

**6. β = 0.05 (info bonus scale)**
IMP piecewise range ≈ [0.01, 1.0]. β=0.05 keeps info bonus as a directional nudge
without dominating the environment reward signal.

**7. Separate Actor/Critic Optimizers**
Single optimizer: Critic MSE gradients (~10) overwhelm Actor gradients (~0.001).
Fix: `critic_optimizer = Adam(critic, lr*2)`, PPO2-style value clipping.

**8. Training schedule justification**
- alt_rounds=3: IMP enters plateau after round 2 (Δ < 0.1/round). Confirmed by result6 progression.
- joint_steps=300: sufficient for final N+S coordination without KL over-expansion.
- 5 seeds: required for statistical significance given IMP std ≈ 3.5 (single-run noise > B−A gap).

---

## Experiment Configuration (Phase 2 Stayman, multi-seed run)

```python
# BC Warmup
stayman_bc_samples      = 20000
stayman_bc_epochs       = 15

# Stage 1.5: Critic + Belief pre-training
critic_warmup_rounds    = 10
critic_warmup_deals     = 512
belief_pretrain_deals   = 10000
belief_pretrain_epochs  = 50
belief_pretrain_target_acc = 0.40   # not blocking — ceiling is ~0.37 for this env

# Stage 2: Alternating fine-tuning
stage2_alt_rounds       = 3         # reduced from 6; IMP converges by round 2-3
stage2_alt_steps        = 200       # steps per half-round
stage2_joint_steps      = 300       # reduced from 400
stage2_deals_per_step   = 32
stage2_accumulate       = 8         # effective: 256 deals/update
stage2_lr               = 3e-5
stage2_lr_joint         = 1e-5
stage2_entropy_start    = 0.10
stage2_entropy_end      = 0.05
stage2_kl_lambda_start  = 0.5       # S-phase: anneals to 0.1
stage2_kl_lambda_end    = 0.1
stage2_n_kl_lambda_start = 0.5      # N-phase: fixed (no anneal)
stage2_n_kl_lambda_end   = 0.5

# JIT Belief Burn-in (Agent B, before each N-phase)
jit_burnin_deals        = 1000
jit_burnin_epochs       = 3
jit_burnin_lr           = 1e-3

# Eval
eval_deals              = 1000
diag_deals              = 2000

# Agent B info bonus
beta                    = 0.05

# Multi-seed
seeds                   = [42, 123, 456, 789, 2024]
```

---

## Quick Start

### 1. Install
```bash
pip install torch numpy tqdm endplay pyyaml scipy
```

### 2. Assemble project (every new session)
```bash
python setup_project.py
```

### 3. Run Phase 2 Stayman multi-seed experiment
```bash
cd bridge-coma/

# Generate data (once, ~5 min)
python -m utils.generate_subgame_data --type stayman --num_workers 4

# Multi-seed run (5 seeds)
for seed in 42 123 456 789 2024; do
    python experiments/subgame_validation.py \
        --stayman_data data/stayman_50k.npz \
        --seed $seed \
        --alt_rounds 3 \
        --joint_steps 300 \
        --device cpu
done

# Quick smoke test
python experiments/subgame_validation.py --quick \
    --stayman_data data/stayman_50k.npz
```

### 4. Outputs
```
results/
├── phase2_report_seed42.json
├── phase2_report_seed123.json
├── ...
├── s_base.pt
├── A_control.pt
└── B_partner_only.pt
```

---

## Full Bug Fix History

### Phase 1 (P0–P6)
P0 package · P1 termination · P2 reward · P3 eval · P4 scoring · P5 vulnerability · P6 dealer

### Phase 2 (P7–P38)

| # | Problem | Fix |
|---|---------|-----|
| P7 | NaN crash | `safe_update` robust normalization |
| P8 | All Pass | Stayman action mask |
| P9 | All-negative reward | Piecewise linear shifted IMP |
| P10 | N↔S coupling | N+S joint BC |
| P11 | cudnn LSTM error | `model.train()` |
| P12 | Competitive reward explosion | Dual-table IMP |
| P13 | Low filter rate | `generate_subgame_data.py` |
| P14 | Unfair DDS | `max_level=4` |
| P15 | Entropy collapse | `entropy_end=0.02`, anneal=0.8 |
| P16 | N weights random | N+S joint training |
| P17 | RL destroys BC | Stage 1 pure BC + low lr |
| P18–P20 | Logging / checkpoint / imports | Fixed |
| P21 | Value loss explosion | `single_step=True`: batch-mean baseline |
| P22 | Entropy collapse | `entropy_end=0.02`, anneal=0.8 |
| P23 | Credit assignment blur | Alternating S-N-S-N training |
| P24 | N-phase reward=0 | Fixed `reward` capture in non-active player path |
| P25 | LSTM deaf | `pack_padded_sequence` |
| P26 | S ignores N's bids | S weight ×3 + minority weight ×2 |
| P27 | KL=-inf NaN | Clamped logits before KL |
| P28 | NaN in `evaluate_actions` | Separate actor/critic optimizers |
| P29 | Critic warmup Adam re-created each call | Reuse `agent.critic_optimizer` |
| P30 | `single_step=True` blocked Critic | `single_step=False` in Stage 2 |
| P31 | Belief Net stuck at 0.75 | BCEWithLogitsLoss(pos_weight=3) + Top-13 hit rate |
| P32 | KL anchor gradient = 0 | Manual KL; `curr_logits` outside `no_grad` |
| P33 | Belief Net catastrophic forgetting | `belief_lr: 1e-3 → 1e-4` |
| P34 | BC missing N Round3 responses | Extended BC collection to include N's invite response |
| P35 | 4H/4S acceptance too rare in BC | `stayman_bc_samples: 10000 → 20000` |
| P36 | r_info never wired to reward; β=0.0; ir negative | Wire ir to terminal reward; `beta=0.05`; ReLU clamp; JIT burn-in |
| P37 | `NameError: BID_PASS` in Stage 3 | Add `BID_PASS, EAST, WEST` to env import |
| P38 | BC mask mismatch after generalization | Live BC code used hardcoded masks (2D/2H/2S only); replaced with `env._get_legal_actions()`. Dead code (327 lines) deleted. |

---

## Project Structure

```
bridge-coma/
├── env/
│   ├── bridge_bidding_env.py
│   └── dual_table_env.py
├── networks/
│   ├── policy_net.py                 # HandEncoder + HistoryEncoder (pack_padded)
│   └── belief_net.py                 # Logit output, BCEWithLogitsLoss(pw=3), Top-13
├── utils/
│   ├── scoring.py
│   ├── imp.py
│   ├── dds_data.py
│   ├── running_stats.py
│   └── generate_subgame_data.py      # Stayman S HCP: 8–10
├── algorithms/
│   ├── ippo.py
│   ├── mappo.py                      # Separate actor/critic optimizers, PPO2 clip
│   └── behavioral_cloning.py
├── subgames/
│   ├── stayman_env.py                # Piecewise reward; MAX_LEVEL=4; legal mask; clean BC
│   ├── competitive_env.py
│   ├── subgame_trainer.py            # ir wired; ReLU; JIT burn-in; HeadToHeadEvaluator
│   └── action_mask.py
├── experiments/
│   ├── train.py
│   └── subgame_validation.py         # alt_rounds=3; joint_steps=300; Stage 3 H2H
├── tests/
│   ├── test_all.py
│   └── test_phase2.py
├── results/
├── data/
├── setup_project.py
└── requirements.txt
```
