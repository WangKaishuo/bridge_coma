# Bridge-COMA

**Dual-Information Credit Assignment for Cooperative-Competitive Multi-Agent Coordination**

MSc Research Project — Addressing relative overgeneralization and miscoordination in bridge bidding via information-theoretic reward shaping with prior asymmetry.

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

#### Stayman Subgame

| Item | Status |
|------|--------|
| Constrained dealing + DDS data generation (50k deals) | ✅ |
| Stayman subgame env (`stayman_env.py`) | ✅ |
| Action mask (`action_mask.py`) | ✅ |
| Subgame trainer (`subgame_trainer.py`) | ✅ |
| N/S rule policies | ✅ |
| BC data generation (N+S joint, weighted loss) | ✅ |
| Piecewise linear reward (IMP-aligned) | ✅ |
| LSTM fix (pack_padded_sequence) | ✅ |
| BC dual weighting (player weight + minority weight) | ✅ |
| KL anchor regularization | ✅ |
| Separate Actor/Critic optimizers | ✅ |
| Critic warmup (dual-track Stage 1.5) | ✅ |
| GAE enabled (`single_step=False`) | ✅ |
| S HCP upper limit (8–10) | ✅ |
| **Belief Net: BCEWithLogitsLoss + pos_weight=3** | ✅ **[THIS SESSION]** |
| **Belief Net: Top-13 hit rate metric (replaces threshold acc)** | ✅ **[THIS SESSION]** |
| **Diagnostics: Fit type breakdown (4-4/5-3/double)** | ✅ **[THIS SESSION]** |
| **Diagnostics: Decision error matrix (cost per error type)** | ✅ **[THIS SESSION]** |
| Stage 1: BC warmup → acc=99.9%, fit=100% | ✅ |
| Stage 1.5: Belief Network pre-training | 🔄 **fix applied, needs re-run** |
| Stage 2: Alternating A vs B | 🔄 **results available, needs re-run post-fix** |
| Multi-seed validation (3–5 seeds + CI) | ⏳ After Belief Net confirmed working |

#### Competitive Subgame

| Item | Status |
|------|--------|
| Constrained dealing + DDS data (100k deals) | ✅ |
| Competitive subgame env (`competitive_env.py`) | ✅ |
| BC warmup (`behavioral_cloning.py`) | ✅ |
| Cross-evaluation (`cross_evaluate`) | ✅ |
| Full experiment | ⏳ After Stayman concluded |

### Phase 3–4: Not started

---

## Current Experimental State

### Latest Run Results (pre-fix, for reference)

**Stage 1 — BC Baseline (S HCP: 8–10):**
```
BC accuracy:   99.5%
3NT rate:      14.4%   (expected ~70% for no-fit; see data skew note below)
4M rate:       85.6%
IMP:           -5.04 ± 4.97
```

**Data skew note:** With S HCP 8–10 and N HCP 15–17, NS total = 23–27 HCP.
At this strength, even a 4-3 major fit can make 4M via power play — DDS correctly
labels many "no 4-4 fit" deals as 4M optimal. The S HCP upper limit of 10 was added
to bring NS total down to the range where suit fit genuinely matters.

**Stage 2 Final Results:**

| Agent | IMP | Δ vs S_base |
|-------|-----|-------------|
| S_base (N=rule) | −5.04 | — |
| A_control (MAPPO) | −5.63 | −0.58 |
| B_partner_only (MAPPO + r_info) | −4.86 | **+0.18** |
| B vs A | — | **+0.77** |

Go/No-Go: ❌ S_base did not converge (IMP not positive enough), but B > A by +0.77 IMP
is directionally correct. The bottleneck was the broken Belief Net (see below).

**N's Policy Shift (final, A vs B):**

| Condition | A_control | B_partner_only |
|-----------|-----------|----------------|
| has_4H → | 2♥ 100% | 2♥ 100% |
| has_4S → | 2♠ 82%, 2♥ 18% | 2♠ 68%, 2♥ 32% |
| no_4M → | 2♠ 58%, 2♥ 24%, 2♦ 18% | 2♠ 45%, 2♥ 41%, 2♦ 14% |

B's N is more expressive on the 4S hand (uses 2♥ more), consistent with r_info
rewarding informative bids. But B collapsed to 99.2% 4M due to broken Belief Net
making ir permanently negative — the fix below should resolve this.

---

## Critical Bug Fixed This Session: Belief Net Class Imbalance (P31)

### Root Cause

```
old belief_acc = 0.75  =  39/52  =  prior baseline
```

Each hand has 13 cards in 52 slots → 39 zeros, 13 ones (3:1 class imbalance).
A network predicting all-zeros achieves 75% accuracy and BCE ≈ 0.47 with zero
learning. The threshold metric `(pred > 0.5) == target` cannot distinguish a
trained network from a trivial one in this imbalanced setting.

Consequence: Belief Net was stuck at prior. When N deviated from BC, Belief Net
OOD predictions worsened → `ce_after > ce_before` → `ir < 0` → information bonus
*punished* informative bids → N learned to always signal 4M → collapse.

### Fixes Applied

**`belief_net.py`:**
- Removed `nn.Sigmoid()` from final layer — outputs raw logits
- `compute_loss` → `BCEWithLogitsLoss(pos_weight=3.0)`
- Added `get_probs()` → `sigmoid(logits)` for info gain computation
- Added `top13_hit_rate()` static method — correct evaluation metric

**`subgame_trainer.py`:**
- `_compute_single_info_gain`: `belief_net(...)` → `belief_net.get_probs(...)`
  (BCE info gain requires probabilities; logits in BCE = NaN)
- `evaluate_belief_accuracy`: full rewrite using Top-13 hit rate

**`subgame_validation.py`:**
- `belief_pretrain_target_acc = 0.40` (was 0.80 under broken metric)
- Log shows `top13_hit=... (random_baseline=0.25)`

### Expected Behavior After Fix

| Metric | Before fix | After fix (expected) |
|--------|------------|---------------------|
| belief loss start | 0.57 | ~0.69 (BCEWithLogitsLoss re-scaled) |
| top13_hit start | — | ~0.25 (random baseline) |
| top13_hit converged | — | 0.35–0.50+ |
| ir during N-phase | always negative | mostly positive |
| B final contract dist | 99.2% 4M (collapsed) | ~70–80% 4M (balanced) |

---

## Pending Tasks for Next Session

### Immediate

- [ ] **Re-run full experiment** with Belief Net fix. Key things to watch:
  - Stage 1.5: `top13_hit` should rise from 0.25 to 0.35+ over 50 epochs
  - Stage 2 N-phase: `ir` should be mostly positive
  - Final contract distribution: B should not collapse to 99% 4M
  - B vs A IMP gap should widen vs pre-fix result (+0.77)
- [ ] Read Decision Error Matrix output — quantify cost of "fit → 3NT" vs "nofit → 4M"

### After confirming Belief Net fix

- [ ] Multi-seed validation (3–5 seeds, paired t-test or bootstrap CI)
- [ ] Investigate N-phase value loss spikes (vl=8–10 occasional)
- [ ] Optional: temperature T=1.2 for KL `bc_logits` (softer anchor; agreed but not implemented)

### After Stayman concluded

- [ ] **Ablation: B_oracle** — 2-bit `[has_4H, has_4S]` belief target as Oracle Upper Bound
- [ ] Competitive subgame experiment
- [ ] Phase 3–4

---

## Architecture

### Network Structures

```
PolicyNetwork (Actor):
  HandEncoder:     52 → 256 → 256 (MLP)
  HistoryEncoder:  (seq_len, 38) → LSTM(2 layers, 256) → h_n[-1]
                   [pack_padded_sequence — valid tokens only, no padding flush]
  Fusion: [hand_256 + history_256 + position_4 + vulnerability_2] → MLP → 38-dim logits

ValueNetwork (Critic):
  Same as Actor + AllHandsEncoder: 4×52 → 256 → 256 (centralized)
  Separate optimizer from Actor (lr × 2, PPO2 value clipping)
  Output: scalar value

BeliefNetwork:
  HandEncoder:    52 → 256 → 256 (MLP)
  HistoryEncoder: LSTM(2 layers, 256) [pack_padded_sequence]
  PositionEmbed:  Embedding(4, 32) × 2 (observer_pos + target_pos)
  Output: 52-dim LOGITS  ← no Sigmoid in forward()
  Probs:  get_probs() = sigmoid(logits)  ← use this for r_info computation
  Loss:   BCEWithLogitsLoss(pos_weight=3.0)
  Metric: top13_hit_rate()  — random baseline ≈ 0.25, target ≥ 0.40
```

### Key Design Decisions

**1. BCEWithLogitsLoss + pos_weight=3 (Belief Net)**

39 zeros vs 13 ones per sample (3:1 imbalance). Standard BCE + threshold accuracy
gives 0.75 for all-zeros prediction. Fix forces equal attention to present cards.
Top-13 hit rate is the unambiguous metric: random ≈ 0.25, trained ≥ 0.35.

**2. Separate Actor/Critic Optimizers**

Single optimizer caused Critic's MSE gradients (~10+) to overwhelm Actor's policy
gradients (~0.001). Fix: `actor_optimizer`, `critic_optimizer = Adam(critic, lr*2)`,
PPO2-style value clipping.

**3. Critic Warmup (dual-track Stage 1.5)**

BC trains Actor only → Critic randomly initialized → GAE produces garbage advantages.
Fix: rollout with current Actor, update Critic with MSE on actual rewards.
Target must be current policy rollout reward (NOT DDS optimal — systematic
overestimation → negative advantages → collapse).

**4. GAE enabled (single_step=False)**

`single_step=True` (batch-mean baseline) is for legacy/debug only. Stage 2 uses
full GAE now that Critic is pretrained.

**5. Alternating Training**

Simultaneous N+S learning blurs credit assignment. Alternating training fixes one
player per half-round → reward changes 100% attributable to active player.
```
Round k: S trains (N frozen) → N trains (S from previous round frozen)
Final:   Joint fine-tune (both active, low lr)
```

**6. BC Dual Weighting**

N's BC gradient (no history needed) suppresses HistoryEncoder features S depends on.
Fix: S sample weight ×3, minority action weight ×2 → S's 3NT samples get ×6 total.

**7. KL Anchor Regularization**

`KL(π || π_BC)` prevents RL from destroying BC's signaling structure.
Lambda anneals from 0.308 → 0.100 over training. Preserves N's 2♦/2♥/2♠ distinction
while allowing RL refinement.

**8. S HCP Constraint: 8–10**

S ≥ 8 with no upper limit allows NS total 28–30 HCP where 4M succeeds on 4-3 fits.
DDS labels pollute the 4M/3NT boundary. Upper limit of 10 keeps NS at 23–27 HCP
where the fit question is genuinely consequential and the learning signal is clean.

---

## Experiment Design (Phase 2 Stayman)

### Stage 1: BC Warmup
```
Pure BC (stage1_steps=0)
N+S joint, 10k samples (N=5000, S=5000)
S weight ×3, minority weight ×2
→ BC acc ≈ 99.5–99.9%, fit detection 100%
```

### Stage 1.5: Belief Network Pre-training
```python
critic_warmup_rounds       = 10
critic_warmup_deals        = 512
belief_pretrain_deals      = 10000
belief_pretrain_epochs     = 50
belief_pretrain_target_acc = 0.40   # Top-13 hit rate (random baseline = 0.25)
eval_deals                 = 1000
diag_deals                 = 2000
```

### Stage 2: Alternating Fine-tuning
```python
stage2_alt_rounds        = 4
stage2_alt_steps         = 200
stage2_joint_steps       = 400
stage2_deals_per_step    = 32
stage2_accumulate        = 8       # 256 deals/update
stage2_lr                = 3e-5
stage2_lr_joint          = 1e-5
stage2_entropy_start     = 0.05
stage2_entropy_end       = 0.02
stage2_entropy_anneal    = 0.8
single_step              = False   # GAE + pretrained Critic
```

Agent A: MAPPO control (no info bonus)
Agent B: MAPPO + r_info (β=0, partner-only, info bonus active in N-phase)

### Stage 3: Evaluation
- IMP vs DDS optimal (S_base / A / B)
- N's policy distribution per hand type
- Decision error matrix (IMP cost per error type)
- Go/No-Go: S_base converged AND B > A

---

## Ablation Study Plan (for paper)

**B_oracle** — 2-bit belief target `[has_4H, has_4S]`:
1. **Main (B)**: 52-dim, BCEWithLogitsLoss(pos_weight=3). Theoretically rigorous MI.
2. **Oracle (B_oracle)**: 2-bit BCE. Perfect belief upper bound in Stayman.
3. **Argument**: B ≈ B_oracle → mechanism is robust. B << B_oracle → room to improve.

This framing prevents the "2-bit cheats" critique — it's explicitly labeled as an
oracle bound in a controlled environment, not the primary contribution.

---

## Full Bug Fix History

### Phase 1 (P0–P6)
P0 package · P1 termination · P2 reward · P3 eval · P4 scoring · P5 vulnerability · P6 dealer

### Phase 2 (P7–P31)

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
| **P31** | **Belief Net stuck at 0.75** | **BCEWithLogitsLoss(pos_weight=3) + Top-13 hit rate** |

---

## Reward Design

### Stayman: Piecewise Linear Shifted IMP

| IMP regret | reward | Bridge semantics |
|-----------|--------|-----------------|
| 0 | 1.00 | Perfect vs restricted DDS optimal |
| −1 | 0.70 | Wrong suit choice |
| −6 | 0.25 | Missed game |
| −13 | 0.01 | Catastrophic |

### Competitive: Dual-Table IMP

Direct dual-table IMP differential, range ±24.

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

### 3. Run Phase 2 Stayman experiment
```bash
cd bridge-coma/

# Generate constrained data (once)
python -m utils.generate_subgame_data --type stayman --num_workers 4

# Full run
python experiments/subgame_validation.py \
    --stayman_data data/stayman_50k.npz \
    --device cuda

# Quick test
python experiments/subgame_validation.py --quick \
    --stayman_data data/stayman_50k.npz
```

### 4. Outputs
```
results/
├── phase2_report.json
├── s_base.pt
├── A_control.pt
└── B_partner_only.pt
```

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
│   ├── stayman_env.py
│   ├── competitive_env.py
│   ├── subgame_trainer.py            # critic_warmup, Top-13 eval, get_probs()
│   └── action_mask.py
├── experiments/
│   ├── train.py
│   └── subgame_validation.py         # target_acc=0.40, fit+error matrix diagnostics
├── tests/
│   ├── test_all.py
│   └── test_phase2.py
├── results/
├── data/
├── setup_project.py
└── requirements.txt
```
