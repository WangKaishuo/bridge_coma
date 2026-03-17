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

#### Stayman Subgame 🔄 Infrastructure complete, clean run pending

| Item | Status |
|------|--------|
| Constrained dealing + DDS data generation (50k deals) | ✅ |
| Stayman subgame env (`stayman_env.py`) | ✅ |
| Action mask → generalized to `env._get_legal_actions()` | ✅ |
| Subgame trainer (`subgame_trainer.py`) | ✅ |
| N/S rule policies | ✅ |
| BC data generation — N+S joint + Round3 N response | ✅ |
| BC samples 50k (HAPPO dual-actor requires separate N/S datasets) | ✅ |
| Running IMP normalization via RunningStats | ✅ |
| HAPPO dual-actor (actor_n + actor_s fully independent) | ✅ |
| Dual independent critic (critic_n + critic_s, P49) | ✅ |
| Entropy revival (temperature=2.0 before Stage 2) | ✅ |
| Adaptive Critic prewarm (convergence-based, not fixed rounds) | ✅ |
| **Critic prewarm: large static buffer (2048 deals, P51)** | ✅ |
| **Critic prewarm: multi-epoch fitting on static buffer (P51)** | ✅ |
| **KL early stopping: epoch-level (not batch-level, P51)** | ✅ |
| **BC global hard ceiling (bc_kl_max, skip actor if exceeded, P51)** | ✅ |
| Post-BC diagnostics: north_rule=True (isolates S quality) | ✅ |
| KL anchor regularization (cross-round macro annealing) | ✅ |
| GAE enabled (`single_step=False`) | ✅ |
| Belief Net: BCEWithLogitsLoss + pos_weight=3 | ✅ |
| Belief Net: 48-dim output (honor 16 + length 32) | ✅ |
| r_info wired to terminal reward | ✅ |
| ReLU clamp on ir | ✅ |
| β fixed to 0.05 | ✅ |
| JIT Belief Burn-in before each N-phase | ✅ |
| Context-level adaptive KL weights | ✅ |
| HeadToHeadEvaluator framework | ✅ |
| Three-way eval (BC / A / B on same held-out deals) | ✅ |
| **Clean run post-P51 to confirm RL improves over BC** | ⏳ Next |
| Multi-seed validation (5 seeds) | ⏳ After clean run |

#### Competitive Subgame

| Item | Status |
|------|--------|
| Constrained dealing + DDS data (100k deals) | ✅ |
| Competitive subgame env (`competitive_env.py`) | ✅ |
| BC warmup (`behavioral_cloning.py`) | ✅ |
| Cross-evaluation (`cross_evaluate`) | ✅ |
| Full experiment | ⏳ After Stayman confirmed |

### Phase 3–4: Not started

---

## Current Experimental State

### Latest Full Run (seed=42, alt_rounds=6, P50 code)

**Stage 1 — BC base:**
```
BC-N: acc=0.999, ent=0.002 (early stop epoch 5)
BC-S: acc=0.999, ent=0.001 (early stop epoch 5)
IMP (S_base): -4.07 ± 3.54
Belief Stage 1.5: overall_acc = 0.756 (target 0.40 ✓)
```

**Stage 2 Final (P50, 6 rounds):**

| Agent | IMP | Δ vs BC |
|-------|-----|---------|
| BC-only | −4.07 | — |
| A_control (MAPPO) | −4.04 | +0.03 |
| B_partner_only (MAPPO+r_info) | −4.08 | −0.01 |
| B vs A | — | −0.04 (p=0.668, not sig) |

**Round-by-round (A / B):**
```
R1: S→-3.96/-3.94  N→-3.79/-4.28
R2: S→-3.93/-3.80  N→-3.86/-4.25
R3: S→-4.25/-4.39  N→-4.00/-4.29
R4: S→-4.21/-4.09  N→-4.41/-3.66
R5: S→-3.77/-3.94  N→-4.54/-4.11
R6: S→-4.38/-4.23  N→-4.20/-5.03  ← B R6 N collapse (kl=0.92)
```

**Conclusion from this run:** RL neither improves nor degrades vs BC (≈ null effect).
B R6 N-phase collapsed (kl=0.92, ent=0.05→B after collapse), confirming epoch-level
KL early stop and BC hard ceiling were needed. P51 addresses both.

### Reference Run (older, seed=42, alt_rounds=3, P42 code — for comparison)

```
S_base (N=rule): IMP = -2.92
A (MAPPO):       IMP = -3.68  (Δ vs S_base: -0.76)   ← RL HURTS
B (MAPPO+r_info):IMP = -3.39  (Δ vs S_base: -0.47)   ← B > A by +0.29
```

Key observation: in this older run, RL *degraded* performance vs BC-S+rule-N baseline.
RL fixed the fit→3NT under-bidding problem (140→83 cases) but created a worse
nofit→4M over-bidding problem (10→47 cases, -7 IMP each). Root cause: critic_s
not converged → noisy advantage → wrong gradient direction for S.

---

## Root Cause Analysis: Why RL Degraded BC

Two confirmed bugs in P50 code, both fixed in P51:

### Bug 1: KL Early Stopping was Batch-Level (Ineffective)

**Symptom:** R6 N-phase kl=0.92 despite kl_early_stop_threshold=0.015.

**Root cause:** P50 computed `approx_kl` per mini-batch and broke out of the inner
batch loop if any single batch exceeded 0.015. But a single mini-batch KL is ~0.001
(few samples, tiny update step) — the threshold was never triggered. Across 4 epochs
× many batches, cumulative policy drift reached 0.2–0.9 undetected.

**P51 fix:** Accumulate KL across all batches in one epoch, check the epoch-average
after the epoch ends, then break. This reflects true per-epoch policy drift.

### Bug 2: Critic Prewarm — Statistical Disaster (128 Deals)

**Symptom:** S-phase critic_s hit max 30 rounds and still showed vl=1.8+ entering
PPO. N-phase critic_n converged in 2–3 rounds.

**Root cause:** 128 deals × 3 substate branches (N bids 2♦/2♥/2♠) = ~40 deals per
branch. S's state space also includes S HCP/shape variation. Critic was severely
overfitting 40 samples per context → vl exploded on next rollout batch.

**P51 fix:** Single large static buffer of 2048 deals collected once, actor frozen,
critic fits for up to 10 epochs with convergence check (relative vl change < 5%).
Returns computed once from frozen actor → no stale targets.

### Bug 3: No Global BC Hard Ceiling

**Symptom:** KL anchor λ decays to 0.1 by round 6 → insufficient restraint → actor
drifts far from BC → nofit→4M hallucinations.

**P51 fix:** Before each PPO update, compute KL(current || BC) over entire buffer.
If > bc_kl_max (0.5), skip actor update entirely (critic still updates). No rollback
— preserves N/S temporal consistency.

---

## Stayman: Scientific Assessment

**The Stayman subgame is a necessary but limited test.** Key evidence:

1. **N's bidding is 100% deterministic** — BC taught the 3-bit Stayman protocol
   (2♦/2♥/2♠) to near-theoretical optimality. N never deviates.
2. **Belief overall_acc = 0.756** — well above 0.40 target; S can already infer N's
   hand type perfectly from N's single bid.
3. **ir ≈ 1.0** — information channel saturated; r_info cannot improve what BC maximized.
4. **EW always pass** — β·opponent_leak term is structurally inactive in Stayman.
5. **Null result is expected** — Stayman validates infrastructure stability, not r_info.

**The real r_info test is the competitive subgame**, where:
- N has a richer signaling space (no single "correct" protocol)
- Active EW interference creates genuine partner/opponent information tension
- BC cannot reach a theoretical ceiling
- Full dual-info formula `I(bid;hand|partner) - β·I(bid;hand|opponent)` is exercised

**Scientific conclusion (paper framing):**
> "The Stayman subgame confirms that the full training infrastructure (BC, KL anchoring,
> JIT burn-in, Belief Net, Critic prewarm) is stable and produces consistent protocols.
> Because BC achieves the theoretical maximum for this 3-bit communication task, r_info's
> partner term cannot demonstrate incremental benefit. The β term is structurally
> untestable here (EW always pass). These constraints motivate the competitive subgame
> as the primary experimental vehicle for the dual-info credit assignment hypothesis."

---

## Pending Tasks

### Immediate

- [ ] **Clean P51 run** — confirm RL now improves over BC baseline
  - Watch for: `🚫actor-skip` frequency (BC ceiling), `⚡KL-stop` frequency (epoch drift)
  - Target: A > BC, B ≥ A (or at minimum B not significantly worse than A)
  - Key diagnostic: critic_s prewarm should now show genuine convergence in logs

### After clean P51 run

- [ ] **Multi-seed run (5 seeds: 42, 123, 456, 789, 2024)**
  - Report mean ± 95% CI for B vs A and A vs BC
  - Paired t-test / Wilcoxon on per-deal IMP

### After Stayman multi-seed

- [ ] **Competitive subgame** — primary r_info validation
  - Full 3-agent comparison: A (MAPPO) / B (β=0) / C (β=0.05)
  - Opponent interference activates the β term

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
  Same as Actor + AllHandsEncoder: 4×52 → 256 → 256 (centralized, CTDE)
  Separate optimizer from Actor (lr × critic_lr_ratio; PPO2 value clipping)
  P49: dual independent critics — critic_n for NORTH, critic_s for SOUTH
       Eliminates cross-phase contamination (shared critic caused vl 2000+ on phase switch)

HAPPOModel (P48+):
  actor_n, actor_s, critic_n, critic_s — four fully independent networks
  S-phase: only actor_s + critic_s updated
  N-phase: only actor_n + critic_n updated
  state_dict format: actor_n.* / actor_s.* / critic_n.* / critic_s.*

BeliefNetwork:
  HandEncoder:    52 → 256 → 256 (MLP)
  HistoryEncoder: LSTM(2 layers, 256) [pack_padded_sequence]
  PositionEmbed:  Embedding(4, 32) × 2 (observer_pos + target_pos)
  Output:         48-dim LOGITS (no Sigmoid in forward)
                  [0:16]  = AKQJ honor flags (4 suits × 4 honors)
                  [16:48] = one-hot suit lengths (4 suits × 8 buckets)
  Probs:          get_probs() = sigmoid(logits)    ← use for r_info
  Loss:           BCEWithLogitsLoss(pos_weight=3.0)
  Metric:         top13_hit_rate  (random baseline ≈ 0.25, target ≥ 0.40)
```

### r_info Design

```
r_info = max(0, I(bid; hand | partner)) - β * max(0, I(bid; hand | opponent))

where I(bid; hand | observer) ≈ CE(belief_before, hand) - CE(belief_after, hand)

ReLU clamp: MI ≥ 0 by definition; negative values are Belief Net lag, not N's fault.
β = 0.05: "gentle breeze" — info bonus supplements IMP, doesn't dominate it.

Applied to: N's terminal step reward only.
Active in:  N-phase for Agent B only.
target:     hand_to_belief_target(hand) → 48-dim (NOT raw 52-dim one-hot)
```

### Key Design Decisions

**1. HAPPO Dual Actor (P48)**
Shared actor caused N/S gradient cross-contamination: S's PPO gradient overwrote
N's learned bidding language. Fix: fully independent actor_n and actor_s.
Each player's gradients only update their own network.

**2. Dual Independent Critic (P49)**
Shared critic caused catastrophic forgetting: S-phase updates destroyed critic's
N-phase value estimates → vl exploded to 2000+ at N-phase start.
Fix: critic_n trained only on N data, critic_s only on S data.
Both critics receive global state (all_hands + obs) → CTDE preserved.

**3. Adaptive Critic Prewarm — Large Static Buffer (P50/P51)**
Before each PPO half-round, critic is pre-warmed using convergence monitoring.

P51 design (current):
- Collect 2048 deals into a static buffer (actor frozen)
- Compute GAE returns once from frozen actor (no stale targets)
- Fit critic for up to 10 epochs, stopping when relative vl change < 5%
- Outer adaptive loop: if critic still not converged, collect new 2048 deals and repeat
  (max 30 outer rounds)

Why 2048 deals (not 128): Stayman has 3 N-bid branches × S hand variation → 128 deals
gives ~40 samples per branch → catastrophic overfitting → vl explodes next rollout.
2048 deals provides statistically meaningful coverage of the joint state distribution.

**4. Epoch-Level KL Early Stopping (P51)**
Within each PPO update, track cumulative KL across all mini-batches in one epoch.
After the epoch ends, if epoch-average KL(π_current ∥ π_old) > 0.015, break.

Why epoch-level (not batch-level as in P50): single mini-batch KL ≈ 0.001 (few
samples, tiny step) → batch-level threshold never triggered. Epoch-average reflects
true policy drift per gradient pass.

**5. BC Global Hard Ceiling (P51)**
Before each PPO update, compute KL(π_current ∥ π_BC) over the entire buffer.
If this global KL exceeds bc_kl_max (0.5), skip actor update (critic still updates).
No rollback — preserves N/S temporal consistency.
Log indicator: `🚫actor-skip(bc_kl=X.XXX)`

**6. Cross-Round Macro KL Annealing (P39)**
Both N and S share a linear KL decay across rounds:
```
Round 1: kl_lambda = 0.5  (strong BC protection)
Round K: kl_lambda = 0.1  (light protection, protocol stable)
```
Original bug: N's KL was fixed at 0.5 forever → N never updated meaningfully →
S's Critic faced non-stationary environment → vl explosion in Round 2+.

**7. JIT Belief Burn-in (P36)**
Belief Net trained at Stage 1.5 on BC rollouts. Once N starts RL exploration,
its protocol evolves → Belief Net goes OOD → ir estimates corrupt.
Fix: Before each N-phase, run 1000 rollouts and fine-tune Belief Net (lr=1e-3, 3 epochs).

**8. Entropy Revival (P45)**
BC trains to near-zero entropy (ent ≈ 0.001). PPO cannot explore from this state.
Fix: Before Stage 2, rescale actor's final-layer logit weights by temperature=2.0,
spreading the distribution back to a trainable entropy level.
Applied independently to actor_n and actor_s (HAPPO).

**9. Post-BC Diagnostics: north_rule=True (P50)**
After BC completes, diagnostics must use north_rule=True (rule-based N) to isolate
S's BC quality. Using north_rule=False caused actor_n to produce OOD bids →
S saw unknown history → nofit→4M false positives ("statistical disaster" in logs).

**10. Separate Actor/Critic Optimizers (critic_lr_ratio=5.0)**
Single optimizer: Critic MSE gradients (~10) overwhelm Actor gradients (~0.001).
Fix: separate Adam optimizers, critic LR = actor LR × 5.0. PPO2-style value clipping.

---

## Experiment Configuration (Phase 2, current — P51)

```python
# BC Warmup
stayman_bc_samples           = 50000
stayman_bc_epochs            = 15
bc_early_stop_acc            = 0.98   # patience=3 consecutive epochs

# Stage 1.5: Critic + Belief pre-training
critic_warmup_rounds         = 10
critic_warmup_deals          = 512
belief_pretrain_deals        = 10000
belief_pretrain_epochs       = 50
belief_pretrain_target_acc   = 0.40

# Stage 2: Alternating fine-tuning (no joint fine-tune — removed P49)
stage2_alt_rounds            = 6
stage2_alt_steps             = 200    # steps per half-round
stage2_deals_per_step        = 32
stage2_accumulate            = 8      # effective: 256 deals/update
stage2_lr                    = 3e-5
stage2_entropy_start         = 0.10
stage2_entropy_end           = 0.05
stage2_entropy_anneal        = 0.5

# KL Anchor (cross-round macro annealing, unified N+S)
stage2_kl_lambda_start       = 0.5   # Round 1
stage2_kl_lambda_end         = 0.1   # Round K (last)
stage2_kl_anneal_frac        = 1.0

# Critic stability (P51)
stage2_critic_lr_ratio       = 5.0
stage2_critic_prewarm_max_rounds = 30   # outer adaptive loop
stage2_critic_prewarm_deals  = 2048     # large static buffer (P51, was 128)
stage2_critic_prewarm_epochs = 10       # epochs on static buffer (P51)
stage2_critic_prewarm_conv_tol = 0.05  # relative vl change threshold

# KL guards (P51)
stage2_kl_early_stop_threshold = 0.015  # epoch-level (P51, was batch-level)
stage2_bc_kl_max             = 0.5      # global BC ceiling, skip actor if exceeded

# JIT Belief Burn-in (Agent B, before each N-phase)
jit_burnin_deals             = 1000
jit_burnin_epochs            = 3
jit_burnin_lr                = 1e-3

# Eval
eval_deals                   = 200
diag_deals                   = 500

# Agent B info bonus
beta                         = 0.05

# Multi-seed
seeds                        = [42, 123, 456, 789, 2024]
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

### 3. Run Phase 2 Stayman experiment
```bash
cd bridge-coma/

# Generate data (once, ~5 min)
python -m utils.generate_subgame_data --type stayman --num_workers 4

# Single run (default seed=42)
python experiments/subgame_validation.py \
    --stayman_data data/stayman_50k.npz \
    --seed 42 \
    --alt_rounds 6 \
    --device cpu

# Multi-seed run (5 seeds)
for seed in 42 123 456 789 2024; do
    python experiments/subgame_validation.py \
        --stayman_data data/stayman_50k.npz \
        --seed $seed \
        --alt_rounds 6 \
        --device cpu
done

# Quick smoke test
python experiments/subgame_validation.py --quick \
    --stayman_data data/stayman_50k.npz
```

### 4. Key log indicators to watch
```
[Critic Prewarm] N rounds × 2048 deals (converged at round K): vl X → Y
  → critic_s should now converge; if hitting max 30, something is wrong

⚡KL-stop
  → epoch-level KL exceeded 0.015; actor update truncated; healthy sign

🚫actor-skip(bc_kl=X.XXX)
  → global BC ceiling triggered; actor skipped this update; should be rare
  → if frequent (>30% of updates), bc_kl_max may be too tight or lr too high
```

### 5. Outputs
```
results/
├── phase2_report.json
├── s_base.pt
├── A_control.pt
└── B_partner_only.pt
```

---

## Full Bug Fix History

### Phase 1 (P0–P6)
P0 package · P1 termination · P2 reward · P3 eval · P4 scoring · P5 vulnerability · P6 dealer

### Phase 2 (P7–P51)

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
| P38 | BC mask mismatch after generalization | Live BC used hardcoded masks; replaced with `env._get_legal_actions()`. 327 lines dead code deleted. |
| P39 | N vl explosion in Round 2+ | N's KL fixed at 0.5 → N not updating → S Critic non-stationary. Fix: unified cross-round macro KL annealing (0.5→0.1) for both N and S |
| P40 | BeliefNetwork 52→48 dim mismatch | `hand_to_belief_target()` not applied in 3 locations; `belief_accuracy()` import missing |
| P41 | Critic prewarm degraded Actor performance | `critic_warmup_step` used `final_reward` flattened to all timesteps. Fix: `store_episodes → compute_returns_and_advantages → Critic update only → buffer.reset()` |
| P42 | Adaptive prewarm removed | Fixed at 3 rounds × 128 deals for reproducibility (later superseded by P50) |
| P43 | BC over-training → entropy=0 | Early stopping at acc≥0.98 × patience=3 |
| P44 | BC-N quality unknown | Added rule-N vs BC-N diagnostic |
| P45 | entropy=0 → PPO cannot update | `_revive_entropy(state_dict, temperature=2.0)` before Stage 2 |
| P46 | temperature=2.0 → N calls OOD bids | (reverted; 2.0 retained as correct value) |
| P47 | N-phase prewarm vl diverges | `stage2_n_critic_prewarm_rounds=6` (superseded by P49) |
| P48 | Shared actor → N/S gradient cross-contamination | HAPPO dual actor (actor_n + actor_s) |
| P48b | Eval loops not passing player= to get_action_and_value | All eval loops fixed with `player = env.current_player` |
| P49 | Shared critic → catastrophic forgetting on phase switch (vl 2000+) | Dual independent centralized critic (critic_n + critic_s) |
| P49b | Joint fine-tune destabilizes coordinated protocol | Removed joint fine-tune; alternating-only training |
| P50 | Fixed prewarm rounds insufficient for critic_s; KL early stop batch-level (never triggers) | Adaptive prewarm (convergence-based); KL early stop moved to batch-level check (later found still insufficient → P51) |
| P50b | Post-BC diagnostics used north_rule=False → OOD bids → false nofit→4M | Changed to north_rule=True to isolate S quality |
| P51 | **KL early stop batch-level never triggered** (single-batch KL ≈ 0.001, cumulative drift 0.2–0.9 undetected) | **Epoch-level KL**: accumulate KL across full epoch, check average after epoch ends |
| P51 | **Critic prewarm 128 deals = statistical disaster** (40 samples/branch → severe overfit → vl explosion) | **Large static buffer**: 2048 deals collected once, actor frozen, multi-epoch fitting until vl convergence |
| P51 | **No global BC ceiling**: λ→0.1 insufficient restraint → nofit→4M hallucinations | **BC hard ceiling**: compute KL(current∥BC) before each update; skip actor if > 0.5 |

---

## Project Structure

```
bridge-coma/
├── env/
│   ├── bridge_bidding_env.py
│   └── dual_table_env.py
├── networks/
│   ├── policy_net.py                 # HandEncoder + HistoryEncoder (pack_padded)
│   └── belief_net.py                 # 48-dim output, BCEWithLogitsLoss(pw=3)
├── utils/
│   ├── scoring.py
│   ├── imp.py
│   ├── dds_data.py
│   ├── running_stats.py
│   ├── hand_features.py              # hand_to_belief_target(), belief_accuracy()
│   └── generate_subgame_data.py      # Stayman S HCP: 8–10
├── algorithms/
│   ├── ippo.py
│   ├── mappo.py                      # P49: dual independent critic; HAPPO dual actor
│   └── behavioral_cloning.py         # player= param; early stopping
├── subgames/
│   ├── stayman_env.py                # running IMP normalization; legal mask
│   ├── competitive_env.py
│   ├── subgame_trainer.py            # P51: large static buffer prewarm; epoch-level KL; BC ceiling
│   └── action_mask.py
├── experiments/
│   ├── train.py
│   └── subgame_validation.py         # P51: adaptive prewarm config; three-way eval; north_rule=True diag
├── tests/
│   ├── test_all.py
│   └── test_phase2.py
├── results/
├── data/
├── setup_project.py
└── requirements.txt
```
