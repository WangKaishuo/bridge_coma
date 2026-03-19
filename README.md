# Bridge-COMA

**Dual-Information Credit Assignment for Cooperative-Competitive Multi-Agent Coordination**

MSc Research Project — Addressing relative overgeneralization and miscoordination in bridge bidding
via information-theoretic reward shaping with prior asymmetry.

Core innovation: `r_info = I(bid;hand|partner) − β·I(bid;hand|opponent)` on top of MAPPO/HAPPO.

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

#### Stayman Subgame 🔄 P53 rewrite complete, clean run pending

| Item | Status |
|------|--------|
| Constrained dealing + DDS data generation (50k deals) | ✅ |
| Stayman subgame env (`stayman_env.py`) | ✅ |
| Action mask → generalized to `env._get_legal_actions()` | ✅ |
| Subgame trainer (`subgame_trainer.py`) | ✅ |
| N/S rule policies | ✅ |
| BC data generation — N+S joint + Round3 N response | ✅ |
| HAPPO dual-actor (actor_n + actor_s fully independent) | ✅ |
| Dual independent critic (critic_n + critic_s) | ✅ |
| Entropy revival (temperature=2.0 before Stage 2) | ✅ |
| KL early stopping (epoch-level) | ✅ |
| BC global hard ceiling (bc_kl_max) | ✅ |
| Post-BC diagnostics: north_rule=True | ✅ |
| **P52: MLP + flat input (no LSTM) — Kita et al. 2024 style** | ✅ |
| **P52: Fictitious Self-Play (FSP) checkpoint pool** | ✅ |
| **P52: lr 3e-5 → 3e-6; steps 200 → 800** | ✅ |
| **P53: PopArt value normalization** | ✅ |
| **Clean run post-P53** | ⏳ Next |
| Multi-seed validation (5 seeds) | ⏳ After clean run |

#### Competitive Subgame

| Item | Status |
|------|--------|
| Constrained dealing + DDS data (100k deals) | ✅ |
| Competitive subgame env (`competitive_env.py`) | ✅ |
| BC warmup (`behavioral_cloning.py`) | ✅ |
| Full experiment | ⏳ After Stayman confirmed |

### Phase 3–4: Not started

---

## Current Experimental State

### P52/P53 Architecture Rewrite (In Progress)

A full architectural overhaul motivated by reviewing prior literature:

**Key references:**
- Kita et al. (2024) "A Simple, Solid, and Reproducible Baseline for Bridge Bidding AI" (IEEE CoG 2024) — MLP+flat input, SL→PPO+FSP recipe, lr=1e-6
- Yu et al. (2022) "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (NeurIPS 2022) — MAPPO, **PopArt as Suggestion #1**, training epoch count guidance

**Changes in P52:**

1. **MLP + global input (no LSTM)** — Following Kita et al. 2024:
   - Deleted `HandEncoder` (52→256 MLP) and `HistoryEncoder` (2-layer LSTM 256)
   - New `encode_history_flat()`: converts `(B, T, 38)` history sequence to `(B, 152)` who-made-it binary vector. For each bid `b`, records which player called it (4-bit one-hot). Lossless for bridge (bids are strictly monotone, each real bid appears at most once).
   - Actor input: `hand(52) + history_flat(152) + position(4) + vuln(2) = 210 dims` [+ `belief_dim` if enabled]
   - Network: 4-layer MLP × 1024 units, ReLU — same as Kita et al.
   - Rationale: Stayman sequences are ≤12 tokens, no long-range dependency; LSTM's 74% parameter share brought no benefit but caused training instability

2. **Fictitious Self-Play (FSP)** — Following Kita et al. 2024 (who followed Brown et al.):
   - `FSPPool` in `mappo.py`: maintains a rolling pool of the last 10 actor snapshots (N and S separately)
   - Non-active player during rollout samples uniformly from pool instead of always using the latest policy
   - Prevents **policy cycling**: the alternating IBR (Improved Best Response) scheme in cooperative games is not guaranteed to converge to optimal Nash; FSP approximates the average policy and stabilizes training
   - Pool empty at start → falls back to latest policy (backward compatible)
   - `agent.fsp_push()` called after each N-phase completes

3. **Learning rate and step count** — Following Kita et al. 2024:
   - `stage2_lr`: 3e-5 → **3e-6** (Kita uses 1e-6; relaxed slightly for subgame scale)
   - `stage2_alt_steps`: 200 → **800** (lower lr requires more steps to accumulate same gradient magnitude)
   - Rationale: previous vl=2-7 explosions in S-phase were caused by large lr destroying BC-learned weights within a few PPO updates

4. **Removed player weighting ×3 from BC** — was a workaround for shared LSTM (N/S shared `HistoryEncoder`, N's gradient "told LSTM history is unimportant"). With independent MLP actors, this coupling no longer exists.

**Changes in P53:**

5. **PopArt value normalization** — Following Yu et al. (2022) MAPPO, Suggestion #1: *"Always use PopArt value normalization."*
   - `PopArtLayer` replaces `nn.Linear(hidden, 1)` in `ValueNetwork`
   - Maintains running `μ` and `σ` of returns via EMA (β=3e-4)
   - **Art step**: when μ/σ update, output layer weights are adjusted to preserve denormalized output continuity — `w ← w·(σ_old/σ_new)`, `b ← (b·σ_old + μ_old - μ_new) / σ_new`
   - `forward()` → denormalized value (for GAE bootstrap)
   - `normalized_forward()` → normalized value (for loss computation)
   - `normalize_target(returns)` → `(returns - μ) / σ` (training target)
   - In `critic_warmup_step` and `safe_update`: call `update_stats(b_ret)` before each batch, compute loss in normalized space
   - **Effect**: vl stays in [0, 0.3] regardless of phase switches; eliminates the vl=0.07→2.8 jump that previously required asymmetric S-phase prewarm (6 rounds ×1024 deals). Prewarm now unified at 2 rounds ×512 deals for both N and S.

### Previous Latest Full Run (P51, seed=42, 6 rounds) — Pre-rewrite baseline

**Stage 2 Final:**

| Agent | IMP | Δ vs BC |
|-------|-----|---------|
| BC-only | −4.07 | — |
| A_control (MAPPO) | −4.11 | −0.04 ← RL slightly degrades |
| B_partner_only (MAPPO+r_info) | pending | — |

**Root causes identified (led to P52/P53 rewrite):**
- `stage2_lr=3e-5` too high → vl=2-7 in S-phase → noisy advantage → wrong gradient direction
- No FSP → policy cycling under alternating IBR → S overfits to latest N rather than learning robust strategy
- LSTM overhead with no benefit in short-sequence subgame

---

## Architecture (P52/P53)

### Actor: `PolicyNetwork`

```
Input (210 dims):
  hand           : (B, 52)   — one-hot cards
  history_flat   : (B, 152)  — who-made-it encoding: 38 bids × 4 players
  position       : (B, 4)    — player one-hot
  vulnerability  : (B, 2)
  [belief]       : (B, K)    — optional BeliefNet output, stop-gradient

Network: Linear(210, 1024) → ReLU → ×3 → Linear(1024, 38)
Output: (B, 38) logits
```

### Critic: `ValueNetwork` (CTDE + PopArt)

```
Input (418 dims):
  hand           : (B, 52)
  history_flat   : (B, 152)
  position       : (B, 4)
  vulnerability  : (B, 2)
  all_hands      : (B, 208)  — 4×52 centralized info

Trunk: Linear(418, 1024) → ReLU → ×3 → (B, 1024)
Head:  PopArtLayer(1024)  → scalar (denormalized)
```

**PopArtLayer internals:**
```
μ, σ: running stats (EMA, β=3e-4)
forward(x)          → Linear(x)·σ + μ      (denorm, for GAE)
normalized_forward(x) → Linear(x)           (norm, for loss)
normalize_target(y) → (y - μ) / σ
update_stats(y)     → update μ,σ + Art weight adjustment
```

### BeliefNetwork (P52, no LSTM)

```
Input: hand(52) + history_flat(152) + pos_embed(32) + target_pos_embed(32) = 268 dims
Network: Linear(268, 512) → ReLU → Linear(512, 512) → ReLU → Linear(512, 48)
Output: (B, 48) logits — 16 honor bits (AKQJ per suit) + 32 length bits (one-hot per suit)
Loss: BCEWithLogitsLoss(pos_weight=3.0)
```

### FSP Pool (`mappo.py`)

```python
FSPPool(pool_size=10)
  .push(actor_n_state, actor_s_state)      # call after N-phase ends
  .sample_actor_state(player)              # uniform sample from pool
  .get_fsp_actor(player) → PolicyNetwork   # returns loaded historical actor
```

Non-active player calls `agent.get_action_for_player_fsp()` during rollout.
Pool empty → falls back to latest policy (first few rounds).

---

## Key Hyperparameters (P53 current)

```python
# Stage 1
stage1_steps                 = 0       # pure BC + critic warmup, no RL
critic_warmup_rounds         = 3
critic_warmup_deals          = 512

# Stage 2
stage2_alt_rounds            = 6
stage2_alt_steps             = 800     # P52: was 200
stage2_lr                    = 3e-6    # P52: was 3e-5 (Kita 2024: 1e-6)
stage2_entropy_start         = 0.05    # P52: was 0.10
stage2_entropy_end           = 0.01    # P52: was 0.05
stage2_fsp_pool_size         = 10      # P52: FSP pool

# Critic prewarm (P53 PopArt: unified, no S-phase special case)
stage2_critic_prewarm_max_rounds = 2
stage2_critic_prewarm_deals  = 512
stage2_critic_prewarm_epochs = 3
stage2_critic_prewarm_conv_tol = 0.05

# KL guards
stage2_kl_early_stop_threshold = 0.015
stage2_bc_kl_max             = 0.5

# PopArt (P53)
popart_beta                  = 3e-4    # EMA update rate for μ/σ

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
# Then manually overwrite:
#   networks/__init__.py  ← remove HandEncoder/HistoryEncoder/ActorCritic exports
```

### 3. Run Phase 2 Stayman experiment
```bash
# Single run (default seed=42)
python experiments/subgame_validation.py \
    --stayman_data data/stayman_50k.npz \
    --seed 42 \
    --device cuda

# With belief actor for S
python experiments/subgame_validation.py \
    --stayman_data data/stayman_50k.npz \
    --belief_actor_south \
    --device cuda

# Quick smoke test
python experiments/subgame_validation.py --quick \
    --stayman_data data/stayman_50k.npz
```

### 4. Key log indicators

```
[Critic Prewarm] 2 rounds × 512 deals: vl X → Y
  → With PopArt, vl should stay < 0.3 even after phase switches
  → If vl > 1.0, PopArt may not be updating (check update_stats calls)

[FSP] pool size N
  → Non-active player sampling from N historical checkpoints

⚡KL-stop
  → epoch-level KL > 0.015; actor update truncated

🚫actor-skip(bc_kl=X.XXX)
  → global BC ceiling triggered; should be rare (<10% of updates)
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

## Stayman: Scientific Assessment

**The Stayman subgame is a necessary but limited test.** Key evidence:

1. **N's bidding is 100% deterministic** — BC taught the 3-bit Stayman protocol (2♦/2♥/2♠) to near-theoretical optimality.
2. **Belief overall_acc = 0.756** — well above 0.40 target; S can already infer N's hand type perfectly from N's single bid.
3. **ir ≈ 1.0** — information channel saturated; r_info cannot improve what BC maximized.
4. **EW always pass** — β·opponent_leak term is structurally inactive in Stayman.
5. **Null result is expected** — Stayman validates infrastructure stability, not r_info.

**The real r_info test is the competitive subgame**, where:
- N has a richer signaling space (no single "correct" protocol)
- Active EW interference creates genuine partner/opponent information tension
- BC cannot reach a theoretical ceiling
- Full dual-info formula `I(bid;hand|partner) - β·I(bid;hand|opponent)` is exercised

---

## Full Bug Fix History

### Phase 1 (P0–P6)
P0 package · P1 termination · P2 reward · P3 eval · P4 scoring · P5 vulnerability · P6 dealer

### Phase 2 (P7–P53)

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
| P34 | BC missing N Round3 responses | Extended BC collection |
| P35 | 4H/4S acceptance too rare in BC | `stayman_bc_samples: 10000 → 20000` |
| P36 | r_info never wired to reward; β=0.0 | Wire ir to terminal reward; `beta=0.05`; ReLU clamp; JIT burn-in |
| P37 | `NameError: BID_PASS` | Add imports |
| P38 | BC mask mismatch | `env._get_legal_actions()`, deleted 327 lines |
| P39 | N vl explosion Round 2+ | Unified cross-round macro KL annealing |
| P40 | BeliefNetwork 52→48 dim mismatch | `hand_to_belief_target()` applied correctly |
| P41 | Critic prewarm degraded Actor | `store_episodes → compute_returns → Critic-only → buffer.reset()` |
| P42 | Adaptive prewarm removed | Fixed 3 rounds × 128 deals |
| P43 | BC over-training entropy=0 | Early stopping acc≥0.98 × patience=3 |
| P44 | BC-N quality unknown | Added rule-N vs BC-N diagnostic |
| P45 | entropy=0 → PPO cannot update | `_revive_entropy(temperature=2.0)` before Stage 2 |
| P46 | temperature=2.0 → OOD bids | (reverted; 2.0 retained) |
| P47 | N-phase prewarm vl diverges | `stage2_n_critic_prewarm_rounds=6` (superseded by P49) |
| P48 | Shared actor → N/S gradient cross-contamination | HAPPO dual actor (actor_n + actor_s) |
| P48b | Eval loops not passing player= | All eval loops fixed |
| P49 | Shared critic → catastrophic forgetting (vl 2000+) | Dual independent centralized critic (critic_n + critic_s) |
| P49b | Joint fine-tune destabilizes protocol | Removed joint fine-tune |
| P50 | Fixed prewarm insufficient; KL stop batch-level | Adaptive prewarm; KL stop at epoch level (later found still insufficient → P51) |
| P50b | Post-BC diagnostics OOD bids | `north_rule=True` diagnostics |
| P51 | KL early stop batch-level never triggered | **Epoch-level KL**: accumulate across full epoch |
| P51 | Critic prewarm 128 deals = statistical disaster | **Large static buffer**: 2048 deals, actor frozen, multi-epoch |
| P51 | No global BC ceiling | **BC hard ceiling**: skip actor if KL(current∥BC) > 0.5 |
| **P52** | **RL degrades BC (vl 2–7, policy cycling)** | **MLP+flat input (no LSTM)** following Kita et al. 2024; **FSP pool** following Kita et al. 2024; **lr 3e-5→3e-6** following Kita et al. 2024 |
| **P52** | **_auto_play_non_agent: OOD bids after game-level** | Auto-pass all players once game-level contract reached (3NT/4M+); BC/RL never trained on post-game positions |
| **P52** | **nofit→4M false positive in diagnostics** | Root cause: S gets OOD turn after N accepts invitation (4H); fixed by game-level auto-pass |
| **P52** | **`networks/__init__.py` exports deleted classes** | Remove `HandEncoder`/`HistoryEncoder`/`ActorCritic`; export `PolicyNetwork`/`ValueNetwork`/`BeliefNetwork` |
| **P52** | **Removed player weighting ×3 in BC** | Was LSTM workaround; MLP actors are independent, coupling no longer exists |
| **P53** | **vl=0.07→2.8 on S-phase phase switch, requires asymmetric prewarm** | **PopArt value normalization** following Yu et al. 2022 (MAPPO Suggestion #1); vl stays in normalized space regardless of reward scale shifts |

---

## Project Structure

```
bridge-coma/
├── env/
│   ├── bridge_bidding_env.py
│   └── dual_table_env.py
├── networks/
│   ├── policy_net.py        # P52: MLP+flat; P53: PopArtLayer in ValueNetwork
│   └── belief_net.py        # P52: MLP+flat (no LSTM); 48-dim, BCEWithLogitsLoss(pw=3)
├── utils/
│   ├── scoring.py
│   ├── imp.py
│   ├── dds_data.py
│   ├── running_stats.py
│   ├── hand_features.py
│   └── generate_subgame_data.py
├── algorithms/
│   ├── ippo.py              # P52: updated to PolicyNetwork/ValueNetwork
│   ├── mappo.py             # P52: FSPPool; P53: PopArt-aware critic calls
│   └── behavioral_cloning.py
├── subgames/
│   ├── stayman_env.py       # P52: game-level auto-pass in _auto_play_non_agent
│   ├── competitive_env.py
│   ├── subgame_trainer.py   # P53: PopArt in warmup+safe_update; P52: FSP rollout
│   └── action_mask.py
├── experiments/
│   ├── train.py
│   └── subgame_validation.py  # P52: FSP config, new lr/steps; P53: simplified prewarm
├── tests/
├── results/
├── data/
├── setup_project.py
└── requirements.txt
```

---

## Key Design Decisions & Learnings

| Decision | Rationale |
|----------|-----------|
| MLP+flat over LSTM | Kita et al. 2024: LSTM provides no benefit for short bridge sequences (≤12 tokens); MLP trains faster and more stably |
| Who-made-it history encoding | Lossless for monotone bidding; 152 dims vs 2280 (naive flatten); Kita et al. use similar 480-dim variant |
| FSP over latest-policy self-play | IBR in cooperative games not guaranteed to converge; FSP approximates average policy, prevents cycling (Kita et al. 2024, Brown 2019) |
| PopArt over prewarm tuning | Yu et al. 2022 Suggestion #1; normalizes value targets, eliminates phase-switch vl explosions at the source rather than treating symptoms |
| lr=3e-6 (not 1e-6) | Kita uses 1e-6 for full-game training on 1M boards; subgame scale allows slight relaxation |
| Dual independent critic (N+S) | Phase switches create non-stationarity; shared critic causes catastrophic forgetting (vl=2000+) |
| Epoch-level KL early stop | Single-batch KL ≈ 0.001 (never triggers at 0.015); epoch-average KL correctly reflects cumulative drift |
| BC hard ceiling (skip actor) | λ→0.1 insufficient restraint late in training; ceiling prevents irreversible BC destruction |
| Stayman as infrastructure test | 3-bit protocol is BC-optimal; r_info signal is structurally saturated; serves as stability check only |
