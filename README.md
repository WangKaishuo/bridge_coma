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
6. **belief_net.py was extended in P97** with EWC support (`compute_fisher`, `ewc_penalty`). These are optional and currently disabled.
7. **subgame_trainer.py was extended in P97b/c** with pretrain replay mixing in `update_belief_on_policy()` and `evaluate_partner_info_gain()` diagnostic.
8. **`sl_pretrain.py` is the BCA (349-dim) version** (P98 refactor). Do NOT use the old 301-dim `sl_pretrain.py`; that file no longer exists. Output: `results/sl_base_bca.pt`.
9. **Paired eval is now built into `subgame_validation.py`** via `--eval-only`. `eval_paired.py`, `diagnose_partner_gain.py`, and `test_phase2.py` have been removed.

---

## Project Progress

### Phase 1: Environment & Infrastructure ✅ Complete

### Phase 2: Subgame Validation ✅ Complete (preparing for thesis write-up)

#### Stayman ⏸ Deferred — null result structurally expected

#### Competitive Subgame ✅ Complete — key findings established

| Item | Status |
|------|--------|
| Env + DDS data (500k deals) | ✅ |
| P54–P86: Core infrastructure + Belief Net rewrite | ✅ |
| P87b: r_info weight 0.02→0.2 | ✅ |
| P88: KL anneal 0.5→0.0, first B>A result (contaminated) | ✅ |
| P93: Dealer eval bug fix + paired eval | ✅ |
| P94: Convention drift discovery + quantification | ✅ |
| P95: λ=0.5, on-policy belief (B>A +0.31, p<0.001) | ✅ |
| P96: λ=0.5, frozen belief (A>B, noise r_info) | ✅ |
| **P97a-d: Belief Net stabilization + communication diagnostic** | ✅ |
| **P97d: λ=0.3, partner_gain +10.8%, but A>B in IMP** | ✅ |

---

## ⚠️ KEY DISCOVERY: Convention Drift (P94)

When KL anchor is weak (λ→0) or moderate (λ=0.3), RL agents develop **convention drift** — bidding semantics diverge from the shared SAYC protocol. This violates bridge's **Full Disclosure principle** and creates an illegitimate advantage in cross-system evaluation.

**Quantitative evidence (P94b, λ=0.3):** DDS regret +1.79 IMP vs vs-SL +3.80 IMP → **~2 IMP convention drift advantage**.

**Impact:** All prior bridge AI papers (Kita 2024, Gong 2024) use SL→RL with no KL constraint. Their reported improvements may partially reflect convention drift, not pure bidding quality. None discussed this confound.

**MARL translation:** "Full disclosure" = shared communication protocol (Hu et al. 2020); "convention drift" = arbitrary handshakes; KL anchor = protocol regularization.

---

## ⚠️ KEY FINDING: r_info Changes Communication But Not Outcomes (P97d)

### The Communication-Outcome Disconnect

Under λ=0.3, r_info produces a **measurable 10.8% increase in partner information gain** (S-position: +10.1%), confirming that the dual-information bonus successfully modifies communication behavior. However, this improved communication does **not translate to better IMP outcomes** — in fact, Agent B (with r_info) performs slightly worse than Agent A in head-to-head evaluation.

This suggests that **optimizing "how clearly to communicate" and "what is the best bid"** are partially conflicting objectives: the auxiliary reward diverts policy optimization away from the primary IMP objective.

---

## Belief Net Degradation: Root Cause + Solution (P97 series)

### Root cause: Pretrain overfitting, not catastrophic forgetting

The original pretrain (10k deals, 300 epochs) produced val_loss=1.76 with severe overfitting (train-val gap: 1.19 vs 1.76). On new RL-trajectory data, loss jumped to 2.19 — this was **distribution shift to unseen samples**, not policy-induced OOD.

### P97 series: systematic investigation

| Experiment | Belief Strategy | Pretrain | length_acc | val_loss trajectory |
|-----------|----------------|----------|------------|-------------------|
| P95 | On-policy 3ep/1e-5 | 10k, 300ep | 0.255 | 1.76→2.19 (Round 1 destruction) |
| P96 | Frozen | 10k, 300ep | 0.220 | N/A (frozen, OOD) |
| P97a | EWC (λ_ewc=5000) | 10k, 300ep | — | ewc_pen=0.000005 (no effect) |
| P97a' | EWC (normalized Fisher) | 10k, 300ep | — | ewc_pen=0.000002 (still no effect) |
| P97b-50/50 | Replay 50/50 mix | 10k, 300ep | — | val_loss=2.34 (worse—conflicting gradients) |
| P97b | Frozen | **100k, 50ep** | 0.278 | No overfitting; frozen still OOD |
| **P97c** | **Unfreeze 1ep + 80/20 replay** | **100k, 50ep** | **0.262** | **1.89→1.94 (stable!)** |
| **P97d** | **Same as P97c, λ=0.3** | **100k, 50ep** | **0.265** | **1.89→1.95 (stable)** |

### Key lessons

1. **EWC failed** because at pretrain convergence, gradients are tiny → Fisher diagonal ≈ 1e-6 → penalty negligible regardless of λ_ewc. The problem is data-level (new samples overwhelm), not weight-level.
2. **50/50 replay was worse** because pretrain gradients pulled weights toward a direction that was bad for on-policy data — conflicting objectives in a single batch.
3. **80/20 replay + 1 epoch works**: val_loss stable at ~1.95 across 15 rounds (Δ=+0.06 from pretrain, vs P95's Δ=+0.43). The key was: (a) 100k pretrain with no overfitting, (b) only 1 epoch per round, (c) 80/20 ratio so on-policy dominates.
4. **length_acc remains at ~0.26-0.28 regardless**: this appears to be the **true ceiling** for this Belief Net architecture on RL trajectories with KL≈0.17-0.22. The pretrain's 0.391 is its accuracy on SL trajectories specifically.

---

## Full Experimental Results (Competitive Subgame)

### Master comparison table

| Exp | λ | Belief | length | A vs B H2H | Paired diff | partner_gain Δ | Conclusion |
|-----|---|--------|--------|------------|-------------|----------------|------------|
| P88 | →0 | JIT 10k | 0.261 | **B>A +3.18** ✅ | -1.89 ✅ | ? | ❌ Drift contaminated |
| P95 | 0.5 | on-policy 3ep 10k | 0.255 | **B>A +0.31** ✅ | +0.01 ns | ? | ✅ First clean B>A (artifact?) |
| P96 | 0.5 | frozen 10k | 0.220 | **A>B +0.30** ✅ | — | ? | ❌ Noise r_info harms B |
| P97b | 0.5 | frozen 100k | 0.278 | A≈B -0.04 ns | -0.05 ns | ? | — No difference |
| P97c | 0.5 | unfreeze+replay 100k | 0.262 | **A>B +0.18** ✅ | -0.007 ns | +1.3% | ❌ r_info slightly harmful |
| **P97d** | **0.3** | **unfreeze+replay 100k** | **0.265** | **A>B +0.31** ✅ | +0.05 ns | **+10.8%** | ⚠️ Communication changes, IMP doesn't |

### P97d detailed results (λ=0.3, 10 rounds, paired eval 5000 deals)

| Matchup | IMP | p-value | |
|---------|-----|---------|--|
| A vs SL | +3.910 | 0.000 ✅ | |
| B vs SL | +3.862 | 0.000 ✅ | |
| A vs B (H2H) | **+0.307** | 0.000 ✅ | **A wins** |
| Paired diff | +0.048 | 0.160 (ns) | No diff vs SL |

### P97d partner info gain diagnostic (500 deals, B's belief net as judge)

| Position | Agent A | Agent B | Δ |
|----------|---------|---------|---|
| N (opener) | 0.0087 | 0.0086 | -0.1% |
| S (responder) | 0.1335 | **0.1470** | **+10.1%** |
| Overall | 0.0844 | **0.0934** | **+10.8%** |

**Interpretation:** r_info successfully increases South's information transmission to North by 10%, but this does not improve (and slightly harms) IMP outcomes. The S-position is where communication decisions happen; N's opener rebids are low-information regardless.

---

## Current Training Pipeline (P97d)

### Stage 1: SL Initialization
Load `results/sl_base.pt` (9.9M SAYC deals, 4 actors with identical weights).

### Stage 1.5: Belief Net Pretrain (Agent B only)
**100k deals** (P97b fix: 10×previous), 50 epochs, no overfitting.
Final: val_loss=1.89, honor=0.762, length=0.391.

### Stage 2: RL Fine-tuning (10 rounds)

Each round:
1. FSP pool: sample checkpoint as opponent (SL permanent, add every round)
2. **Table 1 (NS)**: collect 32768 deals, agent=NS, FSP=EW → r_info bonus → PPO update N+S
3. **Table 2 (EW)**: collect 32768 deals, agent=EW, FSP=NS → r_info bonus → PPO update E+W
4. **Belief Net: Unfreeze, 1 epoch on-policy + 20% pretrain replay** (P97c)
5. **Mini eval**: vs SL H2H (500 deals)

### Stage 3: Evaluation
- A vs SL, B vs SL, A vs B (1000 deals each, Wilcoxon) — runs automatically at end of training
- **Paired eval** (`subgame_validation.py --eval-only`): same deals, Wilcoxon signed-rank + paired diff test
- **Belief eval**: frozen pretrain Belief Net accuracy on RL trajectories
- **Partner info gain diagnostic**: A vs B communication comparison, runs inside Stage 3 when `use_info_bonus=True`

### Hyperparameters (P97d)

| Param | Value | Note |
|-------|-------|------|
| lr | 3e-6 | Kita et al. |
| kl_lambda | **0.3** | P97d: relaxed from 0.5 for more policy freedom |
| deals_per_step | 512 | |
| steps_per_phase | 64 | → 32768 deals/table/round |
| num_rounds | **10** | Converges by Round 8 |
| beta (internal) | 0.05 | I(partner) - β·I(opponent) |
| info_reward_weight | 0.2 | |
| fsp_pool_size | 10 | 1 permanent SL + 9 FIFO |
| fsp_add_interval | **1** | Every round |
| freeze_belief | **False** | P97c: unfreeze with replay |
| belief_update_epochs | **1** | P97c: gentle adaptation |
| belief_update_lr | 1e-5 | |
| belief_pretrain_rounds | **50** | 100k deals total |
| belief_pretrain_max_epochs | **50** | No overfitting |
| replay mixing ratio | **80/20** | On-policy / pretrain |

---

## Key Lessons Learned (P94–P97d)

1. **Convention drift is real and large.** ~2 IMP advantage from drift alone (P94b). Prior bridge AI papers do not account for this confound.

2. **KL anchor is not a training trick — it's protocol compliance.** Without it, agents develop private conventions that violate Full Disclosure.

3. **Belief Net pretrain overfitting was the hidden killer.** 10k deals × 300 epochs → severe overfitting → any new data caused apparent "catastrophic forgetting" that was actually distribution shift to unseen samples. Fix: 100k deals × 50 epochs.

4. **EWC does not work for this problem.** At pretrain convergence, Fisher diagonal ≈ 1e-6 → penalty negligible. The problem is data-level (30万 new samples overwhelm 4.6万 pretrain), not weight-level.

5. **80/20 pretrain replay + 1 epoch is the correct belief update strategy.** Val_loss stable at ~1.95 across 15 rounds (Δ=+0.06), vs P95's Δ=+0.43 destruction.

6. **r_info changes communication behavior** (partner_gain +10.8% at λ=0.3) **but does not improve IMP outcomes** (A>B +0.31). "Saying more clearly" ≠ "bidding better". The auxiliary reward partially conflicts with the primary IMP objective.

7. **P95's B>A +0.308 was likely an artifact** of degraded belief (val_loss 2.19). When belief is properly maintained (P97c/d), B does not beat A.

8. **The only genuine B>A effect comes from convention drift** (P88, λ→0, +3.18 IMP). Under protocol-compliant conditions, r_info has zero or slightly negative effect on IMP.

---

## Scientific Narrative for Thesis

### Three contributions

1. **Convention drift quantification** (novel): First quantitative evidence that RL self-play in bridge produces ~2 IMP illegitimate advantage through private conventions. Identifies a methodological blind spot in Kita 2024, Gong 2024, etc.

2. **Protocol compliance as constrained optimization** (novel framing): Full Disclosure formalized as $\max_\pi J(\pi)$ s.t. $D_{\text{KL}}(\pi \| \pi_{\text{SL}}) \leq \epsilon$. KL anchor reframed from ad-hoc hyperparameter to Lagrange multiplier.

3. **Communication-outcome disconnect** (honest negative finding): r_info successfully modifies communication behavior (+10.8% partner information gain) but this does not translate to IMP improvement. Demonstrates that **information-theoretic optimality and decision-theoretic optimality can diverge** in constrained policy spaces.

### Pareto frontier (λ sweep)

| λ | KL | DDS Regret | vs SL IMP | Drift advantage | r_info effect |
|---|-----|------------|-----------|-----------------|---------------|
| 0 (P88) | high | ~+1.8 | ~+6-8 | ~+4-6 | B>A +3.18 (contaminated) |
| 0.3 (P97d) | ~0.22 | ~+1.5 | ~+3.9 | ~+1-2 | partner_gain +10.8%, IMP ns |
| 0.5 (P97c) | ~0.17 | ~+1.6 | ~+3.6 | small | partner_gain +1.3%, IMP ns |
| 1.5 | ~0.02 | ~-0.5 | ~0 | ~0 | locked at SL |

### Open questions for thesis

1. Would a **policy architecture that conditions on belief** (actor uses belief net output as input) allow r_info to actually improve decisions, not just communication?
2. Can **Semantic Fidelity Score** (KL between SL and RL bid-conditioned hand posteriors) provide a finer-grained measure of convention drift than aggregate KL?
3. Would the full bidding game (not 1H-1S subgame) provide more room for r_info to matter?

---

## Full Bug Fix History

### Phase 1 (P0–P6)
P0 package · P1 termination · P2 reward · P3 eval · P4 scoring · P5 vulnerability · P6 dealer

### Phase 2 (P7–P97d)

| # | Problem | Fix |
|---|---------|-----|
| P7–P53 | Various early issues | See previous README versions |
| P54–P77 | Competitive env infrastructure | Dealer rotation, dual-table, FSP, batch rollout |
| P82 | NS/EW asymmetry | Dual-table symmetric training |
| P83 | r_info drowns base reward | Separate beta vs info_reward_weight |
| P84 | Belief Net catastrophic forgetting | FIFO Replay Buffer (50k) — superseded |
| P86 | Belief Net miscalibration | No pos_weight, CE for length, prior bias init |
| P87b | r_info signal invisible to PPO | info_reward_weight 0.02→0.2 |
| P88 | KL=1.5 locked agent at SL | KL anneal: 0.5→0.0 over 50% rounds |
| P93 | Dealer encoding bug / stale replay / unpaired eval | P93 bundle fix; eval_paired logic later absorbed into subgame_validation.py |
| P94 | Convention drift discovery | Quantified ~2 IMP drift advantage |
| P95 | On-policy belief destroys pretrain | val_loss 1.76→2.19 in Round 1 |
| P96 | Frozen belief OOD | length 0.488→0.220, noise r_info |
| **P97a** | **EWC for belief protection** | **Failed: Fisher ≈ 1e-6 at convergence, penalty negligible** |
| **P97b** | **Pretrain overfitting diagnosis** | **100k deals, 50 epochs → no overfitting, val_loss 1.89** |
| **P97b-replay** | **50/50 replay mixing** | **Failed: conflicting gradients made val_loss worse (2.34)** |
| **P97c** | **80/20 replay + 1 epoch** | **val_loss stable at 1.95 across 15 rounds ✅** |
| **P97d** | **λ=0.3 + partner_gain diagnostic** | **partner_gain +10.8%, but IMP A>B +0.31** |

---

## Project Structure

```
bridge-coma/
├── env/
│   ├── bridge_bidding_env.py       # single-table bidding env
│   └── dual_table_env.py           # dual-table IMP env
├── networks/
│   ├── policy_net.py               # MLPPolicyNetwork (301-dim / 349-dim BCA), encode_obs_flat
│   └── belief_net.py               # P86: dual-head BCE; P97: optional EWC
├── algorithms/
│   ├── mappo.py                    # HAPPO: actor/critic ×4, independent optimizers
│   ├── ippo.py
│   └── behavioral_cloning.py
├── utils/
│   ├── scoring.py                  # bridge score SSOT
│   ├── imp.py                      # IMP conversion table
│   ├── dds_data.py                 # DDS data generation/loading
│   ├── running_stats.py            # Welford online stats
│   ├── hand_features.py
│   ├── fsp_pool.py
│   ├── sl_pretrain.py              # P98: BCA two-stage SL (Belief Net + 349-dim Actor) → sl_base_bca.pt
│   └── generate_subgame_data.py
├── subgames/
│   ├── stayman_env.py
│   ├── competitive_env.py          # P93: dealer fix
│   ├── subgame_trainer.py          # P97c: replay mixing + partner_gain diagnostic
│   ├── subgame_validation.py       # P97d: training entry + --eval-only paired eval mode
│   └── action_mask.py
├── experiments/
│   └── train.py
├── tests/
│   └── test_all.py                 # 35 Phase 1 infrastructure tests
├── results/
└── data/
    └── competitive_500k.npz
```

### Removed files (refactor)
- `sl_pretrain_bca.py` → merged into `sl_pretrain.py`
- `eval_paired.py` → absorbed into `subgame_validation.py --eval-only`
- `diagnose_partner_gain.py` → diagnostic already inside `subgame_trainer.evaluate_partner_info_gain()`
- `test_phase2.py` → superseded by Stage 3 eval in `subgame_validation.py`

---

## Compute Budget (T4 GPU)

| Stage | Time |
|-------|------|
| SL pretrain — Stage A: Belief Net (200k deals, 30 epochs) | ~20 min |
| SL pretrain — Stage B: 349-dim Actor (10 epochs) | ~25 min |
| Belief pretrain (100k deals, 50 epochs) | ~30 min |
| Agent A (10 rounds) | ~40 min |
| Agent B (10 rounds, belief update) | ~45 min |
| Stage 3 eval (3×1000 deals) | ~5 min |
| `--eval-only` paired eval (2000 deals ×3) | ~10 min |
| `evaluate_partner_info_gain` (500 deals, inside Stage 3) | ~5 min |

---

## Pending / Next Steps

1. **Thesis write-up**: Convention drift + communication-outcome disconnect as dual contributions
2. **Optional**: Multi-seed validation of P97d (seeds 42, 123, 456) to confirm partner_gain finding
3. **Optional**: Semantic Fidelity Score implementation for finer drift quantification
4. **Optional**: Stayman re-run under P97c framework (expected: null result, confirming communication ceiling)
5. **Preliminary Report**: LaTeX draft completed (Bristol template), needs revision based on P97d findings

---

*README version: P97d + refactor*
*Last updated: 2026-03-23*
