# Bridge-COMA

**Dual-Information Credit Assignment for Cooperative-Competitive Multi-Agent Coordination**
MSc Research Project — Kaishuo Wang, 2026

$$r_{\text{info}} = I(\text{bid};\,\text{hand} \mid \text{partner}) - \beta \cdot I(\text{bid};\,\text{hand} \mid \text{opponent})$$

---

## ⚠️ CRITICAL WORKFLOW NOTES (read at start of every session)

1. **Claude has NO cross-session memory.** Paste this README at the start of each new conversation.
2. **NEVER use `/mnt/project/` as base for edits.** That directory is the version last manually uploaded and may be several patches behind. Always base edits on the most recent file in `/home/claude/` or `/mnt/user-data/outputs/`.
3. **Config lives in `subgame_validation.py`, not `subgame_trainer.py`.** SubgameConfig kwargs in `subgame_validation.py` override all defaults. Always edit `subgame_validation.py` for hyperparameter changes.
4. **Key files to keep in sync:** `policy_net.py`, `subgame_trainer.py`, `subgame_validation.py`, `competitive_env.py`, `belief_net.py`, `fsp_pool.py`, `drift_sweep.py`, `bid_inspector.py`, `sl_pretrain_bca.py`, `bridge_bidding_env.py`.
5. **P108: All RL obs now use OpenSpiel 571-dim via `hands_to_openspiel_state`.** `encode_obs_flat` has been deleted from the entire codebase. `pip install open_spiel` required every Colab session.
6. **SL checkpoints:** `results/sl_base.pt` (571-dim, 400k iters, non_pass_acc=86.6%), `results/sl_base_bca.pt` (BeliefNet only, honor_acc=0.768, length_acc=0.421), and `results/sl_base_bca_stageB.pt` (667-dim Actor + BeliefNet, non_pass_acc=87.7%). Do NOT retrain any of these. They are trained on None-vul SAYC data; RL learns vul-dependent strategy on top.
7. **`drift_sweep.py` modes:** `--mode 571` (no BCA) and `--mode 667` (BCA). Old `480`/`576` values are gone.
8. **P103: Dealer rotation bug fixed in `competitive_env.py`.** ALL prior RL experiments invalidated.
9. **P102: SAYC deck parsing bug fixed.** All SL checkpoints before P105 are invalid. `competitive_500k.npz` is NOT affected.
10. **Card encoding boundary:** OpenSpiel uses rank-major (`rank*4+suit`). `competitive_env` uses suit-major (`suit*13+rank`). `convert_hands_suit_to_rank()` in `policy_net.py` handles conversion.
11. **OpenSpiel `observation_tensor()` does NOT contain private hand cards.** It is a public-information tensor (bidding history + vulnerability). The SL model learns to predict next bid from bidding history alone — this is correct and intentional (86.6% non_pass_acc confirms it works).
12. **Dealing order for `hands_to_openspiel_state`:** interleaved `p0[0], p1[0], p2[0], p3[0], p0[1], ...` — NOT consecutive 13 cards per player. Wrong order produces different obs.
13. **Colab 12-hour limit:** Use `drift_sweep.py` with `--lambdas` and `--seeds` to split work across sessions. It auto-skips completed runs.
14. **Save results after every Colab session.** Mount Google Drive BEFORE running experiments, then use `--gdrive_dest` for auto-upload. Do NOT write checkpoints directly to Google Drive during training — the write latency will slow training by ~10x.
15. **Stage 3 eval_deals=5000, no per-round H2H.** Per-round H2H was removed (P110) — too noisy. All statistical evaluation deferred to Stage 3.
16. **`_deal_action_cache` size is 8192** (was 2048). Stage 3 with 5000 deals requires larger cache to avoid 0.64s/deal rebuild overhead.
17. **DDS regret IS opponent-dependent.** Do NOT claim it as an opponent-independent metric. actual_score depends on both sides' bidding. This is a known limitation documented in the experiment design.
18. **`bid_inspector.py` now uses cross-table (双桌对战).** play_deal() calls play_mixed() twice, same semantics as cross_evaluate(). Old single-table comparison was scientifically incorrect.
19. **`drift_sweep.py` now supports `--gdrive_dest`.** Mount Drive in a separate cell first (`from google.colab import drive; drive.mount('/content/drive')`), then pass `--gdrive_dest '/content/drive/MyDrive/...'`. The script will auto-upload after all runs complete.
20. **Stage B of BCA SL pretrain: necessary for SL opponent, NOT for Agent A/B.** For Agent A/B: zero-init the 96 belief columns and let RL co-evolve Actor + BeliefNet. For SL as Stage 3 opponent: must run Stage B (`sl_base_bca_stageB.pt`) so the opponent can actually utilise belief features — without Stage B, SL's belief columns are zero and it cannot read the convention card at all.
21. **BCA RL training: freeze_belief=False + belief_warmup_rounds=5.** BeliefNet must update during RL to track agent's drifting protocol. belief_warmup_rounds=5 uses prior features for first 5 rounds to stabilise Actor before co-evolution begins.
22. **Agent bidding differs against different opponents — this is correct.** The 571-dim obs encodes full bidding history. When opponent bids differently, the history diverges, so obs diverges, so agent's response legitimately differs. This is NOT a bug.
23. **FSP is preferred over self-play for training.** Self-play causes near-zero reward signal (both agents symmetric, advantages ≈ 0). FSP with SL as permanent pool entry provides stable distribution anchor. Use `self_play=False` in SubgameConfig.
24. **`load_sl()` and `load_agent()` have different weight mappings (P115).** Never use `load_sl()` to load RL checkpoints. Use `--type1 agent` in bid_inspector for RL checkpoints.
25. **P116 (CRITICAL): `_make_play_mixed_policy` and `make_agent_policy` both had two bugs.** (1) legal_mask from BridgeBiddingEnv instead of OpenSpiel state. (2) actor selected by real seat instead of `(player-dealer)%4`. Fix applied to both. Training code unaffected.
26. **P117 (CRITICAL): dealer rotation caused 50% of training rewards to have wrong sign.** `_compute_score_ns` and `_compute_dds_optimal_score_ns` hardcoded seat 0,2 = opener (NS). When dealer ∈ {E,W}, opener阵营打庄被误判为"EW庄家"，奖励符号反转。Fix: both functions now accept `dealer=` parameter; all callers updated. **All pre-P117 RL training results are invalid.**
27. **P117 is the most severe bug in the project history.** It corrupted 50% of training signal for all experiments.
28. **P118: `bid_inspector.py` alignment labels fixed.** `A(NS) vs B(EW)` labels were hardcoded; now show actual opener/overcaller seats.
29. **P119: BeliefNet API fully migrated to new format.** Old API: `get_probs(oh, h, op, tp)` 4 args. New API: `get_probs(obs_571, target_pos)` 2 args. All call sites updated.
30. **P119: `--belief_checkpoint` parameter added to `subgame_validation.py` and `drift_sweep.py`.**
31. **P120: BCA rollout batching + `bid_inspector.py` 667-dim support.**
32. **P121: `bid_inspector.py` BCA attachment fix; `sl_pretrain_bca.py` `--load_belief`; Stage 3 auto-load StageB SL.**
33. **P122 (CRITICAL): Three bugs fixed + vulnerability randomization. ALL pre-P122 RL results are invalid.**
    - **(a) BeliefNet target_pos bug in `bid_inspector.py`:** Used relative seat `(player-dealer)%4` instead of absolute seat. Training code (`_get_belief_features_single`) uses absolute seats. Mismatch caused wrong belief features for 75% of deals (when dealer≠0). Fix: `partner = (player+2)%4, rho = (player-1)%4` (absolute).
    - **(b) BeliefNet not saved in agent checkpoint:** `MAPPOAgent.save()` only saves actor/critic. Co-evolved BeliefNet was lost. Fix: `subgame_validation.py` now appends `belief_net` state_dict to agent checkpoint. `bid_inspector.py` auto-detects embedded BeliefNet (priority over `--belief_checkpoint`).
    - **(c) Vulnerability randomization:** ALL prior training and evaluation used fixed `(False, False)` (None-vul). OpenSpiel 571-dim obs includes 4-dim vulnerability features that were always zero. Fix: RL training now randomly samples from 4 vul states per deal. Evaluation (`cross_evaluate`, `bid_inspector`) also randomized. SL pretrain unchanged (SAYC data is None-vul). Agent learns vul-dependent strategy during RL.
    - **(d) `sl_pretrain_bca.py --freeze_base`:** Freezes first layer's 571-dim columns during Stage B training. Only belief columns (96-dim) and subsequent layers trainable. Intent: graceful degradation to plain SL when belief is uninformative. Result: insufficient — posterior layers still drift. Documented as negative result.
34. **P122 invalidates ALL prior RL experiments.** Vul randomization changes the training distribution fundamentally. Agent must learn when to be aggressive (non-vul) vs conservative (vul). Prior agents trained only on non-vul developed unconstrained aggressive strategies.
35. **`competitive_500k.npz` does NOT need regeneration.** Vul is assigned dynamically at rollout time, not baked into deal data.
36. **SL checkpoints do NOT need retraining.** SAYC expert data was generated under None-vul. SL learns SAYC convention; RL learns vul-dependent adjustments on top.
37. **P123: `subgame_validation.py` now supports `--agent_b_only`.** Trains only Agent B (MAPPO+BCA+r_info), skipping A and C. Mutually exclusive with `--agent_a_only`. `drift_sweep.py` also supports `--agent_b_only`.
38. **P123: `_sl_bc` bug fixed in `subgame_validation.py`.** Two references to undefined `_sl_bc` (should be `_bc`) would crash when SL eval falls back to belief pretrain. Fixed.
39. **P124: Action encoding mismatch in `sl_pretrain_bca.py`.** `sl_base.pt` uses `action - 52` encoding (Pass=0, 1C=1, 1D=2, ...). `sl_pretrain_bca.py` used `openspiel_raw_to_ours` encoding (Pass=0, Double=1, Redouble=2, 1C=3). All non-pass bids off by 2. Stage B legacy "worked" because full fine-tune re-learned the mapping. ReFine mode (frozen actor) exposed the bug. **Fixed: all data classes in `sl_pretrain_bca.py` now use `action - 52` encoding.**
40. **P124: ReFine Residual Belief Adapter (`sl_pretrain_bca.py --mode refine`).** Freezes plain SL actor entirely; trains lightweight adapter (96→128→1024, ~144k params, 3.7% of total) to inject belief features via residual connection after layer 0. Gate parameter initialized to 0 ensures exact plain-SL behavior at start. Inspired by ReFine (Xu et al. 2025) with no-negative-transfer guarantee.
41. **P124: SL_BCA(StageB) weakness diagnosed.** Stage B full fine-tune changes actor bidding pattern AND makes actor dependent on belief features. `SL vs SL_BCA = -1.1 IMP` is NOT convention card effect — it's negative transfer from Stage B training. The 4.1→1.8 IMP drop (Agent A vs SL → Agent A vs SL_BCA) is primarily pattern-change effect, confirmed by ablation (real vs prior ≈ 0 difference).
42. **P124: SAYC BeliefNet belief output ≠ prior (L1 dist ≈ 0.137), and belief columns contribute ~24% of layer-0 activation.** Despite this, ablation has no effect on decisions. Investigation ongoing — likely related to cross-protocol OOD behavior during actual drifted-bid games (vs initial-position analysis).
43. **P124: ReFine adapter gate converges to ≈0 on SAYC data.** Expected: plain SL already extracts all information from obs_571 that BeliefNet can provide. Belief features are redundant for SAYC SL. ReFine value appears only when co-evolved BeliefNet reads agent's own drifted protocol (see TODO).

---

## Project Progress

| Item | Status |
|------|--------|
| P54–P86: Core infrastructure + Belief Net rewrite | ✅ |
| P87b–P97d: 301-dim experiments | ❌ Invalidated |
| P98–P101: BCA architecture (397-dim) | ❌ Superseded by P105 |
| P102: SAYC deck bug fix | ✅ |
| P103: Dealer rotation bug fix | ✅ |
| P105: OpenSpiel-native 571-dim SL (`sl_pretrain.py`) | ✅ |
| Train `sl_base.pt` (400k iters, non_pass_acc=86.6%) | ✅ Done |
| P106–P108: `hands_to_openspiel_state` fixes | ✅ |
| P109–P121: Infrastructure improvements | ✅ |
| Train `sl_base_bca.pt` (BeliefNet Stage A) | ✅ Done |
| Train `sl_base_bca_stageB.pt` (Stage B legacy, 667-dim) | ✅ Done |
| P122: Vul randomization + BeliefNet save + target_pos fix | ✅ |
| P123: `--agent_b_only` mode + `_sl_bc` bug fix | ✅ |
| P124: Action encoding fix + ReFine adapter + SL_BCA diagnosis | ✅ |
| 571-dim Agent A RL training (seed=100, 25 rounds) | ✅ Done |
| 667-dim Agent A RL training (seed=100, 25 rounds) | ✅ Done |
| 667-dim Agent B RL training (seed=100, 25 rounds, β=0.0) | ✅ Done |
| Train `sl_base_bca_refine.pt` (ReFine, gate≈0) | ✅ Done |
| **subgame_trainer.py performance optimization (P_OPT)** | ✅ Done |
| **bid_inspector ReFine adapter support** | ⏳ TODO |
| **Convention card with co-evolved BeliefNet (Route 2)** | ⏳ TODO |
| **Multi-seed validation (5 seeds)** | ⏳ Pending |
| **BCA core experiment (Agent A vs Agent B, r_info)** | ⏳ Pending |

---

## Valid Experimental Results (Post-P122, seed=100)

### RL Training Results

| Agent | Rounds | Final regret | Belief loss | step_ir |
|-------|--------|-------------|-------------|---------|
| 571-dim Agent A | 25 | +1.625 ± 10.203 | — | — |
| 667-dim Agent A | 25 | +1.491 ± 10.163 | 1.907 | — |
| 667-dim Agent B (β=0.0) | 25 | +1.460 ± 10.204 | 1.940 | 2.505 |

### Head-to-Head Results (bid_inspector, 5000 deals, seed=100)

| Matchup | IMP (model1 perspective) | Wins/Losses/Ties |
|---------|--------------------------|------------------|
| 571 Agent A vs SL | +4.219 ± 11.336 | 3013 / 1556 / 431 |
| 667 Agent A vs SL | +4.105 ± 11.473 | 3003 / 1606 / 391 |
| 667 Agent A vs SL_BCA(StageB) | +1.822 ± 10.557 | 2435 / 2322 / 243 |
| 667 Agent A vs SL_BCA(ablated) | +1.864 ± 10.643 | 2423 / 2339 / 238 |
| 667 Agent B vs SL | +3.753 ± 11.459 | 2942 / 1651 / 407 |
| **667 Agent B vs Agent A** | **+0.678 ± 7.942** | **1110 / 833 / 3057** |
| SL vs SL_BCA(StageB) | -1.105 ± 9.058 | 2077 / 2343 / 580 |

### Key Findings

1. **r_info works: Agent B > Agent A (+0.678 IMP in direct H2H).** Information-theoretic reward shaping improves cooperative bidding quality.
2. **Agent B vs SL (+3.753) < Agent A vs SL (+4.105).** B's more communicative bidding sacrifices some SL-exploit efficiency for partner coordination. This is expected: B optimizes for informativeness, not maximum exploitation of a fixed opponent.
3. **SL_BCA(StageB) ablation ≈ 0 difference.** SAYC BeliefNet cannot interpret drifted bids (cross-protocol failure). The +4.1 → +1.8 drop is primarily pattern-change effect from Stage B full fine-tune, NOT convention card information.
4. **SL_BCA(StageB) is weaker than plain SL (-1.1 IMP).** Stage B causes negative transfer.
5. **ReFine adapter gate ≈ 0 on SAYC data.** Belief features are redundant when actor already sees full bidding history. ReFine correctly avoids negative transfer.

---

## Experiment Design (Post-P122)

### Phase 1: Baselines (571-dim, no BCA)

**Purpose:** Establish that RL improves over SL, and quantify the effect of vul randomization on agent behavior.

| Experiment | Config | Expected outcome |
|---|---|---|
| 571-dim Agent vs SL | λ=0.0, 20 rounds, FSP, 5 seeds | Agent > SL in IMP. With vul, aggressive double exploit should be tempered (high penalties when vul). |

**Key diagnostic:** Compare agent's double frequency in vul vs non-vul deals. If vul-aware, should double less when vul.

### Phase 2: BCA Convention Card Experiment (667-dim)

**Purpose:** Test whether giving the opponent a "convention card" (BeliefNet) reduces the RL agent's advantage.

**Original design (SL_BCA StageB) — FAILED:**
Stage B full fine-tune causes negative transfer (SL_BCA -1.1 IMP weaker than SL) and changes bidding pattern. Ablation (real vs prior) ≈ 0 because SAYC BeliefNet cannot interpret drifted bids. The +4.1 → +1.8 drop is pattern-change effect, not convention card effect.

**Revised design (Route 2: Co-evolved BeliefNet + ReFine Adapter):**

The key insight: to test whether a convention card helps the opponent, the BeliefNet must be trained on the *agent's own protocol*, not on SAYC. Agent A's co-evolved BeliefNet (saved in agent checkpoint, belief_loss=1.91) already understands Agent A's drifted bids.

| Step | What | Purpose |
|------|------|---------|
| 1 | Extract co-evolved BeliefNet from Agent A checkpoint | Protocol-matched convention card |
| 2 | Train ReFine adapter: frozen plain SL + Agent A's BeliefNet | SL opponent that can read Agent A's bids WITHOUT changing bidding pattern |
| 3 | Agent A vs SL_ReFine(real belief) | Convention card effect (if any) |
| 4 | Agent A vs SL_ReFine(ablated belief) | Control: same architecture, belief=prior |
| 5 | Diff of 3 and 4 | **Pure convention card information value** |

**Why this works:**
- SL_ReFine(ablated) ≈ plain SL (gate≈0, same bidding pattern) → clean baseline
- SL_ReFine(real) = plain SL + protocol-matched convention card → if belief helps, gate > 0, accuracy improves
- Diff isolates pure belief information content, free of pattern-change confound

**Theoretical backing:** ReFine (Xu et al. 2025) guarantees no negative transfer: if belief is uninformative, adapter output → 0, SL performance preserved.

### Phase 3: r_info Core Experiment (667-dim)

**Purpose:** Test whether information-theoretic reward shaping improves cooperative bidding.

| Agent | BCA | r_info | β | Role |
|---|---|---|---|---|
| A: MAPPO+BCA | ✓ | ✗ | — | Control |
| B: MAPPO+BCA+r_info | ✓ | ✓ | 0.0 | Partner information incentive |
| C: MAPPO+BCA+r_info | ✓ | ✓ | 0.05 | Full dual-information |

**Question chain:**
- B vs A → does partner information shaping help?
- C vs B → does opponent leakage penalty add value?
- Both vs SL → ecological validity

### Phase 4: Convention Drift Quantification

**Purpose:** Methodological contribution — first quantitative measurement of convention drift in bridge AI.

| λ_KL | Purpose |
|---|---|
| 0.0 | Maximum drift |
| 0.1, 0.3, 0.5 | Intermediate |
| 1.0 | Near-SL behavior |

**Metric:** vs-SL IMP as function of λ. Framed as KL-compliance Pareto frontier (explicit confounding acknowledgment: λ affects both drift and policy quality).

### Multi-seed Protocol

All experiments: 5 seeds × 5000 eval deals. Report mean ± std across seeds. Wilcoxon signed-rank test per seed, then meta-analysis across seeds.

---

## ⚠️ Experiment Design Issues (Updated Post-P122)

### Issue 1: DDS regret is NOT opponent-independent
Unchanged. `actual_score` depends on both sides' bidding.

### Issue 2: λ sweep confounds drift and policy quality
Unchanged. Present as KL-compliance Pareto frontier with honest confounding acknowledgment.

### Issue 3: SAYC BeliefNet cannot cross-protocol interpret drifted bids
**Confirmed in P124.** Ablation (real belief vs prior) showed ≈ 0 IMP difference. Despite belief output ≠ prior (L1 dist ≈ 0.137) and belief contributing ~24% of layer-0 activation, the actor's decisions are unchanged. Root cause: SAYC BeliefNet interprets drifted bids through SAYC lens. **Solution: Route 2 (co-evolved BeliefNet + ReFine adapter).**

### Issue 4: SL_BCA(StageB) is weaker than plain SL — ROOT CAUSE IDENTIFIED
**P124 diagnosis:** Stage B full fine-tune causes two entangled problems: (1) actor bidding pattern changes (pure negative transfer), (2) actor becomes dependent on belief features. The +4.1 → +1.8 drop is primarily (1), not convention card effect. **Solution: ReFine adapter freezes SL weights, eliminates pattern change.**

### Issue 6: Agent B vs SL weaker than Agent A vs SL despite B > A in H2H
**New finding (P124).** Agent B (+3.753 vs SL) < Agent A (+4.105 vs SL), but Agent B > Agent A (+0.678 in direct H2H). Explanation: r_info makes B's bidding more communicative to partner, which trades some SL-exploit efficiency for partner coordination. When playing against SL (which doesn't benefit from B's communicative bids), the exploit loss shows. In H2H (where partner benefits from better communication), the coordination gain dominates. **This is not a problem — it's the expected behavior of the information-theoretic objective.**

### Issue 5: SL trained only on None-vul
SAYC training data and OpenSpiel's reference SL pipeline (Lockhart et al., Kita et al.) all use fixed `(False, False)` vulnerability. Our SL baseline inherits this limitation. RL training adds vul-awareness, but the SL anchor in FSP pool remains vul-naive. This is acceptable because: (1) SL is the initialization, not the final policy; (2) prior work has the same limitation; (3) the FSP SL anchor's vul-naivety is conservative (makes it easier to beat, not harder).

---

## BCA Architecture (current design)

**Actor input:** 571 (OpenSpiel obs, includes 4-dim vul) + 48 (partner belief) + 48 (RHO belief) = **667-dim**

**BeliefNet API (post-P119):**
- `get_probs(obs_571, target_pos)` → (B, 48) probabilities
- `compute_loss(obs_571, target_pos, target_features)` → scalar loss
- Input is OpenSpiel 571-dim observation_tensor() (public info only)
- **target_pos uses ABSOLUTE seat numbers** (not relative to dealer)

**Training flow:**
1. Stage A: Train BeliefNet on SAYC data → `sl_base_bca.pt` ✅ Done
2. **Skip Stage B for Agent A/B.** Zero-init the 96 belief columns; let RL co-evolve.
3. **Run Stage B for SL opponent only.** → `sl_base_bca_stageB.pt` ✅ Done
4. RL phase: Actor + BeliefNet co-evolve with vul-randomized rollouts.

**P122: Co-evolved BeliefNet saved in agent checkpoint.**
`subgame_validation.py` appends `belief_net` + `belief_hidden_dim` to the checkpoint dict.
`bid_inspector.py` auto-loads from checkpoint (priority) or `--belief_checkpoint` (fallback).

---

## CLI Quick Reference

```bash
# Required every Colab session
pip install open_spiel

# Mount Google Drive BEFORE running experiments
from google.colab import drive
drive.mount('/content/drive')

# ── 571-dim baseline (with vul randomization) ──────────────────────────
python drift_sweep.py \
    --mode 571 \
    --sl_checkpoint results/sl_base.pt \
    --data data/competitive_500k.npz \
    --lambdas 0.0 --seeds 42 100 123 456 789 --rounds 20 \
    --eval_deals 5000 \
    --gdrive_dest '/content/drive/MyDrive/bridge_coma/results'

# ── 667-dim BCA (Agent A only) ─────────────────────────────────────────
python drift_sweep.py \
    --mode 667 \
    --sl_checkpoint results/sl_base.pt \
    --belief_checkpoint results/sl_base_bca.pt \
    --data data/competitive_500k.npz \
    --lambdas 0.0 --seeds 42 100 123 456 789 --rounds 20 \
    --eval_deals 5000 \
    --agent_a_only \
    --gdrive_dest '/content/drive/MyDrive/bridge_coma/results'

# ── Bid inspector (agent checkpoint auto-loads co-evolved BeliefNet) ───
python bid_inspector.py \
    --model1 results/drift_sweep_667/lambda0.0_seed100/agent_a_seed100.pt \
    --model2 results/sl_base.pt --type2 sl \
    --data data/competitive_500k.npz \
    --num_deals 5000 --quiet --seed 42

# ── Bid inspector with ablation ────────────────────────────────────────
python bid_inspector.py \
    --model1 results/drift_sweep_667/lambda0.0_seed100/agent_a_seed100.pt \
    --model2 results/sl_base_bca_stageB.pt --type2 sl \
    --belief_checkpoint results/sl_base_bca.pt \
    --ablate_belief \
    --data data/competitive_500k.npz \
    --num_deals 5000 --quiet --seed 42
```

---

## Key Lessons Learned

- **OpenSpiel `observation_tensor()` is public-info only.** No private hand cards. Includes 4-dim vulnerability.
- **OpenSpiel obs includes vulnerability (4 dim).** But all prior work (Lockhart, Kita) trains with fixed None-vul. Our P122 fix randomizes vul during RL, making agents vul-aware.
- **SAYC data has no vul variation.** OpenSpiel's `bridge_supervised_learning.py` uses `GAME = pyspiel.load_game('bridge(use_double_dummy_result=false)')` with default `dealer_vul=false, non_dealer_vul=false`. Every trajectory is replayed under None-vul. This is a limitation shared by ALL prior work using this pipeline.
- **Vul affects strategy dramatically.** Non-vul: aggressive doubles/overcalls are cheap (low penalties). Vul: same actions carry severe penalties. Agents trained only on non-vul learn unconstrained aggression that fails in vul conditions.
- **BeliefNet target_pos must use absolute seat numbers.** BeliefNet was trained with SAYC data where dealer=0, so absolute=relative. RL training uses absolute seats. bid_inspector must match. Using relative seats (pre-P122 bug) caused wrong belief features for 75% of deals.
- **Co-evolved BeliefNet must be saved with agent checkpoint.** `MAPPOAgent.save()` doesn't include it. `subgame_validation.py` now appends it. `bid_inspector.py` auto-detects and loads.
- **SAYC-trained BeliefNet cannot interpret drifted bids.** Ablation showed real vs prior belief ≈ 0 IMP difference. Cross-protocol inference failure. A convention card must be trained on the agent's own protocol to be useful.
- **SL_BCA(StageB) weakness is negative transfer, not convention card failure.** Stage B full fine-tune changes bidding pattern AND creates belief dependency. The +4.1→+1.8 drop is primarily pattern-change effect, confirmed by ablation ≈ 0.
- **ReFine adapter (Xu et al. 2025) solves the SL_BCA negative transfer problem.** Freeze plain SL actor, inject belief via zero-init residual adapter. Gate=0 guarantees exact plain-SL behavior when belief is uninformative. Theoretically proven no-negative-transfer.
- **Action encoding mismatch between sl_pretrain.py and sl_pretrain_bca.py (P124).** sl_base.pt uses `action-52` (Pass=0, 1C=1). sl_pretrain_bca.py used `openspiel_raw_to_ours` (Pass=0, Double=1, Redouble=2, 1C=3). Off by 2 for all non-pass bids. Stage B legacy masked the bug via full fine-tune; ReFine exposed it.
- **Zero-init adapter trap (P124).** If both `up.weight=0` AND `gate=0`, gradient is zero everywhere (double dead zone). Fix: only zero-init gate; use Xavier for up. Gate=0 alone guarantees zero initial output.
- **Belief features are redundant for SAYC SL.** ReFine gate converges to ≈0 on SAYC data: plain SL already extracts all useful information from obs_571 (full bidding history). Belief features add value only when BeliefNet reads a protocol that obs alone cannot decode.
- **r_info trades SL-exploit for partner coordination.** Agent B > Agent A in H2H (+0.678), but Agent B < Agent A vs SL (-0.35). Communicative bidding helps partners but is "wasted" on an SL opponent that can't interpret the signals.
- **Dealing order matters for obs.** OpenSpiel SAYC trajectories use interleaved dealing, not consecutive per-player.
- **Always use `game(dealer=0)` for inference.** SL trained on dealer=0 only. P122 adds `dealer_vul`/`non_dealer_vul` params to the game instance.
- **Reward normalization must persist across rounds.** Re-instantiating `RunningStats` resets normalization.
- **Fixed hyperparameters over adaptive mechanisms** for scientific reproducibility.
- **FSP with SL as permanent anchor is the correct training setup.** Self-play causes near-zero gradient signal.
- **P117 was the most severe bug before P122.** Dealer rotation caused 50% of rewards to have wrong sign.
- **P122 is the second most severe set of fixes.** Three independent bugs (vul, target_pos, BeliefNet save) all affected 667-dim experiments. Plus vul randomization fundamentally changes the training distribution.

---

## ⚠️ P106–P108: `hands_to_openspiel_state` Fix History

### P106: Wrong dealing start seat
**Fix:** `for i in range(4): player = (dealer + i) % 4`

### P107: Wrong game instance for non-North dealers
**Fix:** Always use `game(dealer=0)`. Roll hands by `-dealer` so the opener always sits at index 0.

### P108: Wrong dealing order (consecutive vs interleaved)
```python
cards_per_player = [sorted(np.where(hands_to_deal[p] > 0.5)[0]) for p in range(4)]
for i in range(13):
    for p in range(4):
        state.apply_action(int(cards_per_player[p][i]))
```

### P122: Vulnerability parameter added
```python
# hands_to_openspiel_state now accepts vulnerability=(ns_vul, ew_vul)
# Converted to OpenSpiel's (dealer_vul, non_dealer_vul) relative to rolled dealer
# get_openspiel_game cache key expanded to (dealer, dealer_vul, non_dealer_vul)
```

---

## ⏳ TODO (Next Session)

### TODO 1: bid_inspector ReFine adapter support
`bid_inspector.py` 的 `_make_play_mixed_policy` 需要支持 ReFine 模式的 SL 对手。当前逻辑把 belief features 拼接到 571-dim obs 构成 667-dim 输入喂给 `MLPPolicyNetwork`。ReFine 模式需要分开传 obs_571 和 belief_96 给 `ReFineActor`。

需要改的地方：
- `_make_play_mixed_policy`: 检测 model 是否为 `ReFineActor`，走不同的 forward path
- `load_sl`: 支持加载 `sl_base_bca_refine.pt`（`encoding='openspiel_667_refine'`），构建 `ReFineActor` 而不是 `MLPPolicyNetwork`
- 消融模式：`--ablate_belief` 在 ReFine 模式下把 belief_96 替换为 prior

### TODO 2: Convention Card with Co-evolved BeliefNet (Route 2)
**核心实验：证明 convention card 确实减少了密约优势。**

步骤：
1. 从 Agent A checkpoint (`agent_a_seed100.pt`) 提取 co-evolved BeliefNet（belief_loss=1.91，能读懂 Agent A 的 drifted protocol）
2. 用这个 BeliefNet 重跑 ReFine adapter 训练：`python sl_pretrain_bca.py --mode refine --init_from results/sl_base.pt --load_belief results/drift_sweep_667/lambda0.0_seed100/agent_a_seed100.pt --out results/sl_bca_refine_coevolved_a.pt`
3. 如果 gate > 0 且 non_pass_acc > baseline：co-evolved BeliefNet 为 SL 提供了有用信息
4. 在 bid_inspector 中对战：Agent A vs SL_ReFine_coevolved(real) vs Agent A vs SL_ReFine_coevolved(ablated)
5. 如果 real < ablated（agent 更难赢）：convention card 有效

**预期结果：** gate > 0（co-evolved BeliefNet 有边际信息价值），消融实验显示 real belief 使 agent 优势减少。

**科学合理性：** 不是在测 "某个 NN 训练方案是否有效"，而是在测 "如果对手拥有能读懂 agent protocol 的 convention card，agent 的信息不对称优势是否减少"。ReFine 冻结 SL actor 确保 bidding pattern 不变，唯一变量是 belief features 的信息内容。

### TODO 3: Stage 3 eval 对手统一
当前 667-dim 模式的 Stage 3 自动使用 SL_BCA(StageB) 作为 eval 对手，导致 Stage 3 得分与 bid_inspector（用 plain SL）不一致。加 `--sl_eval_plain` 参数让 Stage 3 也用 plain SL。

### TODO 4: Multi-seed validation
5 seeds × Agent A + Agent B，确认 B > A 的稳定性。当前只有 seed=100 的单次结果。

---

*README version: P124*
*Last updated: 2026-03-31*
