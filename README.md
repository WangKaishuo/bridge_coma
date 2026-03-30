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
37. **P123 (CRITICAL): Eval IMP sign bug — 50% of deals had inverted score.** `cross_evaluate`, `_cross_eval_fixed_deals`, and `bid_inspector` all compute `IMP = score_to_imp(score_1 - score_2)`. `play_mixed`'s `ns_policy` controls **opener seats** (not physical NS), but `_compute_score_ns` returns **physical NS perspective**. When dealer ∈ {E,W}, opener = physical EW, so A playing well → NS score low → `score_1 - score_2 < 0` → wrong sign. Fix: flip to `score_2 - score_1` when `dealer % 2 == 1`. **Training code is unaffected** (uses single-table DDS regret, not `play_mixed` dual-table). All pre-P123 eval results have attenuated IMP (≈50% sign-inverted deals pull mean toward 0).
38. **P123 is the third most severe bug.** It did not corrupt training, but made all eval results unreliable. The "vul training hurts" finding (≈0 IMP vs SL) was entirely an artifact — with fixed eval, vul-trained agent achieves +3.3 IMP, comparable to no-vul agent (+3.0 IMP).
39. **`entropy_coef` default is 0.001.** Was accidentally set to 0.01 in a previous upload. 0.01 causes asymmetric entropy collapse across players. Always verify `entropy_coef=0.001` in the header log.
40. **Early stop is disabled by default.** `early_stop_enabled=False` in SubgameConfig. PopArt value normalization compresses VL to ~0.5-0.9, which falsely triggers the plateau detector (threshold 0.15 was calibrated for raw VL ~20-40). Fixed schedules preferred for reproducibility.
41. **PopArt value normalization: tested but not required.** MAPPO paper recommends it, and it does equalize VL across players with vul. However, post-P123 eval fix shows agents train correctly without it. Available in `utils/value_norm.py` if needed for future experiments with higher reward variance.

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
| Train `sl_base_bca_stageB.pt` (Stage B, 667-dim) | ✅ Done |
| P122: Vul randomization + BeliefNet save + target_pos fix | ✅ |
| P123: Eval IMP sign fix + entropy_coef fix + early_stop off | ✅ |
| **571-dim RL baseline (with vul, λ=0.0, 1 seed)** | ✅ +3.3 IMP vs SL (seed=100) |
| **571-dim RL multi-seed (5 seeds)** | ⏳ Pending |
| **667-dim BCA RL training (with vul)** | ⏳ Pending |
| **BCA ablation experiment** | ⏳ Pending |
| **BCA core experiment (Agent A vs Agent B, r_info)** | ⏳ Pending |

---

## Valid Experimental Results

**All pre-P123 eval results are unreliable** due to IMP sign inversion on 50% of deals (dealer ∈ {E,W}).

### Post-P123 Results (571-dim baseline, seed=100, λ=0.0, 20 rounds, eval=5000 deals)

| Agent | Training vul | Eval vul | vs SL IMP | Win rate |
|---|---|---|---|---|
| RL (no vul training) | None-vul only | Random | +2.97 ± 11.6 | 57.0% |
| RL (with vul training) | Random | Random | +3.31 ± 11.4 | 56.9% |

**Key finding:** Vul randomization during training does NOT hurt — slight improvement (+3.3 vs +3.0). The apparent "vul kills training" finding was entirely the P123 eval bug.

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

| Experiment | Config | What it measures |
|---|---|---|
| AgentA vs plain SL (571-dim) | Same agent, SL has no belief | Total advantage including convention drift |
| AgentA vs SL_BCA(StageB) | SL has SAYC-trained BeliefNet | Advantage after opponent gets convention card |
| AgentA vs SL_BCA(StageB, ablated) | SL has StageB actor but belief=prior | Isolates belief information content vs Stage B training effect |

**Ablation logic:**
- (AgentA vs SL) − (AgentA vs SL_BCA) = total BCA effect (convention card + Stage B actor change)
- (AgentA vs SL_BCA_ablated) − (AgentA vs SL_BCA_real) = pure belief information value
- SL_BCA vs plain SL = Stage B's own strength change (known to be negative ~1.4 IMP pre-P122; recheck post-P122)

**Key discovery from pre-P122 experiments:** SAYC-trained BeliefNet cannot cross-protocol interpret drifted bids. Ablation showed real vs prior belief difference ≈ 0. This is a meaningful negative result — convention cards require protocol-matched training.

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
**New finding from P122 experiments.** Ablation (real belief vs prior) showed ≈ 0 IMP difference. The SAYC-trained BeliefNet interprets drifted bids through a SAYC lens, producing no useful signal. A proper convention card would require fine-tuning on the agent's own rollout data — but this raises scientific concerns (see Key Lessons).

### Issue 4: SL_BCA(StageB) is weaker than plain SL
**New finding.** SL_BCA(StageB) loses ~1.4 IMP to plain SL in direct play. Stage B training makes the actor dependent on belief features; when those features are noisy/inaccurate, performance degrades below plain SL. This actually strengthens the BCA argument: a weaker opponent equipped with convention card still reduces agent advantage more than a stronger opponent without it.

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
- **SL_BCA(StageB) is weaker than plain SL.** Stage B makes actor dependent on belief features; imperfect belief signals cause worse decisions than ignoring them entirely. This is analogous to an inexperienced player misreading a convention card.
- **Fine-tuning BeliefNet on agent rollout data is scientifically problematic.** It tests "can a specific neural network training scheme help" rather than "does the convention card concept work." The conclusion would depend on engineering quality, not the concept.
- **Dealing order matters for obs.** OpenSpiel SAYC trajectories use interleaved dealing, not consecutive per-player.
- **Always use `game(dealer=0)` for inference.** SL trained on dealer=0 only. P122 adds `dealer_vul`/`non_dealer_vul` params to the game instance.
- **Reward normalization must persist across rounds.** Re-instantiating `RunningStats` resets normalization.
- **Fixed hyperparameters over adaptive mechanisms** for scientific reproducibility.
- **FSP with SL as permanent anchor is the correct training setup.** Self-play causes near-zero gradient signal.
- **P117 was the most severe bug before P122.** Dealer rotation caused 50% of rewards to have wrong sign.
- **P122 is the second most severe set of fixes.** Three independent bugs (vul, target_pos, BeliefNet save) all affected 667-dim experiments. Plus vul randomization fundamentally changes the training distribution.
- **P123 eval sign bug: `play_mixed` ns_policy ≠ physical NS.** `ns_policy` controls opener seats, but `_compute_score_ns` returns physical NS perspective. When dealer is EW, the IMP sign inverts. This bug only affected eval, not training. Symptom: all agents appeared to tie with SL (~0 IMP) despite regret improving during training.
- **Eval bugs can masquerade as training failures.** P123 wasted multiple sessions investigating training-side fixes (PopArt, entropy tuning, batch size) when the actual problem was a sign error in the evaluation function. Always verify the eval pipeline independently before diagnosing training issues.
- **PopArt value normalization equalizes VL across players but is not necessary for correct training.** Tested and confirmed: agents train correctly without it. Available as `utils/value_norm.py` for future use if needed.

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

*README version: P123*
*Last updated: 2026-03-30*
