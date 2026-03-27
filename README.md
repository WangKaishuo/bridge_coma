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
6. **SL checkpoint:** `results/sl_base.pt` — trained 400k iters, `non_pass_acc=86.6%`, `encoding=openspiel_571`. This file is valid and does NOT need retraining.
7. **`drift_sweep.py` modes:** `--mode 571` (no BCA) and `--mode 667` (BCA). Old `480`/`576` values are gone.
8. **P103: Dealer rotation bug fixed in `competitive_env.py`.** ALL prior RL experiments invalidated.
9. **P102: SAYC deck parsing bug fixed.** All SL checkpoints before P105 are invalid. `competitive_500k.npz` is NOT affected.
10. **Card encoding boundary:** OpenSpiel uses rank-major (`rank*4+suit`). `competitive_env` uses suit-major (`suit*13+rank`). `convert_hands_suit_to_rank()` in `policy_net.py` handles conversion.
11. **OpenSpiel `observation_tensor()` does NOT contain private hand cards.** It is a public-information tensor (bidding history only). The SL model learns to predict next bid from bidding history alone — this is correct and intentional (86.6% non_pass_acc confirms it works).
12. **Dealing order for `hands_to_openspiel_state`:** interleaved `p0[0], p1[0], p2[0], p3[0], p0[1], ...` — NOT consecutive 13 cards per player. Wrong order produces different obs.
13. **Colab 12-hour limit:** Use `drift_sweep.py` with `--lambdas` and `--seeds` to split work across sessions. It auto-skips completed runs.
14. **Save results after every Colab session.** `zip -r results_partial.zip results/` and download. Upload + unzip at start of next session for auto-skip to work.
15. **Stage 3 eval_deals=5000, no per-round H2H.** Per-round H2H was removed (P110) — too noisy. All statistical evaluation deferred to Stage 3.
16. **`_deal_action_cache` size is 8192** (was 2048). Stage 3 with 5000 deals requires larger cache to avoid 0.64s/deal rebuild overhead.
17. **DDS regret IS opponent-dependent.** Do NOT claim it as an opponent-independent metric. actual_score depends on both sides' bidding. This is a known limitation documented in the experiment design.
18. **`bid_inspector.py` now uses cross-table (双桌对战).** play_deal() calls play_mixed() twice, same semantics as cross_evaluate(). Old single-table comparison was scientifically incorrect.

---

## Project Progress

| Item | Status |
|------|--------|
| P54–P86: Core infrastructure + Belief Net rewrite | ✅ |
| P87b–P97d: 301-dim experiments | ❌ Invalidated by P102+P103+P104 |
| P98–P101: BCA architecture (397-dim) | ❌ Superseded by P105 |
| P102: SAYC deck bug fix | ✅ |
| P103: Dealer rotation bug fix | ✅ |
| P104/P104c: Custom 480-dim encoding | ❌ Superseded by P105 |
| P105: OpenSpiel-native 571-dim SL (`sl_pretrain.py`) | ✅ |
| Train `sl_base.pt` (400k iters, non_pass_acc=86.6%) | ✅ **Done** |
| P106: Fix `hands_to_openspiel_state` dealer order | ✅ |
| P107: Fix `hands_to_openspiel_state` always use dealer=0 game + roll | ✅ |
| P108: Fix dealing order (interleaved); migrate `subgame_trainer.py` to 571-dim | ✅ |
| P109: obs cache performance improvements | ✅ |
| P110: Remove per-round H2H; Stage 3 eval_deals=5000 | ✅ |
| P111: Fix `bridge_bidding_env.py` X/XX legal actions state machine | ✅ |
| P112: Fix `bid_inspector.py` to use cross-table play_mixed | ✅ |
| P113: Fix `sl_pretrain_bca.py` Stage B to use 571-dim OpenSpiel obs (was encode_obs_flat 301-dim) | ✅ |
| **Drift Sweep Exp 1** (571-dim, λ=0.0 and λ=1.0, seed=100) | ✅ **Partial results** |
| Full drift sweep (λ ∈ {0.0, 0.1, 0.3, 0.5, 1.0}, 5 seeds) | ⏳ Pending — experiment design under review |
| BCA SL pretrain (`sl_base_bca.pt`, 667-dim) | ⏳ Pending |
| BCA core experiment (A/B/C, 667-dim) | ⏳ Pending |

---

## Drift Sweep Results (571-dim, seed=100, partial)

| λ | A vs SL IMP | std | win_rate | notes |
|---|------------|-----|----------|-------|
| 0.0 | +7.406 | 10.223 | 72.2% | unconstrained drift |
| 1.0 | +1.244 | 5.692 | 20.0% | near-SL behavior; 9/10 bid inspector deals are tie |

**Key observations:**
- λ=0.0 advantage is largely due to **better competitive judgment** (doubling decisions, sacrifice bids), NOT convention drift confusing opponents. Confirmed by bid inspector qualitative analysis.
- λ=1.0 agent almost identical to SL (KL constraint too strong), confirming the constraint mechanism works.
- The gap λ=0.0 vs λ=1.0 (+6.2 IMP) **cannot be cleanly attributed to drift** — KL constraint also limits policy improvement independently of drift. See "Experiment Design Issues" below.

---

## ⚠️ Experiment Design Issues (Identified 2026-03-26)

### Issue 1: DDS regret is NOT opponent-independent
`dds_oracle_evaluate` computes `regret = actual_score - dds_optimal_score`. But `actual_score` depends on BOTH sides' bidding. So regret is still opponent-dependent. The decomposition `drift_advantage = vs_SL_IMP - DDS_regret` does NOT cleanly isolate drift.

### Issue 2: λ sweep confounds drift and policy quality
λ↓ simultaneously causes:
1. More convention drift (D_KL increases)
2. Less constrained policy optimization (agent can explore more)

These two effects are inseparable in the current design. The +6.2 IMP gap between λ=0.0 and λ=1.0 could be entirely due to (2) with zero contribution from (1).

### Issue 3: Correct experiment to isolate drift advantage
Fix the agent (use the trained λ=0.0 agent), vary the **opponent's understanding**:
- Opponent A: plain SL (cannot interpret drifted bids)
- Opponent B: SL + belief net (understands drifted bids via convention card)

If A_IMP > B_IMP, drift genuinely helps against unaware opponents. This is a clean causal test. Requires BCA infrastructure to be complete first.

### Issue 4: λ sweep framing is still salvageable
Frame as: "we measure the causal effect of protocol compliance constraint on vs-SL IMP". The sweep is NOT claiming to isolate drift — it establishes that **relaxing the constraint increases vs-SL IMP**, which is a valid observation with correct framing (analogous to KL-reward Pareto frontier in RLHF). The limitation (confounding with policy quality) must be acknowledged explicitly.

---

## Research Narrative (confirmed 2026-03-26)

**Core MARL question:** In dual-audience signaling (protocol-constrained, same bid interpreted by partner AND opponent), does information-theoretic reward shaping improve coordination?

**Story structure:**
- Bridge is testbed, not the goal
- BCA is infrastructure that closes the perception-action loop (fair test prerequisite)
- r_info is the proposed method
- Convention drift is a methodological bonus contribution discovered en route

**Experiment matrix (revised):**
- A vs SL → sanity check (does RL + closed perception loop improve over SL?)
- B vs A → **core RQ1**: does partner information incentive improve coordination?
- C vs B → **core RQ2**: does opponent leakage penalty add value?
- Drift sweep → independent methodological contribution (with honest framing of limitations)

**Both result routes are publishable:**
- r_info works → NeurIPS/ICML: wiretap-channel reward shaping improves coordination
- r_info doesn't work → AAMAS/CoG: in constrained policy spaces, task reward already contains communication incentives (valuable negative result)

---

## ⚠️ P106–P108: `hands_to_openspiel_state` Fix History (2026-03-26)

### P106: Wrong dealing start seat
**Bug:** Loop always started at `player 0` (North), but OpenSpiel deals starting from the dealer seat.
**Fix:** `for i in range(4): player = (dealer + i) % 4`

### P107: Wrong game instance for non-North dealers
**Bug:** Used `game(dealer=X)` for each dealer. SAYC training data always has `dealer=North (0)`. A `game(dealer=2)` produces observations with different semantics.
**Fix:** Always use `game(dealer=0)`. Roll hands by `-dealer` so the opener always sits at index 0.

### P108: Wrong dealing order (consecutive vs interleaved)
**Fix:**
```python
cards_per_player = [sorted(np.where(hands_to_deal[p] > 0.5)[0]) for p in range(4)]
for i in range(13):
    for p in range(4):
        state.apply_action(int(cards_per_player[p][i]))
```

### Key insight: OpenSpiel obs does NOT contain private hand cards
`observation_tensor()` (571-dim) contains **only public information** (bidding history + game metadata). The 86.6% SL accuracy comes entirely from learning bidding history patterns.

---

## ⚠️ P111: bridge_bidding_env.py X/XX Legal Actions Fix

**Bug:** `_get_legal_actions()` used two independent boolean loops for Double/Redouble. After Redouble, `doubled=False` incorrectly re-allowed Double.

**Fix:** Unified `double_state` state machine (0=undoubled, 1=doubled, 2=redoubled). After Redouble (state=2), neither X nor XX is legal.

---

## ⚠️ P113: sl_pretrain_bca.py Stage B Fix

**Bug:** Stage B `collect_stage_b_data()` called `encode_obs_flat()` (deleted in P108), producing 301-dim base obs instead of 571-dim OpenSpiel obs.

**Fix:** Stage B now uses `hands_to_openspiel_state` + `get_openspiel_obs` → 571-dim base obs. Combined with belief features: 571 + 48 + 48 = **667-dim** actor input (was incorrectly 397-dim = 301 + 96).

---

## Architecture: SL→RL Bridge (P108, resolved)

**Decision: Option 3 (Map at RL time)** — implemented in P108.

At each RL step, `subgame_trainer._encode_for_actor()` converts the current env state to an OpenSpiel obs:
1. `convert_hands_suit_to_rank(hands_sm)` — suit-major → rank-major
2. `hands_to_openspiel_state(hands_rm, dealer)` — build OpenSpiel state with P108 interleaved dealing
3. Replay `history_int` via `ours_to_openspiel_raw(a)`
4. `get_openspiel_obs(state)` → 571-dim obs

---

## CLI Quick Reference

```bash
# Required every Colab session
pip install open_spiel

# SL training (already done — sl_base.pt exists)
python sl_pretrain.py --iterations 400000 --batch_size 128 --device cuda

# BCA SL training (667-dim, uses sl_base.pt as init)
python sl_pretrain_bca.py \
    --train data/sayc_train.txt \
    --valid data/sayc_valid.txt \
    --out results/sl_base_bca.pt \
    --init_from results/sl_base.pt \
    --epochs 30 --device cuda

# Drift sweep (Exp 1, 571-dim, no BCA)
python drift_sweep.py \
    --mode 571 \
    --sl_checkpoint results/sl_base.pt \
    --data data/competitive_500k.npz \
    --lambdas 0.0 0.1 0.3 0.5 1.0 \
    --seeds 42 123 456 789 2024 \
    --rounds 20 --eval_deals 3000

# Bid inspector (cross-table, agent vs SL)
python bid_inspector.py \
    --agent results/drift_sweep_571/lambda0.0_seed100/agent_a_seed100.pt \
    --sl results/sl_base.pt \
    --data data/competitive_500k.npz \
    --num_deals 10

# Save results before session ends
zip -r results_partial.zip results/  # download from Colab
```

---

## Key Lessons Learned

- **OpenSpiel `observation_tensor()` is public-info only.** No private hand cards. SL works because bidding history implies hand constraints under SAYC.
- **Dealing order matters for obs.** OpenSpiel SAYC trajectories use interleaved dealing, not consecutive per-player.
- **Always use `game(dealer=0)` for inference.** SL trained on dealer=0 only.
- **Reward normalization must persist across rounds.** Re-instantiating `RunningStats` resets normalization.
- **Fixed hyperparameters over adaptive mechanisms** for scientific reproducibility.
- **Critic targets must match training path.** Use GAE returns, not flattened final_reward.
- **Stage 3 performance bottleneck:** `hands_to_openspiel_state` costs ~0.64s/deal on first call (full rebuild). With 5000 deals this is ~53 min. `_deal_action_cache` (8192 entries) caches the deal sequence but NOT the game state itself. Per-trainer `_obs_state_cache` caches states but only for deals seen during training.
- **DDS regret is opponent-dependent** — cannot be used as an opponent-independent ground truth metric.
- **Per-round H2H is too noisy** (IMP std ≈ 9, 500 deals insufficient). All evaluation deferred to Stage 3 with 5000 deals.
- **bid_inspector cross-table** — old single-table comparison inflated agent scores by letting each side play against themselves. Cross-table (play_mixed twice) is the correct evaluation.
- **λ sweep confounds drift and policy quality** — cannot cleanly attribute IMP gap to convention drift alone. Correct isolation requires fixing agent and varying opponent understanding capability.

---

## Next Steps (for new conversation window)

1. **Discuss experiment design** — how to cleanly isolate drift advantage; whether λ sweep is salvageable with correct framing; whether the "fix agent, vary opponent understanding" experiment is worth adding.
2. **Run remaining drift sweep seeds** (seeds 42, 123, 456, 789, 2024) for λ ∈ {0.0, 0.1, 0.3, 0.5, 1.0} — but only after experiment design questions are resolved.
3. **BCA SL pretrain** — run `sl_pretrain_bca.py` with `--init_from results/sl_base.pt` to get `sl_base_bca.pt` (667-dim).
4. **BCA core experiment** — A/B/C comparison with 667-dim actors.
5. **Write up preliminary report** after BCA results are in.

---

*README version: P113 (experiment design issues documented; drift sweep partial results)*
*Last updated: 2026-03-26*
