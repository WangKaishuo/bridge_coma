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
14. **Save results after every Colab session.** Mount Google Drive BEFORE running experiments, then use `--gdrive_dest` for auto-upload. Do NOT write checkpoints directly to Google Drive during training — the write latency will slow training by ~10x.
15. **Stage 3 eval_deals=5000, no per-round H2H.** Per-round H2H was removed (P110) — too noisy. All statistical evaluation deferred to Stage 3.
16. **`_deal_action_cache` size is 8192** (was 2048). Stage 3 with 5000 deals requires larger cache to avoid 0.64s/deal rebuild overhead.
17. **DDS regret IS opponent-dependent.** Do NOT claim it as an opponent-independent metric. actual_score depends on both sides' bidding. This is a known limitation documented in the experiment design.
18. **`bid_inspector.py` now uses cross-table (双桌对战).** play_deal() calls play_mixed() twice, same semantics as cross_evaluate(). Old single-table comparison was scientifically incorrect.
19. **`drift_sweep.py` now supports `--gdrive_dest`.** Mount Drive in a separate cell first (`from google.colab import drive; drive.mount('/content/drive')`), then pass `--gdrive_dest '/content/drive/MyDrive/...'`. The script will auto-upload after all runs complete.
20. **Stage B of BCA SL pretrain can be skipped.** Stage B trains a 667-dim Actor on SAYC with frozen SL BeliefNet. Since BeliefNet is updated from round 1 of RL anyway, Stage B's learned "how to use belief features" is immediately obsolete. Skip Stage B; instead zero-init the 96 new belief columns in the Actor and let RL co-evolve Actor + BeliefNet together.
21. **BCA RL training: freeze_belief=False + BeliefReplayBuffer.** BeliefNet must update during RL to track agent's drifting protocol. Use belief_warmup_rounds=5 (freeze_belief=True for first 5 rounds) to let Actor stabilize before co-evolution begins. Buffer capacity=50000 is sufficient.
22. **Agent bidding differs against different opponents — this is correct.** The 571-dim obs encodes full bidding history. When opponent bids differently (e.g., SL vs another agent), the history diverges, so obs diverges, so agent's response legitimately differs. This is NOT a bug.
23. **bid_inspector stochastic inference: RESOLVED (P115).** Both inference paths (`make_policy_with_probs_openspiel` and `_make_play_mixed_policy`) use deterministic inference (`logits.argmax` / `deterministic=True`). The observed difference between `--sl_only` and `--agent` modes was caused by `--sl_only` using `load_sl()` to load an RL checkpoint — `load_sl()` has a different weight-mapping path than `load_agent()`, producing incorrect actor weights. **Fix:** new `--agent_only` mode uses `load_agent()` for correct RL checkpoint loading.
24. **`bid_inspector.py --agent_only` mode (P115).** Uses `load_agent` + cross-table `play_deal`, both sides use the same agent object. Replaces the broken pattern of `--sl agent_a.pt --sl_only` which silently loaded RL weights through the wrong path.
25. **FSP is preferred over self-play for training.** Self-play without external anchor causes mutual escalation (7-level乱叫). FSP with `sl_competitive.pt` as permanent pool entry provides stable distribution anchor. Use `self_play=False` in SubgameConfig.
26. **`sl_competitive.pt` trained on 1H-1S filtered SAYC data (17,401 games, non_pass_acc=84.6%).** Behaves normally in self-play (2-4 level contracts). Use as FSP anchor and RL init checkpoint for competitive subgame.
27. **P116 (CRITICAL): `_make_play_mixed_policy` in `bid_inspector.py` had two bugs.** (1) `legal_mask` was taken from `obs['legal_actions']` (BridgeBiddingEnv) instead of OpenSpiel state — these can be inconsistent. (2) actor was selected by real seat number instead of OpenSpiel-relative seat `(player-dealer)%4`. Both bugs caused degenerate bidding (7NT chaos) in cross-table evaluation. **Fix:** legal_mask now reconstructed from `os_state.legal_actions()`; actor selection uses `(player-dealer)%4`. BridgeBiddingEnv and OpenSpiel legal actions are logically equivalent (verified), so training code is unaffected.
28. **P116: All prior cross-table IMP results from bid_inspector are INVALID.** +7.406, +1.244, +8.777 IMP figures were artifacts of the legal_mask bug. Corrected evaluation: λ=0.5 agent (25 rounds, FSP, sl_competitive init) vs SL = **-0.117 ± 11.738 IMP** (statistically indistinguishable from zero). λ=0.0 agent (60 rounds) vs SL = **-1.680 ± 11.563 IMP** (SL wins marginally). RL training has not yet produced a measurable advantage over SL.
29. **P116: `make_agent_policy` in `competitive_env.py` has the same bugs as bid_inspector.** H2H evaluation inside `subgame_validation.py` (cross_evaluate) is also invalid. All reported Stage 3 IMP numbers from prior runs must be discarded.
30. **Training code (`_collect_episodes_batch` in `subgame_trainer.py`) is NOT affected by P116.** `BridgeBiddingEnv` and OpenSpiel legal actions are logically equivalent. Actor selection by role name (`actor_n/e/s/w`) matches obs correctly because all 4 actors have identical SL weights.
31. **Open research question: why does RL not improve over SL?** vl converges (28→14), pl decreases slightly (-0.007→-0.003), entropy increases (0.330→0.364). Critic learns but actor may not receive effective gradient. Hypothesis: IMP reward signal has very high variance (std≈8-11 IMP) relative to signal, causing near-zero advantage estimates after GAE normalization. Needs investigation in next session.

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
| P114: Add `--gdrive_dest` to `drift_sweep.py` for auto Google Drive upload | ✅ |
| **Drift Sweep Exp 1** (571-dim, λ=0.0, seed=100, 20 rounds) | ✅ +7.406 IMP |
| **Drift Sweep Exp 2** (571-dim, λ=1.0, seed=100, 20 rounds) | ✅ +1.244 IMP |
| **Drift Sweep Exp 3** (571-dim, λ=0.0, seed=100, 60 rounds) | ✅ +8.777 IMP |
| P115: Fix `bid_inspector.py` — add `--agent_only` mode, resolve `load_sl` vs `load_agent` confusion | ✅ |
| P115: Switch training from FSP to pure self-play (`self_play=True` in SubgameConfig) | ✅ |
| BCA SL pretrain Stage A (`sl_base_bca.pt`, 667-dim) | ⏳ In progress |
| BCA core experiment (Agent A/B, 667-dim, full env) | ⏳ Pending |
| Drift advantage isolation experiment (fix agent, vary opponent understanding) | ⏳ Pending — requires BCA complete |

---

## Drift Sweep Results (571-dim, seed=100)

⚠️ **ALL PRIOR CROSS-TABLE RESULTS ARE INVALID — see P116 bug below.**

| λ | rounds | A vs SL IMP (corrected) | std | notes |
|---|--------|------------------------|-----|-------|
| 0.5 | 25 | -0.117 | 11.738 | FSP, sl_competitive init, corrected eval |
| 0.0 | 60 | -0.117→TBD | — | needs re-eval with corrected bid_inspector |

**All prior +7.406 / +1.244 / +8.777 IMP numbers are artifacts of P116 bug and must be discarded.**

---

## ⚠️ Experiment Design Issues

### Issue 1: DDS regret is NOT opponent-independent
`dds_oracle_evaluate` computes `regret = actual_score - dds_optimal_score`. But `actual_score` depends on BOTH sides' bidding. So regret is still opponent-dependent.

### Issue 2: λ sweep confounds drift and policy quality
λ↓ simultaneously causes (1) more convention drift and (2) less constrained policy optimization. These are inseparable. The IMP gap between λ=0.0 and λ=1.0 cannot be attributed to drift alone.

**Salvageable framing:** Present λ sweep as "measuring the causal effect of protocol compliance constraint on vs-SL IMP" — analogous to KL-reward Pareto frontier in RLHF. Limitation must be explicitly acknowledged.

### Issue 3: Correct experiment to isolate drift advantage
Fix the trained λ=0.0 agent. Vary opponent understanding:
- Opponent A: plain SL (cannot interpret drifted bids)
- Opponent B: SL policy + BeliefNet fine-tuned on agent's rollout data (understands drifted bids)

If A_IMP > B_IMP → drift genuinely helps against unaware opponents.

**Critical constraint:** Opponent B's BeliefNet must be fine-tuned on the drifted agent's rollout data AFTER training. The standard SL BCA BeliefNet is trained on SAYC and cannot understand drifted bids — it is equally blind to drift as plain SL. This experiment requires: (1) train λ=0.0 agent to convergence, (2) generate rollout data, (3) fine-tune BeliefNet on rollout data, (4) use fine-tuned BeliefNet as Opponent B's convention card.

**This experiment is the only scientifically clean way to quantify drift advantage.**

### Issue 4: FSP training produces anti-SL exploits, not general strategies (P115)
The drift sweep (λ=0.0, 60 rounds) agent shows +8.777 IMP vs SL but performs catastrophically in self-play (mean score ≈ −450, 28% contracts made). The agent learns to exploit SL-specific weaknesses (e.g., SL never plays XX, doesn't counter aggressive high-level bidding) rather than learning genuinely better bridge.

**Root cause:** FSP pool is seeded with SL baseline and only contains SL-derived snapshots. The agent never faces an opponent that adapts to its own strategy, so it has no incentive to develop robust play.

**Fix (P115):** `self_play=True` in SubgameConfig disables FSP entirely. Opponent = current agent's own weights (pure self-play). This forces strategies toward Nash equilibrium: aggressive exploit tactics that backfire against a competent opponent are punished immediately.

**Trade-off:** Self-play may converge slower and to a lower vs-SL IMP. But the resulting strategy will be sound bridge rather than a fragile anti-SL exploit. Vs-SL IMP remains a secondary metric; the primary metrics (I(bid;hand|partner), I(bid;hand|opponent)) are measured independently.

---

## Research Narrative (confirmed 2026-03-27)

**Core MARL question:** In dual-audience signaling (same bid interpreted by partner AND opponent), does information-theoretic reward shaping improve coordination?

**Story structure:**
- Bridge is testbed, not the goal
- BCA closes the perception-action loop (full disclosure prerequisite — all players must have equal access to bidding system interpretation)
- r_info is the proposed method
- Convention drift is a methodological bonus contribution

**Experiment matrix:**
- Agent A vs SL → sanity check (does RL improve over SL baseline?)
- Agent B vs Agent A → **core RQ1**: does r_info improve partner information transmission?
- Drift isolation experiment → methodological contribution (requires BCA complete)
- λ sweep → framing as KL-compliance Pareto frontier (with honest confounding acknowledgment)

**Primary metric (MARL language):**
- I(bid; hand | partner): does Agent B's bidding carry more information for partner?
- I(bid; hand | opponent): does Agent B's bidding leak less to opponent?
- These are computed via BeliefNet at evaluation time, independent of training objective

**Secondary metric:** vs-SL IMP (ecological validity check)

**Both result routes are publishable:**
- r_info works → NeurIPS/ICML/AAMAS: wiretap-channel reward shaping improves dual-audience coordination
- r_info doesn't work → CoG/workshop: task reward subsumes information incentives in constrained protocol spaces (valuable negative result)

---

## BCA Architecture (current design, 2026-03-27)

**Actor input:** 571 (OpenSpiel obs) + 48 (partner belief) + 48 (RHO belief) = **667-dim**

**Training flow:**
1. Stage A: Train BeliefNet on SAYC data (honor BCE + length CE losses). Target: honor_acc > 0.76, length_acc > 0.40.
2. **Skip Stage B.** Zero-init the 96 belief columns in Actor; load 571-dim weights from `sl_base.pt`.
3. RL phase: Actor + BeliefNet co-evolve. Use `belief_warmup_rounds=5` (freeze BeliefNet for first 5 rounds). After round 5, `freeze_belief=False` with BeliefReplayBuffer (capacity=50000) to prevent catastrophic forgetting.

**Full disclosure compliance:** All four players (including SL opponents) must use BCA. SL opponent policy remains `sl_base.pt` (571-dim), but receives belief features from the shared BeliefNet. This satisfies the bridge full disclosure rule.

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

---

## CLI Quick Reference

```bash
# Required every Colab session
pip install open_spiel

# Mount Google Drive BEFORE running experiments
# (run in a separate cell)
from google.colab import drive
drive.mount('/content/drive')

# SL training (already done — sl_base.pt exists)
python sl_pretrain.py --iterations 400000 --batch_size 128 --device cuda

# BCA SL training — Stage A only (skip Stage B)
python sl_pretrain_bca.py \
    --train data/sayc_train.txt \
    --valid data/sayc_valid.txt \
    --out results/sl_base_bca.pt \
    --init_from results/sl_base.pt \
    --epochs 50 --device cuda

# Drift sweep (571-dim diagnostic, no BCA)
python drift_sweep.py \
    --mode 571 \
    --sl_checkpoint results/sl_base.pt \
    --data data/competitive_500k.npz \
    --lambdas 0.0 \
    --seeds 42 \
    --rounds 60 \
    --eval_deals 5000 \
    --gdrive_dest '/content/drive/MyDrive/bridge_coma/drift_sweep_60r'

# Core experiment (BCA, Agent A vs Agent B, full env)
# To be defined after BCA pretrain complete

# Bid inspector — agent vs SL (cross-table)
python bid_inspector.py \
    --agent results/drift_sweep_571/lambda0.0_seed100/agent_a_seed100.pt \
    --sl results/sl_base.pt \
    --data data/competitive_500k.npz \
    --num_deals 50

# Bid inspector — agent self-play (4×agent, cross-table, correct load path)
python bid_inspector.py \
    --agent results/agent_a.pt \
    --data data/competitive_500k.npz \
    --num_deals 50 \
    --agent_only
```

---

## Key Lessons Learned

- **OpenSpiel `observation_tensor()` is public-info only.** No private hand cards. SL works because bidding history implies hand constraints under SAYC.
- **Dealing order matters for obs.** OpenSpiel SAYC trajectories use interleaved dealing, not consecutive per-player.
- **Always use `game(dealer=0)` for inference.** SL trained on dealer=0 only.
- **Reward normalization must persist across rounds.** Re-instantiating `RunningStats` resets normalization.
- **Fixed hyperparameters over adaptive mechanisms** for scientific reproducibility.
- **Critic targets must match training path.** Use GAE returns, not flattened final_reward.
- **DDS regret is opponent-dependent** — cannot be used as an opponent-independent ground truth metric.
- **Per-round H2H is too noisy** (IMP std ≈ 9-11, 500 deals insufficient). All evaluation deferred to Stage 3 with 5000 deals.
- **λ sweep confounds drift and policy quality** — cannot cleanly attribute IMP gap to convention drift alone.
- **Convention drift isolation requires fine-tuned BeliefNet** — SL BCA BeliefNet trained on SAYC cannot understand drifted bids, making it equivalent to plain SL as an "aware" opponent. Fine-tuning on agent rollout data is required.
- **Agent bidding varies against different opponents — this is correct.** obs encodes full bidding history; different opponent bids → different history → different obs → different agent response.
- **Do NOT write checkpoints to Google Drive during training.** Mount Drive beforehand; use `--gdrive_dest` for post-training upload only.
- **Stage B of BCA pretrain is unnecessary.** Zero-init belief columns; co-evolve Actor + BeliefNet in RL.
- **XX (redouble) emerges after ~40-60 rounds** of unconstrained training. Agent uses XX as a penalty trap; SL has near-zero XX probability.
- **Agent performs significantly worse in self-play** (28% contracts made) vs vs-SL (74% win rate). Strategy is opponent-specific, not universally superior.
- **FSP-seeded training produces fragile exploits (P115).** Agent learns anti-SL tactics (XX traps, high-level sacrifice) that backfire against any non-SL opponent. Self-play training (`self_play=True`) is required for robust strategies.
- **`load_sl()` and `load_agent()` have different weight mappings (P115).** Never use `load_sl()` to load RL checkpoints — it silently produces incorrect actor weights. Use `--agent_only` mode in bid_inspector for RL-vs-RL comparison.
- **FSP with sl_competitive as permanent anchor is the correct training setup (P116).** Self-play causes 7-level escalation; FSP provides stable distribution anchor. Use `self_play=False`.
- **`sl_competitive.pt` (17k filtered SAYC games, 84.6% acc) is the correct subgame init.** `sl_base.pt` produces OOD behavior in 1H-1S competitive context when facing RL agent bids.
- **P116 (CRITICAL): `_make_play_mixed_policy` in `bid_inspector.py` had two bugs causing 7NT乱叫 in cross-table eval.** (1) legal_mask from BridgeBiddingEnv instead of OpenSpiel state. (2) actor selected by real seat instead of `(player-dealer)%4`. Fix applied. Training code unaffected (BridgeBiddingEnv and OpenSpiel legal actions are logically equivalent; all actors have identical SL weights).
- **All prior cross-table IMP numbers are invalid (P116).** +7.406/+1.244/+8.777 IMP were artifacts. Corrected: λ=0.5/25rounds = -0.117 IMP; λ=0.0/60rounds = -1.680 IMP. RL has not demonstrated measurable improvement over SL.
- **RL improvement failure under investigation.** vl converges, pl decreases slightly, entropy increases — critic learns but actor gradient may be near zero. Hypothesis: high IMP variance (std≈8-11) causes advantage collapse after GAE normalization.

---

*README version: P116 (bid_inspector legal_mask+actor-seat fix; all prior cross-table IMP invalid; corrected RL vs SL ≈ 0 IMP; FSP+sl_competitive is correct setup)*
*Last updated: 2026-03-27*
