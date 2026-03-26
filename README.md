# Bridge-COMA

**Dual-Information Credit Assignment for Cooperative-Competitive Multi-Agent Coordination**
MSc Research Project — Kaishuo Wang, 2026

$$r_{\text{info}} = I(\text{bid};\,\text{hand} \mid \text{partner}) - \beta \cdot I(\text{bid};\,\text{hand} \mid \text{opponent})$$

---

## ⚠️ CRITICAL WORKFLOW NOTES (read at start of every session)

1. **Claude has NO cross-session memory.** Paste this README at the start of each new conversation.
2. **NEVER use `/mnt/project/` as base for edits.** That directory is the version last manually uploaded and may be several patches behind. Always base edits on the most recent file in `/home/claude/` or `/mnt/user-data/outputs/`.
3. **Config lives in `subgame_validation.py`, not `subgame_trainer.py`.** SubgameConfig kwargs in `subgame_validation.py` override all defaults. Always edit `subgame_validation.py` for hyperparameter changes.
4. **Key files to keep in sync:** `policy_net.py`, `subgame_trainer.py`, `subgame_validation.py`, `competitive_env.py`, `belief_net.py`, `fsp_pool.py`, `drift_sweep.py`.
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
| Validate SL with `bid_inspector.py` | ✅ Bidding is coherent and hand-dependent |
| **Start RL training (`subgame_validation.py`)** | ⏳ **Next** |
| Exp 1: drift sweep (`drift_sweep.py --mode 571`) | ⏳ |

---

## ⚠️ P106–P108: `hands_to_openspiel_state` Fix History (2026-03-26)

Three compounding bugs were found and fixed in `hands_to_openspiel_state` (in `policy_net.py`). This function is used at **RL inference time** to convert `competitive_env` hands into an OpenSpiel state for observation generation.

### P106: Wrong dealing start seat
**Bug:** Loop always started at `player 0` (North), but OpenSpiel deals starting from the dealer seat.
**Fix:** `for i in range(4): player = (dealer + i) % 4`

### P107: Wrong game instance for non-North dealers
**Bug:** Used `game(dealer=X)` for each dealer. SAYC training data always has `dealer=North (0)`. A `game(dealer=2)` produces observations with different semantics than `game(dealer=0)`, so the model — which only saw `dealer=0` observations during training — bids incoherently.
**Fix:** Always use `game(dealer=0)`. Roll hands by `-dealer` so the opener always sits at index 0.

### P108: Wrong dealing order (consecutive vs interleaved)
**Bug:** Even with correct dealer and game, dealing 13 cards consecutively per player (`p0[0..12], p1[0..12], ...`) produces a different `observation_tensor()` than the training data.
**Root cause:** OpenSpiel's SAYC training trajectories deal cards in **interleaved** order: `p0[0], p1[0], p2[0], p3[0], p0[1], p1[1], ...` (confirmed by debug).
**Fix:** 
```python
cards_per_player = [sorted(np.where(hands_to_deal[p] > 0.5)[0]) for p in range(4)]
for i in range(13):
    for p in range(4):
        state.apply_action(int(cards_per_player[p][i]))
```

### Key insight: OpenSpiel obs does NOT contain private hand cards
During debugging we discovered that `observation_tensor()` (571-dim) and `information_state_tensor()` (also 571-dim) are **identical** and contain **only public information** (bidding history + game metadata). There are no private hand cards in the obs. The 86.6% SL accuracy comes entirely from learning bidding history patterns — which is correct for this task.

---

## ⚠️ P105: OpenSpiel-native SL (2026-03-26)

SL model trained with `sl_pretrain.py` using `state.observation_tensor()` directly from OpenSpiel trajectory replay. No custom encoding.

- **Checkpoint:** `results/sl_base.pt` (`encoding=openspiel_571`, `non_pass_acc=0.866`)
- **Architecture:** 4×1024 MLP, 571-dim input, 38-dim output
- **Training:** 400k iterations, batch=128, lr=1e-4, ~72 min on T4

### Action mapping (our ordering ↔ OpenSpiel)

| Action | OpenSpiel raw | Our index |
|--------|--------------|-----------|
| Pass | 52 | 0 |
| Double | 88 | 1 |
| Redouble | 89 | 2 |
| 1♣ | 53 | 3 |
| ... | ... | ... |
| 7NT | 87 | 37 |

---

## ⚠️ P103: Dealer Rotation Bug (2026-03-25)

`competitive_env.py` `generate_deal()` used `np.roll(hands, -rotation)` but should be `np.roll(hands, +rotation)`. Fixed. ALL prior RL experiments invalidated.

---

## ⚠️ P102: SAYC Deck Parsing Bug

SAYC data format is `deck[position] = card_id`, but code assumed `deck[card] = player`. Fixed. All SL checkpoints before P105 are invalid. `competitive_500k.npz` is NOT affected.

---

## Architecture: SL→RL Bridge (P108, resolved)

**Decision: Option 3 (Map at RL time)** — implemented in P108.

At each RL step, `subgame_trainer._encode_for_actor()` converts the current env state to an OpenSpiel obs:
1. `convert_hands_suit_to_rank(hands_sm)` — suit-major → rank-major
2. `hands_to_openspiel_state(hands_rm, dealer)` — build OpenSpiel state with P108 interleaved dealing
3. Replay `history_int` via `ours_to_openspiel_raw(a)` 
4. `get_openspiel_obs(state)` → 571-dim obs

This is slower than a native encoding but correct. Acceptable for the competitive subgame scale.

---

## CLI Quick Reference

```bash
# Required every Colab session
pip install open_spiel

# SL training (already done — sl_base.pt exists)
python sl_pretrain.py --iterations 400000 --batch_size 128 --device cuda

# Bid inspector (SL quality check)
python bid_inspector.py --sl results/sl_base.pt --data data/competitive_500k.npz --sl_only --num_deals 10

# RL training (subgame validation, Agent A vs Agent B)
python subgame_validation.py \
    --data data/competitive_500k.npz \
    --sl_checkpoint results/sl_base.pt \
    --seed 42 --rounds 20

# Drift sweep (Exp 1, no BCA)
python drift_sweep.py \
    --mode 571 \
    --sl_checkpoint results/sl_base.pt \
    --data data/competitive_500k.npz \
    --lambdas 0.0 0.1 0.3 0.5 1.0 \
    --seeds 42 123 456 \
    --rounds 10 --eval_deals 2000

# Save results before session ends
zip -r results_partial.zip results/ && # download from Colab
```

---

## Key Lessons Learned

- **OpenSpiel `observation_tensor()` is public-info only.** No private hand cards. The 571-dim obs encodes bidding history + metadata. SL works because bidding history implies hand constraints under SAYC.
- **Dealing order matters for obs.** OpenSpiel SAYC trajectories use interleaved dealing (`p0[0],p1[0],...`), not consecutive per-player. Wrong order = wrong obs = broken inference.
- **Always use `game(dealer=0)` for inference.** SL trained on dealer=0 only. Using `game(dealer=X)` for X≠0 produces obs with different structure.
- **ALWAYS use the reference implementation for observation generation.** `encode_obs_flat` accumulated 3 independent bugs. `state.observation_tensor()` avoids all of them.
- **Validate obs identity between train and inference before starting RL.** Three rounds of debug were needed to find P106→P107→P108.
- **Card encoding boundary:** OpenSpiel rank-major vs `competitive_env` suit-major. Always convert at the boundary with `convert_hands_suit_to_rank()`.
- **Reward normalization must persist across rounds.** Re-instantiating `RunningStats` resets normalization.
- **Fixed hyperparameters over adaptive mechanisms** for scientific reproducibility.
- **Critic targets must match training path.** Use GAE returns, not flattened final_reward.

---

## Next Steps (for new conversation window)

1. **Upload to Colab:** `policy_net.py`, `subgame_trainer.py`, `subgame_validation.py`, `drift_sweep.py`
2. **Verify SL checkpoint exists:** `ls results/sl_base.pt` — if missing, retrain with `sl_pretrain.py`
3. **Run quick smoke test:**
   ```bash
   python subgame_validation.py --data data/competitive_500k.npz \
       --sl_checkpoint results/sl_base.pt --seed 42 --rounds 3 --quick
   ```
4. **Check logs** for reasonable IMP values and no crashes
5. **Run full Exp 1** (drift sweep, `--mode 571`, lambdas 0.0→1.0, 3–5 seeds)
6. **Assess Agent B vs Agent A** IMP difference at each λ

---

*README version: P108 (RL infrastructure migrated to OpenSpiel 571-dim)*
*Last updated: 2026-03-26*
