# Bridge-COMA

**Dual-Information Credit Assignment for Cooperative-Competitive Multi-Agent Coordination**
MSc Research Project — Kaishuo Wang, 2026

$$r_{\text{info}} = I(\text{bid};\,\text{hand} \mid \text{partner}) - \beta \cdot I(\text{bid};\,\text{hand} \mid \text{opponent})$$

---

## ⚠️ CRITICAL WORKFLOW NOTES (read at start of every session)

1. **Claude has NO cross-session memory.** Paste this README at the start of each new conversation.
2. **NEVER use `/mnt/project/` as base for edits.** That directory is the version last manually uploaded by Titus and may be several patches behind. Always base edits on the most recent file in `/home/claude/` or `/mnt/user-data/outputs/`.
3. **Config lives in `subgame_validation.py`, not `subgame_trainer.py`.** SubgameConfig kwargs in `subgame_validation.py` override all defaults. Always edit `subgame_validation.py` for hyperparameter changes.
4. **Key files to keep in sync:** `policy_net.py`, `subgame_trainer.py`, `subgame_validation.py`, `competitive_env.py`, `belief_net.py`, `fsp_pool.py`, `drift_sweep.py`.
5. **P104: OBS_DIM = 480 (OpenSpiel standard).** All prior 301-dim checkpoints are INCOMPATIBLE. Must retrain SL from scratch.
6. **Two SL pretrain files exist:** `sl_pretrain.py` (480-dim, outputs `sl_base.pt`) and `sl_pretrain_bca.py` (576-dim BCA, outputs `sl_base_bca_v2.pt`). Do NOT conflate them.
7. **P102: SAYC deck parsing bug fixed.** All prior SL checkpoints were trained on corrupted data (1 card/player). `competitive_500k.npz` is NOT affected.
8. **P103: Dealer rotation bug fixed in `competitive_env.py`.** See section below. ALL prior RL experiments invalidated.
9. **P104: Actor input is 480-dim** (OpenSpiel standard). BCA mode: 480 + 48 (partner) + 48 (RHO) = 576. Old 301/397-dim is superseded.
10. **P100: BCA is the standard baseline for ALL agents.** The only experimental variable is r_info.
11. **Colab 12-hour limit:** Use `drift_sweep.py` with `--lambdas` and `--seeds` to split work across sessions. It auto-skips completed runs.
12. **Save results after every Colab session.** `zip -r drift_sweep_partial.zip results/drift_sweep_480/` and download. Upload + unzip at start of next session for auto-skip to work.

---

## Project Progress

| Item | Status |
|------|--------|
| P54–P86: Core infrastructure + Belief Net rewrite | ✅ |
| P87b–P97d: 301-dim experiments | ❌ Invalidated by P102 + P103 + P104 |
| P98–P101: BCA architecture (397-dim, partner + RHO) | ❌ Superseded by P104 (576-dim) |
| P102: SAYC deck bug fix | ✅ |
| P103: Dealer rotation bug fix + speed optimizations | ✅ |
| **P104: 480-dim OpenSpiel observation encoding** | ✅ |
| Retrain `sl_base.pt` (480-dim, correct encoding) | ⏳ **Next** |
| Retrain `sl_base_bca.pt` (576-dim) | ⏳ |
| Exp 1: 480-dim drift sweep | ⏳ |
| Exp 2: 576-dim BCA drift sweep | ⏳ |
| Exp 3: r_info A vs B vs C | ⏳ |

---

## ⚠️ P104: Observation Encoding Rewrite (2026-03-26)

### The Problem

The original 301-dim encoding had three fatal flaws discovered via `bid_inspector.py`:

1. **Absolute player positions** (N/E/S/W) instead of **relative** (self/LHO/partner/RHO). The network had to implicitly infer "who am I" from the hand to interpret bidding history — an unnecessary learning burden.
2. **Complete loss of Pass information.** Only substance bids were recorded in `who_called`. The crucial information of "who passed before opening" and "who passed at each step" was entirely missing.
3. **No temporal order preservation.** The encoding collapsed all bids into a flat (35×4) matrix, losing the sequential structure.

These flaws caused the SL baseline to produce nonsensical bidding in competitive scenarios (escalation spirals like `2♣→4♣→5♣→6♣→6♠...`), despite achieving `non_pass_acc=34.3%` on overall SAYC data.

### The Fix: OpenSpiel 480-dim Standard

Replaced with the standard encoding used by ALL prior work (JPS/Tian+20, Lockhart+20, Kita+24, OpenSpiel, Pgx):

```
obs[  0:  4]  Vulnerability (one-hot)                    4
obs[  4:  8]  "Pass before opening" per relative player   4
obs[  8:428]  Bidding history: 35 bids × 12 bits         420
                Per bid: [self_bid, LHO_bid, partner_bid, RHO_bid,
                          self_dbl, LHO_dbl, partner_dbl, RHO_dbl,
                          self_rdbl, LHO_rdbl, partner_rdbl, RHO_rdbl]
obs[428:480]  My hand (52-dim, 13-hot)                   52
─────────────────────────────────────────────────────────────
Total                                                    480
```

Key design features:
- **Relative positions**: player 0 = self, 1 = LHO, 2 = partner, 3 = RHO
- **Pass before opening**: explicitly encoded (4 bits)
- **Per-player bid/double/redouble**: 12 bits per bid preserving full attribution
- **Binary features**: all values are 0 or 1

### Impact

| Component | Affected? |
|-----------|-----------|
| `policy_net.py` | ✅ **Rewritten**: `encode_obs_flat()` + `OBS_DIM=480` |
| `sl_base.pt` (all prior) | ❌ **Incompatible** — must retrain |
| `sl_base_bca.pt` (all prior) | ❌ **Incompatible** — must retrain |
| ALL prior RL experiments | ❌ **Invalidated** — wrong encoding |
| `competitive_500k.npz` | ❌ Not affected (raw data) |
| `subgame_trainer.py` | ✅ Comments updated (code auto-adapts via import) |
| `subgame_validation.py` | ✅ Comments updated |
| `mappo.py` | ✅ Comments updated |
| `drift_sweep.py` | ✅ Mode `301`→`480`, `397`→`576` |

---

## ⚠️ P103: Dealer Rotation Bug (2026-03-25)

### The Bug

`competitive_env.py` `generate_deal()` used `np.roll(hands, -rotation)` but should be `np.roll(hands, +rotation)`. This caused the opener/overcaller constraints to be assigned to **wrong players** for 75% of deals.

### The Fix

```python
# BEFORE (wrong):
hands    = np.roll(hands, -rotation, axis=0)
dd_table = np.roll(dd_table, -rotation, axis=1)

# AFTER (correct):
hands    = np.roll(hands, rotation, axis=0)
dd_table = np.roll(dd_table, rotation, axis=1)
```

---

## ⚠️ P102: SAYC Deck Parsing Bug

SAYC data format is `deck[position] = card_id` (values 0-51), but code assumed `deck[card] = player` (values 0-3). Result: only ~1 card per player. **All SL checkpoints invalid. `competitive_500k.npz` is NOT affected.**

---

## Experiment Design

See **`experiment_plan.md`** for full details. Summary:

| Experiment | Architecture | Purpose | Runs |
|------------|-------------|---------|------|
| Exp 1 | 480-dim (no BCA) | Quantify drift advantage (prior work scenario) | 5λ × 5 seeds = 25 |
| Exp 2 | 576-dim (BCA) | Quantify drift advantage with BCA | 5λ × 5 seeds = 25 |
| Exp 3 | 576-dim (BCA) | Test r_info at optimal λ | 3 configs × 5 seeds = 15 |

---

## CLI Quick Reference

### `drift_sweep.py` (Exp 1 & 2)

```bash
# Full sweep (split across Colab sessions — auto-skips completed runs)
python drift_sweep.py --mode 480 \
    --sl_checkpoint results/sl_base.pt \
    --data data/competitive_500k.npz \
    --lambdas 0.0 0.1 0.3 --seeds 42 123 456 789 2024 \
    --rounds 10 --eval_deals 2000 --verbose

# Quick smoke test
python drift_sweep.py --mode 480 \
    --sl_checkpoint results/sl_base.pt \
    --data data/competitive_500k.npz \
    --lambdas 0.0 0.3 --seeds 42 --rounds 3 --quick --verbose
```

### `bid_inspector.py` (diagnostic)

```bash
python bid_inspector.py \
    --agent results/drift_sweep_480/lambda0.0_seed42/agent_a_seed42.pt \
    --sl results/sl_base.pt \
    --data data/competitive_500k.npz \
    --num_deals 10
```

---

## Project Structure

```
bridge-coma/
├── networks/
│   ├── policy_net.py               # P104: 480/576-dim (OpenSpiel standard)
│   └── belief_net.py               # dual-head (honor + length)
├── utils/
│   ├── sl_pretrain.py              # P104: 480-dim → sl_base.pt
│   ├── sl_pretrain_bca.py          # P104: 576-dim → sl_base_bca.pt
│   └── hand_features.py, fsp_pool.py, ...
├── subgames/
│   ├── subgame_trainer.py          # P104: auto-adapts via OBS_DIM import
│   ├── competitive_env.py          # P103 fixed: dealer rotation
│   └── subgame_validation.py       # P104: updated dimension comments
├── experiments/
│   ├── drift_sweep.py              # P104: mode 480/576
│   └── bid_inspector.py            # diagnostic bidding comparison
└── data/
    └── competitive_500k.npz        # ✅ Not affected by any patch
```

---

## Files Modified in P104 (must update in Colab)

| File | Change |
|------|--------|
| `policy_net.py` | **Critical:** Complete rewrite of `encode_obs_flat()`. OBS_DIM 301→480, BELIEF_OBS_DIM 397→576 |
| `sl_pretrain.py` | Docstring updated (code auto-adapts via import) |
| `sl_pretrain_bca.py` | Dimension comments updated: 301→480, 397→576 |
| `mappo.py` | Comments updated |
| `subgame_trainer.py` | Comments updated |
| `subgame_validation.py` | Comments updated: dimension references |
| `competitive_env.py` | Comment updated |
| `drift_sweep.py` | Mode choices: 301→480, 397→576 |
| `bid_inspector.py` | Path references updated |

---

## Next Steps (for new conversation window)

1. **Deploy P104 files to Colab** — replace all 9 modified files
2. **Delete ALL existing checkpoints and results** — `rm -rf results/sl_base*.pt results/drift_sweep_*`
3. **Retrain SL:** `python sl_pretrain.py --epochs 30 --device cuda` (~30 min)
4. **Sanity check:** Run `bid_inspector.py` on 10 competitive deals — verify no escalation spirals
5. **Quick smoke test:** `drift_sweep.py --mode 480 --lambdas 0.0 0.3 --seeds 42 --rounds 3 --quick`
6. **Start Exp 1:** `drift_sweep.py --mode 480 --lambdas 0.0 0.1 0.3 --seeds 42 123 456 789 2024 --rounds 10`

---

*README version: P104 (480-dim OpenSpiel encoding)*
*Last updated: 2026-03-26*
