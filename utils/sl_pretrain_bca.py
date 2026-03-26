"""
SL Pretrain with Belief-Conditioned Actor (P98, P100, P101)
=============================================================

Two-stage SL pretraining for 576-dim belief-conditioned actors:

Stage A: Train Belief Net on SAYC data
  - Replay SAYC games, collect (observer_hand, history, observer_pos,
    target_pos, target_features) tuples
  - P100: Collect ALL target types (partner + LHO + RHO) per observer step.
    This teaches the Belief Net to predict any player's hand, not just partner's.
    Needed for r_info's opponent term and convention-sharing evaluation.
    ~3× more samples than partner-only mode.
  - Train Belief Net to predict target hand features (48-dim)

Stage B: Train 576-dim Actor on SAYC data with Belief Net features
  - Replay SAYC games, at each step:
    1. Encode base obs (480-dim) via encode_obs_flat
    2. Query Belief Net for partner hand prediction (48-dim)
    3. Query Belief Net for RHO hand prediction (48-dim)
    4. Concatenate → 576-dim input
    5. Train Actor on cross-entropy loss

Output: sl_base_bca.pt containing:
  - actor_n/s/e/w: 576-dim MLPPolicyNetwork state_dicts
  - belief_net: BeliefNetwork state_dict
  - obs_dim: 576

Usage:
  python sl_pretrain_bca.py \
      --train data/sayc_train.txt \
      --valid data/sayc_valid.txt \
      --out   results/sl_base_bca.pt \
      --epochs 30 --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from env import (
    BridgeBiddingEnv, NUM_BIDS, NUM_PLAYERS,
    BID_PASS, BID_DOUBLE, BID_REDOUBLE, BID_1C,
    NORTH, EAST, SOUTH, WEST,
)
from networks.policy_net import (
    encode_obs_flat, MLPPolicyNetwork, OBS_DIM, BELIEF_OBS_DIM,
    BELIEF_FEAT_DIM, make_belief_features_prior, append_belief_features,
)
from networks.belief_net import BeliefNetwork
from utils.hand_features import hand_to_belief_target, belief_accuracy, BELIEF_DIM


# ==============================================================================
# Action mapping (same as sl_pretrain.py)
# ==============================================================================

def openspiel_to_our_bid(action: int) -> int:
    if action == 52:
        return BID_PASS
    elif action == 88:
        return BID_DOUBLE
    elif action == 89:
        return BID_REDOUBLE
    elif 53 <= action <= 87:
        return BID_1C + (action - 53)
    else:
        return -1


# ==============================================================================
# Stage A: Collect Belief Training Data from SAYC
# ==============================================================================

def collect_all_data_from_sayc(
    filepath: str,
    max_lines: int = None,
    progress_every: int = 10000,
    include_opponent_targets: bool = True,
) -> List[dict]:
    """
    Stage A data collection: replay SAYC games and return belief training samples.
    Each sample: {observer_hand (52,), hist_presence (NUM_BIDS,) bool,
                  observer_pos, target_pos, target_features (48,)}.

    P101 memory fix: stores hist_presence (38 bool = 38 bytes) instead of
    full hist_enc (60×38 float32 = 9120 bytes). 240× memory reduction.
    BeliefNetwork.encode_history_flat does max-pool anyway, so a (1, 38)
    fake history with bid-presence flags is equivalent.

    P100: When include_opponent_targets=True (default), also collect samples
    with target=LHO and target=RHO for each observer. This teaches the Belief
    Network to predict ANY player's hand from the bidding, not just partner's.
    With opponent targets: ~3× more samples (partner + LHO + RHO per step).
    """
    env = BridgeBiddingEnv(max_history_len=60)
    samples = []

    print(f"[Belief Data] Collecting from {filepath}..."
          f" (opponent_targets={'ON' if include_opponent_targets else 'OFF'})")
    n_lines = 0
    with open(filepath) as f:
        for line_idx, line in enumerate(f):
            if max_lines and line_idx >= max_lines:
                break

            nums = list(map(int, line.split()))
            if len(nums) < 53:
                continue
            n_lines += 1

            deck = np.array(nums[:52], dtype=np.uint8)
            hands = np.zeros((4, 52), dtype=np.float32)
            # SAYC format: deck[position] = card_id (0-51)
            # positions 0-12 = player 0, 13-25 = player 1, etc.
            for player in range(4):
                for card_id in deck[player * 13 : (player + 1) * 13]:
                    hands[player, card_id] = 1.0

            dealer = line_idx % NUM_PLAYERS
            obs = env.reset(hands, dealer=dealer)
            history_int: List[int] = []

            for os_action in nums[52:]:
                our_action = openspiel_to_our_bid(os_action)
                if our_action < 0:
                    break
                if obs['legal_actions'][our_action] < 0.5:
                    break

                player  = env.state.current_player
                partner = (player + 2) % 4
                lho     = (player + 1) % 4   # left-hand opponent
                rho     = (player + 3) % 4   # right-hand opponent

                # Compact history: bool bitmap of which bids have appeared
                hist_presence = np.zeros(NUM_BIDS, dtype=bool)
                for bid in history_int:
                    hist_presence[bid] = True

                # P100: collect targets for partner + both opponents
                targets = [partner]
                if include_opponent_targets:
                    targets.extend([lho, rho])

                for target in targets:
                    samples.append({
                        'observer_hand':   hands[player].copy(),
                        'hist_presence':   hist_presence,
                        'observer_pos':    player,
                        'target_pos':      target,
                        'target_features': hand_to_belief_target(hands[target]),
                    })

                history_int.append(our_action)
                obs, _, done, _ = env.step(our_action)
                if done:
                    break

            if n_lines % progress_every == 0:
                print(f"  {n_lines:,} games / {len(samples):,} steps...")

    print(f"[Belief Data] Done: {n_lines:,} games, {len(samples):,} samples"
          f" ({'3× with opponent targets' if include_opponent_targets else 'partner only'}).")
    return samples


def collect_belief_data_from_sayc(
    filepath: str,
    max_lines: int = None,
    include_opponent_targets: bool = True,
) -> List[dict]:
    """Alias for collect_all_data_from_sayc (backward compatibility)."""
    return collect_all_data_from_sayc(
        filepath, max_lines=max_lines,
        include_opponent_targets=include_opponent_targets,
    )


def collect_stage_b_data(
    filepath: str,
    max_lines: int = None,
    progress_every: int = 50000,
) -> dict:
    """
    Lightweight single-pass replay for Stage B only.
    Collects base_obs (480), hist_presence (38 bool), actions, positions.
    Does NOT build hist_enc (60×38) — ~10× faster than collect_all_data_from_sayc.

    P101: Also collects rho_pos for RHO belief query in 576-dim mode.

    Returns dict with: base_obs, actions, observer_hands,
                       hist_presence, observer_pos, partner_pos, rho_pos
    """
    env = BridgeBiddingEnv(max_history_len=60)

    base_obs_list  = []
    actions_list   = []
    oh_list        = []
    hist_pres_list = []
    op_list        = []
    partner_list   = []
    rho_list       = []

    print(f"[Stage B Data] Collecting from {filepath}...")
    n_lines = 0
    with open(filepath) as f:
        for line_idx, line in enumerate(f):
            if max_lines and line_idx >= max_lines:
                break

            nums = list(map(int, line.split()))
            if len(nums) < 53:
                continue
            n_lines += 1

            deck = np.array(nums[:52], dtype=np.uint8)
            hands = np.zeros((4, 52), dtype=np.float32)
            # SAYC format: deck[position] = card_id (0-51)
            # positions 0-12 = player 0, 13-25 = player 1, etc.
            for player in range(4):
                for card_id in deck[player * 13 : (player + 1) * 13]:
                    hands[player, card_id] = 1.0

            dealer = line_idx % NUM_PLAYERS
            obs = env.reset(hands, dealer=dealer)
            history_int: List[int] = []

            for os_action in nums[52:]:
                our_action = openspiel_to_our_bid(os_action)
                if our_action < 0:
                    break
                if obs['legal_actions'][our_action] < 0.5:
                    break

                player  = env.state.current_player
                partner = (player + 2) % 4
                rho     = (player - 1) % 4    # right-hand opponent

                hist_presence = np.zeros(NUM_BIDS, dtype=bool)
                for bid in history_int:
                    hist_presence[bid] = True

                base_obs_list.append(encode_obs_flat(obs, dealer, history_int))
                actions_list.append(our_action)
                oh_list.append(hands[player].copy())
                hist_pres_list.append(hist_presence)
                op_list.append(player)
                partner_list.append(partner)
                rho_list.append(rho)

                history_int.append(our_action)
                obs, _, done, _ = env.step(our_action)
                if done:
                    break

            if n_lines % progress_every == 0:
                print(f"  {n_lines:,} games / {len(actions_list):,} steps...")

    N = len(actions_list)
    print(f"[Stage B Data] Done: {n_lines:,} games, {N:,} steps.")
    return {
        'base_obs':      np.stack(base_obs_list),
        'actions':       np.array(actions_list, dtype=np.int64),
        'observer_hands':np.stack(oh_list),
        'hist_presence': np.stack(hist_pres_list),
        'observer_pos':  np.array(op_list,      dtype=np.int64),
        'partner_pos':   np.array(partner_list,  dtype=np.int64),
        'rho_pos':       np.array(rho_list,      dtype=np.int64),
    }


def build_belief_dataset_from_data(
    data: dict,
    belief_net: BeliefNetwork,
    device: str = 'cuda',
    batch_size: int = 8192,
) -> List[Tuple[np.ndarray, int]]:
    """
    P101: Batch-compute belief features from pre-collected data and return
    (flat_obs_576, action) pairs for SAYCBeliefDataset.

    Queries belief net twice per sample: partner (48-dim) + RHO (48-dim) = 96-dim.
    Concatenated with base_obs (480-dim) → 576-dim actor input.

    Uses hist_presence (N, NUM_BIDS) bool bitmap — lossless compression of
    history for Belief Net input (encode_history_flat does max-pool anyway).
    ~2-3 min for 1.7M samples on T4 (2× inference vs old 349-dim).
    """
    N = len(data['actions'])
    print(f"[Belief Inference] Batch GPU inference on {N:,} samples "
          f"(batch={batch_size}), querying partner + RHO...")

    # Expand hist_presence (N, NUM_BIDS) bool → (N, 1, NUM_BIDS) float
    hist_pres_f = data['hist_presence'].astype(np.float32)  # (N, NUM_BIDS)
    hist_1step = hist_pres_f[:, np.newaxis, :]  # (N, 1, NUM_BIDS)

    partner_feats = np.empty((N, 48), dtype=np.float32)
    rho_feats     = np.empty((N, 48), dtype=np.float32)
    belief_net.eval()

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            oh_t  = torch.tensor(data['observer_hands'][start:end], dtype=torch.float32).to(device)
            h_t   = torch.tensor(hist_1step[start:end],             dtype=torch.float32).to(device)
            op_t  = torch.tensor(data['observer_pos'][start:end],   dtype=torch.long).to(device)

            # Query partner
            tp_partner = torch.tensor(data['partner_pos'][start:end], dtype=torch.long).to(device)
            partner_feats[start:end] = belief_net.get_probs(oh_t, h_t, op_t, tp_partner).cpu().numpy()

            # Query RHO
            tp_rho = torch.tensor(data['rho_pos'][start:end], dtype=torch.long).to(device)
            rho_feats[start:end] = belief_net.get_probs(oh_t, h_t, op_t, tp_rho).cpu().numpy()

            if (start // batch_size) % 20 == 0:
                print(f"  {end:,}/{N:,} ({end/N:.0%})")

    belief_feats = np.concatenate([partner_feats, rho_feats], axis=1)  # (N, 96)
    combined = np.concatenate([data['base_obs'], belief_feats], axis=1)  # (N, 576)
    print(f"[Belief Inference] Done. obs_dim={combined.shape[1]}")
    return list(zip(combined, data['actions']))


class SAYCBeliefDataset(Dataset):
    """
    Thin Dataset wrapper around pre-computed (flat_obs_576, action) pairs.
    Build via build_belief_dataset_from_data() before instantiating.
    """

    def __init__(self, samples: List[Tuple[np.ndarray, int]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flat_obs, action = self.samples[idx]
        return (torch.tensor(flat_obs, dtype=torch.float32),
                torch.tensor(action,   dtype=torch.int64))


def train_belief_net(
    samples: List[dict],
    device: str = 'cuda',
    epochs: int = 30,
    batch_size: int = 2048,
    lr: float = 1e-3,
    hidden_dim: int = 1024,
) -> BeliefNetwork:
    """
    Train Belief Net on collected SAYC data.

    P101 memory fix: samples now use compact hist_presence (38 bool) instead
    of full hist_enc (60×38 float32). We stack into numpy arrays (not tensors)
    and convert to tensors only per-batch on GPU. This reduces peak memory
    from ~50 GB to ~1.5 GB for 5M samples.
    """
    N = len(samples)
    print(f"\n[Stage A] Training Belief Net: {N:,} samples, {epochs} epochs")

    # Stack into numpy arrays (compact, stays on CPU)
    oh  = np.stack([s['observer_hand']    for s in samples])           # (N, 52) float32
    hp  = np.stack([s['hist_presence']    for s in samples])           # (N, 38) bool
    op  = np.array([s['observer_pos']     for s in samples], dtype=np.int64)  # (N,)
    tp  = np.array([s['target_pos']       for s in samples], dtype=np.int64)  # (N,)
    tgt = np.stack([s['target_features']  for s in samples])           # (N, 48) float32

    # Free the list of dicts to reclaim memory
    del samples

    # Expand hist_presence to (N, 1, NUM_BIDS) float32 — lossless compression.
    # BeliefNetwork.forward → encode_history_flat does max-pool over time axis,
    # so a single-step history with bid-presence flags gives identical results.
    hp_f = hp.astype(np.float32)[:, np.newaxis, :]   # (N, 1, 38)
    del hp

    # 90/10 train/val split
    perm = np.random.permutation(N)
    n_val = max(1000, N // 10)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # Pre-extract small val set to avoid repeated slicing
    val_oh  = torch.tensor(oh[val_idx],  dtype=torch.float32)
    val_h   = torch.tensor(hp_f[val_idx], dtype=torch.float32)
    val_op  = torch.tensor(op[val_idx],  dtype=torch.long)
    val_tp  = torch.tensor(tp[val_idx],  dtype=torch.long)
    val_tgt = torch.tensor(tgt[val_idx], dtype=torch.float32)

    belief_net = BeliefNetwork(hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(belief_net.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        belief_net.train()
        np.random.shuffle(train_idx)
        train_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_idx), batch_size):
            idx = train_idx[start:start + batch_size]
            # Convert per-batch to tensor and move to GPU
            b_oh  = torch.tensor(oh[idx],   dtype=torch.float32).to(device)
            b_h   = torch.tensor(hp_f[idx], dtype=torch.float32).to(device)
            b_op  = torch.tensor(op[idx],   dtype=torch.long).to(device)
            b_tp  = torch.tensor(tp[idx],   dtype=torch.long).to(device)
            b_tgt = torch.tensor(tgt[idx],  dtype=torch.float32).to(device)

            loss = belief_net.compute_loss(b_oh, b_h, b_op, b_tp, b_tgt)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(belief_net.parameters(), 0.5)
            opt.step()
            train_loss += loss.item()
            n_batches += 1

        # Validation (in chunks to avoid GPU OOM on large val sets)
        belief_net.eval()
        with torch.no_grad():
            # Val loss on full val set (chunked)
            val_loss_sum = 0.0
            val_chunks = 0
            VAL_CHUNK = 8192
            for vs in range(0, len(val_idx), VAL_CHUNK):
                ve = min(vs + VAL_CHUNK, len(val_idx))
                vl = belief_net.compute_loss(
                    val_oh[vs:ve].to(device), val_h[vs:ve].to(device),
                    val_op[vs:ve].to(device), val_tp[vs:ve].to(device),
                    val_tgt[vs:ve].to(device),
                ).item()
                val_loss_sum += vl * (ve - vs)
                val_chunks += (ve - vs)
            val_loss = val_loss_sum / max(1, val_chunks)

            # Accuracy on first 2000 val samples
            n_acc = min(2000, len(val_idx))
            val_probs = belief_net.get_probs(
                val_oh[:n_acc].to(device), val_h[:n_acc].to(device),
                val_op[:n_acc].to(device), val_tp[:n_acc].to(device),
            )
            acc = belief_accuracy(val_probs, val_tgt[:n_acc].to(device))

        avg_train = train_loss / max(1, n_batches)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  train_loss={avg_train:.4f}  "
                  f"val_loss={val_loss:.4f}  honor={acc['honor_acc']:.3f}  "
                  f"length={acc['length_acc']:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in belief_net.state_dict().items()}

    belief_net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"  [Stage A] Best val_loss={best_val_loss:.4f}")
    return belief_net


# ==============================================================================
# Stage B: SL Pretrain with Belief Features (576-dim)
# ==============================================================================

def train_sl_bca(
    train_file:  str,
    valid_file:  str,
    out_path:    str,
    belief_net:  BeliefNetwork,
    epochs:      int   = 10,
    batch_size:  int   = 2048,
    lr:          float = 3e-4,
    hidden_dim:  int   = 1024,
    device:      str   = 'cuda',
    max_lines:   int   = None,
    patience:    int   = 3,
    target_acc:  float = 0.36,
    init_from:   str   = None,
):
    """
    Stage B: Train 576-dim actors on SAYC data with belief features.

    P101: Actor input = base obs (480) + partner belief (48) + RHO belief (48) = 576.

    P99: When init_from is provided (path to sl_base.pt), load 480-dim weights
    into the 576-dim model with belief columns zero-initialised. This ensures:
    - Base obs (480-dim) weights start at sl_base.pt quality
    - Belief features (96-dim) influence starts from zero
    - Finetune teaches the model to USE belief features gradually
    - Final model has high-quality base weights + learned belief utilisation
    """
    print(f"\n[Stage B] SL Pretrain (576-dim BCA: partner + RHO)")
    print(f"  device={device}  epochs={epochs}  batch={batch_size}")
    if init_from:
        print(f"  init_from={init_from} (480→576 zero-init + finetune)")

    class_weight = torch.ones(NUM_BIDS, device=device)
    class_weight[BID_PASS] = 0.1

    # Lightweight collection (no hist_enc, no belief_targets)
    train_data = collect_stage_b_data(train_file, max_lines=max_lines)
    valid_data = collect_stage_b_data(valid_file, max_lines=50000)

    # Batched GPU belief inference → (N, 576) combined obs
    train_samples = build_belief_dataset_from_data(train_data, belief_net, device)
    valid_samples = build_belief_dataset_from_data(valid_data, belief_net, device)
    del train_data, valid_data  # free ~3 GB base arrays

    train_ds = SAYCBeliefDataset(train_samples)
    valid_ds  = SAYCBeliefDataset(valid_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    model = MLPPolicyNetwork(obs_dim=BELIEF_OBS_DIM, hidden_dim=hidden_dim).to(device)

    # P99: Load 480-dim weights from sl_base.pt, zero-init belief columns
    if init_from and os.path.exists(init_from):
        ckpt_base = torch.load(init_from, map_location=device)
        # sl_base.pt stores actor_n/s/e/w — all identical, use any one
        sl_key = None
        for k in ['actor_s', 'actor_n', 'actor_e', 'actor_w']:
            if k in ckpt_base:
                sl_key = k
                break
        if sl_key:
            sl_sd = {k: v.to(device) for k, v in ckpt_base[sl_key].items()}
            target_sd = model.state_dict()
            for param_name, sl_val in sl_sd.items():
                if param_name in target_sd:
                    tgt_shape = target_sd[param_name].shape
                    if sl_val.shape == tgt_shape:
                        target_sd[param_name] = sl_val
                    elif param_name == 'net.0.weight' and len(sl_val.shape) == 2:
                        # First layer: (hidden, 576) ← (hidden, 480) + zeros
                        # or (hidden, 576) ← (hidden, 349) + zeros for 349→576 upgrade
                        target_sd[param_name][:, :sl_val.shape[1]] = sl_val
                        target_sd[param_name][:, sl_val.shape[1]:] = 0.0
                    else:
                        target_sd[param_name] = sl_val
            model.load_state_dict(target_sd)
            print(f"  [P99] Loaded 480-dim weights from {init_from} ({sl_key}), "
                  f"belief columns zero-init")

            # P99/P101: Freeze all parameters, then unfreeze only what needs training.
            # Strategy: only train net.0.weight[:, 480:576] (belief columns) and
            # net.0.bias. All other layers are frozen — 480-dim quality preserved.
            # This allows higher lr (3e-4) without destroying base weights.
            for param in model.parameters():
                param.requires_grad_(False)
            # Unfreeze first layer weight and bias (belief columns learn, base columns frozen via zero grad mask)
            model.net[0].weight.requires_grad_(True)
            model.net[0].bias.requires_grad_(True)
            # We'll mask gradients for the base columns in the training loop
            _freeze_base_cols = True
            _base_dim = sl_val.shape[1]  # 480 (base obs dim)
            print(f"  [P99] Frozen all layers except net.0 (belief columns + bias)")
        else:
            print(f"  ⚠️  No actor found in {init_from}, training from scratch")
            _freeze_base_cols = False
            _base_dim = 0
    else:
        _freeze_base_cols = False
        _base_dim = 0

    # When init_from is used, only net.0 params are trainable → use higher lr
    _trainable = [p for p in model.parameters() if p.requires_grad]
    _eff_lr = lr if not _freeze_base_cols else max(lr, 3e-4)
    opt   = torch.optim.Adam(_trainable, lr=_eff_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if _freeze_base_cols:
        print(f"  [P99] Effective lr={_eff_lr} (only {sum(p.numel() for p in _trainable):,} "
              f"trainable params out of {sum(p.numel() for p in model.parameters()):,})")

    best_acc     = 0.0
    patience_cnt = 0
    best_state   = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0; n_batches = 0

        for flat_obs, actions in train_loader:
            flat_obs = flat_obs.to(device)
            actions  = actions.to(device)
            legal    = torch.ones(flat_obs.size(0), NUM_BIDS,
                                  dtype=torch.float32, device=device)

            logits = model(flat_obs, legal)
            loss   = F.cross_entropy(logits, actions, weight=class_weight)

            opt.zero_grad()
            loss.backward()
            # P99: Zero out gradients on base columns ([:, :480]) to preserve them
            if _freeze_base_cols:
                with torch.no_grad():
                    model.net[0].weight.grad[:, :_base_dim] = 0.0
            nn.utils.clip_grad_norm_(_trainable, 0.5)
            opt.step()

            total_loss += loss.item()
            n_batches  += 1

        train_loss = total_loss / max(1, n_batches)
        sched.step()

        # Validation
        model.eval()
        correct = total = 0
        correct_np = total_np = 0
        with torch.no_grad():
            for flat_obs, actions in valid_loader:
                flat_obs = flat_obs.to(device)
                actions  = actions.to(device)
                legal    = torch.ones(flat_obs.size(0), NUM_BIDS,
                                      dtype=torch.float32, device=device)
                logits   = model(flat_obs, legal)
                pred     = logits.argmax(dim=-1)
                correct += (pred == actions).sum().item()
                total   += actions.size(0)
                mask     = (actions != BID_PASS)
                correct_np += (pred[mask] == actions[mask]).sum().item()
                total_np   += mask.sum().item()

        val_acc    = correct    / max(1, total)
        val_acc_np = correct_np / max(1, total_np)
        print(f"  Epoch {epoch:3d}/{epochs}  loss={train_loss:.4f}  "
              f"val_acc={val_acc:.4f}  non_pass_acc={val_acc_np:.4f}")

        if val_acc_np > best_acc:
            best_acc   = val_acc_np
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if val_acc_np >= target_acc:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stop: non_pass_acc >= {target_acc} for {patience} epochs.")
                break
        else:
            patience_cnt = 0

    # Save: 4 actors (identical init) + belief net
    print(f"\n[Stage B] Best non_pass_acc={best_acc:.4f}. Saving → {out_path}")
    os.makedirs(Path(out_path).parent, exist_ok=True)

    belief_state = {k: v.cpu().clone() for k, v in belief_net.state_dict().items()}

    torch.save({
        'actor_n':          best_state,
        'actor_s':          best_state,
        'actor_e':          best_state,
        'actor_w':          best_state,
        'belief_net':       belief_state,
        'val_acc':          best_acc,
        'obs_dim':          BELIEF_OBS_DIM,
        'hidden_dim':       hidden_dim,          # Actor hidden dim
        'belief_hidden_dim': belief_net.trunk[0].out_features,  # BeliefNet hidden dim
    }, out_path)
    print(f"[SL Pretrain BCA] Done.")
    return best_acc


# ==============================================================================
# Main
# ==============================================================================

def main():
    p = argparse.ArgumentParser(description='SL Pretrain with Belief-Conditioned Actor (P98)')
    p.add_argument('--train',      default='data/sayc_train.txt')
    p.add_argument('--valid',      default='data/sayc_valid.txt')
    p.add_argument('--out',        default='results/sl_base_bca.pt')
    p.add_argument('--epochs',     type=int,   default=30)
    p.add_argument('--belief_epochs', type=int, default=30,
                   help='Epochs for Stage A (Belief Net training)')
    p.add_argument('--batch_size', type=int,   default=2048)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--hidden_dim', type=int,   default=1024)
    p.add_argument('--device',     default='cuda')
    p.add_argument('--max_lines',  type=int,   default=None,
                   help='Limit Stage B training lines (for quick test). '
                        'Stage A always uses --belief_max_lines.')
    p.add_argument('--belief_max_lines', type=int, default=200000,
                   help='Lines for Belief Net training (default 200k ≈ 1.7M samples). '
                        'Independent of --max_lines.')
    p.add_argument('--init_from', type=str, default=None,
                   help='P99/P101: Path to sl_base.pt (480-dim) or sl_base_bca.pt (576-dim). '
                        'Stage B will load these weights into 576-dim model with belief '
                        'columns zero-init, then finetune.')
    args = p.parse_args()

    # ── Stage A: Train Belief Net ──────────────────────────────────────
    # Read only --belief_max_lines lines. Fast, bounded memory.
    print("\n[Stage A] Collecting Belief Net training data "
          f"(max {args.belief_max_lines:,} lines)...")
    belief_samples = collect_all_data_from_sayc(
        args.train, max_lines=args.belief_max_lines)

    belief_net = train_belief_net(
        belief_samples,
        device=args.device,
        epochs=args.belief_epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
    )
    del belief_samples  # free ~15 GB before Stage B

    # ── Stage B: Train 576-dim Actor ──────────────────────────────────
    # collect_stage_b_data reads the full file (or --max_lines),
    # stores only lightweight fields (no 60×38 hist_enc).
    print("\n[Stage B] Training 576-dim Actor (partner + RHO belief)...")
    train_sl_bca(
        train_file  = args.train,
        valid_file  = args.valid,
        out_path    = args.out,
        belief_net  = belief_net,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        hidden_dim  = args.hidden_dim,
        device      = args.device,
        max_lines   = args.max_lines,
        init_from   = args.init_from,
    )


if __name__ == '__main__':
    main()
