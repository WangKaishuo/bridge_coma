"""
SL Pretrain with Belief-Conditioned Actor (P105 + P123 ReFine)
================================================================

Three-stage SL pretraining for 667-dim belief-conditioned actors:

Stage A: Train Belief Net on SAYC data (unchanged)

Stage B (original): Full fine-tune 667-dim Actor — DEPRECATED
  → Causes negative transfer: SL_BCA weaker than plain SL
  → Actor weights drift, bidding pattern changes even when belief is uninformative

Stage B-ReFine (P123): Residual Belief Adapter
  → Freeze entire plain SL actor (571-dim weights untouched)
  → Train a lightweight adapter: belief_96 → residual signal injected after layer 0
  → Zero-init gate ensures: at training start, adapter output = 0, actor = plain SL
  → Theoretical guarantee (Xu et al. 2025, ReFine): no negative transfer

  Architecture:
    obs_571 ──→ [frozen SL layer 0] ──→ h0 (1024-dim)
                                          │
    belief_96 → [adapter: down→ReLU→up] → δh (1024-dim) × gate(init=0)
                                          │
                                      h0 + δh ──→ [frozen SL layers 1-4] ──→ logits

  Training: cross-entropy on SAYC data, only adapter params updated.
  Result: when belief=prior, gate≈0, actor=plain SL. When belief informative, adapter adjusts.

Output: sl_base_bca_refine.pt

Usage:
  # Stage B-ReFine (recommended)
  python sl_pretrain_bca.py \
      --train data/sayc_train.txt \
      --valid data/sayc_valid.txt \
      --out   results/sl_base_bca_refine.pt \
      --init_from results/sl_base.pt \
      --load_belief results/sl_base_bca.pt \
      --mode refine \
      --iterations 400000 --device cuda

  # Stage B legacy (full fine-tune, not recommended)
  python sl_pretrain_bca.py \
      --train data/sayc_train.txt \
      --valid data/sayc_valid.txt \
      --out   results/sl_base_bca_stageB.pt \
      --init_from results/sl_base.pt \
      --load_belief results/sl_base_bca.pt \
      --mode legacy \
      --iterations 400000 --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
import random
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pyspiel

from env import (
    NUM_BIDS, NUM_PLAYERS,
    BID_PASS, BID_DOUBLE, BID_REDOUBLE, BID_1C,
)
from networks.policy_net import (
    MLPPolicyNetwork, OBS_DIM, BELIEF_OBS_DIM, BELIEF_FEAT_DIM,
    openspiel_raw_to_ours, rank_major_to_suit_major,
)
from networks.belief_net import BeliefNetwork
from utils.hand_features import hand_to_belief_target, belief_accuracy, BELIEF_DIM

# ── Constants ────────────────────────────────────────────────────────────────
_NUM_CARDS = 52
_GAME = pyspiel.load_game('bridge(use_double_dummy_result=false)')


# ==============================================================================
# Data loading (shared between Stage A and B)
# ==============================================================================

def load_trajectories(filepath: str, max_lines: int = None) -> List[Tuple[int, ...]]:
    """Load SAYC trajectories, stripping play phase."""
    trajectories = []
    print(f"[Data] Loading {filepath}...")
    t0 = time.time()
    with open(filepath) as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            actions = tuple(int(x) for x in line.split())
            if len(actions) == _NUM_CARDS + 4:
                pass  # all-pass, no play
            elif len(actions) > _NUM_CARDS + 4:
                actions = actions[:-_NUM_CARDS]
            if len(actions) > _NUM_CARDS:
                trajectories.append(actions)
    print(f"[Data] {len(trajectories):,} games in {time.time()-t0:.1f}s.")
    return trajectories


def _rebuild_hands_sm(traj) -> np.ndarray:
    """Rebuild (4, 52) suit-major hands from dealing actions."""
    hands_sm = np.zeros((4, 52), dtype=np.float32)
    for i, card_rm in enumerate(traj[:_NUM_CARDS]):
        player = i % 4  # interleaved dealing
        sm_idx = rank_major_to_suit_major(card_rm)
        hands_sm[player, sm_idx] = 1.0
    return hands_sm


# ==============================================================================
# Stage A: Train Belief Net (unchanged from original)
# ==============================================================================

def _sample_belief_batch(
    trajectories: list,
    n_samples: int,
    include_opponents: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample belief training data by replaying OpenSpiel states.

    Returns:
        obs_arr:    (N, 571) float32 — OpenSpiel observation
        tp_arr:     (N,) int64 — target player (absolute position)
        tgt_arr:    (N, 48) float32 — target hand features
    """
    obs_list = []
    tp_list  = []
    tgt_list = []

    while len(tp_list) < n_samples:
        traj = random.choice(trajectories)
        action_index = random.randint(_NUM_CARDS, len(traj) - 1)

        state = _GAME.new_initial_state()
        for action in traj[:action_index]:
            state.apply_action(action)

        obs = np.array(state.observation_tensor(), dtype=np.float32)
        player = state.current_player()
        hands_sm = _rebuild_hands_sm(traj)

        # Targets: partner (always) + LHO + RHO (if include_opponents)
        partner = (player + 2) % 4
        targets = [partner]
        if include_opponents:
            targets.append((player + 1) % 4)  # LHO
            targets.append((player + 3) % 4)  # RHO

        for tgt_player in targets:
            obs_list.append(obs)
            tp_list.append(tgt_player)
            tgt_list.append(hand_to_belief_target(hands_sm[tgt_player]))

    return (
        np.stack(obs_list[:n_samples]),
        np.array(tp_list[:n_samples], dtype=np.int64),
        np.stack(tgt_list[:n_samples]),
    )


def train_belief_net(
    trajectories: list,
    device: str = 'cuda',
    epochs: int = 30,
    batch_size: int = 2048,
    lr: float = 1e-3,
    hidden_dim: int = 512,
    include_opponents: bool = True,
) -> BeliefNetwork:
    """Train Belief Net on SAYC trajectories using OpenSpiel observations."""

    N_traj = len(trajectories)
    print(f"\n[Stage A] Training Belief Net: {N_traj:,} trajectories, {epochs} epochs")

    belief_net = BeliefNetwork(obs_dim=OBS_DIM, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(belief_net.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_state = None

    # Fixed validation set
    print(f"  Building validation set...")
    val_size = min(50000, N_traj // 10)
    val_obs, val_tp, val_tgt = _sample_belief_batch(
        trajectories, val_size, include_opponents)
    val_obs_t = torch.tensor(val_obs, dtype=torch.float32)
    val_tp_t  = torch.tensor(val_tp,  dtype=torch.long)
    val_tgt_t = torch.tensor(val_tgt, dtype=torch.float32)
    print(f"  Validation set: {val_size:,} samples.")

    for epoch in range(1, epochs + 1):
        belief_net.train()
        train_loss = 0.0
        n_batches = 0
        t0 = time.time()

        # Fresh training data each epoch
        n_train = min(500000, N_traj * 3 if include_opponents else N_traj)
        tr_obs, tr_tp, tr_tgt = _sample_belief_batch(
            trajectories, n_train, include_opponents)

        # Shuffle
        perm = np.random.permutation(n_train)
        tr_obs, tr_tp, tr_tgt = tr_obs[perm], tr_tp[perm], tr_tgt[perm]

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            b_obs = torch.tensor(tr_obs[start:end], dtype=torch.float32).to(device)
            b_tp  = torch.tensor(tr_tp[start:end],  dtype=torch.long).to(device)
            b_tgt = torch.tensor(tr_tgt[start:end], dtype=torch.float32).to(device)

            loss = belief_net.compute_loss(b_obs, b_tp, b_tgt)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(belief_net.parameters(), 0.5)
            opt.step()
            train_loss += loss.item()
            n_batches += 1

        # Validation
        belief_net.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            CHUNK = 8192
            for vs in range(0, len(val_tp_t), CHUNK):
                ve = min(vs + CHUNK, len(val_tp_t))
                vl = belief_net.compute_loss(
                    val_obs_t[vs:ve].to(device),
                    val_tp_t[vs:ve].to(device),
                    val_tgt_t[vs:ve].to(device),
                ).item()
                val_loss_sum += vl * (ve - vs)
            val_loss = val_loss_sum / max(1, len(val_tp_t))

            n_acc = min(2000, len(val_tp_t))
            val_probs = belief_net.get_probs(
                val_obs_t[:n_acc].to(device),
                val_tp_t[:n_acc].to(device))
            acc = belief_accuracy(val_probs, val_tgt_t[:n_acc].to(device))

        avg_train = train_loss / max(1, n_batches)
        elapsed = time.time() - t0
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  train_loss={avg_train:.4f}  "
                  f"val_loss={val_loss:.4f}  honor={acc['honor_acc']:.3f}  "
                  f"length={acc['length_acc']:.3f}  [{elapsed:.0f}s]")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in belief_net.state_dict().items()}

    belief_net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"  [Stage A] Best val_loss={best_val_loss:.4f}")
    return belief_net


# ==============================================================================
# P123: Residual Belief Adapter (ReFine-inspired)
# ==============================================================================

class BeliefAdapter(nn.Module):
    """
    Lightweight residual adapter that injects belief information into a frozen actor.

    Architecture:
        belief_96 → Linear(96, bottleneck) → ReLU → Linear(bottleneck, hidden_dim) → × gate
        output: δh of shape (B, hidden_dim), added to frozen layer-0 output

    Key design choices (inspired by ReFine / LoRA / Side-Tuning):
    1. gate initialized to 0 → at start, δh=0 → actor = plain SL exactly
    2. Bottleneck keeps adapter small (~50k params vs actor's ~5M)
    3. Only adapter params are trained; actor is completely frozen
    4. No negative transfer guarantee: if belief is uninformative,
       optimal adapter output is 0 (via gate → 0 or weights → 0)
    """

    def __init__(self, belief_dim: int = BELIEF_FEAT_DIM,
                 hidden_dim: int = 1024, bottleneck: int = 128):
        super().__init__()
        self.down = nn.Linear(belief_dim, bottleneck)
        self.up   = nn.Linear(bottleneck, hidden_dim)
        # Learnable gate scalar, initialized to 0 → adapter has zero effect at start
        self.gate = nn.Parameter(torch.zeros(1))

        # Init: Xavier for both down and up.
        # gate=0 alone guarantees δh=0 at initialization (no dual zero-lock).
        # up must have non-zero weights so that d(loss)/d(gate) ≠ 0.
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, belief_feats: torch.Tensor) -> torch.Tensor:
        """
        belief_feats: (B, 96)
        returns: (B, hidden_dim) residual signal
        """
        h = F.relu(self.down(belief_feats))
        h = self.up(h)
        return h * self.gate


class ReFineActor(nn.Module):
    """
    Frozen SL actor + Residual Belief Adapter.

    Forward pass:
        1. obs_571 → frozen layer 0 → h0
        2. belief_96 → adapter → δh
        3. h0 + δh → frozen layers 1-4 → logits

    The actor's 571-dim weights are NEVER modified.
    Only the adapter's ~50k parameters are trained.
    """

    def __init__(self, frozen_actor: MLPPolicyNetwork,
                 adapter: BeliefAdapter):
        super().__init__()
        self.frozen_actor = frozen_actor
        self.adapter = adapter

        # Freeze all actor parameters
        for p in self.frozen_actor.parameters():
            p.requires_grad = False

    def _get_hidden_and_rest(self, obs_571: torch.Tensor):
        """Run frozen layer 0 (Linear + ReLU), return h0 and remaining layers."""
        # frozen_actor.net = Sequential(Linear, ReLU, Linear, ReLU, ..., Linear)
        # Layer 0: net[0] (Linear) + net[1] (ReLU)
        h0 = self.frozen_actor.net[1](self.frozen_actor.net[0](obs_571))
        return h0

    def _run_rest(self, h: torch.Tensor):
        """Run frozen layers 1-4 (after layer 0)."""
        # net[2:] = Linear, ReLU, Linear, ReLU, Linear, ReLU, Linear(out)
        return self.frozen_actor.net[2:](h)

    def forward(self, obs_571: torch.Tensor, belief_96: torch.Tensor,
                legal_actions: torch.Tensor):
        """
        obs_571:       (B, 571)
        belief_96:     (B, 96) — from BeliefNet
        legal_actions: (B, num_bids)
        returns: masked logits (B, num_bids)
        """
        h0 = self._get_hidden_and_rest(obs_571)
        delta_h = self.adapter(belief_96)
        h = h0 + delta_h
        logits = self._run_rest(h)
        return logits - 1e9 * (1.0 - legal_actions)

    def get_action(self, obs_571, belief_96, legal_actions, deterministic=False):
        logits = self.forward(obs_571, belief_96, legal_actions)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def adapter_param_count(self):
        return sum(p.numel() for p in self.adapter.parameters())

    def total_param_count(self):
        return sum(p.numel() for p in self.parameters())

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# Stage B data classes (shared between legacy and ReFine)
# ==============================================================================

class SAYCGamesBCA:
    """Iteration-based dataset for Stage B. Queries belief net on-the-fly."""

    def __init__(self, filepath: str, max_lines: int = None):
        self.trajectories = load_trajectories(filepath, max_lines)

    def sample_batch(
        self, batch_size: int, belief_net: BeliefNetwork, device: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample (obs_571, belief_96, action) batch.
        Returns separate obs and belief arrays for ReFine compatibility.
        """
        obs_list = []
        act_list = []
        pa_list  = []
        rh_list  = []

        for _ in range(batch_size):
            traj = random.choice(self.trajectories)
            action_index = random.randint(_NUM_CARDS, len(traj) - 1)

            state = _GAME.new_initial_state()
            for action in traj[:action_index]:
                state.apply_action(action)

            obs_571 = np.array(state.observation_tensor(), dtype=np.float32)
            # Use same encoding as sl_pretrain.py: target = raw_action - 52
            # This matches sl_base.pt's action space (0-37: Pass=0, 1C=1, 1D=2, ...)
            raw_action = traj[action_index]
            target = raw_action - _NUM_CARDS  # 52-89 → 0-37
            if not (0 <= target < NUM_BIDS):
                target = 0  # fallback to Pass

            player  = state.current_player()
            partner = (player + 2) % 4
            rho     = (player + 3) % 4

            obs_list.append(obs_571)
            act_list.append(target)
            pa_list.append(partner)
            rh_list.append(rho)

        # Batched belief query
        obs_t = torch.tensor(np.stack(obs_list), dtype=torch.float32).to(device)
        pa_t  = torch.tensor(pa_list, dtype=torch.long).to(device)
        rh_t  = torch.tensor(rh_list, dtype=torch.long).to(device)

        belief_net.eval()
        with torch.no_grad():
            partner_feats = belief_net.get_probs(obs_t, pa_t).cpu().numpy()
            rho_feats     = belief_net.get_probs(obs_t, rh_t).cpu().numpy()

        base_obs = np.stack(obs_list)                                    # (B, 571)
        belief   = np.concatenate([partner_feats, rho_feats], axis=1)    # (B, 96)

        return base_obs, belief, np.array(act_list, dtype=np.int64)


class SAYCValidationReFine(Dataset):
    """Deterministic validation dataset — returns (obs_571, belief_96, action)."""

    def __init__(self, filepath: str, belief_net: BeliefNetwork,
                 device: str, max_lines: int = 50000):
        print(f"[SAYCValidationReFine] Building from {filepath}...")
        t0 = time.time()

        obs_list = []
        act_list = []
        pa_list  = []
        rh_list  = []

        with open(filepath) as f:
            for line_idx, line in enumerate(f):
                if max_lines and line_idx >= max_lines:
                    break
                actions = tuple(int(x) for x in line.split())
                if len(actions) == _NUM_CARDS + 4:
                    pass
                elif len(actions) > _NUM_CARDS + 4:
                    actions = actions[:-_NUM_CARDS]
                if len(actions) <= _NUM_CARDS:
                    continue

                state = _GAME.new_initial_state()
                for action in actions[:_NUM_CARDS]:
                    state.apply_action(action)

                for step_idx in range(_NUM_CARDS, len(actions)):
                    obs_571 = np.array(state.observation_tensor(), dtype=np.float32)
                    # Use same encoding as sl_pretrain.py: target = raw_action - 52
                    raw_action = actions[step_idx]
                    target = raw_action - _NUM_CARDS  # 52-89 → 0-37
                    if not (0 <= target < NUM_BIDS):
                        state.apply_action(raw_action)
                        continue

                    player  = state.current_player()
                    partner = (player + 2) % 4
                    rho     = (player + 3) % 4

                    obs_list.append(obs_571)
                    act_list.append(target)
                    pa_list.append(partner)
                    rh_list.append(rho)

                    state.apply_action(raw_action)

        N = len(act_list)
        print(f"[SAYCValidationReFine] {N:,} samples, running belief inference...")

        obs_arr = np.stack(obs_list)
        pa_arr  = np.array(pa_list, dtype=np.int64)
        rh_arr  = np.array(rh_list, dtype=np.int64)

        partner_feats = np.empty((N, 48), dtype=np.float32)
        rho_feats     = np.empty((N, 48), dtype=np.float32)
        belief_net.eval()
        BS = 4096
        with torch.no_grad():
            for start in range(0, N, BS):
                end = min(start + BS, N)
                obs_t = torch.tensor(obs_arr[start:end], dtype=torch.float32).to(device)
                pa_t  = torch.tensor(pa_arr[start:end], dtype=torch.long).to(device)
                rh_t  = torch.tensor(rh_arr[start:end], dtype=torch.long).to(device)
                partner_feats[start:end] = belief_net.get_probs(obs_t, pa_t).cpu().numpy()
                rho_feats[start:end]     = belief_net.get_probs(obs_t, rh_t).cpu().numpy()

        belief = np.concatenate([partner_feats, rho_feats], axis=1)

        self.obs_571 = torch.tensor(obs_arr, dtype=torch.float32)
        self.belief  = torch.tensor(belief, dtype=torch.float32)
        self.actions = torch.tensor(np.array(act_list), dtype=torch.int64)
        print(f"[SAYCValidationReFine] Done. {N:,} samples, t={time.time()-t0:.1f}s")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.obs_571[idx], self.belief[idx], self.actions[idx]


# ==============================================================================
# Stage B-ReFine: Train Residual Belief Adapter
# ==============================================================================

def train_sl_bca_refine(
    train_file: str, valid_file: str, out_path: str,
    belief_net: BeliefNetwork,
    sl_checkpoint: str,
    iterations: int = 400000, batch_size: int = 128,
    lr: float = 3e-4, hidden_dim: int = 1024,
    bottleneck: int = 128,
    device: str = 'cuda', max_lines: int = None,
    eval_every: int = 10000,
):
    """
    Stage B-ReFine: Train a Residual Belief Adapter on frozen plain SL.

    The plain SL actor is loaded from sl_checkpoint and completely frozen.
    Only the adapter's parameters (~50k) are trained.
    """
    print(f"\n[Stage B-ReFine] Residual Belief Adapter Training")
    print(f"  device={device}  iterations={iterations}  batch={batch_size}  lr={lr}")
    print(f"  bottleneck={bottleneck}  hidden_dim={hidden_dim}")

    # ── Load frozen SL actor ──────────────────────────────────────────────
    print(f"\n  Loading frozen SL actor from: {sl_checkpoint}")
    frozen_actor = MLPPolicyNetwork(obs_dim=OBS_DIM, hidden_dim=hidden_dim).to(device)
    ckpt = torch.load(sl_checkpoint, map_location=device, weights_only=False)
    sl_sd = ckpt.get('model_state') or next(
        (ckpt[k] for k in ['actor_s', 'actor_n', 'actor_e', 'actor_w'] if k in ckpt), None)
    if sl_sd:
        sl_sd = {k: v.to(device) for k, v in sl_sd.items()}
        frozen_actor.load_state_dict(sl_sd)
        print(f"  SL actor loaded (obs_dim={OBS_DIM}, hidden_dim={hidden_dim})")
    else:
        print(f"  WARNING: Could not load SL weights from {sl_checkpoint}")

    # ── Build ReFine actor ────────────────────────────────────────────────
    adapter = BeliefAdapter(
        belief_dim=BELIEF_FEAT_DIM, hidden_dim=hidden_dim,
        bottleneck=bottleneck,
    ).to(device)

    refine_actor = ReFineActor(frozen_actor, adapter).to(device)

    n_total = refine_actor.total_param_count()
    n_train = refine_actor.trainable_param_count()
    n_adapter = refine_actor.adapter_param_count()
    print(f"  Total params:     {n_total:,}")
    print(f"  Trainable params: {n_train:,} (adapter only)")
    print(f"  Adapter params:   {n_adapter:,}")
    print(f"  Frozen params:    {n_total - n_train:,}")
    print(f"  Adapter/Total:    {n_adapter/n_total*100:.2f}%")

    # ── Data ──────────────────────────────────────────────────────────────
    train_games = SAYCGamesBCA(train_file, max_lines=max_lines)
    valid_ds    = SAYCValidationReFine(valid_file, belief_net, device, max_lines=50000)
    valid_loader = DataLoader(valid_ds, batch_size=2048, shuffle=False)

    # ── Optimizer (only adapter params) ───────────────────────────────────
    opt = torch.optim.Adam(adapter.parameters(), lr=lr)
    best_np_acc = 0.0
    best_state  = None
    running_loss = 0.0
    t_start = time.time()

    # ── Baseline: plain SL accuracy (adapter gate=0) ──────────────────────
    print(f"\n  [Baseline] Computing plain SL accuracy (gate=0)...")
    refine_actor.eval()
    _c = _t = _cnp = _tnp = 0
    with torch.no_grad():
        for obs_v, belief_v, targets_v in valid_loader:
            obs_v     = obs_v.to(device)
            belief_v  = belief_v.to(device)
            targets_v = targets_v.to(device)
            legal_v   = torch.ones(obs_v.size(0), NUM_BIDS,
                                   dtype=torch.float32, device=device)
            logits_v  = refine_actor(obs_v, belief_v, legal_v)
            pred = logits_v.argmax(dim=-1)
            _c  += (pred == targets_v).sum().item()
            _t  += targets_v.size(0)
            mask = (targets_v != BID_PASS)
            _cnp += (pred[mask] == targets_v[mask]).sum().item()
            _tnp += mask.sum().item()
    baseline_acc    = _c / max(1, _t)
    baseline_np_acc = _cnp / max(1, _tnp)
    print(f"  [Baseline] plain SL: val_acc={baseline_acc:.4f}  "
          f"non_pass_acc={baseline_np_acc:.4f}  (gate={adapter.gate.item():.6f})")

    # ── Training loop ─────────────────────────────────────────────────────
    for it in range(1, iterations + 1):
        obs_np, belief_np, act_np = train_games.sample_batch(
            batch_size, belief_net, device)
        obs_t    = torch.tensor(obs_np, dtype=torch.float32, device=device)
        belief_t = torch.tensor(belief_np, dtype=torch.float32, device=device)
        actions  = torch.tensor(act_np, dtype=torch.int64, device=device)
        legal    = torch.ones(obs_t.size(0), NUM_BIDS,
                              dtype=torch.float32, device=device)

        refine_actor.train()
        logits = refine_actor(obs_t, belief_t, legal)
        loss = F.cross_entropy(logits, actions)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(adapter.parameters(), 0.5)
        opt.step()
        running_loss += loss.item()

        if it % eval_every == 0 or it == iterations:
            refine_actor.eval()
            correct = total = correct_np = total_np = 0
            with torch.no_grad():
                for obs_v, belief_v, targets_v in valid_loader:
                    obs_v     = obs_v.to(device)
                    belief_v  = belief_v.to(device)
                    targets_v = targets_v.to(device)
                    legal_v   = torch.ones(obs_v.size(0), NUM_BIDS,
                                           dtype=torch.float32, device=device)
                    logits_v  = refine_actor(obs_v, belief_v, legal_v)
                    pred = logits_v.argmax(dim=-1)
                    correct  += (pred == targets_v).sum().item()
                    total    += targets_v.size(0)
                    mask = (targets_v != BID_PASS)
                    correct_np += (pred[mask] == targets_v[mask]).sum().item()
                    total_np   += mask.sum().item()

            val_acc    = correct    / max(1, total)
            val_np_acc = correct_np / max(1, total_np)
            avg_loss   = running_loss / eval_every
            running_loss = 0.0
            elapsed = time.time() - t_start
            gate_val = adapter.gate.item()
            print(f"  Iter {it:7d}/{iterations}  loss={avg_loss:.4f}  "
                  f"val_acc={val_acc:.4f}  non_pass_acc={val_np_acc:.4f}  "
                  f"gate={gate_val:.4f}  [{elapsed/60:.1f}min]")

            if val_np_acc > best_np_acc:
                best_np_acc = val_np_acc
                # Save: frozen actor state + adapter state + belief net
                best_actor_state = {
                    k: v.cpu().clone() for k, v in frozen_actor.state_dict().items()
                }
                best_adapter_state = {
                    k: v.cpu().clone() for k, v in adapter.state_dict().items()
                }
                belief_state = {
                    k: v.cpu().clone() for k, v in belief_net.state_dict().items()
                }
                os.makedirs(Path(out_path).parent, exist_ok=True)
                torch.save({
                    'model_state':       best_actor_state,
                    'adapter_state':     best_adapter_state,
                    'belief_net':        belief_state,
                    'non_pass_acc':      best_np_acc,
                    'baseline_np_acc':   baseline_np_acc,
                    'obs_dim':           OBS_DIM,  # 571 (frozen actor input)
                    'belief_obs_dim':    BELIEF_OBS_DIM,  # 667 (for compatibility)
                    'hidden_dim':        hidden_dim,
                    'bottleneck':        bottleneck,
                    'belief_hidden_dim': belief_net.trunk[0].out_features,
                    'iteration':         it,
                    'encoding':          'openspiel_667_refine',
                    'gate':              gate_val,
                    'mode':              'refine',
                }, out_path)

    print(f"\n[Stage B-ReFine] Best non_pass_acc={best_np_acc:.4f} "
          f"(baseline={baseline_np_acc:.4f}, Δ={best_np_acc-baseline_np_acc:+.4f})")
    print(f"  Final gate={adapter.gate.item():.4f}")
    print(f"  Saved → {out_path}")
    return best_np_acc


# ==============================================================================
# Stage B Legacy: Full fine-tune 667-dim (unchanged, for comparison)
# ==============================================================================

class SAYCValidationBCA(Dataset):
    """Deterministic validation dataset for Stage B legacy."""

    def __init__(self, filepath: str, belief_net: BeliefNetwork,
                 device: str, max_lines: int = 50000):
        print(f"[SAYCValidationBCA] Building from {filepath}...")
        t0 = time.time()

        obs_list = []
        act_list = []
        pa_list  = []
        rh_list  = []

        with open(filepath) as f:
            for line_idx, line in enumerate(f):
                if max_lines and line_idx >= max_lines:
                    break
                actions = tuple(int(x) for x in line.split())
                if len(actions) == _NUM_CARDS + 4:
                    pass
                elif len(actions) > _NUM_CARDS + 4:
                    actions = actions[:-_NUM_CARDS]
                if len(actions) <= _NUM_CARDS:
                    continue

                state = _GAME.new_initial_state()
                for action in actions[:_NUM_CARDS]:
                    state.apply_action(action)

                for step_idx in range(_NUM_CARDS, len(actions)):
                    obs_571 = np.array(state.observation_tensor(), dtype=np.float32)
                    # Use same encoding as sl_pretrain.py: target = raw_action - 52
                    raw_action = actions[step_idx]
                    target = raw_action - _NUM_CARDS  # 52-89 → 0-37
                    if not (0 <= target < NUM_BIDS):
                        state.apply_action(raw_action)
                        continue

                    player  = state.current_player()
                    partner = (player + 2) % 4
                    rho     = (player + 3) % 4

                    obs_list.append(obs_571)
                    act_list.append(target)
                    pa_list.append(partner)
                    rh_list.append(rho)

                    state.apply_action(raw_action)

        N = len(act_list)
        print(f"[SAYCValidationBCA] {N:,} samples, running belief inference...")

        obs_arr = np.stack(obs_list)
        pa_arr  = np.array(pa_list, dtype=np.int64)
        rh_arr  = np.array(rh_list, dtype=np.int64)

        partner_feats = np.empty((N, 48), dtype=np.float32)
        rho_feats     = np.empty((N, 48), dtype=np.float32)
        belief_net.eval()
        BS = 4096
        with torch.no_grad():
            for start in range(0, N, BS):
                end = min(start + BS, N)
                obs_t = torch.tensor(obs_arr[start:end], dtype=torch.float32).to(device)
                pa_t  = torch.tensor(pa_arr[start:end], dtype=torch.long).to(device)
                rh_t  = torch.tensor(rh_arr[start:end], dtype=torch.long).to(device)
                partner_feats[start:end] = belief_net.get_probs(obs_t, pa_t).cpu().numpy()
                rho_feats[start:end]     = belief_net.get_probs(obs_t, rh_t).cpu().numpy()

        belief = np.concatenate([partner_feats, rho_feats], axis=1)
        obs_667 = np.concatenate([obs_arr, belief], axis=1)

        self.obs     = torch.tensor(obs_667, dtype=torch.float32)
        self.actions = torch.tensor(np.array(act_list), dtype=torch.int64)
        print(f"[SAYCValidationBCA] Done. obs_dim={obs_667.shape[1]}, t={time.time()-t0:.1f}s")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


def train_sl_bca(
    train_file: str, valid_file: str, out_path: str,
    belief_net: BeliefNetwork,
    iterations: int = 400000, batch_size: int = 128,
    lr: float = 1e-4, hidden_dim: int = 1024,
    device: str = 'cuda', max_lines: int = None,
    eval_every: int = 10000, init_from: str = None,
    freeze_base: bool = False,
):
    """Stage B legacy: Full fine-tune 667-dim actor (original, not recommended)."""
    print(f"\n[Stage B Legacy] SL Pretrain (667-dim BCA, iteration-based)")
    print(f"  device={device}  iterations={iterations}  batch={batch_size}  lr={lr}")
    if freeze_base:
        print(f"  freeze_base=True → net.0.weight[:, :571] frozen")
    if init_from:
        print(f"  init_from={init_from} (571→667 zero-init)")

    # Legacy sample_batch returns (obs_667, action) — need wrapper
    train_games = SAYCGamesBCA(train_file, max_lines=max_lines)
    valid_ds    = SAYCValidationBCA(valid_file, belief_net, device, max_lines=50000)
    valid_loader = DataLoader(valid_ds, batch_size=2048, shuffle=False)

    model = MLPPolicyNetwork(obs_dim=BELIEF_OBS_DIM, hidden_dim=hidden_dim).to(device)

    # Init from base SL checkpoint (571→667)
    if init_from and os.path.exists(init_from):
        ckpt = torch.load(init_from, map_location=device, weights_only=False)
        sl_sd = ckpt.get('model_state') or next(
            (ckpt[k] for k in ['actor_s','actor_n','actor_e','actor_w'] if k in ckpt), None)
        if sl_sd:
            sl_sd = {k: v.to(device) for k, v in sl_sd.items()}
            target_sd = model.state_dict()
            for pn, sv in sl_sd.items():
                if pn not in target_sd:
                    continue
                if sv.shape == target_sd[pn].shape:
                    target_sd[pn] = sv
                elif pn == 'net.0.weight' and len(sv.shape) == 2:
                    target_sd[pn][:, :sv.shape[1]] = sv
                    target_sd[pn][:, sv.shape[1]:] = 0.0
                else:
                    target_sd[pn] = sv
            model.load_state_dict(target_sd)
            print(f"  [init] Base weights loaded, belief cols zero-init")

    _grad_hook = None
    if freeze_base:
        base_dim = OBS_DIM
        w0 = model.net[0].weight
        grad_mask = torch.zeros_like(w0)
        grad_mask[:, base_dim:] = 1.0
        _grad_hook = w0.register_hook(lambda grad: grad * grad_mask)
        print(f"  [freeze] net.0.weight: base cols frozen, belief cols trainable")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_np_acc = 0.0
    best_state  = None
    running_loss = 0.0
    t_start = time.time()

    for it in range(1, iterations + 1):
        obs_np, belief_np, act_np = train_games.sample_batch(batch_size, belief_net, device)
        # Legacy: concatenate to 667-dim
        obs_667 = np.concatenate([obs_np, belief_np], axis=1)
        flat_obs = torch.tensor(obs_667, dtype=torch.float32, device=device)
        actions  = torch.tensor(act_np, dtype=torch.int64,  device=device)
        legal    = torch.ones(flat_obs.size(0), NUM_BIDS, dtype=torch.float32, device=device)

        model.train()
        logits = model(flat_obs, legal)
        loss   = F.cross_entropy(logits, actions)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        running_loss += loss.item()

        if it % eval_every == 0 or it == iterations:
            model.eval()
            correct = total = correct_np = total_np = 0
            with torch.no_grad():
                for flat_v, targets_v in valid_loader:
                    flat_v    = flat_v.to(device)
                    targets_v = targets_v.to(device)
                    legal_v   = torch.ones(flat_v.size(0), NUM_BIDS,
                                           dtype=torch.float32, device=device)
                    logits_v  = model(flat_v, legal_v)
                    pred      = logits_v.argmax(dim=-1)
                    correct  += (pred == targets_v).sum().item()
                    total    += targets_v.size(0)
                    mask      = (targets_v != BID_PASS)
                    correct_np += (pred[mask] == targets_v[mask]).sum().item()
                    total_np   += mask.sum().item()

            val_acc    = correct    / max(1, total)
            val_np_acc = correct_np / max(1, total_np)
            avg_loss   = running_loss / eval_every
            running_loss = 0.0
            elapsed = time.time() - t_start
            print(f"  Iter {it:7d}/{iterations}  loss={avg_loss:.4f}  "
                  f"val_acc={val_acc:.4f}  non_pass_acc={val_np_acc:.4f}  "
                  f"[{elapsed/60:.1f}min]")

            if val_np_acc > best_np_acc:
                best_np_acc = val_np_acc
                best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                belief_state = {k: v.cpu().clone() for k, v in belief_net.state_dict().items()}
                os.makedirs(Path(out_path).parent, exist_ok=True)
                torch.save({
                    'model_state':       best_state,
                    'belief_net':        belief_state,
                    'non_pass_acc':      best_np_acc,
                    'obs_dim':           BELIEF_OBS_DIM,
                    'hidden_dim':        hidden_dim,
                    'belief_hidden_dim': belief_net.trunk[0].out_features,
                    'iteration':         it,
                    'encoding':          'openspiel_667',
                    'mode':              'legacy',
                }, out_path)

    print(f"\n[Stage B Legacy] Best non_pass_acc={best_np_acc:.4f}. Saved → {out_path}")
    if _grad_hook is not None:
        _grad_hook.remove()
    return best_np_acc


# ==============================================================================
# Main
# ==============================================================================

def main():
    p = argparse.ArgumentParser(description='SL Pretrain BCA (P105 + P123 ReFine)')
    p.add_argument('--train',      default='data/sayc_train.txt')
    p.add_argument('--valid',      default='data/sayc_valid.txt')
    p.add_argument('--out',        default='results/sl_base_bca.pt')
    p.add_argument('--iterations', type=int, default=400000)
    p.add_argument('--belief_epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr',         type=float, default=None,
                   help='Learning rate. Default: 3e-4 for refine, 1e-4 for legacy')
    p.add_argument('--hidden_dim', type=int, default=1024)
    p.add_argument('--bottleneck', type=int, default=128,
                   help='Adapter bottleneck dim (only for --mode refine)')
    p.add_argument('--device',     default='cuda')
    p.add_argument('--eval_every', type=int, default=10000)
    p.add_argument('--max_lines',  type=int, default=None)
    p.add_argument('--belief_max_lines', type=int, default=200000)
    p.add_argument('--init_from',   type=str, default=None,
                   help='Path to sl_base.pt (571-dim) for init')
    p.add_argument('--load_belief', type=str, default=None,
                   help='Path to existing BeliefNet checkpoint (e.g. sl_base_bca.pt). '
                        'If set, skip Stage A and load BeliefNet directly.')
    p.add_argument('--mode', choices=['refine', 'legacy'], default='refine',
                   help='refine = P123 Residual Belief Adapter (recommended). '
                        'legacy = original full fine-tune (for comparison).')
    p.add_argument('--freeze_base', action='store_true',
                   help='[legacy mode only] Freeze first layer base-obs columns.')
    args = p.parse_args()

    # Default LR depends on mode
    if args.lr is None:
        args.lr = 3e-4 if args.mode == 'refine' else 1e-4

    # Stage A — skip if --load_belief provided
    if args.load_belief:
        print(f"\n[Stage A] Skipped — loading BeliefNet from: {args.load_belief}")
        _bn_ckpt   = torch.load(args.load_belief, map_location=args.device,
                                 weights_only=False)
        _bn_sd     = _bn_ckpt.get('belief_net', _bn_ckpt)
        _bn_hidden = _bn_ckpt.get('belief_hidden_dim',
                                   next(iter(_bn_sd.values())).shape[0])
        belief_net = BeliefNetwork(obs_dim=OBS_DIM, hidden_dim=_bn_hidden).to(args.device)
        belief_net.load_state_dict({k: v.to(args.device) for k, v in _bn_sd.items()})
        print(f"  BeliefNet loaded (hidden_dim={_bn_hidden})")
    else:
        print(f"\n[Stage A] Collecting data (max {args.belief_max_lines:,} lines)...")
        trajs = load_trajectories(args.train, max_lines=args.belief_max_lines)
        belief_net = train_belief_net(
            trajs, device=args.device, epochs=args.belief_epochs,
            batch_size=2048, hidden_dim=512)
        del trajs

    # Stage B
    if args.mode == 'refine':
        if not args.init_from:
            print("ERROR: --mode refine requires --init_from (path to sl_base.pt)")
            sys.exit(1)
        print(f"\n[Stage B-ReFine] Training Residual Belief Adapter...")
        train_sl_bca_refine(
            train_file=args.train, valid_file=args.valid, out_path=args.out,
            belief_net=belief_net, sl_checkpoint=args.init_from,
            iterations=args.iterations, batch_size=args.batch_size,
            lr=args.lr, hidden_dim=args.hidden_dim, bottleneck=args.bottleneck,
            device=args.device, max_lines=args.max_lines,
            eval_every=args.eval_every)
    else:
        print(f"\n[Stage B Legacy] Training 667-dim Actor (full fine-tune)...")
        train_sl_bca(
            train_file=args.train, valid_file=args.valid, out_path=args.out,
            belief_net=belief_net, iterations=args.iterations,
            batch_size=args.batch_size, lr=args.lr, hidden_dim=args.hidden_dim,
            device=args.device, max_lines=args.max_lines,
            eval_every=args.eval_every, init_from=args.init_from,
            freeze_base=args.freeze_base)


if __name__ == '__main__':
    main()
