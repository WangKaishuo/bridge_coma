"""
SL Pretrain with Belief-Conditioned Actor (P105)
==================================================

Two-stage SL pretraining for 667-dim belief-conditioned actors:

Stage A: Train Belief Net on SAYC data
  - Replay SAYC games via OpenSpiel, at each bidding step:
    1. Get 571-dim obs from state.observation_tensor()
    2. Determine current_player (observer) and target players
    3. Rebuild hands from dealing actions → hand_to_belief_target(target)
    4. Train: obs_571 + target_pos_embed → predict target's 48-dim features

Stage B: Train 667-dim Actor on SAYC data with Belief Net features
  - Same OpenSpiel replay, at each step:
    1. Get 571-dim obs from state.observation_tensor()
    2. Query Belief Net: obs_571 → partner features (48) + RHO features (48)
    3. Concatenate → 667-dim input
    4. Train Actor on cross-entropy loss

Output: sl_base_bca.pt

Usage:
  python sl_pretrain_bca.py \
      --train data/sayc_train.txt \
      --valid data/sayc_valid.txt \
      --out   results/sl_base_bca.pt \
      --init_from results/sl_base.pt \
      --belief_epochs 30 --iterations 400000 --device cuda
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
# Stage A: Train Belief Net
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
# Stage B: Train 667-dim Actor
# ==============================================================================

class SAYCGamesBCA:
    """Iteration-based dataset for Stage B. Queries belief net on-the-fly."""

    def __init__(self, filepath: str, max_lines: int = None):
        self.trajectories = load_trajectories(filepath, max_lines)

    def sample_batch(
        self, batch_size: int, belief_net: BeliefNetwork, device: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample (obs_667, action) batch."""
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
            our_action = openspiel_raw_to_ours(traj[action_index])
            if not (0 <= our_action < NUM_BIDS):
                our_action = BID_PASS

            player  = state.current_player()
            partner = (player + 2) % 4
            rho     = (player + 3) % 4

            obs_list.append(obs_571)
            act_list.append(our_action)
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
        belief   = np.concatenate([partner_feats, rho_feats], axis=1)   # (B, 96)
        obs_667  = np.concatenate([base_obs, belief], axis=1)           # (B, 667)

        return obs_667, np.array(act_list, dtype=np.int64)


class SAYCValidationBCA(Dataset):
    """Deterministic validation dataset for Stage B."""

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
                    our_action = openspiel_raw_to_ours(actions[step_idx])
                    if not (0 <= our_action < NUM_BIDS):
                        state.apply_action(actions[step_idx])
                        continue

                    player  = state.current_player()
                    partner = (player + 2) % 4
                    rho     = (player + 3) % 4

                    obs_list.append(obs_571)
                    act_list.append(our_action)
                    pa_list.append(partner)
                    rh_list.append(rho)

                    state.apply_action(actions[step_idx])

        N = len(act_list)
        print(f"[SAYCValidationBCA] {N:,} samples, running belief inference...")

        # Batch belief inference
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
):
    """Stage B: Train 667-dim actors with belief features."""
    print(f"\n[Stage B] SL Pretrain (667-dim BCA, iteration-based)")
    print(f"  device={device}  iterations={iterations}  batch={batch_size}  lr={lr}")
    if init_from:
        print(f"  init_from={init_from} (571→667 zero-init)")

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

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_np_acc = 0.0
    best_state  = None
    running_loss = 0.0
    t_start = time.time()

    for it in range(1, iterations + 1):
        obs_np, act_np = train_games.sample_batch(batch_size, belief_net, device)
        flat_obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
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
                }, out_path)

    print(f"\n[Stage B] Best non_pass_acc={best_np_acc:.4f}. Saved → {out_path}")
    return best_np_acc


# ==============================================================================
# Main
# ==============================================================================

def main():
    p = argparse.ArgumentParser(description='SL Pretrain BCA (P105)')
    p.add_argument('--train',      default='data/sayc_train.txt')
    p.add_argument('--valid',      default='data/sayc_valid.txt')
    p.add_argument('--out',        default='results/sl_base_bca.pt')
    p.add_argument('--iterations', type=int, default=400000)
    p.add_argument('--belief_epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--hidden_dim', type=int, default=1024)
    p.add_argument('--device',     default='cuda')
    p.add_argument('--eval_every', type=int, default=10000)
    p.add_argument('--max_lines',  type=int, default=None)
    p.add_argument('--belief_max_lines', type=int, default=200000)
    p.add_argument('--init_from',   type=str, default=None,
                   help='Path to sl_base.pt (571-dim) for Stage B init')
    p.add_argument('--load_belief', type=str, default=None,
                   help='Path to existing BeliefNet checkpoint (e.g. sl_base_bca.pt). '
                        'If set, skip Stage A and load BeliefNet directly. '
                        'Use with --init_from sl_base.pt to run Stage B only.')
    args = p.parse_args()

    # Stage A — skip if --load_belief provided
    if args.load_belief:
        import torch as _torch
        print(f"\n[Stage A] Skipped — loading BeliefNet from: {args.load_belief}")
        _bn_ckpt   = _torch.load(args.load_belief, map_location=args.device,
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
    print(f"\n[Stage B] Training 667-dim Actor...")
    train_sl_bca(
        train_file=args.train, valid_file=args.valid, out_path=args.out,
        belief_net=belief_net, iterations=args.iterations,
        batch_size=args.batch_size, lr=args.lr, hidden_dim=args.hidden_dim,
        device=args.device, max_lines=args.max_lines,
        eval_every=args.eval_every, init_from=args.init_from)


if __name__ == '__main__':
    main()
