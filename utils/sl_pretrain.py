"""
SL Pretraining with Native OpenSpiel Observations (P105)
==========================================================

P105: Uses OpenSpiel's state.observation_tensor() directly instead of
hand-crafted encode_obs_flat(). This guarantees observation encoding
matches the reference implementation exactly.

Key differences from P104c:
  1. OBS_DIM = 571 (OpenSpiel native), not 480 (hand-crafted)
  2. observation_tensor() from pyspiel.State, not encode_obs_flat()
  3. Dealer determined by game state replay, not line_idx % 4
  4. Action mapping: target = openspiel_action - 52 (verified match)

The 571-dim observation is a superset of the 480-dim encoding described
in Lockhart+20/Kita+24. The extra 91 dims encode additional game state
that OpenSpiel tracks internally. Using the full 571-dim tensor ensures
zero information loss and zero encoding bugs.

Data format (each line = one game trajectory):
  First 52 integers: dealing actions (chance events)
  Remaining integers: bidding actions + playing actions (OpenSpiel encoding)

Usage:
  # Full training (OpenSpiel/Kita parameters, ~1-2 hours on T4)
  python sl_pretrain.py --iterations 400000 --batch_size 128 --device cuda

  # Quick test (5 minutes)
  python sl_pretrain.py --iterations 10000 --batch_size 128 --max_lines 10000 --device cuda
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyspiel

# ── OpenSpiel Constants ────────────────────────────────────────────────────────
GAME = pyspiel.load_game('bridge(use_double_dummy_result=false)')
OBS_DIM = GAME.observation_tensor_shape()[0]   # 571
NUM_ACTIONS = 38    # Pass + Dbl + RDbl + 35 bids
MIN_ACTION = 52     # OpenSpiel action offset for bidding
NUM_CARDS = 52
NUM_PLAYERS = 4


# ==============================================================================
# Trajectory Parser
# ==============================================================================

def _no_play_trajectory(line: str) -> Tuple[int, ...]:
    """
    Strip playing phase from trajectory, keeping only deal + bidding.
    Matches OpenSpiel's bridge_supervised_learning.py exactly.
    """
    actions = tuple(int(x) for x in line.split())
    # If all 4 players pass, there's no play phase
    if len(actions) == NUM_CARDS + NUM_PLAYERS:
        return actions
    else:
        # Remove last NUM_CARDS actions (playing phase)
        return actions[:-NUM_CARDS]


# ==============================================================================
# Dataset: OpenSpiel-native (random step per game)
# ==============================================================================

class SAYCGamesOpenSpiel:
    """
    Pre-parse SAYC trajectories. At training time, each game yields ONE
    random (obs, action) sample — matching OpenSpiel's make_dataset().

    Unlike P104c, this class replays through OpenSpiel's game state to
    generate observation tensors, guaranteeing correct dealer, relative
    positions, and observation encoding.
    """

    def __init__(self, filepath: str, max_lines: int = None):
        self.trajectories: List[Tuple[int, ...]] = []
        print(f"[SAYCGamesOpenSpiel] Loading {filepath}...")
        t0 = time.time()
        with open(filepath) as f:
            for line_idx, line in enumerate(f):
                if max_lines and line_idx >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                traj = _no_play_trajectory(line)
                # Must have at least 52 deal actions + 1 bidding action
                if len(traj) > NUM_CARDS:
                    self.trajectories.append(traj)
        elapsed = time.time() - t0
        print(f"[SAYCGamesOpenSpiel] {len(self.trajectories):,} games loaded "
              f"in {elapsed:.1f}s.")

    def sample_one(self) -> Tuple[np.ndarray, int]:
        """
        Sample one (obs, target_action) pair from a random game
        at a random bidding step.

        This is the exact algorithm from OpenSpiel's make_dataset():
          1. Pick random trajectory
          2. Pick random action_index in [52, len(trajectory))
          3. Replay state up to action_index
          4. yield (observation_tensor, action - MIN_ACTION)
        """
        traj = random.choice(self.trajectories)
        # Random bidding step (index 52 = first bidding action)
        action_index = random.randint(NUM_CARDS, len(traj) - 1)

        state = GAME.new_initial_state()
        for action in traj[:action_index]:
            state.apply_action(action)

        obs = np.array(state.observation_tensor(), dtype=np.float32)
        target = traj[action_index] - MIN_ACTION  # 0-37

        assert 0 <= target < NUM_ACTIONS, \
            f"Invalid target {target} from action {traj[action_index]}"
        return obs, target

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a batch of (obs, target) pairs."""
        obs_batch = np.zeros((batch_size, OBS_DIM), dtype=np.float32)
        act_batch = np.zeros(batch_size, dtype=np.int64)
        for i in range(batch_size):
            obs_batch[i], act_batch[i] = self.sample_one()
        return obs_batch, act_batch


# ==============================================================================
# Validation Dataset (deterministic, all steps)
# ==============================================================================

class SAYCValidationOpenSpiel:
    """
    Expand all bidding steps for deterministic validation.
    Uses OpenSpiel state replay for correct observations.
    """

    def __init__(self, filepath: str, max_lines: int = None):
        self.samples: List[Tuple[np.ndarray, int]] = []
        print(f"[SAYCValidationOpenSpiel] Loading {filepath}...")
        t0 = time.time()

        with open(filepath) as f:
            for line_idx, line in enumerate(f):
                if max_lines and line_idx >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue

                traj = _no_play_trajectory(line)
                if len(traj) <= NUM_CARDS:
                    continue

                state = GAME.new_initial_state()
                # Apply dealing
                for action in traj[:NUM_CARDS]:
                    state.apply_action(action)

                # Extract (obs, action) for each bidding step
                for step_idx in range(NUM_CARDS, len(traj)):
                    obs = np.array(state.observation_tensor(), dtype=np.float32)
                    target = traj[step_idx] - MIN_ACTION
                    if 0 <= target < NUM_ACTIONS:
                        self.samples.append((obs, target))
                    state.apply_action(traj[step_idx])

        elapsed = time.time() - t0
        print(f"[SAYCValidationOpenSpiel] {len(self.samples):,} samples "
              f"in {elapsed:.1f}s.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs, target = self.samples[idx]
        return (torch.tensor(obs, dtype=torch.float32),
                torch.tensor(target, dtype=torch.int64))


# ==============================================================================
# Network (same 4x1024 MLP, different input dim)
# ==============================================================================

class BridgePolicyMLP(nn.Module):
    """
    4x1024 MLP policy network.
    Input: 571-dim OpenSpiel observation tensor.
    Output: 38-dim logits (Pass/Dbl/RDbl/1C-7NT).
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = 1024,
                 num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x):
        return self.net(x)


# ==============================================================================
# SL Training Loop
# ==============================================================================

def train_sl(
    train_file:  str,
    valid_file:  str,
    out_path:    str,
    iterations:  int   = 400000,
    batch_size:  int   = 128,
    lr:          float = 1e-4,
    hidden_dim:  int   = 1024,
    device:      str   = 'cuda',
    max_lines:   int   = None,
    eval_every:  int   = 10000,
):
    print(f"\n{'='*60}")
    print(f"  SL Pretrain P105 (OpenSpiel native)")
    print(f"  OBS_DIM={OBS_DIM}  device={device}  iterations={iterations}")
    print(f"  batch={batch_size}  lr={lr}  hidden={hidden_dim}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────
    train_games = SAYCGamesOpenSpiel(train_file, max_lines=max_lines)
    valid_ds = SAYCValidationOpenSpiel(valid_file, max_lines=50000)

    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=2048, shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────
    model = BridgePolicyMLP(obs_dim=OBS_DIM, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {total_params:,} parameters")

    best_np_acc = 0.0
    best_state = None
    running_loss = 0.0
    t_start = time.time()

    for it in range(1, iterations + 1):
        # ── Sample batch (OpenSpiel replay) ──────────────────────────────
        obs_np, act_np = train_games.sample_batch(batch_size)
        flat_obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
        actions  = torch.tensor(act_np, dtype=torch.int64, device=device)

        # ── Forward + backward ───────────────────────────────────────────
        model.train()
        logits = model(flat_obs)
        loss = F.cross_entropy(logits, actions)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()

        running_loss += loss.item()

        # ── Evaluate ─────────────────────────────────────────────────────
        if it % eval_every == 0 or it == iterations:
            model.eval()
            correct = total = 0
            correct_np = total_np = 0
            BID_PASS = 0  # target label for Pass
            with torch.no_grad():
                for flat_v, targets_v in valid_loader:
                    flat_v = flat_v.to(device)
                    targets_v = targets_v.to(device)
                    logits_v = model(flat_v)
                    pred = logits_v.argmax(dim=-1)
                    correct += (pred == targets_v).sum().item()
                    total += targets_v.size(0)
                    # Non-pass accuracy
                    mask = (targets_v != BID_PASS)
                    correct_np += (pred[mask] == targets_v[mask]).sum().item()
                    total_np += mask.sum().item()

            val_acc = correct / max(1, total)
            val_np_acc = correct_np / max(1, total_np)
            avg_loss = running_loss / eval_every
            running_loss = 0.0
            elapsed = time.time() - t_start

            print(f"  Iter {it:7d}/{iterations}  loss={avg_loss:.4f}  "
                  f"val_acc={val_acc:.4f}  non_pass_acc={val_np_acc:.4f}  "
                  f"[{elapsed/60:.1f}min]")

            if val_np_acc > best_np_acc:
                best_np_acc = val_np_acc
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                os.makedirs(Path(out_path).parent, exist_ok=True)
                torch.save({
                    'model_state': best_state,
                    'obs_dim': OBS_DIM,
                    'hidden_dim': hidden_dim,
                    'num_actions': NUM_ACTIONS,
                    'val_acc': val_acc,
                    'non_pass_acc': best_np_acc,
                    'iteration': it,
                    'encoding': 'openspiel_571',
                }, out_path)

    # ── Final save ───────────────────────────────────────────────────────
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    torch.save({
        'model_state': best_state,
        'obs_dim': OBS_DIM,
        'hidden_dim': hidden_dim,
        'num_actions': NUM_ACTIONS,
        'val_acc': val_acc,
        'non_pass_acc': best_np_acc,
        'iteration': iterations,
        'encoding': 'openspiel_571',
    }, out_path)

    print(f"\n[SL Pretrain] Best non_pass_acc={best_np_acc:.4f}. Saved -> {out_path}")
    return best_np_acc


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="SL Pretrain with OpenSpiel native observations (P105)")
    p.add_argument('--train', default='data/sayc_train.txt')
    p.add_argument('--valid', default='data/sayc_valid.txt')
    p.add_argument('--out',   default='results/sl_base_571.pt')
    p.add_argument('--iterations', type=int, default=400000,
                   help='Training iterations (OpenSpiel/Kita: 400k)')
    p.add_argument('--batch_size', type=int, default=128,
                   help='Batch size (OpenSpiel/Kita: 128)')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--hidden_dim', type=int, default=1024)
    p.add_argument('--eval_every', type=int, default=10000)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max_lines', type=int, default=None,
                   help='Limit training games (for quick test)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_sl(
        train_file  = args.train,
        valid_file  = args.valid,
        out_path    = args.out,
        iterations  = args.iterations,
        batch_size  = args.batch_size,
        lr          = args.lr,
        hidden_dim  = args.hidden_dim,
        device      = args.device,
        max_lines   = args.max_lines,
        eval_every  = args.eval_every,
    )
