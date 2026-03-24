"""
SL Pretrain with Belief-Conditioned Actor (P98)
================================================

Two-stage SL pretraining for 349-dim belief-conditioned actors:

Stage A: Train Belief Net on SAYC data
  - Replay SAYC games, collect (observer_hand, history, observer_pos,
    target_pos, target_features) tuples
  - Train Belief Net to predict partner's hand features (48-dim)

Stage B: Train 349-dim Actor on SAYC data with Belief Net features
  - Replay SAYC games, at each step:
    1. Encode base obs (301-dim) via encode_obs_flat
    2. Query Belief Net for partner hand prediction (48-dim)
    3. Concatenate → 349-dim input
    4. Train Actor on cross-entropy loss

Output: sl_base_bca.pt containing:
  - actor_n/s/e/w: 349-dim MLPPolicyNetwork state_dicts
  - belief_net: BeliefNetwork state_dict
  - obs_dim: 349

Usage:
  python sl_pretrain.py \
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

def collect_belief_data_from_sayc(
    filepath: str,
    max_lines: int = None,
) -> List[dict]:
    """
    Replay SAYC games and collect belief training samples.

    For each bidding step by player P, we create a sample:
      observer = P, target = partner(P)
      observer_hand = hands[P]
      history = one-hot encoded history up to current point
      target_features = hand_to_belief_target(hands[partner])
    """
    env = BridgeBiddingEnv(max_history_len=60)
    samples = []

    print(f"[Belief Data] Collecting from {filepath}...")
    with open(filepath) as f:
        for line_idx, line in enumerate(f):
            if max_lines and line_idx >= max_lines:
                break

            nums = list(map(int, line.split()))
            if len(nums) < 53:
                continue

            deck = np.array(nums[:52], dtype=np.uint8)
            hands = np.zeros((4, 52), dtype=np.float32)
            for card, player in enumerate(deck):
                if player < 4:
                    hands[player, card] = 1.0

            dealer = line_idx % NUM_PLAYERS
            openspiel_actions = nums[52:]

            obs = env.reset(hands, dealer=dealer)
            history_int: List[int] = []

            for os_action in openspiel_actions:
                our_action = openspiel_to_our_bid(os_action)
                if our_action < 0:
                    break
                if obs['legal_actions'][our_action] < 0.5:
                    break

                player = env.current_player
                partner = (player + 2) % 4

                # Encode history as one-hot
                max_len = 60
                hist_enc = np.zeros((max_len, NUM_BIDS), dtype=np.float32)
                for i, bid in enumerate(history_int[-max_len:]):
                    hist_enc[i, bid] = 1.0

                samples.append({
                    'observer_hand': hands[player].copy(),
                    'history': hist_enc,
                    'observer_pos': player,
                    'target_pos': partner,
                    'target_features': hand_to_belief_target(hands[partner]),
                })

                history_int.append(our_action)
                obs, _, done, _ = env.step(our_action)
                if done:
                    break

    print(f"[Belief Data] {len(samples):,} samples collected.")
    return samples


def train_belief_net(
    samples: List[dict],
    device: str = 'cuda',
    epochs: int = 30,
    batch_size: int = 2048,
    lr: float = 1e-3,
    hidden_dim: int = 512,
) -> BeliefNetwork:
    """Train Belief Net on collected SAYC data."""
    print(f"\n[Stage A] Training Belief Net: {len(samples):,} samples, {epochs} epochs")

    oh  = torch.tensor(np.stack([s['observer_hand']    for s in samples]), dtype=torch.float32)
    h   = torch.tensor(np.stack([s['history']          for s in samples]), dtype=torch.float32)
    op  = torch.tensor(np.array([s['observer_pos']     for s in samples]), dtype=torch.long)
    tp  = torch.tensor(np.array([s['target_pos']       for s in samples]), dtype=torch.long)
    tgt = torch.tensor(np.stack([s['target_features']  for s in samples]), dtype=torch.float32)

    N = len(samples)
    # 90/10 train/val split
    perm = np.random.permutation(N)
    n_val = max(1000, N // 10)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

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
            loss = belief_net.compute_loss(
                oh[idx].to(device), h[idx].to(device),
                op[idx].to(device), tp[idx].to(device),
                tgt[idx].to(device),
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(belief_net.parameters(), 0.5)
            opt.step()
            train_loss += loss.item()
            n_batches += 1

        # Validation
        belief_net.eval()
        with torch.no_grad():
            val_loss = belief_net.compute_loss(
                oh[val_idx].to(device), h[val_idx].to(device),
                op[val_idx].to(device), tp[val_idx].to(device),
                tgt[val_idx].to(device),
            ).item()

            val_probs = belief_net.get_probs(
                oh[val_idx[:2000]].to(device), h[val_idx[:2000]].to(device),
                op[val_idx[:2000]].to(device), tp[val_idx[:2000]].to(device),
            )
            acc = belief_accuracy(val_probs, tgt[val_idx[:2000]].to(device))

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
# Stage B: SL Pretrain with Belief Features (349-dim)
# ==============================================================================

class SAYCBeliefDataset(Dataset):
    """
    SAYC dataset with belief features appended.

    At each step, queries the Belief Net for partner hand prediction
    and appends 48-dim features to the 301-dim base obs.
    """

    def __init__(self, filepath: str, belief_net: BeliefNetwork,
                 device: str = 'cuda', max_lines: int = None):
        self.samples: List[Tuple[np.ndarray, int]] = []  # (flat_obs_349, action)

        env = BridgeBiddingEnv(max_history_len=60)
        belief_net.eval()

        print(f"[SAYCBeliefDataset] Loading {filepath} with belief features...")

        with open(filepath) as f:
            for line_idx, line in enumerate(f):
                if max_lines and line_idx >= max_lines:
                    break

                nums = list(map(int, line.split()))
                if len(nums) < 53:
                    continue

                deck = np.array(nums[:52], dtype=np.uint8)
                hands = np.zeros((4, 52), dtype=np.float32)
                for card, player in enumerate(deck):
                    if player < 4:
                        hands[player, card] = 1.0

                dealer = line_idx % NUM_PLAYERS
                openspiel_actions = nums[52:]

                obs = env.reset(hands, dealer=dealer)
                history_int: List[int] = []

                for os_action in openspiel_actions:
                    our_action = openspiel_to_our_bid(os_action)
                    if our_action < 0:
                        break
                    if obs['legal_actions'][our_action] < 0.5:
                        break

                    player = env.current_player
                    partner = (player + 2) % 4

                    # Base obs (301)
                    flat_obs = encode_obs_flat(obs, dealer, history_int)

                    # Belief features (48) via Belief Net
                    max_len = 60
                    hist_enc = np.zeros((max_len, NUM_BIDS), dtype=np.float32)
                    for i, bid in enumerate(history_int[-max_len:]):
                        hist_enc[i, bid] = 1.0

                    with torch.no_grad():
                        oh_t = torch.tensor(hands[player], dtype=torch.float32
                                            ).unsqueeze(0).to(device)
                        h_t  = torch.tensor(hist_enc, dtype=torch.float32
                                            ).unsqueeze(0).to(device)
                        op_t = torch.tensor([player],  dtype=torch.long).to(device)
                        tp_t = torch.tensor([partner], dtype=torch.long).to(device)
                        probs = belief_net.get_probs(oh_t, h_t, op_t, tp_t)
                        bf = probs.squeeze(0).cpu().numpy()

                    # Combined (349)
                    combined = append_belief_features(flat_obs, bf)
                    self.samples.append((combined, our_action))

                    history_int.append(our_action)
                    obs, _, done, _ = env.step(our_action)
                    if done:
                        break

        print(f"[SAYCBeliefDataset] {len(self.samples):,} state-action pairs loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flat_obs, action = self.samples[idx]
        return (torch.tensor(flat_obs, dtype=torch.float32),
                torch.tensor(action,   dtype=torch.int64))


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
):
    """Stage B: Train 349-dim actors on SAYC data with belief features."""
    print(f"\n[Stage B] SL Pretrain (349-dim BCA)")
    print(f"  device={device}  epochs={epochs}  batch={batch_size}")

    class_weight = torch.ones(NUM_BIDS, device=device)
    class_weight[BID_PASS] = 0.1

    train_ds = SAYCBeliefDataset(train_file, belief_net, device, max_lines=max_lines)
    valid_ds = SAYCBeliefDataset(valid_file, belief_net, device, max_lines=50000)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=False)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    model = MLPPolicyNetwork(obs_dim=BELIEF_OBS_DIM, hidden_dim=hidden_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

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
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
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
        'actor_n':    best_state,
        'actor_s':    best_state,
        'actor_e':    best_state,
        'actor_w':    best_state,
        'belief_net': belief_state,
        'val_acc':    best_acc,
        'obs_dim':    BELIEF_OBS_DIM,
        'hidden_dim': hidden_dim,
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
                   help='Limit training lines (for quick test)')
    p.add_argument('--belief_max_lines', type=int, default=200000,
                   help='Lines for belief net training (200k default, ~1M samples)')
    args = p.parse_args()

    # ── Stage A: Train Belief Net ────────────────────────────────────
    belief_samples = collect_belief_data_from_sayc(
        args.train, max_lines=args.belief_max_lines)
    belief_net = train_belief_net(
        belief_samples,
        device=args.device,
        epochs=args.belief_epochs,
        batch_size=args.batch_size,
    )
    del belief_samples  # free memory

    # ── Stage B: Train 349-dim Actor ─────────────────────────────────
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
    )


if __name__ == '__main__':
    main()
