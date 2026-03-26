"""
OpenSpiel SAYC Data Loader + SL Pretraining
============================================

数据格式（每行一局）：
  前52个整数：发牌，deck[card] = 持有者 (0=N,1=E,2=S,3=W)
  第53个整数起：叫牌动作序列（OpenSpiel编码）

OpenSpiel → 我们的环境 动作映射：
  52        → BID_PASS     (0)
  53-87     → BID_1C+0 到 BID_1C+34  (3-37)
  88        → BID_DOUBLE   (1)
  89        → BID_REDOUBLE (2)

SL预训练目标：
  给定当前玩家的 flat_obs (480维, P104 OpenSpiel标准)，预测该步叫品（交叉熵损失）。
  覆盖所有四方，dealer轮换。

用法：
  python utils/sl_pretrain.py \
      --train data/sayc_train.txt \
      --valid data/sayc_valid.txt \
      --out   results/sl_base.pt \
      --epochs 30 \
      --batch_size 512 \
      --device cuda
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
from networks.policy_net import encode_obs_flat, MLPPolicyNetwork, OBS_DIM


# ==============================================================================
# 动作映射
# ==============================================================================

# OpenSpiel action → 我们的 BID index
def openspiel_to_our_bid(action: int) -> int:
    if action == 52:
        return BID_PASS        # 0
    elif action == 88:
        return BID_DOUBLE      # 1
    elif action == 89:
        return BID_REDOUBLE    # 2
    elif 53 <= action <= 87:
        return BID_1C + (action - 53)   # 3-37
    else:
        return -1  # 无效（发牌阶段的数字，不应出现）


# ==============================================================================
# Dataset
# ==============================================================================

class SAYCDataset(Dataset):
    """
    将 SAYC txt 文件解析为 (flat_obs, action) 样本对。

    每行一局，dealer 从行索引推算（0%4=N, 1%4=E, 2%4=S, 3%4=W），
    实现dealer轮换，四方样本均匀。

    内存策略：先全部解析存成 numpy array，RAM约需 2-3GB（300万样本）。
    如果RAM不够，可改为按行懒加载（设 lazy=True）。
    """

    def __init__(self, filepath: str, max_lines: int = None, lazy: bool = False):
        self.filepath   = filepath
        self.lazy       = lazy
        self.samples: List[Tuple[np.ndarray, int]] = []  # (flat_obs, action)

        print(f"[SAYCDataset] Loading {filepath}...")
        self._load(max_lines)
        print(f"[SAYCDataset] {len(self.samples):,} state-action pairs loaded.")

    def _load(self, max_lines):
        env = BridgeBiddingEnv(max_history_len=60)

        with open(self.filepath) as f:
            for line_idx, line in enumerate(f):
                if max_lines and line_idx >= max_lines:
                    break

                nums = list(map(int, line.split()))
                if len(nums) < 53:
                    continue

                # 发牌
                deck  = np.array(nums[:52], dtype=np.uint8)
                hands = np.zeros((4, 52), dtype=np.float32)
                # SAYC format: deck[position] = card_id (0-51)
                # positions 0-12 = player 0, 13-25 = player 1, etc.
                for player in range(4):
                    for card_id in deck[player * 13 : (player + 1) * 13]:
                        hands[player, card_id] = 1.0

                # dealer 轮换
                dealer = line_idx % NUM_PLAYERS

                # 叫牌动作序列（OpenSpiel编码，从第53个数字起）
                openspiel_actions = nums[52:]

                # 重放叫牌，收集每步的 (flat_obs, action)
                obs = env.reset(hands, dealer=dealer)
                history_int: List[int] = []

                for os_action in openspiel_actions:
                    our_action = openspiel_to_our_bid(os_action)
                    if our_action < 0:
                        break

                    # 检查合法性（跳过非法动作，数据集里偶有噪音）
                    if obs['legal_actions'][our_action] < 0.5:
                        break

                    # 编码当前状态
                    flat_obs = encode_obs_flat(obs, dealer, history_int)
                    self.samples.append((flat_obs, our_action))

                    history_int.append(our_action)
                    obs, _, done, _ = env.step(our_action)
                    if done:
                        break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flat_obs, action = self.samples[idx]
        return (torch.tensor(flat_obs, dtype=torch.float32),
                torch.tensor(action,   dtype=torch.int64))


# ==============================================================================
# SL 训练
# ==============================================================================

def train_sl(
    train_file:  str,
    valid_file:  str,
    out_path:    str,
    epochs:      int   = 10,
    batch_size:  int   = 2048,
    lr:          float = 3e-4,
    hidden_dim:  int   = 1024,
    device:      str   = 'cuda',
    max_lines:   int   = None,
    patience:    int   = 3,       # early stop: acc >= 0.90 连续 patience 轮
    target_acc:  float = 0.36,
):
    print(f"\n[SL Pretrain] device={device}  epochs={epochs}  batch={batch_size}")

    # ── Class weight: Pass(bid=0) 占57%，降权至0.1，其余均为1.0 ────────────
    class_weight = torch.ones(NUM_BIDS, device=device)
    class_weight[BID_PASS] = 0.1

    # ── 数据集 ──────────────────────────────────────────────────────────────
    train_ds = SAYCDataset(train_file, max_lines=max_lines)
    valid_ds = SAYCDataset(valid_file, max_lines=50000)  # valid 只取5万条够用

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)

    # ── 网络：共享一个 policy network，所有四方共用 ──────────────────────────
    # SL阶段：一个网络学习"通用桥牌叫牌"，不区分NS/EW
    # RL阶段：再分叉成四个独立网络做fine-tune
    model = MLPPolicyNetwork(obs_dim=OBS_DIM, hidden_dim=hidden_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc     = 0.0
    patience_cnt = 0
    best_state   = None

    for epoch in range(1, epochs + 1):
        # ── Train ───────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0; n_batches = 0

        for flat_obs, actions in train_loader:
            flat_obs = flat_obs.to(device)
            actions  = actions.to(device)
            # legal_actions 在 SL 阶段不限制（让模型自己学合法性）
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

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        correct_np = total_np = 0  # non-pass
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
                # 非 Pass 准确率（Pass = bid 0）
                mask     = (actions != BID_PASS)
                correct_np += (pred[mask] == actions[mask]).sum().item()
                total_np   += mask.sum().item()

        val_acc    = correct    / max(1, total)
        val_acc_np = correct_np / max(1, total_np)
        print(f"  Epoch {epoch:3d}/{epochs}  loss={train_loss:.4f}  "
              f"val_acc={val_acc:.4f}  non_pass_acc={val_acc_np:.4f}")

        # ── Early stop ──────────────────────────────────────────────────────
        if val_acc_np > best_acc:
            best_acc   = val_acc_np
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # 每次更新 best 立即保存，防止中断丢失
            os.makedirs(Path(out_path).parent, exist_ok=True)
            torch.save({
                'actor_n': best_state, 'actor_s': best_state,
                'actor_e': best_state, 'actor_w': best_state,
                'val_acc': best_acc, 'epoch': epoch,
                'obs_dim': OBS_DIM, 'hidden_dim': hidden_dim,
            }, out_path)

        if val_acc_np >= target_acc:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stop: non_pass_acc={val_acc_np:.4f} >= {target_acc} for {patience} epochs.")
                break
        else:
            patience_cnt = 0

    # ── 保存：把最佳权重复制到四个独立 actor ────────────────────────────────
    # 格式对齐 MAPPOAgent.save()，四方各自一份（初始相同，RL后分化）
    print(f"\n[SL Pretrain] Best non_pass_acc={best_acc:.4f}. Saving → {out_path}")
    os.makedirs(Path(out_path).parent, exist_ok=True)

    torch.save({
        'actor_n':  best_state,
        'actor_s':  best_state,
        'actor_e':  best_state,
        'actor_w':  best_state,
        'val_acc':  best_acc,
        'obs_dim':  OBS_DIM,
        'hidden_dim': hidden_dim,
    }, out_path)
    print(f"[SL Pretrain] Done.")
    return best_acc


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train',      default='data/sayc_train.txt')
    p.add_argument('--valid',      default='data/sayc_valid.txt')
    p.add_argument('--out',        default='results/sl_base.pt')
    p.add_argument('--epochs',     type=int,   default=30)
    p.add_argument('--batch_size', type=int,   default=512)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--hidden_dim', type=int,   default=1024)
    p.add_argument('--device',     default='cuda')
    p.add_argument('--max_lines',  type=int,   default=None,
                   help='Limit training lines (for quick test)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_sl(
        train_file  = args.train,
        valid_file  = args.valid,
        out_path    = args.out,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        hidden_dim  = args.hidden_dim,
        device      = args.device,
        max_lines   = args.max_lines,
    )
