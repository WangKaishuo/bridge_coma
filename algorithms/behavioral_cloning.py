"""
Behavioral Cloning
==================

BC 预热 — 用于 Stayman 和 Competitive 子博弈。

- BCDataset / behavioral_cloning_warmup: 通用 BC 训练组件 (两个子博弈共用)
- create_bc_dataset_for_competitive: 生成 Competitive 子博弈 BC 数据
  (Stayman BC 数据由 stayman_env.create_bc_dataset_for_stayman 生成)
- 开叫规则 + 竞叫应叫规则 (启发式, 偏简单保守)
- evaluate_pass_rate: 用于诊断 agent 是否退化为全 Pass
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader

from env import (
    BridgeBiddingEnv,
    BID_PASS, BID_DOUBLE,
    bid_to_string, string_to_bid,
    NORTH, EAST, SOUTH, WEST,
)
from subgames.action_mask import count_hcp, count_suit_length, is_balanced, suit_lengths


# ============================================================================
# Opening bid rules
# ============================================================================

def select_simple_opening(hand: np.ndarray) -> int:
    """
    简单开叫选择 → bid index.

    规则 (偏保守, 不要太完美):
      22+ HCP           → 2C
      15-17 HCP 均型    → 1NT
      12+ 5+ S          → 1S
      12+ 5+ H          → 1H
      12+ longest minor → 1D / 1C
      < 12              → Pass
    """
    hcp = count_hcp(hand)

    if hcp < 12:
        return BID_PASS

    if hcp >= 22:
        return string_to_bid("2C")

    if is_balanced(hand) and 15 <= hcp <= 17:
        return string_to_bid("1NT")

    c, d, h, s = suit_lengths(hand)

    if s >= 5:
        return string_to_bid("1S")
    if h >= 5:
        return string_to_bid("1H")
    if d >= c:
        return string_to_bid("1D")
    return string_to_bid("1C")


# ============================================================================
# Competitive response rules (1H - 1S - ?)
# ============================================================================

def competitive_response_after_1h_1s(hand: np.ndarray) -> int:
    """
    南家应叫规则: 进程 1H - 1S - ?

    规则:
      X  (负加倍)     : 11+ HCP, 0-2 张 H
      1NT             : 8-10 HCP, 3 张 H
      2C              : 8-11 HCP, 5+ C
      2D              : 8-11 HCP, 5+ D
      2H              : 5-7 HCP, 3 张 H
      2S (强争叫)     : 11+ HCP, 3 张 H
      2NT             : 11+ HCP, 4+ 张 H
      3H (阻击加叫)   : 5-10 HCP, 4 张 H  ← 弱牌 4H
      3C              : 12+ HCP, 6+ C
      3D              : 12+ HCP, 6+ D
      3NT             : 12-15 HCP, 均型, S有止张 (2+S)
      Pass            : default (太弱或不符合以上)
    """
    hcp = count_hcp(hand)
    c, d, h, s = suit_lengths(hand)

    # 11+ HCP, 4+ H → 2NT
    if hcp >= 11 and h >= 4:
        return string_to_bid("2NT")

    # 12+ HCP, 6+ C → 3C
    if hcp >= 12 and c >= 6:
        return string_to_bid("3C")

    # 12+ HCP, 6+ D → 3D
    if hcp >= 12 and d >= 6:
        return string_to_bid("3D")

    # 12-15 HCP, 均型, S有止张 → 3NT
    if 12 <= hcp <= 15 and is_balanced(hand) and s >= 2:
        return string_to_bid("3NT")

    # 11+ HCP, 3 H → 2S (强争叫)
    if hcp >= 11 and h == 3:
        return string_to_bid("2S")

    # 11+ HCP, 0-2 H → X (负加倍)
    if hcp >= 11 and h <= 2:
        return BID_DOUBLE

    # 8-10 HCP, 3 H → 1NT
    if 8 <= hcp <= 10 and h == 3:
        return string_to_bid("1NT")

    # 8-11 HCP, 5+ C → 2C
    if 8 <= hcp <= 11 and c >= 5:
        return string_to_bid("2C")

    # 8-11 HCP, 5+ D → 2D
    if 8 <= hcp <= 11 and d >= 5:
        return string_to_bid("2D")

    # 5-10 HCP, 4 H → 3H (阻击加叫, 弱牌 4H)
    if 5 <= hcp <= 10 and h >= 4:
        return string_to_bid("3H")

    # 5-7 HCP, 3 H → 2H
    if 5 <= hcp <= 7 and h == 3:
        return string_to_bid("2H")

    return BID_PASS


def responder_rebid_after_1h_1s(hand: np.ndarray, history: list) -> int:
    """
    北家续叫规则: 1H - 1S - (应叫) - ?

    仅处理应叫人叫品低于 2S 的情况:
      X  (负加倍)     : 11+ HCP, 0-2 S
      2S              : 6-10 HCP, 3 S
      2NT             : 11+ HCP, 4+ S
      3S              : 6-10 HCP, 4+ S
    其他              : Pass (简化)
    """
    hcp = count_hcp(hand)
    _, _, h, s = suit_lengths(hand)

    # 仅当存在需要支持 S 的情况才续叫
    # 简化: 如果有 4+S 或 3S 考虑加叫
    if hcp >= 11 and s >= 4:
        return string_to_bid("2NT")
    if hcp >= 11 and s <= 2:
        return BID_DOUBLE
    if 6 <= hcp <= 10 and s >= 4:
        return string_to_bid("3S")
    if 6 <= hcp <= 10 and s == 3:
        return string_to_bid("2S")

    return BID_PASS


# ============================================================================
# BC Dataset
# ============================================================================

class BCDataset(Dataset):
    """
    BC 训练数据集.

    每条数据: (obs_dict, target_action_idx)
    obs_dict 与 BridgeBiddingEnv._get_observation 格式一致。
    """

    def __init__(self, data: List[Dict]):
        """data: list of {'obs': {hand, history, legal_actions, position, vulnerability}, 'action': int}"""
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        obs = {k: torch.tensor(v, dtype=torch.float32) for k, v in d['obs'].items()}
        action = torch.tensor(d['action'], dtype=torch.long)
        return obs, action


def create_bc_dataset_for_competitive(
    data_path: str,
    num_samples: int = 50000,
    max_history_len: int = 60,
) -> BCDataset:
    """
    为 Competitive 子博弈 (1H-1S) 生成 BC 数据.

    流程:
    1. 从 DDS loader 采样牌
    2. 筛选符合 1H-1S 约束的 (N: 5+H 12+ HCP, E: 5+S 8+ HCP)
    3. 对开叫后的每一步, 用启发式规则标注目标叫品
    4. 构造 obs → target_action 对
    """
    from utils.dds_data import create_loader

    loader = create_loader(data_path)
    env = BridgeBiddingEnv(max_history_len)
    data = []
    attempts = 0
    max_attempts = num_samples * 20  # 筛选率 ~5-10%

    while len(data) < num_samples and attempts < max_attempts:
        attempts += 1
        hands, dd_table = loader.sample_one()

        # 约束: N(0) 5+H 12-21 HCP, E(1) 5+S 8-16 HCP
        n_hand, e_hand = hands[0], hands[1]
        n_hcp = count_hcp(n_hand)
        e_hcp = count_hcp(e_hand)
        n_h = count_suit_length(n_hand, 2)  # hearts
        e_s = count_suit_length(e_hand, 3)  # spades

        if not (12 <= n_hcp <= 21 and n_h >= 5):
            continue
        if not (8 <= e_hcp <= 16 and e_s >= 5):
            continue

        # 模拟 1H - 1S 前缀 + 启发式续叫
        dealer = NORTH  # N 开叫
        obs = env.reset(hands, dealer=dealer, vulnerability=(False, False))

        # Step 1: N opens 1H
        bid_1h = string_to_bid("1H")
        if env._is_valid_action(bid_1h):
            data.append({'obs': _copy_obs(obs), 'action': bid_1h})
            obs, _, done, _ = env.step(bid_1h)
        else:
            continue

        if done:
            continue

        # Step 2: E overcalls 1S
        bid_1s = string_to_bid("1S")
        if env._is_valid_action(bid_1s):
            data.append({'obs': _copy_obs(obs), 'action': bid_1s})
            obs, _, done, _ = env.step(bid_1s)
        else:
            continue

        if done:
            continue

        # Step 3: S responds (competitive_response)
        s_hand = hands[2]  # South
        s_bid = competitive_response_after_1h_1s(s_hand)
        if env._is_valid_action(s_bid):
            data.append({'obs': _copy_obs(obs), 'action': s_bid})
            obs, _, done, _ = env.step(s_bid)
        else:
            # fallback to Pass
            data.append({'obs': _copy_obs(obs), 'action': BID_PASS})
            obs, _, done, _ = env.step(BID_PASS)

        if done:
            continue

        # Step 4: W responds (simple: pass for now)
        data.append({'obs': _copy_obs(obs), 'action': BID_PASS})
        obs, _, done, _ = env.step(BID_PASS)

        if done:
            continue

        # Step 5: N rebids (opener rebid)
        n_rebid = responder_rebid_after_1h_1s(n_hand, env.state.history)
        if env._is_valid_action(n_rebid):
            data.append({'obs': _copy_obs(obs), 'action': n_rebid})
        else:
            data.append({'obs': _copy_obs(obs), 'action': BID_PASS})

    print(f"BC dataset: {len(data)} samples from {attempts} attempts "
          f"(acceptance ~{len(data)/max(1,attempts):.1%})")

    return BCDataset(data[:num_samples])


def _copy_obs(obs: dict) -> dict:
    """Deep copy numpy obs."""
    return {k: v.copy() for k, v in obs.items()}


# ============================================================================
# BC Training
# ============================================================================

def behavioral_cloning_warmup(
    agent,
    dataset: BCDataset,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 256,
    minority_weight: float = 2.0,
) -> dict:
    """
    对 agent 的 policy network 做交叉熵监督训练.

    Args:
        agent: MAPPOAgent or IPPOAgent (需要有 model.actor)
        dataset: BCDataset
        epochs: 训练轮数
        lr: 学习率
        batch_size: batch 大小
        minority_weight: 少数类 (非 4H/4S 动作) 的 loss 权重.
            默认 2.0: 让 3NT/2NT 等叫品的梯度贡献翻倍,
            迫使网络不能靠猜多数派 (4M) 过关.

    Returns:
        训练统计: {'final_loss', 'final_acc'}
    """
    from env import string_to_bid
    # 高花成局叫品 — 多数类, 权重 1.0
    majority_actions = {string_to_bid("4H"), string_to_bid("4S")}

    model = agent.model.actor
    device = agent.device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=_collate_bc)

    model.train()
    stats = {'losses': [], 'accs': []}

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for obs_batch, action_batch in loader:
            # Move to device
            obs_d = {k: v.to(device) for k, v in obs_batch.items()}
            targets = action_batch.to(device)

            # Forward
            logits = model(obs_d)  # (B, 38)

            # Mask illegal actions, then CE loss with per-sample weights
            mask = obs_d['legal_actions']
            logits = logits - 1e9 * (1 - mask)

            # ========================================================
            # Per-sample 权重 (双重加权):
            #
            # 1) 动作加权: 非 4H/4S 少数类 → minority_weight
            #    迫使网络不能靠猜多数派过关.
            #
            # 2) 玩家加权: S 的样本 → ×3
            #    N 和 S 共享 HistoryEncoder, 但 N 的任务不需要 history
            #    (N 只看手牌叫 2D/2H/2S). N 的 5000 个样本的梯度
            #    告诉 HistoryEncoder "history 不重要", 直接和 S 的梯度
            #    打架. S 的加权确保 HistoryEncoder 被 S 的需求主导.
            # ========================================================
            weights = torch.ones(targets.shape[0], device=device)

            # 动作加权
            is_minority = torch.ones(targets.shape[0], dtype=torch.bool, device=device)
            for maj_act in majority_actions:
                is_minority = is_minority & (targets != maj_act)
            weights[is_minority] = minority_weight

            # 玩家加权: position one-hot 的 argmax == SOUTH(2) 的样本
            positions = obs_d['position']  # (B, 4)
            is_south = positions.argmax(dim=-1) == 2  # SOUTH = 2
            weights[is_south] = weights[is_south] * 3.0

            loss = F.cross_entropy(logits, targets, reduction='none')
            loss = (loss * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Accuracy
            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == targets).sum().item()
            epoch_total += targets.size(0)
            epoch_loss += loss.item() * targets.size(0)

        avg_loss = epoch_loss / max(1, epoch_total)
        avg_acc = epoch_correct / max(1, epoch_total)
        stats['losses'].append(avg_loss)
        stats['accs'].append(avg_acc)

        print(f"  BC Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, acc={avg_acc:.3f}")

    model.train()  # Restore train mode (critical: LSTM backward requires train mode)
    return {
        'final_loss': stats['losses'][-1] if stats['losses'] else 0,
        'final_acc': stats['accs'][-1] if stats['accs'] else 0,
        'epochs': epochs,
    }


def _collate_bc(batch):
    """Custom collate for BC dataset."""
    obs_list, action_list = zip(*batch)
    keys = obs_list[0].keys()
    obs_batch = {k: torch.stack([o[k] for o in obs_list]) for k in keys}
    action_batch = torch.stack(action_list)
    return obs_batch, action_batch


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_pass_rate(agent, env: BridgeBiddingEnv, num_deals: int = 200) -> float:
    """
    评估 agent 在随机牌中的 pass rate.

    如果开叫位全部 Pass, pass_rate → 1.0.
    BC 成功后应该 < 0.5.
    """
    total_bids = 0
    pass_bids = 0

    for _ in range(num_deals):
        obs = env.reset(dealer=np.random.randint(4))
        done = False
        while not done:
            with torch.no_grad():
                obs_t = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(agent.device)
                         for k, v in obs.items()}
                action, _, _ = agent.model.actor.get_action(obs_t)
                action = action.item()

            total_bids += 1
            if action == BID_PASS:
                pass_bids += 1

            obs, _, done, _ = env.step(action)

    return pass_bids / max(1, total_bids)
