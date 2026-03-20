"""
Belief Network
==============

推断他人手牌的网络，用于计算 Dual-Info Bonus。

v2: 预测目标从 52 维 one-hot 改为 48 维二值语义特征
    详见 hand_features.py

    [0 :16]  荣誉牌归属 (AKQJ × 4门)  — 16维独立 binary
    [16:48]  套长 one-hot (0~7+ × 4门) — 32维，每门花色互斥 8档

    优势:
    - AKQJ 覆盖全部大牌点来源，T以下对叫牌决策贡献极小
    - one-hot 套长避免阶梯式编码的单调性冗余：
      "5张黑桃"只激活1个bit，r_info 不会重复计算
    - pos_weight 统一=3.0，简洁无额外超参数
    - r_info / BCEWithLogitsLoss 公式完全不变
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from env import NUM_BIDS, NUM_PLAYERS
from utils.hand_features import (
    BELIEF_DIM, HONOR_DIM, build_pos_weight, belief_accuracy
)


class BeliefNetwork(nn.Module):
    """
    Belief Network (P52): 预测目标玩家的 48 维二值语义特征。

    P52 变更: 删除 LSTM history encoder, 改用 MLP + who-made-it 展平编码.
    与 PolicyNetwork 保持一致的 history 表示.

    输入:
        observer_hand (52) + history_flat (NUM_BIDS×NUM_PLAYERS=152)
        + observer_pos_embed (32) + target_pos_embed (32)
        = 268 dims
    输出: (batch, 48) logits，未经 Sigmoid
    """

    def __init__(self, hand_dim: int = 256, history_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        # hand_dim / history_dim 参数保留向后兼容，实际不再使用
        from networks.policy_net import encode_history_flat, NUM_BIDS, NUM_PLAYERS
        self._encode_history_flat = encode_history_flat

        self.position_embed = nn.Embedding(4, 32)

        # 输入: hand(52) + history_flat(NUM_BIDS*NUM_PLAYERS) + pos×2(64)
        input_dim = 52 + NUM_BIDS * NUM_PLAYERS + 32 + 32
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, BELIEF_DIM),
            # 不加 Sigmoid; forward 返回 logits
        )

        self.register_buffer('pos_weight', build_pos_weight())

    def forward(
        self,
        observer_hand: torch.Tensor,
        history: torch.Tensor,
        observer_pos: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            history: (batch, max_len, NUM_BIDS) one-hot 序列
        Returns:
            logits: (batch, 48)
        """
        hist_flat = self._encode_history_flat(history)   # (B, NUM_BIDS*NUM_PLAYERS)
        obs_embed = self.position_embed(observer_pos)
        tgt_embed = self.position_embed(target_pos)
        x = torch.cat([observer_hand, hist_flat, obs_embed, tgt_embed], dim=-1)
        return self.fc(x)

    def get_probs(
        self,
        observer_hand: torch.Tensor,
        history: torch.Tensor,
        observer_pos: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        """返回 sigmoid 概率 (batch, 48)，供 r_info 计算使用。"""
        return torch.sigmoid(self.forward(observer_hand, history, observer_pos, target_pos))

    def compute_loss(
        self,
        observer_hand: torch.Tensor,
        history: torch.Tensor,
        observer_pos: torch.Tensor,
        target_pos: torch.Tensor,
        target_features: torch.Tensor,   # (batch, 48)
    ) -> torch.Tensor:
        """
        BCEWithLogitsLoss with uniform pos_weight=3.0.

        Args:
            target_features: (batch, 48) 由 hand_to_belief_target() 生成
        """
        logits = self.forward(observer_hand, history, observer_pos, target_pos)
        return F.binary_cross_entropy_with_logits(
            logits, target_features, pos_weight=self.pos_weight
        )

    @staticmethod
    def evaluate_accuracy(probs: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        评估指标，替代旧版 top13_hit_rate。

        Returns:
            honor_acc  / length_acc / overall_acc
        """
        return belief_accuracy(probs, targets)


class DualInfoComputer:
    """
    计算 Dual-Info Bonus

    r_info = I(bid; hand | partner) - β * I(bid; hand | opponent)

    近似:
    I(bid; hand) ≈ CE(belief_before, target) - CE(belief_after, target)

    注: target 现为 48 维特征，CE 计算方式与原 52 维完全相同。
    reduction='mean' 归一化维度数量，避免量纲随特征维度变化漂移。
    """

    def __init__(self, belief_net: BeliefNetwork, beta: float = 0.5):
        self.belief_net = belief_net
        self.beta = beta

    def compute_info_gain(
        self,
        belief_before: torch.Tensor,    # (batch, 48) sigmoid probs
        belief_after: torch.Tensor,     # (batch, 48) sigmoid probs
        target_features: torch.Tensor,  # (batch, 48) 0/1
    ) -> torch.Tensor:
        """
        信息增益 = max(0, CE(before, target) - CE(after, target))

        ReLU 截断保证非负：互信息 I(X;Y) ≥ 0。
        Belief Net 未收敛时 CE 差值可能为负（随机噪声），
        不截断会给 actor 加随机方向的噪声惩罚，破坏训练。
        """
        ce_before = F.binary_cross_entropy(
            belief_before, target_features, reduction='none'
        ).mean(dim=-1)
        ce_after = F.binary_cross_entropy(
            belief_after, target_features, reduction='none'
        ).mean(dim=-1)
        return torch.relu(ce_before - ce_after)

    def compute_dual_info_bonus(
        self,
        partner_gain: torch.Tensor,
        opponent_leak: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Dual-Info Bonus = partner_gain - β * opponent_leak

        Returns:
            bonus:   (batch,) tensor
            metrics: 详细指标 dict
        """
        bonus = partner_gain - self.beta * opponent_leak
        metrics = {
            'partner_gain':    partner_gain.mean().item(),
            'opponent_leak':   opponent_leak.mean().item(),
            'info_ratio':      (partner_gain.mean() / (opponent_leak.mean() + 1e-8)).item(),
            'dual_info_bonus': bonus.mean().item(),
        }
        return bonus, metrics
