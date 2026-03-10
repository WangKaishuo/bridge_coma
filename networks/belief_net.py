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

from env import NUM_BIDS
from utils.hand_features import (
    BELIEF_DIM, HONOR_DIM, build_pos_weight, belief_accuracy
)


class BeliefNetwork(nn.Module):
    """
    Belief Network: 预测目标玩家的 48 维二值语义特征。

    输入: 观察者手牌 (52) + 叫牌历史 + 观察者位置 + 目标位置
    输出: (batch, 48) logits，未经 Sigmoid

    pos_weight:
        统一 3.0，(48,) tensor
        荣誉牌理论值恰好 = 3.0 (P=0.25)
        套长 one-hot 互斥特性使不平衡影响较小，3.0 作为统一值

    评估指标 (替代旧版 top13_hit_rate):
        honor_acc  — AKQJ 归属准确率 (threshold=0.5)
        length_acc — 套长 argmax 准确率 (等价于分类准确率)
        overall_acc — 全部48维准确率
    """

    def __init__(self, hand_dim: int = 256, history_dim: int = 256, hidden_dim: int = 256):
        super().__init__()

        self.hand_encoder = nn.Sequential(
            nn.Linear(52, 256),
            nn.ReLU(),
            nn.Linear(256, hand_dim),
            nn.ReLU(),
        )

        self.history_encoder = nn.LSTM(NUM_BIDS, history_dim, num_layers=2, batch_first=True)

        self.position_embed = nn.Embedding(4, 32)

        input_dim = hand_dim + history_dim + 32 + 32
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, BELIEF_DIM),   # 52 → 48
            # 不加 Sigmoid; forward 返回 logits
        )

        # pos_weight 注册为 buffer（随模型保存/迁移设备）
        self.register_buffer('pos_weight', build_pos_weight())

    def forward(
        self,
        observer_hand: torch.Tensor,
        history: torch.Tensor,
        observer_pos: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            logits: (batch, 48) — 未经 sigmoid 的 raw scores
        """
        hand_feat = self.hand_encoder(observer_hand)

        lengths = (history.sum(dim=-1) > 0).sum(dim=-1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            history, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.history_encoder(packed)
        hist_feat = h_n[-1]

        obs_embed = self.position_embed(observer_pos)
        tgt_embed = self.position_embed(target_pos)

        x = torch.cat([hand_feat, hist_feat, obs_embed, tgt_embed], dim=-1)
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
        信息增益 = CE(before, target) - CE(after, target)
        用 mean(dim=-1) 归一化，消除维度数量对量纲的影响。
        """
        ce_before = F.binary_cross_entropy(
            belief_before, target_features, reduction='none'
        ).mean(dim=-1)   # mean 而非 sum，归一化48维
        ce_after = F.binary_cross_entropy(
            belief_after, target_features, reduction='none'
        ).mean(dim=-1)
        return ce_before - ce_after

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
