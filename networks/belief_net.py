"""
Belief Network
==============

推断他人手牌的网络，用于计算 Dual-Info Bonus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from env import NUM_BIDS


class BeliefNetwork(nn.Module):
    """
    Belief Network: 推断目标玩家的手牌

    输入: 观察者手牌 + 叫牌历史 + 观察者位置 + 目标位置
    输出: 目标手牌 52 张牌的 logits (未经 Sigmoid)

    关键修复 — 类别不平衡问题:
    每手 13 张牌, 目标向量中 39 个 0, 13 个 1 (比例 3:1).
    若用普通 BCE + Sigmoid, 网络只需无脑预测全 0 就能拿到
    acc = 39/52 = 0.75, Loss ≈ 0.47, 实际上什么都没学.

    修复方案:
    1. 最后一层去掉 Sigmoid, 直接输出 logits.
    2. 用 BCEWithLogitsLoss(pos_weight=3.0): 正样本 (有牌) 权重
       ×3, 强迫网络关注 13 张实际存在的牌.
    3. 评估改用 Top-13 命中率 (随机基线 = 3.25/13 ≈ 25%),
       而非阈值 acc (随机基线 = 75%, 完全无法区分好坏).

    forward 返回 logits; 外部调用 sigmoid(logits) 得概率.
    """

    # 正样本权重: 0/1 比例 = 3:1
    POS_WEIGHT = 3.0

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
            nn.Linear(hidden_dim, 52),
            # 注意: 不加 Sigmoid, forward 返回 logits
            # 概率 = torch.sigmoid(logits)
        )

    def forward(
        self,
        observer_hand: torch.Tensor,
        history: torch.Tensor,
        observer_pos: torch.Tensor,
        target_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns:
            logits: (batch, 52) — 未经 sigmoid 的 raw scores
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
        target_pos: torch.Tensor
    ) -> torch.Tensor:
        """返回 sigmoid 概率, 供信息增益计算使用."""
        return torch.sigmoid(self.forward(observer_hand, history, observer_pos, target_pos))

    def compute_loss(
        self,
        observer_hand: torch.Tensor,
        history: torch.Tensor,
        observer_pos: torch.Tensor,
        target_pos: torch.Tensor,
        target_hand: torch.Tensor
    ) -> torch.Tensor:
        """
        BCEWithLogitsLoss + pos_weight=3.

        等价于: BCE(sigmoid(logits), target) 但数值更稳定,
        且正样本梯度权重 ×3, 解决类别不平衡问题.
        """
        logits = self.forward(observer_hand, history, observer_pos, target_pos)
        pw = torch.tensor([self.POS_WEIGHT], device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, target_hand, pos_weight=pw)

    @staticmethod
    def top13_hit_rate(probs: torch.Tensor, target_hand: torch.Tensor) -> float:
        """
        Top-13 命中率: 取概率最高的 13 张牌, 与真实手牌求交集.

        随机基线: E[交集] = 13×13/52 = 3.25 张 (25%)
        完美预测: 13 张全中 (100%)

        Args:
            probs: (batch, 52) sigmoid 概率
            target_hand: (batch, 52) 0/1
        Returns:
            平均命中率 (0.0 ~ 1.0)
        """
        top13 = probs.topk(13, dim=-1).indices  # (batch, 13)
        hits = 0.0
        for i in range(probs.shape[0]):
            pred_set = set(top13[i].tolist())
            true_set = set(target_hand[i].nonzero(as_tuple=False).squeeze(-1).tolist())
            hits += len(pred_set & true_set)
        return hits / (probs.shape[0] * 13)


class DualInfoComputer:
    """
    计算 Dual-Info Bonus
    
    r_info = I(bid; hand | partner) - β * I(bid; hand | opponent)
    
    近似计算:
    I(bid; hand) ≈ H(hand | before) - H(hand | after)
                 ≈ CE(belief_before, hand) - CE(belief_after, hand)
    """
    
    def __init__(self, belief_net: BeliefNetwork, beta: float = 0.5):
        self.belief_net = belief_net
        self.beta = beta
    
    def compute_info_gain(
        self,
        belief_before: torch.Tensor,
        belief_after: torch.Tensor,
        target_hand: torch.Tensor
    ) -> torch.Tensor:
        """
        计算信息增益
        
        = CE(before, target) - CE(after, target)
        = uncertainty_before - uncertainty_after
        """
        ce_before = F.binary_cross_entropy(belief_before, target_hand, reduction='none').sum(dim=-1)
        ce_after = F.binary_cross_entropy(belief_after, target_hand, reduction='none').sum(dim=-1)
        return ce_before - ce_after
    
    def compute_dual_info_bonus(
        self,
        partner_gain: torch.Tensor,
        opponent_leak: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算 Dual-Info Bonus
        
        Returns:
            bonus: partner_gain - β * opponent_leak
            metrics: 详细指标
        """
        bonus = partner_gain - self.beta * opponent_leak
        
        metrics = {
            'partner_gain': partner_gain.mean().item(),
            'opponent_leak': opponent_leak.mean().item(),
            'info_ratio': (partner_gain.mean() / (opponent_leak.mean() + 1e-8)).item(),
            'dual_info_bonus': bonus.mean().item(),
        }
        
        return bonus, metrics
