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
    输出: 目标手牌的 52 张牌概率
    """
    
    def __init__(self, hand_dim: int = 256, history_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        
        # 手牌编码
        self.hand_encoder = nn.Sequential(
            nn.Linear(52, 256),
            nn.ReLU(),
            nn.Linear(256, hand_dim),
            nn.ReLU(),
        )
        
        # 历史编码
        self.history_encoder = nn.LSTM(NUM_BIDS, history_dim, num_layers=2, batch_first=True)
        
        # 位置嵌入
        self.position_embed = nn.Embedding(4, 32)
        
        # 输出层
        input_dim = hand_dim + history_dim + 32 + 32
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 52),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        observer_hand: torch.Tensor,
        history: torch.Tensor,
        observer_pos: torch.Tensor,
        target_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            observer_hand: (batch, 52)
            history: (batch, seq_len, num_bids)
            observer_pos: (batch,) int
            target_pos: (batch,) int
        
        Returns:
            belief: (batch, 52) 每张牌在目标手中的概率
        """
        hand_feat = self.hand_encoder(observer_hand)
        
        # 同 HistoryEncoder 修复: 只处理有效 token, 忽略零 padding
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
    
    def compute_loss(
        self,
        observer_hand: torch.Tensor,
        history: torch.Tensor,
        observer_pos: torch.Tensor,
        target_pos: torch.Tensor,
        target_hand: torch.Tensor
    ) -> torch.Tensor:
        """计算 BCE 损失"""
        belief = self.forward(observer_hand, history, observer_pos, target_pos)
        return F.binary_cross_entropy(belief, target_hand)


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
