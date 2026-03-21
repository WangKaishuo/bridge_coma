"""
Belief Network (P86 重构)
=========================

核心修正:
1. Honor (16维): BCE **无 pos_weight** + sigmoid → 校准概率
2. Length (32维): CrossEntropyLoss + softmax → 互斥分类的正确建模
3. Bias 初始化: honor bias = log(0.25/0.75) ≈ -1.098
               length bias = log(prior_dist) 使初始输出 = 先验分布
4. get_probs(): honor→sigmoid, length→softmax (per suit)
5. compute_info_gain(): honor用BCE, length用CE, 加权合并

为什么 pos_weight 有害:
    pos_weight 放大正例 loss → 网络输出概率系统性偏高 (0.41 vs 真实 0.25)
    → r_info = log P(after) - log P(before) 被偏移污染
    → 即使没有信息的叫牌 (Pass) 也产生非零 r_info → 噪声奖励

为什么 Length 必须用 CrossEntropy:
    花色长度是互斥分类 (8选1)，BCE 把 8 个 bin 当独立预测，
    允许概率和 > 1.0 → 概率论上不合法 → 互信息计算错误
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from env import NUM_BIDS, NUM_PLAYERS
from utils.hand_features import (
    BELIEF_DIM, HONOR_DIM, LENGTH_DIM, LENGTH_BINS, NUM_SUITS,
    HONOR_BIAS_INIT, LENGTH_BIAS_INIT,
    belief_accuracy,
)


class BeliefNetwork(nn.Module):
    """
    Belief Network (P86): 双头架构，honor + length 分别预测。

    输入:
        observer_hand (52) + history_flat (NUM_BIDS×NUM_PLAYERS=152)
        + observer_pos_embed (32) + target_pos_embed (32)
        = 268 dims

    输出:
        honor_logits: (batch, 16) — 用 sigmoid 得到概率
        length_logits: (batch, 4, 8) — 用 softmax(dim=-1) 得到概率
    """

    def __init__(self, hand_dim: int = 256, history_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        from networks.policy_net import encode_history_flat, NUM_BIDS, NUM_PLAYERS
        self._encode_history_flat = encode_history_flat

        self.position_embed = nn.Embedding(4, 32)

        # 共享 trunk
        input_dim = 52 + NUM_BIDS * NUM_PLAYERS + 32 + 32
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Honor head: 16 独立 binary logits
        self.honor_head = nn.Linear(hidden_dim, HONOR_DIM)
        # Length head: 4 suits × 8 bins = 32 logits
        self.length_head = nn.Linear(hidden_dim, LENGTH_DIM)

        # ── Bias 初始化（P86 核心）──────────────────────────────────────
        self._init_bias()

        # 保留 pos_weight buffer 向后兼容（实际不再使用）
        from utils.hand_features import build_pos_weight
        self.register_buffer('pos_weight', build_pos_weight())

    def _init_bias(self):
        """
        先验 bias 初始化:
        - Honor: bias = log(0.25/0.75) ≈ -1.098 → sigmoid 输出 ≈ 0.25
        - Length: bias = log(prior_dist) → softmax 输出 ≈ 先验分布
        网络从先验开始学习，无信息时自然回退到先验 → r_info ≈ 0
        """
        with torch.no_grad():
            self.honor_head.bias.fill_(HONOR_BIAS_INIT)
            self.honor_head.weight.mul_(0.01)

            length_bias = torch.tensor(LENGTH_BIAS_INIT * NUM_SUITS, dtype=torch.float32)
            self.length_head.bias.copy_(length_bias)
            self.length_head.weight.mul_(0.01)

    def forward(
        self,
        observer_hand: torch.Tensor,
        history: torch.Tensor,
        observer_pos: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns: (batch, 48) raw logits.
        [0:16] = honor logits (for BCE / sigmoid)
        [16:48] = length logits (for CE / softmax per suit)
        """
        hist_flat = self._encode_history_flat(history)
        obs_embed = self.position_embed(observer_pos)
        tgt_embed = self.position_embed(target_pos)
        x = torch.cat([observer_hand, hist_flat, obs_embed, tgt_embed], dim=-1)
        h = self.trunk(x)
        honor_logits = self.honor_head(h)
        length_logits = self.length_head(h)
        return torch.cat([honor_logits, length_logits], dim=-1)

    def get_probs(
        self,
        observer_hand: torch.Tensor,
        history: torch.Tensor,
        observer_pos: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        返回校准概率 (batch, 48):
        [0:16]  = sigmoid(honor_logits)  — 独立二值概率
        [16:48] = softmax(length_logits per suit) — 互斥分类概率
        """
        logits = self.forward(observer_hand, history, observer_pos, target_pos)
        return self._logits_to_probs(logits)

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """logits (B, 48) → calibrated probs (B, 48)."""
        honor_probs = torch.sigmoid(logits[:, :HONOR_DIM])
        length_logits = logits[:, HONOR_DIM:].view(-1, NUM_SUITS, LENGTH_BINS)
        length_probs = F.softmax(length_logits, dim=-1).view(-1, LENGTH_DIM)
        return torch.cat([honor_probs, length_probs], dim=-1)

    def compute_loss(
        self,
        observer_hand: torch.Tensor,
        history: torch.Tensor,
        observer_pos: torch.Tensor,
        target_pos: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        P86 混合 loss:
        - Honor [0:16]: BCEWithLogitsLoss（无 pos_weight）
        - Length [16:48]: CrossEntropyLoss（4门花色各自 8 分类）
        """
        logits = self.forward(observer_hand, history, observer_pos, target_pos)

        honor_loss = F.binary_cross_entropy_with_logits(
            logits[:, :HONOR_DIM],
            target_features[:, :HONOR_DIM],
            reduction='mean',
        )

        length_logits = logits[:, HONOR_DIM:].view(-1, NUM_SUITS, LENGTH_BINS)
        length_target = target_features[:, HONOR_DIM:].view(-1, NUM_SUITS, LENGTH_BINS)
        length_labels = length_target.argmax(dim=-1)

        length_loss = F.cross_entropy(
            length_logits.reshape(-1, LENGTH_BINS),
            length_labels.reshape(-1),
            reduction='mean',
        )

        return honor_loss + length_loss

    @staticmethod
    def evaluate_accuracy(probs: torch.Tensor, targets: torch.Tensor) -> dict:
        return belief_accuracy(probs, targets)


class DualInfoComputer:
    """
    计算 Dual-Info Bonus (P86 适配)

    r_info = I(bid; hand | partner) - β * I(bid; hand | opponent)

    P86: honor 和 length 分别计算信息增益，加权合并。
    """

    def __init__(self, belief_net: BeliefNetwork, beta: float = 0.5):
        self.belief_net = belief_net
        self.beta = beta

    def compute_info_gain(
        self,
        belief_before: torch.Tensor,
        belief_after: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        信息增益 = max(0, CE(before, target) - CE(after, target))
        Honor: BCE, Length: CE (per suit), 等权平均。
        """
        # ── Honor ──────────────────────────────────────────────────────
        h_before = belief_before[:, :HONOR_DIM].clamp(1e-7, 1-1e-7)
        h_after  = belief_after[:, :HONOR_DIM].clamp(1e-7, 1-1e-7)
        h_target = target_features[:, :HONOR_DIM]

        honor_ce_before = F.binary_cross_entropy(
            h_before, h_target, reduction='none').mean(dim=-1)
        honor_ce_after = F.binary_cross_entropy(
            h_after, h_target, reduction='none').mean(dim=-1)

        # ── Length ─────────────────────────────────────────────────────
        l_before = belief_before[:, HONOR_DIM:].view(-1, NUM_SUITS, LENGTH_BINS)
        l_after  = belief_after[:, HONOR_DIM:].view(-1, NUM_SUITS, LENGTH_BINS)
        l_target = target_features[:, HONOR_DIM:].view(-1, NUM_SUITS, LENGTH_BINS)
        l_labels = l_target.argmax(dim=-1)

        l_ce_before = -torch.log(
            l_before.clamp(1e-7, 1.0).gather(2, l_labels.unsqueeze(-1)).squeeze(-1)
        ).mean(dim=-1)
        l_ce_after = -torch.log(
            l_after.clamp(1e-7, 1.0).gather(2, l_labels.unsqueeze(-1)).squeeze(-1)
        ).mean(dim=-1)

        # ── 合并 ──────────────────────────────────────────────────────
        ce_before = (honor_ce_before + l_ce_before) / 2.0
        ce_after  = (honor_ce_after  + l_ce_after)  / 2.0

        return torch.relu(ce_before - ce_after)

    def compute_dual_info_bonus(
        self,
        partner_gain: torch.Tensor,
        opponent_leak: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        bonus = partner_gain - self.beta * opponent_leak
        metrics = {
            'partner_gain':    partner_gain.mean().item(),
            'opponent_leak':   opponent_leak.mean().item(),
            'info_ratio':      (partner_gain.mean() / (opponent_leak.mean() + 1e-8)).item(),
            'dual_info_bonus': bonus.mean().item(),
        }
        return bonus, metrics
