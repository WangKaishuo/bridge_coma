"""
Policy and Value Networks
=========================

Actor-Critic 网络架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from env import NUM_BIDS


class HandEncoder(nn.Module):
    """手牌编码器"""
    
    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(52, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.ReLU(),
        )
    
    def forward(self, hand: torch.Tensor) -> torch.Tensor:
        return self.net(hand)


class HistoryEncoder(nn.Module):
    """
    叫牌历史编码器 (LSTM)

    关键修复: 只处理有效 token, 忽略零 padding.

    原版直接对 (batch, 60, 38) 跑 LSTM 取 h_n[-1],
    但实际叫牌历史只有 4-20 步, 剩余 40-56 步全是零向量.
    LSTM 处理几十步零输入后 hidden state 退化到固定点,
    冲刷掉了关键叫牌 (如 N 的 2D/2H/2S) 的信息.

    修复: 检测每条序列的实际长度, 用 pack_padded_sequence
    让 LSTM 只处理有效时间步, 取实际最后一步的 h_n.
    """
    
    def __init__(self, input_dim: int = NUM_BIDS, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.output_dim = hidden_dim
    
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: (batch, max_seq_len, num_bids) — one-hot, 零 padding
        batch_size = history.shape[0]

        # 计算每条序列的实际长度: 非零行的数量
        # history.sum(dim=-1) > 0 → (batch, max_seq_len) bool mask
        lengths = (history.sum(dim=-1) > 0).sum(dim=-1)  # (batch,)
        lengths = lengths.clamp(min=1)  # 防止全零序列导致 pack 崩溃

        # pack → LSTM → unpack
        # 需要 CPU 上的 lengths for pack_padded_sequence
        lengths_cpu = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            history, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers, batch, hidden_dim) — 已经是每条序列实际最后一步的状态
        return h_n[-1]  # (batch, hidden_dim)


class PolicyNetwork(nn.Module):
    """策略网络

    belief_dim > 0 时接受 obs['belief'] 作为额外输入.
    belief 特征由外部 BeliefNetwork 推断后注入 obs, stop-gradient.

    架构支持任意 player 的双向 belief 输入 (N推断S, S推断N, E推断W等).
    当前实验只为 SOUTH 的 actor 启用 (belief_dim=BELIEF_DIM),
    其余 player 的 actor 保持 belief_dim=0 (向后兼容).
    """

    def __init__(self, hand_dim: int = 256, history_dim: int = 256,
                 hidden_dim: int = 256, belief_dim: int = 0):
        super().__init__()
        self.hand_encoder = HandEncoder(hand_dim)
        self.history_encoder = HistoryEncoder(NUM_BIDS, history_dim)
        self.belief_dim = belief_dim

        # 融合: hand + history + position + vulnerability [+ belief (可选)]
        input_dim = hand_dim + history_dim + 4 + 2 + belief_dim

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_BIDS),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        hand_feat = self.hand_encoder(obs['hand'])
        hist_feat = self.history_encoder(obs['history'])

        features = [hand_feat, hist_feat, obs['position'], obs['vulnerability']]

        if self.belief_dim > 0:
            # belief: (B, belief_dim) — 由外部 BeliefNetwork 推断, stop-gradient
            # BC 阶段 obs['belief'] 不存在时用零向量填充, 保持网络结构一致
            if 'belief' in obs:
                features.append(obs['belief'].detach())
            else:
                features.append(torch.zeros(
                    hand_feat.shape[0], self.belief_dim,
                    device=hand_feat.device, dtype=hand_feat.dtype))

        x = torch.cat(features, dim=-1)
        return self.fc(x)
    
    def get_action(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样动作"""
        logits = self.forward(obs)
        
        # 应用 legal action mask
        mask = obs['legal_actions']
        logits = logits - 1e9 * (1 - mask)
        
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy
    
    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估动作"""
        logits = self.forward(obs)
        mask = obs['legal_actions']
        logits = logits - 1e9 * (1 - mask)
        
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        return dist.log_prob(actions), dist.entropy()


class ValueNetwork(nn.Module):
    """价值网络"""
    
    def __init__(self, hand_dim: int = 256, history_dim: int = 256, hidden_dim: int = 256,
                 centralized: bool = False):
        super().__init__()
        self.centralized = centralized
        
        self.hand_encoder = HandEncoder(hand_dim)
        self.history_encoder = HistoryEncoder(NUM_BIDS, history_dim)
        
        if centralized:
            # 集中式: 看所有手牌
            self.all_hands_encoder = nn.Sequential(
                nn.Linear(4 * 52, 256),
                nn.ReLU(),
                nn.Linear(256, hand_dim),
                nn.ReLU(),
            )
            input_dim = hand_dim + history_dim + hand_dim + 4 + 2
        else:
            input_dim = hand_dim + history_dim + 4 + 2
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        all_hands: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hand_feat = self.hand_encoder(obs['hand'])
        hist_feat = self.history_encoder(obs['history'])
        
        features = [hand_feat, hist_feat, obs['position'], obs['vulnerability']]
        
        if self.centralized and all_hands is not None:
            batch_size = all_hands.shape[0]
            all_hands_flat = all_hands.view(batch_size, -1)
            all_hands_feat = self.all_hands_encoder(all_hands_flat)
            features.append(all_hands_feat)
        
        x = torch.cat(features, dim=-1)
        return self.fc(x).squeeze(-1)


class ActorCritic(nn.Module):
    """Actor-Critic 联合网络"""
    
    def __init__(
        self,
        hand_dim: int = 256,
        history_dim: int = 256,
        hidden_dim: int = 256,
        lstm_layers: int = 2,
        centralized_critic: bool = False
    ):
        super().__init__()
        self.actor = PolicyNetwork(hand_dim, history_dim, hidden_dim)
        self.critic = ValueNetwork(hand_dim, history_dim, hidden_dim, centralized_critic)
    
    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        all_hands: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(obs)
        value = self.critic(obs, all_hands)
        return logits, value
    
    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        all_hands: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action, log_prob, entropy = self.actor.get_action(obs, deterministic)
        value = self.critic(obs, all_hands)
        return action, log_prob, entropy, value
