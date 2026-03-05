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
    """叫牌历史编码器 (LSTM)"""
    
    def __init__(self, input_dim: int = NUM_BIDS, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.output_dim = hidden_dim
    
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: (batch, seq_len, num_bids)
        _, (h_n, _) = self.lstm(history)
        return h_n[-1]  # 最后一层的隐藏状态


class PolicyNetwork(nn.Module):
    """策略网络"""
    
    def __init__(self, hand_dim: int = 256, history_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.hand_encoder = HandEncoder(hand_dim)
        self.history_encoder = HistoryEncoder(NUM_BIDS, history_dim)
        
        # 融合: hand + history + position + vulnerability
        input_dim = hand_dim + history_dim + 4 + 2
        
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
        
        x = torch.cat([hand_feat, hist_feat, obs['position'], obs['vulnerability']], dim=-1)
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
