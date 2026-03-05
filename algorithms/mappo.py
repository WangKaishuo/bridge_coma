"""
MAPPO (Multi-Agent PPO)
=======================

集中式 Critic 的 PPO (CTDE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from env import NUM_PLAYERS
from networks import ActorCritic
from algorithms.ippo import PPOConfig, RolloutBuffer


@dataclass
class MAPPOConfig(PPOConfig):
    """MAPPO 配置"""
    centralized_critic: bool = True


class MAPPORolloutBuffer(RolloutBuffer):
    """MAPPO 缓冲区（额外存储所有手牌）"""
    
    def reset(self):
        super().reset()
        self.all_hands = []
    
    def add(self, obs, action, log_prob, reward, value, done, all_hands=None):
        super().add(obs, action, log_prob, reward, value, done)
        if all_hands is not None:
            self.all_hands.append(torch.tensor(all_hands, dtype=torch.float32))
    
    def get_batches(self, batch_size: int):
        n = len(self.actions)
        indices = np.random.permutation(n)
        
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            
            batch_obs = {key: torch.stack([self.observations[i][key] for i in batch_idx]).to(self.device)
                         for key in self.observations[0].keys()}
            
            batch = {
                'obs': batch_obs,
                'actions': torch.stack([self.actions[i] for i in batch_idx]).to(self.device),
                'old_log_probs': torch.stack([self.log_probs[i] for i in batch_idx]).to(self.device),
                'advantages': self.advantages[batch_idx].to(self.device),
                'returns': self.returns[batch_idx].to(self.device),
            }
            
            if self.all_hands:
                batch['all_hands'] = torch.stack([self.all_hands[i] for i in batch_idx]).to(self.device)
            
            yield batch


class MAPPOAgent:
    """Multi-Agent PPO with Centralized Critic"""
    
    def __init__(self, config: MAPPOConfig):
        self.config = config
        self.device = config.device
        
        self.model = ActorCritic(
            hand_dim=config.hand_dim,
            history_dim=config.history_dim,
            hidden_dim=config.hidden_dim,
            centralized_critic=True
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.buffers = {p: MAPPORolloutBuffer(self.device) for p in range(NUM_PLAYERS)}
    
    def get_action(self, obs: Dict[str, np.ndarray], all_hands: Optional[np.ndarray] = None,
                   deterministic: bool = False) -> Tuple[int, Dict]:
        obs_tensor = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device) for k, v in obs.items()}
        all_hands_tensor = torch.tensor(all_hands, dtype=torch.float32).unsqueeze(0).to(self.device) if all_hands is not None else None
        
        with torch.no_grad():
            action, log_prob, _, value = self.model.get_action_and_value(obs_tensor, all_hands_tensor, deterministic)
        
        return action.item(), {'log_prob': log_prob.squeeze(0), 'value': value.squeeze(0)}
    
    def store_transition(self, player: int, obs, action, log_prob, reward, value, done, all_hands=None):
        obs_tensor = {k: torch.tensor(v, dtype=torch.float32) for k, v in obs.items()}
        self.buffers[player].add(obs_tensor, torch.tensor(action), log_prob, reward, value, done, all_hands)
    
    def update(self) -> Dict[str, float]:
        total_loss = total_policy = total_value = total_entropy = num_updates = 0
        
        for player in range(NUM_PLAYERS):
            buffer = self.buffers[player]
            if not buffer.actions:
                continue
            
            with torch.no_grad():
                last_obs = {k: v.unsqueeze(0).to(self.device) for k, v in buffer.observations[-1].items()}
                last_hands = buffer.all_hands[-1].unsqueeze(0).to(self.device) if buffer.all_hands else None
                last_value = self.model.critic(last_obs, last_hands).item()
            
            buffer.compute_returns_and_advantages(last_value, self.config.gamma, self.config.gae_lambda)
            
            for _ in range(self.config.num_epochs):
                for batch in buffer.get_batches(self.config.batch_size):
                    log_probs, entropy = self.model.actor.evaluate_actions(batch['obs'], batch['actions'])
                    values = self.model.critic(batch['obs'], batch.get('all_hands'))
                    
                    ratio = torch.exp(log_probs - batch['old_log_probs'])
                    adv = (batch['advantages'] - batch['advantages'].mean()) / (batch['advantages'].std() + 1e-8)
                    
                    policy_loss = -torch.min(ratio * adv,
                                             torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * adv).mean()
                    value_loss = F.mse_loss(values, batch['returns'])
                    loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy.mean()
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    total_policy += policy_loss.item()
                    total_value += value_loss.item()
                    total_entropy += entropy.mean().item()
                    num_updates += 1
            
            buffer.reset()
        
        return {'loss': total_loss / max(1, num_updates), 'policy_loss': total_policy / max(1, num_updates),
                'value_loss': total_value / max(1, num_updates), 'entropy': total_entropy / max(1, num_updates)} if num_updates else {}
    
    def save(self, path: str):
        torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, path)
    
    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
