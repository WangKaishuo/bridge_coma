"""
IPPO (Independent PPO)
======================

每个 Agent 独立训练的 PPO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

from env import NUM_PLAYERS
from networks import ActorCritic


@dataclass
class PPOConfig:
    """PPO 配置"""
    hand_dim: int = 256
    history_dim: int = 256
    hidden_dim: int = 256
    lstm_layers: int = 2
    
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    
    lr: float = 3e-4
    batch_size: int = 256
    num_epochs: int = 4
    
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    device: str = 'cpu'


class RolloutBuffer:
    """经验缓冲区"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.reset()
    
    def reset(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = None
        self.returns = None
    
    def add(self, obs, action, log_prob, reward, value, done):
        self.observations.append({k: v.cpu() if torch.is_tensor(v) else torch.tensor(v) 
                                  for k, v in obs.items()})
        self.actions.append(action.cpu() if torch.is_tensor(action) else torch.tensor(action))
        self.log_probs.append(log_prob.cpu() if torch.is_tensor(log_prob) else torch.tensor(log_prob))
        self.rewards.append(reward)
        self.values.append(value.cpu() if torch.is_tensor(value) else torch.tensor(value))
        self.dones.append(done)
    
    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        rewards = np.array(self.rewards)
        values = np.array([v.item() for v in self.values])
        dones = np.array(self.dones)
        
        n = len(rewards)
        advantages = np.zeros(n)
        last_gae = 0
        
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else values[t + 1]
            next_non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        self.returns = torch.tensor(advantages + values, dtype=torch.float32)
        self.advantages = torch.tensor(advantages, dtype=torch.float32)
    
    def get_batches(self, batch_size: int):
        n = len(self.actions)
        indices = np.random.permutation(n)
        
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            
            batch_obs = {key: torch.stack([self.observations[i][key] for i in batch_idx]).to(self.device)
                         for key in self.observations[0].keys()}
            
            yield {
                'obs': batch_obs,
                'actions': torch.stack([self.actions[i] for i in batch_idx]).to(self.device),
                'old_log_probs': torch.stack([self.log_probs[i] for i in batch_idx]).to(self.device),
                'advantages': self.advantages[batch_idx].to(self.device),
                'returns': self.returns[batch_idx].to(self.device),
            }


class IPPOAgent:
    """Independent PPO Agent"""
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = config.device
        
        self.model = ActorCritic(
            hand_dim=config.hand_dim,
            history_dim=config.history_dim,
            hidden_dim=config.hidden_dim,
            centralized_critic=False
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.buffers = {p: RolloutBuffer(self.device) for p in range(NUM_PLAYERS)}
    
    def get_action(self, obs: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[int, Dict]:
        obs_tensor = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device) 
                      for k, v in obs.items()}
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.model.get_action_and_value(obs_tensor, deterministic=deterministic)
        
        return action.item(), {'log_prob': log_prob.squeeze(0), 'value': value.squeeze(0)}
    
    def store_transition(self, player: int, obs, action, log_prob, reward, value, done):
        obs_tensor = {k: torch.tensor(v, dtype=torch.float32) for k, v in obs.items()}
        self.buffers[player].add(obs_tensor, torch.tensor(action), log_prob, reward, value, done)
    
    def update(self) -> Dict[str, float]:
        total_loss = total_policy = total_value = total_entropy = num_updates = 0
        
        for player in range(NUM_PLAYERS):
            buffer = self.buffers[player]
            if not buffer.actions:
                continue
            
            with torch.no_grad():
                last_obs = {k: v.unsqueeze(0).to(self.device) for k, v in buffer.observations[-1].items()}
                last_value = self.model.critic(last_obs).item()
            
            buffer.compute_returns_and_advantages(last_value, self.config.gamma, self.config.gae_lambda)
            
            for _ in range(self.config.num_epochs):
                for batch in buffer.get_batches(self.config.batch_size):
                    log_probs, entropy = self.model.actor.evaluate_actions(batch['obs'], batch['actions'])
                    values = self.model.critic(batch['obs'])
                    
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
