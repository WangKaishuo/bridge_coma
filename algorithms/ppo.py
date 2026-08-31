"""Shared PPO configuration and a lightweight compatibility buffer."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PPOConfig:
    hidden_dim: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    batch_size: int = 256
    num_epochs: int = 4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    device: str = "cpu"


class RolloutBuffer:
    """Minimal rollout buffer retained for MAPPO checkpoint compatibility."""

    def __init__(self, device: str = "cpu"):
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
        self.observations.append({
            key: item.cpu() if torch.is_tensor(item) else torch.tensor(item)
            for key, item in obs.items()
        })
        self.actions.append(action.cpu() if torch.is_tensor(action) else torch.tensor(action))
        self.log_probs.append(
            log_prob.cpu() if torch.is_tensor(log_prob) else torch.tensor(log_prob)
        )
        self.rewards.append(reward)
        self.values.append(value.cpu() if torch.is_tensor(value) else torch.tensor(value))
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        rewards = np.asarray(self.rewards)
        values = np.asarray([value.item() for value in self.values])
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae = 0.0
        for index in reversed(range(len(rewards))):
            next_value = last_value if index == len(rewards) - 1 else values[index + 1]
            non_terminal = 1.0 - float(self.dones[index])
            delta = rewards[index] + gamma * next_value * non_terminal - values[index]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[index] = last_gae
        self.advantages = torch.tensor(advantages)
        self.returns = torch.tensor(advantages + values, dtype=torch.float32)

    def get_batches(self, batch_size):
        indices = np.random.permutation(len(self.actions))
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            observations = {
                key: torch.stack([self.observations[i][key] for i in selected]).to(self.device)
                for key in self.observations[0]
            }
            yield {
                "obs": observations,
                "actions": torch.stack([self.actions[i] for i in selected]).to(self.device),
                "old_log_probs": torch.stack(
                    [self.log_probs[i] for i in selected]
                ).to(self.device),
                "advantages": self.advantages[selected].to(self.device),
                "returns": self.returns[selected].to(self.device),
            }
