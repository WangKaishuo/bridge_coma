"""
MAPPO (Multi-Agent PPO) — HAPPO-style Heterogeneous Actor
==========================================================

架构改动 (P48): 从共享 Actor 改为双 Actor + 共享 Critic.

动机:
  原架构 actor 参数 N 和 S 共享. S-phase 梯度流入 N 的 actor,
  N-phase 梯度流入 S 的 actor. 每轮切换后 active player 的
  BC 初始化被污染 → N-phase IMP 系统性崩溃 (~-5.0 vs BC level -3.8).

HAPPO 解法 (Kuba et al. 2021):
  异质智能体 (Heterogeneous Agent) 采用独立的非共享 policy,
  只共享集中式 critic. 保证 S-phase 梯度只更新 actor_s,
  N-phase 梯度只更新 actor_n. Critic 仍然看全局, 价值估计不退化.

接口变化:
  - agent.model.actor         → 已弃用 (兼容层: 指向 actor_s)
  - agent.get_actor(player)   → 按 player 返回对应 actor
  - agent.model.critic        → 不变, 共享 critic
  - save/load: 分别保存 actor_n, actor_s, critic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from env import NUM_PLAYERS, NORTH, SOUTH
from networks import ActorCritic
from networks.policy_net import PolicyNetwork, ValueNetwork
from algorithms.ippo import PPOConfig, RolloutBuffer


@dataclass
class MAPPOConfig(PPOConfig):
    """MAPPO 配置"""
    centralized_critic: bool = True
    critic_lr_ratio: float = 3.0   # critic_lr = lr * ratio; 默认3x加速Critic收敛


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


class _SharedCriticWrapper(nn.Module):
    """
    包装容器: 让 subgame_trainer 中 agent.model.critic 接口保持不变.
    包含 actor_n, actor_s, critic 三个子模块.
    """
    def __init__(self, actor_n: PolicyNetwork, actor_s: PolicyNetwork,
                 critic: ValueNetwork):
        super().__init__()
        self.actor_n = actor_n          # NORTH actor (player 0)
        self.actor_s = actor_s          # SOUTH actor (player 2)
        self.critic = critic            # 共享 centralized critic

        # 向后兼容: agent.model.actor 指向 actor_s
        self.actor = actor_s

    def forward(self, obs, all_hands=None):
        logits = self.actor_s(obs)
        value = self.critic(obs, all_hands)
        return logits, value

    def get_action_and_value(self, obs, all_hands=None, deterministic=False,
                             player: int = SOUTH):
        actor = self.actor_n if player == NORTH else self.actor_s
        action, log_prob, entropy = actor.get_action(obs, deterministic)
        value = self.critic(obs, all_hands)
        return action, log_prob, entropy, value


class MAPPOAgent:
    """
    HAPPO-style Multi-Agent PPO with Centralized Critic.

    - actor_n (NORTH) 和 actor_s (SOUTH) 独立参数, 独立 optimizer
    - critic 共享, 独立 optimizer
    - S-phase PPO 只更新 actor_s; N-phase PPO 只更新 actor_n
    - critic 在所有 phase 都可以更新
    """

    def __init__(self, config: MAPPOConfig):
        self.config = config
        self.device = config.device

        actor_n = PolicyNetwork(
            hand_dim=config.hand_dim,
            history_dim=config.history_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        actor_s = PolicyNetwork(
            hand_dim=config.hand_dim,
            history_dim=config.history_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        critic = ValueNetwork(
            hand_dim=config.hand_dim,
            history_dim=config.history_dim,
            hidden_dim=config.hidden_dim,
            centralized=True,
        ).to(self.device)

        self.model = _SharedCriticWrapper(actor_n, actor_s, critic).to(self.device)

        self.actor_n_optimizer = torch.optim.Adam(
            self.model.actor_n.parameters(), lr=config.lr)
        self.actor_s_optimizer = torch.optim.Adam(
            self.model.actor_s.parameters(), lr=config.lr)
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic.parameters(), lr=config.lr * config.critic_lr_ratio)

        # 向后兼容
        self.actor_optimizer = self.actor_s_optimizer
        self.optimizer = self.actor_s_optimizer

        self.buffers = {p: MAPPORolloutBuffer(self.device) for p in range(NUM_PLAYERS)}

    def get_actor(self, player: int) -> PolicyNetwork:
        """按 player 返回对应 actor. NORTH=actor_n; 其余=actor_s."""
        return self.model.actor_n if player == NORTH else self.model.actor_s

    def get_actor_optimizer(self, player: int):
        """按 player 返回对应 actor 的 optimizer."""
        return self.actor_n_optimizer if player == NORTH else self.actor_s_optimizer

    def get_action(self, obs: Dict[str, np.ndarray], all_hands=None,
                   deterministic: bool = False) -> Tuple[int, Dict]:
        """向后兼容接口: 默认使用 actor_s. 推荐使用 get_action_for_player."""
        return self._get_action_for_player(obs, SOUTH, all_hands, deterministic)

    def get_action_for_player(self, obs: Dict[str, np.ndarray], player: int,
                               all_hands=None,
                               deterministic: bool = False) -> Tuple[int, Dict]:
        """按 player 选择对应 actor 采样动作 (推荐接口)."""
        return self._get_action_for_player(obs, player, all_hands, deterministic)

    def _get_action_for_player(self, obs, player, all_hands, deterministic):
        obs_tensor = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device)
                      for k, v in obs.items()}
        all_hands_tensor = (torch.tensor(all_hands, dtype=torch.float32).unsqueeze(0).to(self.device)
                            if all_hands is not None else None)
        actor = self.get_actor(player)
        with torch.no_grad():
            action, log_prob, _ = actor.get_action(obs_tensor, deterministic)
            value = self.model.critic(obs_tensor, all_hands_tensor)
        return action.item(), {'log_prob': log_prob.squeeze(0), 'value': value.squeeze(0)}

    def store_transition(self, player: int, obs, action, log_prob, reward, value, done,
                         all_hands=None):
        obs_tensor = {k: torch.tensor(v, dtype=torch.float32) for k, v in obs.items()}
        self.buffers[player].add(obs_tensor, torch.tensor(action), log_prob, reward,
                                 value, done, all_hands)

    def update(self) -> Dict[str, float]:
        """标准 update. 推荐使用 SubgameTrainer.safe_update() 替代."""
        total_loss = total_policy = total_value = total_entropy = num_updates = 0

        for player in range(NUM_PLAYERS):
            buffer = self.buffers[player]
            if not buffer.actions:
                continue

            actor = self.get_actor(player)
            actor_opt = self.get_actor_optimizer(player)

            with torch.no_grad():
                last_obs = {k: v.unsqueeze(0).to(self.device)
                            for k, v in buffer.observations[-1].items()}
                last_hands = (buffer.all_hands[-1].unsqueeze(0).to(self.device)
                              if buffer.all_hands else None)
                last_value = self.model.critic(last_obs, last_hands).item()

            buffer.compute_returns_and_advantages(
                last_value, self.config.gamma, self.config.gae_lambda)

            for _ in range(self.config.num_epochs):
                for batch in buffer.get_batches(self.config.batch_size):
                    log_probs, entropy = actor.evaluate_actions(
                        batch['obs'], batch['actions'])
                    ratio = torch.exp(log_probs - batch['old_log_probs'])
                    adv = (batch['advantages'] - batch['advantages'].mean()) / (
                        batch['advantages'].std() + 1e-8)

                    policy_loss = -torch.min(
                        ratio * adv,
                        torch.clamp(ratio, 1 - self.config.clip_ratio,
                                    1 + self.config.clip_ratio) * adv
                    ).mean()
                    actor_loss = policy_loss - self.config.entropy_coef * entropy.mean()

                    actor_opt.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), self.config.max_grad_norm)
                    actor_opt.step()

                    values = self.model.critic(batch['obs'], batch.get('all_hands'))
                    old_values = values.detach()
                    v_clipped = old_values + (values - old_values).clamp(
                        -self.config.clip_ratio, self.config.clip_ratio)
                    value_loss = torch.max(
                        F.mse_loss(values, batch['returns']),
                        F.mse_loss(v_clipped, batch['returns'])
                    )

                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.model.critic.parameters(),
                                             self.config.max_grad_norm)
                    self.critic_optimizer.step()

                    total_loss += actor_loss.item() + value_loss.item()
                    total_policy += policy_loss.item()
                    total_value += value_loss.item()
                    total_entropy += entropy.mean().item()
                    num_updates += 1

            buffer.reset()

        if num_updates == 0:
            return {}
        return {
            'loss': total_loss / num_updates,
            'policy_loss': total_policy / num_updates,
            'value_loss': total_value / num_updates,
            'entropy': total_entropy / num_updates,
        }

    def save(self, path: str):
        torch.save({
            'actor_n': self.model.actor_n.state_dict(),
            'actor_s': self.model.actor_s.state_dict(),
            'critic': self.model.critic.state_dict(),
            'actor_n_optimizer': self.actor_n_optimizer.state_dict(),
            'actor_s_optimizer': self.actor_s_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        if 'actor_n' in ckpt:
            self.model.actor_n.load_state_dict(ckpt['actor_n'])
            self.model.actor_s.load_state_dict(ckpt['actor_s'])
            self.model.critic.load_state_dict(ckpt['critic'])
            if 'actor_n_optimizer' in ckpt:
                self.actor_n_optimizer.load_state_dict(ckpt['actor_n_optimizer'])
                self.actor_s_optimizer.load_state_dict(ckpt['actor_s_optimizer'])
                self.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
        elif 'model' in ckpt:
            # 旧格式迁移
            old_state = ckpt['model']
            actor_state = {k[len('actor.'):]: v for k, v in old_state.items()
                           if k.startswith('actor.')}
            critic_state = {k[len('critic.'):]: v for k, v in old_state.items()
                            if k.startswith('critic.')}
            self.model.actor_n.load_state_dict(actor_state)
            self.model.actor_s.load_state_dict(actor_state)
            self.model.critic.load_state_dict(critic_state)

    def state_dict(self) -> dict:
        """返回合并的 state dict, 格式: actor_n.* / actor_s.* / critic.*"""
        d = {}
        for k, v in self.model.actor_n.state_dict().items():
            d[f'actor_n.{k}'] = v
        for k, v in self.model.actor_s.state_dict().items():
            d[f'actor_s.{k}'] = v
        for k, v in self.model.critic.state_dict().items():
            d[f'critic.{k}'] = v
        return d

    def load_state_dict(self, state: dict):
        """加载 state_dict, 兼容新格式 (actor_n.*) 和旧格式 (actor.*)."""
        if any(k.startswith('actor_n.') for k in state):
            actor_n_state = {k[len('actor_n.'):]: v for k, v in state.items()
                             if k.startswith('actor_n.')}
            actor_s_state = {k[len('actor_s.'):]: v for k, v in state.items()
                             if k.startswith('actor_s.')}
            critic_state = {k[len('critic.'):]: v for k, v in state.items()
                            if k.startswith('critic.')}
            self.model.actor_n.load_state_dict(actor_n_state)
            self.model.actor_s.load_state_dict(actor_s_state)
            self.model.critic.load_state_dict(critic_state)
        elif any(k.startswith('actor.') for k in state):
            # 旧格式: 共享 actor → 复制到 actor_n 和 actor_s
            actor_state = {k[len('actor.'):]: v for k, v in state.items()
                           if k.startswith('actor.')}
            critic_state = {k[len('critic.'):]: v for k, v in state.items()
                            if k.startswith('critic.')}
            self.model.actor_n.load_state_dict(actor_state)
            self.model.actor_s.load_state_dict(actor_state)
            self.model.critic.load_state_dict(critic_state)
        else:
            raise ValueError(
                f"Unrecognized state_dict format. Keys sample: {list(state.keys())[:5]}")
