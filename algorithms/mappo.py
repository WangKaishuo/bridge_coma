"""
MAPPO (Multi-Agent PPO) — HAPPO + FSP
======================================

P52 变更:
- PolicyNetwork / ValueNetwork 换成 MLP+全局拼接版本 (无 LSTM)
- 加入 Fictitious Self-Play (FSP) checkpoint pool
  FSP 原理: 非 active player 从历史 checkpoint pool 中均匀采样策略执行,
  而不是总用最新策略. 防止 policy cycling (交替训练的常见失效模式).
- MAPPOConfig 新增 fsp_pool_size (默认 10)
"""

import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field

from env import NUM_PLAYERS, NORTH, SOUTH
from networks.policy_net import PolicyNetwork, ValueNetwork
from algorithms.ippo import PPOConfig, RolloutBuffer


@dataclass
class MAPPOConfig(PPOConfig):
    """MAPPO 配置"""
    centralized_critic: bool = True
    critic_lr_ratio: float = 3.0
    # belief_dims: player → belief_dim; None = 全部0 (向后兼容)
    belief_dims: Dict[int, int] = None
    # FSP: checkpoint pool 最大容量. 0 = 关闭 FSP (纯 latest-policy self-play)
    fsp_pool_size: int = 10


class MAPPORolloutBuffer(RolloutBuffer):
    """MAPPO 缓冲区 (额外存储所有手牌)"""

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
            batch_obs = {
                key: torch.stack([self.observations[i][key] for i in batch_idx]).to(self.device)
                for key in self.observations[0].keys()
            }
            batch = {
                'obs': batch_obs,
                'actions': torch.stack([self.actions[i] for i in batch_idx]).to(self.device),
                'old_log_probs': torch.stack([self.log_probs[i] for i in batch_idx]).to(self.device),
                'advantages': self.advantages[batch_idx].to(self.device),
                'returns': self.returns[batch_idx].to(self.device),
            }
            if self.all_hands:
                batch['all_hands'] = torch.stack(
                    [self.all_hands[i] for i in batch_idx]).to(self.device)
            yield batch


class FSPPool:
    """
    Fictitious Self-Play checkpoint pool.

    维护最近 pool_size 个 actor 快照.
    非 active player 调用 sample_actor() 从 pool 中均匀采样一个历史策略执行.
    这样非 active player 的行为来自"过去自己的平均", 防止 policy cycling.
    """

    def __init__(self, pool_size: int, device: str):
        self.pool_size = pool_size
        self.device = device
        self._pool: List[Dict] = []   # list of state_dicts

    def push(self, actor_n_state: dict, actor_s_state: dict):
        """把当前 N/S actor 快照加入 pool."""
        if self.pool_size <= 0:
            return
        entry = {
            NORTH: copy.deepcopy(actor_n_state),
            SOUTH: copy.deepcopy(actor_s_state),
        }
        self._pool.append(entry)
        if len(self._pool) > self.pool_size:
            self._pool.pop(0)

    def sample_actor_state(self, player: int) -> Optional[dict]:
        """从 pool 中均匀采样一个历史 actor state_dict.
        pool 为空时返回 None (caller 用最新策略)."""
        if not self._pool:
            return None
        entry = random.choice(self._pool)
        return entry[player]

    def __len__(self):
        return len(self._pool)


class _HAPPOModel(nn.Module):
    """HAPPO 模型容器 (P52: MLP 版)."""

    def __init__(self, actor_n: PolicyNetwork, actor_s: PolicyNetwork,
                 critic_n: ValueNetwork, critic_s: ValueNetwork):
        super().__init__()
        self.actor_n  = actor_n
        self.actor_s  = actor_s
        self.critic_n = critic_n
        self.critic_s = critic_s
        # 向后兼容别名
        self.actor  = actor_s
        self.critic = critic_s

    def get_action_and_value(self, obs, all_hands=None, deterministic=False,
                             player: int = SOUTH):
        actor  = self.actor_n  if player == NORTH else self.actor_s
        critic = self.critic_n if player == NORTH else self.critic_s
        action, log_prob, entropy = actor.get_action(obs, deterministic)
        value = critic(obs, all_hands)
        return action, log_prob, entropy, value


class MAPPOAgent:
    """
    HAPPO Multi-Agent PPO (P52):
    - MLP+全局拼接 actor/critic (无 LSTM)
    - FSP checkpoint pool (防 policy cycling)
    """

    def __init__(self, config: MAPPOConfig):
        self.config = config
        self.device = config.device

        _belief_dims = config.belief_dims or {}

        def _make_actor(player: int) -> PolicyNetwork:
            bdim = _belief_dims.get(player, 0)
            return PolicyNetwork(
                hidden_dim=1024,
                num_layers=4,
                belief_dim=bdim,
            ).to(self.device)

        def _make_critic() -> ValueNetwork:
            return ValueNetwork(
                hidden_dim=1024,
                num_layers=4,
                centralized=True,
            ).to(self.device)

        self.model = _HAPPOModel(
            actor_n=_make_actor(NORTH), actor_s=_make_actor(SOUTH),
            critic_n=_make_critic(),    critic_s=_make_critic(),
        ).to(self.device)

        lr = config.lr
        clr = config.lr * config.critic_lr_ratio
        self.actor_n_optimizer  = torch.optim.Adam(self.model.actor_n.parameters(),  lr=lr)
        self.actor_s_optimizer  = torch.optim.Adam(self.model.actor_s.parameters(),  lr=lr)
        self.critic_n_optimizer = torch.optim.Adam(self.model.critic_n.parameters(), lr=clr)
        self.critic_s_optimizer = torch.optim.Adam(self.model.critic_s.parameters(), lr=clr)

        # 向后兼容
        self.actor_optimizer  = self.actor_s_optimizer
        self.critic_optimizer = self.critic_s_optimizer
        self.optimizer        = self.actor_s_optimizer

        self.buffers = {p: MAPPORolloutBuffer(self.device) for p in range(NUM_PLAYERS)}

        # FSP pool
        fsp_size = getattr(config, 'fsp_pool_size', 10)
        self.fsp_pool = FSPPool(pool_size=fsp_size, device=self.device)

    # ── Per-player 接口 ───────────────────────────────────────────────────

    def get_actor(self, player: int) -> PolicyNetwork:
        return self.model.actor_n if player == NORTH else self.model.actor_s

    def get_critic(self, player: int) -> ValueNetwork:
        return self.model.critic_n if player == NORTH else self.model.critic_s

    def get_actor_optimizer(self, player: int):
        return self.actor_n_optimizer if player == NORTH else self.actor_s_optimizer

    def get_critic_optimizer(self, player: int):
        return self.critic_n_optimizer if player == NORTH else self.critic_s_optimizer

    # ── FSP ──────────────────────────────────────────────────────────────

    def fsp_push(self):
        """将当前 actor_n / actor_s 快照加入 FSP pool."""
        self.fsp_pool.push(
            self.model.actor_n.state_dict(),
            self.model.actor_s.state_dict(),
        )

    def get_fsp_actor(self, player: int) -> Optional[PolicyNetwork]:
        """
        返回一个临时 actor (从 FSP pool 采样的历史策略).
        pool 为空时返回 None, caller 用最新策略.
        """
        state = self.fsp_pool.sample_actor_state(player)
        if state is None:
            return None
        # 创建同结构网络并加载历史权重
        actor = PolicyNetwork(
            hidden_dim=1024,
            num_layers=4,
            belief_dim=self.get_actor(player).belief_dim,
        ).to(self.device)
        actor.load_state_dict(state)
        actor.eval()
        return actor

    # ── Action sampling ──────────────────────────────────────────────────

    def get_action(self, obs, all_hands=None, deterministic=False):
        return self._get_action_for_player(obs, SOUTH, all_hands, deterministic)

    def get_action_for_player(self, obs, player: int, all_hands=None,
                               deterministic=False):
        return self._get_action_for_player(obs, player, all_hands, deterministic)

    def _get_action_for_player(self, obs, player, all_hands, deterministic,
                                actor_override=None):
        obs_t = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device)
                 for k, v in obs.items()}
        all_h_t = (torch.tensor(all_hands, dtype=torch.float32).unsqueeze(0).to(self.device)
                   if all_hands is not None else None)
        actor  = actor_override if actor_override is not None else self.get_actor(player)
        critic = self.get_critic(player)
        with torch.no_grad():
            action, log_prob, _ = actor.get_action(obs_t, deterministic)
            value = critic(obs_t, all_h_t)
        return action.item(), {'log_prob': log_prob.squeeze(0), 'value': value.squeeze(0)}

    def get_action_for_player_fsp(self, obs, player: int, all_hands=None,
                                   deterministic=False):
        """
        FSP 版 action: 从 pool 采样历史策略.
        用于非 active player 的 rollout 执行.
        pool 为空时退化为最新策略.
        """
        fsp_actor = self.get_fsp_actor(player)
        return self._get_action_for_player(obs, player, all_hands, deterministic,
                                           actor_override=fsp_actor)

    def store_transition(self, player, obs, action, log_prob, reward, value, done,
                         all_hands=None):
        obs_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in obs.items()}
        self.buffers[player].add(obs_t, torch.tensor(action), log_prob, reward,
                                 value, done, all_hands)

    # ── Standard update (向后兼容) ─────────────────────────────────────────

    def update(self) -> Dict[str, float]:
        total_loss = total_policy = total_value = total_entropy = num_updates = 0
        for player in range(NUM_PLAYERS):
            buffer = self.buffers[player]
            if not buffer.actions:
                continue
            actor  = self.get_actor(player)
            critic = self.get_critic(player)
            actor_opt  = self.get_actor_optimizer(player)
            critic_opt = self.get_critic_optimizer(player)

            with torch.no_grad():
                last_obs  = {k: v.unsqueeze(0).to(self.device) for k, v in buffer.observations[-1].items()}
                last_h    = buffer.all_hands[-1].unsqueeze(0).to(self.device) if buffer.all_hands else None
                last_value = critic(last_obs, last_h).item()
            buffer.compute_returns_and_advantages(last_value, self.config.gamma, self.config.gae_lambda)

            for _ in range(self.config.num_epochs):
                for batch in buffer.get_batches(self.config.batch_size):
                    log_probs, entropy = actor.evaluate_actions(batch['obs'], batch['actions'])
                    ratio = torch.exp(log_probs - batch['old_log_probs'])
                    adv = (batch['advantages'] - batch['advantages'].mean()) / (batch['advantages'].std() + 1e-8)
                    policy_loss = -torch.min(
                        ratio * adv,
                        torch.clamp(ratio, 1-self.config.clip_ratio, 1+self.config.clip_ratio) * adv
                    ).mean()
                    actor_loss = policy_loss - self.config.entropy_coef * entropy.mean()
                    actor_opt.zero_grad(); actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), self.config.max_grad_norm)
                    actor_opt.step()

                    values = critic(batch['obs'], batch.get('all_hands'))
                    old_v  = values.detach()
                    v_clip = old_v + (values - old_v).clamp(-self.config.clip_ratio, self.config.clip_ratio)
                    value_loss = torch.max(F.mse_loss(values, batch['returns']),
                                          F.mse_loss(v_clip,  batch['returns']))
                    critic_opt.zero_grad(); value_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), self.config.max_grad_norm)
                    critic_opt.step()

                    total_loss    += actor_loss.item() + value_loss.item()
                    total_policy  += policy_loss.item()
                    total_value   += value_loss.item()
                    total_entropy += entropy.mean().item()
                    num_updates   += 1
            buffer.reset()

        if num_updates == 0:
            return {}
        return {'loss': total_loss/num_updates, 'policy_loss': total_policy/num_updates,
                'value_loss': total_value/num_updates, 'entropy': total_entropy/num_updates}

    # ── Serialization ─────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            'actor_n':  self.model.actor_n.state_dict(),
            'actor_s':  self.model.actor_s.state_dict(),
            'critic_n': self.model.critic_n.state_dict(),
            'critic_s': self.model.critic_s.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        if 'actor_n' in ckpt:
            self.model.actor_n.load_state_dict(ckpt['actor_n'])
            self.model.actor_s.load_state_dict(ckpt['actor_s'])
            self.model.critic_n.load_state_dict(ckpt['critic_n'])
            self.model.critic_s.load_state_dict(ckpt['critic_s'])

    def state_dict(self) -> dict:
        d = {}
        for prefix, net in [('actor_n',  self.model.actor_n),
                             ('actor_s',  self.model.actor_s),
                             ('critic_n', self.model.critic_n),
                             ('critic_s', self.model.critic_s)]:
            for k, v in net.state_dict().items():
                d[f'{prefix}.{k}'] = v
        return d

    @staticmethod
    def _load_actor_state(actor: PolicyNetwork, state_dict: dict):
        """兼容 belief_dim 扩展的 state_dict 加载 (支持双向尺寸不匹配)."""
        current_sd = actor.state_dict()
        new_sd = {}
        for k, ckpt_v in state_dict.items():
            if k not in current_sd:
                continue
            curr_v = current_sd[k]
            if curr_v.shape == ckpt_v.shape:
                new_sd[k] = ckpt_v
            elif (curr_v.dim() == 2 and ckpt_v.dim() == 2
                  and curr_v.shape[0] == ckpt_v.shape[0]):
                if ckpt_v.shape[1] > curr_v.shape[1]:
                    # 大→小: 截断
                    new_sd[k] = ckpt_v[:, :curr_v.shape[1]]
                else:
                    # 小→大: 新增列保持随机初始化
                    merged = curr_v.clone()
                    merged[:, :ckpt_v.shape[1]] = ckpt_v
                    new_sd[k] = merged
            else:
                import warnings
                warnings.warn(f'_load_actor_state: skipping {k} '
                              f'(ckpt {ckpt_v.shape} vs current {curr_v.shape})')
                new_sd[k] = curr_v
        actor.load_state_dict(new_sd, strict=False)

    def load_state_dict(self, state: dict):
        """兼容新格式 (actor_n.*/critic_n.*) 和旧格式 (actor.*/critic.*)."""
        if any(k.startswith('actor_n.') for k in state):
            def _extract(prefix):
                return {k[len(prefix)+1:]: v for k, v in state.items()
                        if k.startswith(prefix+'.')}
            self._load_actor_state(self.model.actor_n, _extract('actor_n'))
            self._load_actor_state(self.model.actor_s, _extract('actor_s'))
            self.model.critic_n.load_state_dict(_extract('critic_n'))
            self.model.critic_s.load_state_dict(_extract('critic_s'))
        elif any(k.startswith('actor.') for k in state):
            actor_state  = {k[len('actor.'):]:  v for k, v in state.items() if k.startswith('actor.')}
            critic_state = {k[len('critic.'): ]: v for k, v in state.items() if k.startswith('critic.')}
            self._load_actor_state(self.model.actor_n, actor_state)
            self._load_actor_state(self.model.actor_s, actor_state)
            self.model.critic_n.load_state_dict(critic_state)
            self.model.critic_s.load_state_dict(critic_state)
        else:
            raise ValueError(f"Unrecognized state_dict format. Keys: {list(state.keys())[:5]}")
