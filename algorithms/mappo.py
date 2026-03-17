"""
MAPPO (Multi-Agent PPO) — HAPPO: 独立 Actor + 独立 Centralized Critic
======================================================================

架构改动 (P49): 从共享 Critic 改为双独立 Critic.

P48 的问题:
  共享 Critic 在 S-phase 用 S 的状态数据更新, 切到 N-phase 时
  Critic 面对完全不同的状态分布 → 灾难性遗忘 → vl 爆到 2000+.
  根本原因是 N (round1/3) 和 S (round2) 的状态分布在叫牌树上
  处于完全不同的深度, 一个 Critic 无法同时服务两个分布.

P49 解法 (Gemini 建议 + HAPPO 标准操作):
  N 和 S 各自拥有独立的 critic (critic_n, critic_s).
  两个 critic 均接受全局状态 (all_hands + obs), 满足 CTDE 要求.
  S-phase: critic_s 更新, critic_n 完全冻结.
  N-phase: critic_n 更新, critic_s 完全冻结.
  切换 phase 时 critic 从不相互污染 → vl 平滑衔接上轮状态.

接口:
  - agent.get_actor(player)          → actor_n / actor_s
  - agent.get_critic(player)         → critic_n / critic_s
  - agent.get_actor_optimizer(player)
  - agent.get_critic_optimizer(player)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from env import NUM_PLAYERS, NORTH, SOUTH
from networks.policy_net import PolicyNetwork, ValueNetwork
from algorithms.ippo import PPOConfig, RolloutBuffer


@dataclass
class MAPPOConfig(PPOConfig):
    """MAPPO 配置"""
    centralized_critic: bool = True
    critic_lr_ratio: float = 3.0
    # belief_dim > 0: 为对应 player 的 actor 注入 BeliefNetwork 推断特征.
    # 按 player 配置: {NORTH: 0, SOUTH: 48, EAST: 0, WEST: 0} 等.
    # 默认全0 (向后兼容). 实验入口在 _make_sub_config 里按需传入.
    belief_dims: Dict[int, int] = None   # player → belief_dim; None=全部0


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


class _HAPPOModel(nn.Module):
    """
    HAPPO 模型容器.
    包含 actor_n, actor_s, critic_n, critic_s 四个独立子网络.
    向后兼容: model.actor → actor_s, model.critic → critic_s.
    """

    def __init__(self, actor_n: PolicyNetwork, actor_s: PolicyNetwork,
                 critic_n: ValueNetwork, critic_s: ValueNetwork):
        super().__init__()
        self.actor_n  = actor_n
        self.actor_s  = actor_s
        self.critic_n = critic_n
        self.critic_s = critic_s

        # 向后兼容别名
        self.actor  = actor_s
        self.critic = critic_s   # 旧代码 agent.model.critic 默认指向 critic_s

    def get_action_and_value(self, obs, all_hands=None, deterministic=False,
                             player: int = SOUTH):
        actor  = self.actor_n  if player == NORTH else self.actor_s
        critic = self.critic_n if player == NORTH else self.critic_s
        action, log_prob, entropy = actor.get_action(obs, deterministic)
        value = critic(obs, all_hands)
        return action, log_prob, entropy, value


class MAPPOAgent:
    """
    HAPPO Multi-Agent PPO: 独立 Actor + 独立 Centralized Critic.

    N-phase: 只更新 actor_n + critic_n; actor_s + critic_s 完全冻结.
    S-phase: 只更新 actor_s + critic_s; actor_n + critic_n 完全冻结.
    """

    def __init__(self, config: MAPPOConfig):
        self.config = config
        self.device = config.device

        # belief_dims: player → belief_dim (None = 全部0, 向后兼容)
        _belief_dims = config.belief_dims or {}

        def _make_actor(player: int):
            bdim = _belief_dims.get(player, 0)
            return PolicyNetwork(
                hand_dim=config.hand_dim,
                history_dim=config.history_dim,
                hidden_dim=config.hidden_dim,
                belief_dim=bdim,
            ).to(self.device)

        def _make_critic():
            return ValueNetwork(
                hand_dim=config.hand_dim,
                history_dim=config.history_dim,
                hidden_dim=config.hidden_dim,
                centralized=True,
            ).to(self.device)

        self.model = _HAPPOModel(
            actor_n=_make_actor(NORTH), actor_s=_make_actor(SOUTH),
            critic_n=_make_critic(), critic_s=_make_critic(),
        ).to(self.device)

        self.actor_n_optimizer  = torch.optim.Adam(self.model.actor_n.parameters(),  lr=config.lr)
        self.actor_s_optimizer  = torch.optim.Adam(self.model.actor_s.parameters(),  lr=config.lr)
        self.critic_n_optimizer = torch.optim.Adam(self.model.critic_n.parameters(), lr=config.lr * config.critic_lr_ratio)
        self.critic_s_optimizer = torch.optim.Adam(self.model.critic_s.parameters(), lr=config.lr * config.critic_lr_ratio)

        # 向后兼容
        self.actor_optimizer  = self.actor_s_optimizer
        self.critic_optimizer = self.critic_s_optimizer
        self.optimizer        = self.actor_s_optimizer

        self.buffers = {p: MAPPORolloutBuffer(self.device) for p in range(NUM_PLAYERS)}

    # ------------------------------------------------------------------
    # Per-player 接口
    # ------------------------------------------------------------------

    def get_actor(self, player: int) -> PolicyNetwork:
        return self.model.actor_n if player == NORTH else self.model.actor_s

    def get_critic(self, player: int) -> ValueNetwork:
        return self.model.critic_n if player == NORTH else self.model.critic_s

    def get_actor_optimizer(self, player: int):
        return self.actor_n_optimizer if player == NORTH else self.actor_s_optimizer

    def get_critic_optimizer(self, player: int):
        return self.critic_n_optimizer if player == NORTH else self.critic_s_optimizer

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------

    def get_action(self, obs: Dict[str, np.ndarray], all_hands=None,
                   deterministic: bool = False) -> Tuple[int, Dict]:
        """向后兼容接口, 默认用 actor_s/critic_s."""
        return self._get_action_for_player(obs, SOUTH, all_hands, deterministic)

    def get_action_for_player(self, obs: Dict[str, np.ndarray], player: int,
                               all_hands=None,
                               deterministic: bool = False) -> Tuple[int, Dict]:
        return self._get_action_for_player(obs, player, all_hands, deterministic)

    def _get_action_for_player(self, obs, player, all_hands, deterministic):
        obs_t = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device)
                 for k, v in obs.items()}
        all_h_t = (torch.tensor(all_hands, dtype=torch.float32).unsqueeze(0).to(self.device)
                   if all_hands is not None else None)
        actor  = self.get_actor(player)
        critic = self.get_critic(player)
        with torch.no_grad():
            action, log_prob, _ = actor.get_action(obs_t, deterministic)
            value = critic(obs_t, all_h_t)
        return action.item(), {'log_prob': log_prob.squeeze(0), 'value': value.squeeze(0)}

    def store_transition(self, player, obs, action, log_prob, reward, value, done,
                         all_hands=None):
        obs_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in obs.items()}
        self.buffers[player].add(obs_t, torch.tensor(action), log_prob, reward,
                                 value, done, all_hands)

    # ------------------------------------------------------------------
    # Standard update (向后兼容, SubgameTrainer.safe_update 优先)
    # ------------------------------------------------------------------

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
                        torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * adv
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

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            'actor_n':  self.model.actor_n.state_dict(),
            'actor_s':  self.model.actor_s.state_dict(),
            'critic_n': self.model.critic_n.state_dict(),
            'critic_s': self.model.critic_s.state_dict(),
            'actor_n_optimizer':  self.actor_n_optimizer.state_dict(),
            'actor_s_optimizer':  self.actor_s_optimizer.state_dict(),
            'critic_n_optimizer': self.critic_n_optimizer.state_dict(),
            'critic_s_optimizer': self.critic_s_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        if 'actor_n' in ckpt:
            self.model.actor_n.load_state_dict(ckpt['actor_n'])
            self.model.actor_s.load_state_dict(ckpt['actor_s'])
            self.model.critic_n.load_state_dict(ckpt['critic_n'])
            self.model.critic_s.load_state_dict(ckpt['critic_s'])
            for key in ('actor_n_optimizer', 'actor_s_optimizer',
                        'critic_n_optimizer', 'critic_s_optimizer'):
                if key in ckpt:
                    getattr(self, key).load_state_dict(ckpt[key])
        elif 'model' in ckpt:
            # 旧格式迁移: 共享 actor/critic → 复制到 N 和 S
            old = ckpt['model']
            actor_state  = {k[len('actor.'):]:  v for k, v in old.items() if k.startswith('actor.')}
            critic_state = {k[len('critic.'): ]: v for k, v in old.items() if k.startswith('critic.')}
            self.model.actor_n.load_state_dict(actor_state)
            self.model.actor_s.load_state_dict(actor_state)
            self.model.critic_n.load_state_dict(critic_state)
            self.model.critic_s.load_state_dict(critic_state)

    def state_dict(self) -> dict:
        """合并 state dict: actor_n.* / actor_s.* / critic_n.* / critic_s.*"""
        d = {}
        for prefix, net in [('actor_n',  self.model.actor_n),
                             ('actor_s',  self.model.actor_s),
                             ('critic_n', self.model.critic_n),
                             ('critic_s', self.model.critic_s)]:
            for k, v in net.state_dict().items():
                d[f'{prefix}.{k}'] = v
        return d

    def load_state_dict(self, state: dict):
        """兼容新格式 (actor_n.*/critic_n.*) 和旧格式 (actor.*/critic.*)."""
        if any(k.startswith('actor_n.') for k in state):
            def _extract(prefix):
                return {k[len(prefix)+1:]: v for k, v in state.items() if k.startswith(prefix+'.')}
            self.model.actor_n.load_state_dict(_extract('actor_n'))
            self.model.actor_s.load_state_dict(_extract('actor_s'))
            self.model.critic_n.load_state_dict(_extract('critic_n'))
            self.model.critic_s.load_state_dict(_extract('critic_s'))
        elif any(k.startswith('actor.') for k in state):
            # 旧格式: 共享网络 → 复制到 N 和 S
            actor_state  = {k[len('actor.'):]:  v for k, v in state.items() if k.startswith('actor.')}
            critic_state = {k[len('critic.'): ]: v for k, v in state.items() if k.startswith('critic.')}
            self.model.actor_n.load_state_dict(actor_state)
            self.model.actor_s.load_state_dict(actor_state)
            self.model.critic_n.load_state_dict(critic_state)
            self.model.critic_s.load_state_dict(critic_state)
        else:
            raise ValueError(f"Unrecognized state_dict format. Keys: {list(state.keys())[:5]}")
