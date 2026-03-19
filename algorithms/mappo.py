"""
MAPPO (Multi-Agent PPO) — HAPPO: 独立 Actor + 独立 Centralized Critic
======================================================================

新架构变更（对齐 301 维 MLP policy_net.py）:

1. MAPPOConfig 新增 hidden_dim=1024（旧的 hand_dim/history_dim/lstm_layers
   已废弃但保留，避免旧代码 dataclass 初始化报错）。

2. _make_actor() 改用 MLPPolicyNetwork(obs_dim=301, hidden_dim)；
   _make_critic() 改用 MLPValueNetwork(obs_dim=301, hidden_dim, centralized=True)。
   两者接口从 (obs_dict, all_hands) 改为 (flat_obs, legal_actions, all_hands)。

3. _HAPPOModel.get_action_and_value 签名对应更新，
   接受 (flat_obs, legal_actions, all_hands, player)。

4. get_action_for_player / _get_action_for_player 接受 flat_obs + legal_actions
   而非 obs_dict；向后兼容旧 get_action() 接口已移除（新代码全用
   SubgameTrainer.FlatRolloutBuffer，不再走 agent.buffers）。

5. MAPPORolloutBuffer 保留（供 agent.update() 向后兼容），但新 SubgameTrainer
   使用自己的 FlatRolloutBuffer，不依赖 agent.buffers。

6. save() / load() 格式不变（actor_n/actor_s/critic_n/critic_s keys）。

7. FSP 相关逻辑移到 utils/fsp_pool.py，此文件不负责 FSP。

兼容性:
    - 旧的 MAPPORolloutBuffer / agent.buffers 保留，不影响已有测试。
    - state_dict() / load_state_dict() 格式向后兼容。
    - MAPPOConfig 中的 hand_dim / history_dim / lstm_layers 保留，
      只是在网络构建时被忽略（无副作用）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from env import NUM_PLAYERS, NORTH, SOUTH
from networks.policy_net import MLPPolicyNetwork, MLPValueNetwork, OBS_DIM
from algorithms.ippo import PPOConfig, RolloutBuffer


# ==============================================================================
# Config
# ==============================================================================

@dataclass
class MAPPOConfig(PPOConfig):
    """
    MAPPO / HAPPO 配置.

    新字段:
        hidden_dim : 4×1024 MLP 的隐藏层宽度（覆盖 PPOConfig 的 hidden_dim=256）

    保留旧字段（不再使用，仅防止旧代码 dataclass 报错）:
        hand_dim, history_dim, lstm_layers — deprecated, ignored in network construction
        belief_dims, fsp_pool_size         — deprecated, managed externally
    """
    # ── 网络 ────────────────────────────────────────────────────────────────
    hidden_dim:         int   = 1024      # 覆盖 PPOConfig.hidden_dim=256
    obs_dim:            int   = OBS_DIM   # 301

    # ── Critic ──────────────────────────────────────────────────────────────
    centralized_critic: bool  = True
    critic_lr_ratio:    float = 3.0

    # ── 向后兼容（不再使用）────────────────────────────────────────────────
    hand_dim:           int   = 256       # deprecated
    history_dim:        int   = 256       # deprecated
    lstm_layers:        int   = 2         # deprecated
    belief_dims:        Optional[dict] = None   # deprecated, managed by SubgameTrainer
    fsp_pool_size:      int   = 0         # deprecated, managed by SubgameTrainer


# ==============================================================================
# MAPPORolloutBuffer（向后兼容，新 SubgameTrainer 不使用此类）
# ==============================================================================

class MAPPORolloutBuffer(RolloutBuffer):
    """
    向后兼容 Buffer（存旧格式 obs_dict）.

    新的 SubgameTrainer 使用 FlatRolloutBuffer，此类仅供旧代码和测试使用.
    """

    def reset(self):
        super().reset()
        self.all_hands = []

    def add(self, obs, action, log_prob, reward, value, done, all_hands=None):
        super().add(obs, action, log_prob, reward, value, done)
        if all_hands is not None:
            self.all_hands.append(torch.tensor(all_hands, dtype=torch.float32))

    def get_batches(self, batch_size: int):
        n       = len(self.actions)
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx      = indices[start:start + batch_size]
            batch_obs = {
                key: torch.stack([self.observations[i][key] for i in idx]).to(self.device)
                for key in self.observations[0].keys()
            }
            batch = {
                'obs':          batch_obs,
                'actions':      torch.stack([self.actions[i]   for i in idx]).to(self.device),
                'old_log_probs':torch.stack([self.log_probs[i] for i in idx]).to(self.device),
                'advantages':   self.advantages[idx].to(self.device),
                'returns':      self.returns[idx].to(self.device),
            }
            if self.all_hands:
                batch['all_hands'] = torch.stack(
                    [self.all_hands[i] for i in idx]).to(self.device)
            yield batch


# ==============================================================================
# _HAPPOModel
# ==============================================================================

class _HAPPOModel(nn.Module):
    """
    HAPPO 模型容器.

    包含 actor_n, actor_s, critic_n, critic_s 四个独立子网络.

    新 API（对齐 MLPPolicyNetwork）:
        actor.get_action(flat_obs, legal_actions, deterministic)
        critic(flat_obs, all_hands)

    向后兼容别名:
        model.actor  → actor_s
        model.critic → critic_s
    """

    def __init__(
        self,
        actor_n:  MLPPolicyNetwork,
        actor_s:  MLPPolicyNetwork,
        critic_n: MLPValueNetwork,
        critic_s: MLPValueNetwork,
    ):
        super().__init__()
        self.actor_n  = actor_n
        self.actor_s  = actor_s
        self.critic_n = critic_n
        self.critic_s = critic_s

        # 向后兼容别名
        self.actor  = actor_s
        self.critic = critic_s

    def get_action_and_value(
        self,
        flat_obs:      torch.Tensor,
        legal_actions: torch.Tensor,
        all_hands:     Optional[torch.Tensor] = None,
        deterministic: bool = False,
        player:        int  = SOUTH,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor  = self.actor_n  if player == NORTH else self.actor_s
        critic = self.critic_n if player == NORTH else self.critic_s

        action, log_prob, entropy = actor.get_action(flat_obs, legal_actions, deterministic)
        value = critic(flat_obs, all_hands)
        return action, log_prob, entropy, value


# ==============================================================================
# MAPPOAgent
# ==============================================================================

class MAPPOAgent:
    """
    HAPPO Multi-Agent PPO: 独立 Actor + 独立 Centralized Critic.

    N-phase: 只更新 actor_n + critic_n; actor_s + critic_s 完全冻结.
    S-phase: 只更新 actor_s + critic_s; actor_n + critic_n 完全冻结.

    对 competitive 子博弈的扩展:
        EW 方在竞叫中也参与决策，但通常由 FSP pool 中的 frozen snapshot 控制,
        不需要为 EW 单独维护 critic（FSP actor only）.
        若需要训练 EW，直接用 get_actor(EAST/WEST) 映射到 actor_n/actor_s.
    """

    def __init__(self, config: MAPPOConfig):
        self.config = config
        self.device = config.device

        obs_dim    = getattr(config, 'obs_dim', OBS_DIM)
        hidden_dim = getattr(config, 'hidden_dim', 1024)

        def _make_actor() -> MLPPolicyNetwork:
            return MLPPolicyNetwork(
                obs_dim    = obs_dim,
                hidden_dim = hidden_dim,
            ).to(self.device)

        def _make_critic() -> MLPValueNetwork:
            return MLPValueNetwork(
                obs_dim     = obs_dim,
                hidden_dim  = hidden_dim,
                centralized = config.centralized_critic,
            ).to(self.device)

        self.model = _HAPPOModel(
            actor_n  = _make_actor(),
            actor_s  = _make_actor(),
            critic_n = _make_critic(),
            critic_s = _make_critic(),
        )

        critic_lr = config.lr * config.critic_lr_ratio

        self.actor_n_optimizer  = torch.optim.Adam(
            self.model.actor_n.parameters(),  lr=config.lr)
        self.actor_s_optimizer  = torch.optim.Adam(
            self.model.actor_s.parameters(),  lr=config.lr)
        self.critic_n_optimizer = torch.optim.Adam(
            self.model.critic_n.parameters(), lr=critic_lr)
        self.critic_s_optimizer = torch.optim.Adam(
            self.model.critic_s.parameters(), lr=critic_lr)

        # 向后兼容别名
        self.actor_optimizer  = self.actor_s_optimizer
        self.critic_optimizer = self.critic_s_optimizer
        self.optimizer        = self.actor_s_optimizer

        # 向后兼容旧 buffer（新代码不使用）
        self.buffers = {p: MAPPORolloutBuffer(self.device) for p in range(NUM_PLAYERS)}

    # ------------------------------------------------------------------
    # Per-player 接口
    # ------------------------------------------------------------------

    def get_actor(self, player: int) -> MLPPolicyNetwork:
        """
        返回对应 player 的 actor.

        映射规则（适配 competitive 4方决策）:
            NORTH (0) → actor_n
            EAST  (1) → actor_n  (FSP 对手用 frozen snapshot，训练时同 N)
            SOUTH (2) → actor_s
            WEST  (3) → actor_s  (FSP 对手用 frozen snapshot，训练时同 S)
        """
        if player % 2 == 0:  # N / S 阵营
            return self.model.actor_n if player == NORTH else self.model.actor_s
        else:                 # E / W 阵营（映射到同侧 actor，FSP 会临时覆盖）
            return self.model.actor_n if player == 1 else self.model.actor_s

    def get_critic(self, player: int) -> MLPValueNetwork:
        if player % 2 == 0:
            return self.model.critic_n if player == NORTH else self.model.critic_s
        else:
            return self.model.critic_n if player == 1 else self.model.critic_s

    def get_actor_optimizer(self, player: int):
        if player % 2 == 0:
            return self.actor_n_optimizer if player == NORTH else self.actor_s_optimizer
        else:
            return self.actor_n_optimizer if player == 1 else self.actor_s_optimizer

    def get_critic_optimizer(self, player: int):
        if player % 2 == 0:
            return self.critic_n_optimizer if player == NORTH else self.critic_s_optimizer
        else:
            return self.critic_n_optimizer if player == 1 else self.critic_s_optimizer

    # ------------------------------------------------------------------
    # Action sampling（新 API：接受 flat_obs + legal_actions）
    # ------------------------------------------------------------------

    def get_action_for_player(
        self,
        flat_obs:      np.ndarray,
        legal_actions: np.ndarray,
        player:        int,
        all_hands:     Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[int, Dict]:
        """
        给定 flat_obs (301,) 和 legal_actions (38,)，返回 (action_int, extras).

        extras: {'log_prob': Tensor, 'value': Tensor}
        """
        flat_t  = torch.tensor(flat_obs,      dtype=torch.float32
                               ).unsqueeze(0).to(self.device)
        legal_t = torch.tensor(legal_actions, dtype=torch.float32
                               ).unsqueeze(0).to(self.device)
        ah_t    = (torch.tensor(all_hands,    dtype=torch.float32
                                ).unsqueeze(0).to(self.device)
                   if all_hands is not None else None)

        actor  = self.get_actor(player)
        critic = self.get_critic(player)

        with torch.no_grad():
            action, log_prob, _ = actor.get_action(flat_t, legal_t, deterministic)
            value               = critic(flat_t, ah_t)

        return action.item(), {
            'log_prob': log_prob.squeeze(0),
            'value':    value.squeeze(0),
        }

    def store_transition(self, player, flat_obs, legal_actions, action,
                         log_prob, reward, value, done, all_hands=None):
        """
        向后兼容接口：存入旧版 MAPPORolloutBuffer.

        新代码（SubgameTrainer）使用 FlatRolloutBuffer，不调此方法.
        """
        # 旧 buffer 存的是 obs_dict，这里做最小兼容：存 flat_obs 的伪装字典
        obs_compat = {
            'flat_obs':      torch.tensor(flat_obs,      dtype=torch.float32),
            'legal_actions': torch.tensor(legal_actions, dtype=torch.float32),
        }
        self.buffers[player].add(
            obs_compat, torch.tensor(action), log_prob, reward, value, done, all_hands)

    # ------------------------------------------------------------------
    # Standard update（向后兼容，新 SubgameTrainer 不调此方法）
    # ------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        """
        标准 PPO update（旧接口，新代码用 SubgameTrainer._safe_update）.

        注: 旧 buffer 里存的是 obs_dict，这里直接取 flat_obs / legal_actions.
        若 buffer 里没有这两个 key，会 KeyError — 这是预期行为（提醒切换新接口）.
        """
        total_loss = total_policy = total_value = total_entropy = num_updates = 0
        for player in range(NUM_PLAYERS):
            buffer = self.buffers[player]
            if not buffer.actions:
                continue

            actor      = self.get_actor(player)
            critic     = self.get_critic(player)
            actor_opt  = self.get_actor_optimizer(player)
            critic_opt = self.get_critic_optimizer(player)

            last_obs     = buffer.observations[-1]
            flat_obs     = last_obs['flat_obs'].unsqueeze(0).to(self.device)
            legal_obs    = last_obs['legal_actions'].unsqueeze(0).to(self.device)
            last_ah      = (buffer.all_hands[-1].unsqueeze(0).to(self.device)
                            if buffer.all_hands else None)

            with torch.no_grad():
                last_value = critic(flat_obs, last_ah).item()

            buffer.compute_returns_and_advantages(
                last_value, self.config.gamma, self.config.gae_lambda)

            for _ in range(self.config.num_epochs):
                for batch in buffer.get_batches(self.config.batch_size):
                    b_flat  = batch['obs']['flat_obs']
                    b_legal = batch['obs']['legal_actions']
                    b_act   = batch['actions']
                    b_old   = batch['old_log_probs']
                    b_adv   = batch['advantages']
                    b_ret   = batch['returns']
                    b_ah    = batch.get('all_hands')

                    adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
                    log_probs, entropy = actor.evaluate_actions(b_flat, b_legal, b_act)
                    ratio = torch.exp(log_probs - b_old)
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

                    vals   = critic(b_flat, b_ah)
                    old_v  = vals.detach()
                    v_clip = old_v + (vals - old_v).clamp(
                        -self.config.clip_ratio, self.config.clip_ratio)
                    value_loss = torch.max(F.mse_loss(vals, b_ret),
                                           F.mse_loss(v_clip, b_ret))

                    critic_opt.zero_grad()
                    value_loss.backward()
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
        return {
            'loss':        total_loss    / num_updates,
            'policy_loss': total_policy  / num_updates,
            'value_loss':  total_value   / num_updates,
            'entropy':     total_entropy / num_updates,
        }

    # ------------------------------------------------------------------
    # Serialization（格式与旧版兼容）
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            'actor_n':              self.model.actor_n.state_dict(),
            'actor_s':              self.model.actor_s.state_dict(),
            'critic_n':             self.model.critic_n.state_dict(),
            'critic_s':             self.model.critic_s.state_dict(),
            'actor_n_optimizer':    self.actor_n_optimizer.state_dict(),
            'actor_s_optimizer':    self.actor_s_optimizer.state_dict(),
            'critic_n_optimizer':   self.critic_n_optimizer.state_dict(),
            'critic_s_optimizer':   self.critic_s_optimizer.state_dict(),
            # 元数据，便于检查
            'obs_dim':    getattr(self.config, 'obs_dim', OBS_DIM),
            'hidden_dim': getattr(self.config, 'hidden_dim', 1024),
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
            # 旧格式（共享 actor/critic → 复制到 N 和 S）
            old = ckpt['model']
            actor_state  = {k[len('actor.'):]:  v for k, v in old.items()
                            if k.startswith('actor.')}
            critic_state = {k[len('critic.'): ]: v for k, v in old.items()
                            if k.startswith('critic.')}
            self.model.actor_n.load_state_dict(actor_state)
            self.model.actor_s.load_state_dict(actor_state)
            self.model.critic_n.load_state_dict(critic_state)
            self.model.critic_s.load_state_dict(critic_state)

        else:
            raise ValueError(
                f"Unrecognized checkpoint format. Keys: {list(ckpt.keys())[:8]}")

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
                return {k[len(prefix)+1:]: v
                        for k, v in state.items() if k.startswith(prefix+'.')}
            self.model.actor_n.load_state_dict(_extract('actor_n'))
            self.model.actor_s.load_state_dict(_extract('actor_s'))
            self.model.critic_n.load_state_dict(_extract('critic_n'))
            self.model.critic_s.load_state_dict(_extract('critic_s'))
        elif any(k.startswith('actor.') for k in state):
            # 旧格式：共享网络 → 复制到 N 和 S
            actor_state  = {k[len('actor.'):]:  v for k, v in state.items()
                            if k.startswith('actor.')}
            critic_state = {k[len('critic.'): ]: v for k, v in state.items()
                            if k.startswith('critic.')}
            self.model.actor_n.load_state_dict(actor_state)
            self.model.actor_s.load_state_dict(actor_state)
            self.model.critic_n.load_state_dict(critic_state)
            self.model.critic_s.load_state_dict(critic_state)
        else:
            raise ValueError(
                f"Unrecognized state_dict format. Keys: {list(state.keys())[:5]}")
