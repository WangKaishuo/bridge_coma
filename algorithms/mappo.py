"""See the formal README for the current behavior contract."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from env import NUM_PLAYERS, NORTH, SOUTH
from networks.policy_net import (
    ACTION_MAPPING_VERSION,
    MLPPolicyNetwork,
    MLPValueNetwork,
    OBS_DIM,
    BELIEF_OBS_DIM,
)
from algorithms.ppo import PPOConfig, RolloutBuffer


# ==============================================================================
# Config
# ==============================================================================

@dataclass
class MAPPOConfig(PPOConfig):
    """See the formal README for the current behavior contract."""
    hidden_dim:         int   = 1024
    obs_dim:            int   = OBS_DIM   # 480 (P104)
    actor_belief_conditioned: bool = False
    actor_belief_hidden_dim: Optional[int] = None

    # -- Critic --------------------------------------------------------------
    centralized_critic: bool  = True
    critic_lr_ratio:    float = 3.0

    hand_dim:           int   = 256       # deprecated
    history_dim:        int   = 256       # deprecated
    lstm_layers:        int   = 2         # deprecated
    belief_dims:        Optional[dict] = None   # deprecated, managed by SubgameTrainer
    fsp_pool_size:      int   = 0         # deprecated, managed by SubgameTrainer


# ==============================================================================
# ==============================================================================

class MAPPORolloutBuffer(RolloutBuffer):
    """See the formal README for the current behavior contract."""

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
    """See the formal README for the current behavior contract."""

    def __init__(
        self,
        actor_n:  MLPPolicyNetwork, actor_s:  MLPPolicyNetwork,
        actor_e:  MLPPolicyNetwork, actor_w:  MLPPolicyNetwork,
        critic_n: MLPValueNetwork,  critic_s: MLPValueNetwork,
        critic_e: MLPValueNetwork,  critic_w: MLPValueNetwork,
    ):
        super().__init__()
        self.actor_n  = actor_n;  self.actor_s  = actor_s
        self.actor_e  = actor_e;  self.actor_w  = actor_w
        self.critic_n = critic_n; self.critic_s = critic_s
        self.critic_e = critic_e; self.critic_w = critic_w
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
        actor  = self._actor_for(player)
        critic = self._critic_for(player)
        action, log_prob, entropy = actor.get_action(flat_obs, legal_actions, deterministic)
        value = critic(flat_obs, all_hands)
        return action, log_prob, entropy, value

    def _actor_for(self, player: int) -> MLPPolicyNetwork:
        return {0: self.actor_n, 1: self.actor_e,
                2: self.actor_s, 3: self.actor_w}[player]

    def _critic_for(self, player: int) -> MLPValueNetwork:
        return {0: self.critic_n, 1: self.critic_e,
                2: self.critic_s, 3: self.critic_w}[player]


# ==============================================================================
# MAPPOAgent
# ==============================================================================

class MAPPOAgent:
    """See the formal README for the current behavior contract."""

    def __init__(self, config: MAPPOConfig):
        self.config = config
        self.device = config.device

        obs_dim    = getattr(config, 'obs_dim', OBS_DIM)
        hidden_dim = getattr(config, 'hidden_dim', 1024)

        def _make_actor() -> MLPPolicyNetwork:
            return MLPPolicyNetwork(
                obs_dim    = obs_dim,
                hidden_dim = hidden_dim,
                belief_conditioned=config.actor_belief_conditioned,
                belief_hidden_dim=config.actor_belief_hidden_dim,
            ).to(self.device)

        def _make_critic() -> MLPValueNetwork:
            return MLPValueNetwork(
                obs_dim     = obs_dim,
                hidden_dim  = hidden_dim,
                centralized = config.centralized_critic,
            ).to(self.device)

        self.model = _HAPPOModel(
            actor_n=_make_actor(), actor_s=_make_actor(),
            actor_e=_make_actor(), actor_w=_make_actor(),
            critic_n=_make_critic(), critic_s=_make_critic(),
            critic_e=_make_critic(), critic_w=_make_critic(),
        )

        clr = config.lr * config.critic_lr_ratio
        self.actor_n_optimizer  = torch.optim.Adam(self.model.actor_n.parameters(),  lr=config.lr)
        self.actor_s_optimizer  = torch.optim.Adam(self.model.actor_s.parameters(),  lr=config.lr)
        self.actor_e_optimizer  = torch.optim.Adam(self.model.actor_e.parameters(),  lr=config.lr)
        self.actor_w_optimizer  = torch.optim.Adam(self.model.actor_w.parameters(),  lr=config.lr)
        self.critic_n_optimizer = torch.optim.Adam(self.model.critic_n.parameters(), lr=clr)
        self.critic_s_optimizer = torch.optim.Adam(self.model.critic_s.parameters(), lr=clr)
        self.critic_e_optimizer = torch.optim.Adam(self.model.critic_e.parameters(), lr=clr)
        self.critic_w_optimizer = torch.optim.Adam(self.model.critic_w.parameters(), lr=clr)

        self.actor_optimizer  = self.actor_s_optimizer
        self.critic_optimizer = self.critic_s_optimizer
        self.optimizer        = self.actor_s_optimizer

        self.buffers = {p: MAPPORolloutBuffer(self.device) for p in range(NUM_PLAYERS)}

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def get_actor(self, player: int) -> MLPPolicyNetwork:
        return self.model._actor_for(player)

    def get_critic(self, player: int) -> MLPValueNetwork:
        return self.model._critic_for(player)

    def get_actor_optimizer(self, player: int):
        return {0: self.actor_n_optimizer, 1: self.actor_e_optimizer,
                2: self.actor_s_optimizer, 3: self.actor_w_optimizer}[player]

    def get_critic_optimizer(self, player: int):
        return {0: self.critic_n_optimizer, 1: self.critic_e_optimizer,
                2: self.critic_s_optimizer, 3: self.critic_w_optimizer}[player]

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def get_action_for_player(
        self,
        flat_obs:      np.ndarray,
        legal_actions: np.ndarray,
        player:        int,
        all_hands:     Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[int, Dict]:
        """See the formal README for the current behavior contract."""
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
        """See the formal README for the current behavior contract."""
        obs_compat = {
            'flat_obs':      torch.tensor(flat_obs,      dtype=torch.float32),
            'legal_actions': torch.tensor(legal_actions, dtype=torch.float32),
        }
        self.buffers[player].add(
            obs_compat, torch.tensor(action), log_prob, reward, value, done, all_hands)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        """See the formal README for the current behavior contract."""
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
    # ------------------------------------------------------------------

    def checkpoint_dict(self) -> dict:
        """Return the complete trainable state, including all optimizers."""
        return {
            'actor_n': self.model.actor_n.state_dict(),
            'actor_s': self.model.actor_s.state_dict(),
            'actor_e': self.model.actor_e.state_dict(),
            'actor_w': self.model.actor_w.state_dict(),
            'critic_n': self.model.critic_n.state_dict(),
            'critic_s': self.model.critic_s.state_dict(),
            'critic_e': self.model.critic_e.state_dict(),
            'critic_w': self.model.critic_w.state_dict(),
            'actor_n_optimizer':  self.actor_n_optimizer.state_dict(),
            'actor_s_optimizer':  self.actor_s_optimizer.state_dict(),
            'actor_e_optimizer':  self.actor_e_optimizer.state_dict(),
            'actor_w_optimizer':  self.actor_w_optimizer.state_dict(),
            'critic_n_optimizer': self.critic_n_optimizer.state_dict(),
            'critic_s_optimizer': self.critic_s_optimizer.state_dict(),
            'critic_e_optimizer': self.critic_e_optimizer.state_dict(),
            'critic_w_optimizer': self.critic_w_optimizer.state_dict(),
            'obs_dim':    getattr(self.config, 'obs_dim', OBS_DIM),
            'hidden_dim': getattr(self.config, 'hidden_dim', 1024),
            'actor_belief_conditioned': getattr(
                self.config, 'actor_belief_conditioned', False
            ),
            'actor_belief_hidden_dim': getattr(
                self.config, 'actor_belief_hidden_dim', None
            ),
            'action_mapping_version': ACTION_MAPPING_VERSION,
        }

    def save(self, path: str):
        torch.save(self.checkpoint_dict(), path)

    def load_checkpoint_dict(self, ckpt: dict) -> None:
        """Restore a dictionary produced by :meth:`checkpoint_dict`."""
        for role in ('actor_n','actor_s','actor_e','actor_w',
                     'critic_n','critic_s','critic_e','critic_w'):
            if role in ckpt:
                getattr(self.model, role).load_state_dict(ckpt[role])
        for opt_key in ('actor_n_optimizer','actor_s_optimizer',
                        'actor_e_optimizer','actor_w_optimizer',
                        'critic_n_optimizer','critic_s_optimizer',
                        'critic_e_optimizer','critic_w_optimizer'):
            if opt_key in ckpt:
                getattr(self, opt_key).load_state_dict(ckpt[opt_key])

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.load_checkpoint_dict(ckpt)

    def state_dict(self) -> dict:
        d = {}
        for role in ('actor_n','actor_s','actor_e','actor_w',
                     'critic_n','critic_s','critic_e','critic_w'):
            for k, v in getattr(self.model, role).state_dict().items():
                d[f'{role}.{k}'] = v
        return d

    def load_state_dict(self, state: dict):
        def _extract(prefix):
            return {k[len(prefix)+1:]: v for k, v in state.items()
                    if k.startswith(prefix+'.')}
        for role in ('actor_n','actor_s','actor_e','actor_w',
                     'critic_n','critic_s','critic_e','critic_w'):
            if any(k.startswith(role+'.') for k in state):
                getattr(self.model, role).load_state_dict(_extract(role))
