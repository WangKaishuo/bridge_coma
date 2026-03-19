"""
Subgame Trainer  (新架构版)
============================

适配 301 维 MLP Actor（无 LSTM），HAPPO 独立 Critic，FSP pool，KL anchor。

核心变化（vs 旧版）:
1. collect_episodes 用 encode_obs_flat 生成 flat_obs (301,)，
   不再存 {'hand', 'history', 'position', 'vulnerability'} 字典。
   Buffer 只存 flat_obs + legal_actions + all_hands。

2. FSP pool 集成：每 fsp_add_interval 轮将 actor snapshot 存入 pool，
   rollout 时对手从 pool 中随机采样（anti-cycling）。

3. KL anchor：set_bc_anchor() 存储 BC checkpoint，
   safe_update() 中 loss += kl_lambda * KL(pi_current ∥ pi_bc)，
   kl_lambda 从 kl_lambda_start 线性退火到 kl_lambda_end。

4. JIT Belief Burn-in：每次 N-phase 开始前对 Belief Net 做快速微调，
   避免 N 策略更新后 Belief Net OOD。

5. Critic warmup：大池子静态 buffer + 多 epoch 拟合（P51 设计保留）。

6. Rule-based BC 预热：run_bc_warmup() 使用 competitive_env 的
   generate_rule_based_bc_data()，不依赖外部数据集。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from env import NUM_PLAYERS, NUM_BIDS, BID_PASS, NORTH, EAST, SOUTH, WEST
from networks.belief_net import BeliefNetwork, DualInfoComputer
from networks.policy_net import (
    MLPPolicyNetwork, MLPValueNetwork, encode_obs_flat, OBS_DIM
)
from algorithms.mappo import MAPPOAgent, MAPPOConfig
from utils.running_stats import RunningStats
from utils.hand_features import hand_to_belief_target, belief_accuracy, BELIEF_DIM
from utils.fsp_pool import FSPPool


# ==============================================================================
# Config
# ==============================================================================

@dataclass
class SubgameConfig:
    """
    子博弈训练配置（新架构版）.

    主要参数参考 README §4.2 / Kita et al. 2024.
    """
    # ── 训练规模 ────────────────────────────────────────────────────────────
    num_rounds:       int   = 10          # IBR 轮数（外层循环）
    steps_per_phase:  int   = 500         # 每 phase 采集步数（每步 deals_per_step 局）
    deals_per_step:   int   = 32          # 每步并行 rollout 局数
    accumulate_steps: int   = 4           # 累积 N 步数据后做 1 次 PPO update

    # ── 学习率 ──────────────────────────────────────────────────────────────
    lr:              float  = 1e-6        # Actor lr（Kita et al. 2024）
    critic_lr_ratio: float  = 3.0         # Critic lr = lr × critic_lr_ratio
    belief_lr:       float  = 1e-4        # Belief Net 微调 lr

    # ── PPO ─────────────────────────────────────────────────────────────────
    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    clip_ratio:       float = 0.2
    num_epochs:       int   = 4
    batch_size:       int   = 256         # Kita et al. 2024
    entropy_coef:     float = 1e-3        # Kita et al. 2024
    value_coef:       float = 0.5
    max_grad_norm:    float = 0.5

    # ── KL Anchor ───────────────────────────────────────────────────────────
    kl_lambda_start:  float = 0.5         # README §4.2：0.5 → 0.1 退火
    kl_lambda_end:    float = 0.1
    kl_anneal_frac:   float = 1.0         # 全程退火

    # ── KL Early Stopping ───────────────────────────────────────────────────
    kl_early_stop_threshold: float = 0.015

    # ── r_info ──────────────────────────────────────────────────────────────
    use_info_bonus:   bool  = False
    beta:             float = 0.05        # README: β=0.05 主配置

    # ── FSP ─────────────────────────────────────────────────────────────────
    fsp_pool_size:    int   = 10          # Kita et al. 2024
    fsp_add_interval: int   = 2           # 每 N 轮将 actor 存入 pool

    # ── JIT Belief Burn-in ──────────────────────────────────────────────────
    jit_burnin_deals:  int  = 1000        # 每次 N-phase 前采集局数
    jit_burnin_epochs: int  = 3           # 快速微调 epoch 数

    # ── Critic Warmup ───────────────────────────────────────────────────────
    critic_prewarm_deals:  int   = 2048
    critic_prewarm_epochs: int   = 10
    critic_prewarm_conv_tol: float = 0.05

    # ── BC Warmup（rule-based）──────────────────────────────────────────────
    bc_warmup_samples: int  = 5000        # rule-based BC 样本数
    bc_warmup_epochs:  int  = 10          # BC 训练 epoch 数
    bc_warmup_lr:      float = 1e-4

    # ── 网络 ────────────────────────────────────────────────────────────────
    hidden_dim:       int   = 1024        # 4 × 1024 MLP

    # ── 运行控制 ────────────────────────────────────────────────────────────
    active_players:   Optional[List[int]] = None   # None = 全部四方
    eval_interval:    int   = 200
    log_interval:     int   = 50
    device:           str   = 'cpu'


# ==============================================================================
# Rollout Buffer（适配 flat_obs）
# ==============================================================================

class FlatRolloutBuffer:
    """
    存储 flat_obs (301,) + legal_actions (38,) + all_hands (4,52).

    区别于旧版 RolloutBuffer（存字典 obs）:
        - flat_obs 已经是 tensor-ready 的一维向量，无需重新 encode
        - 批量化更高效
    """

    def __init__(self, device: str):
        self.device = device
        self.reset()

    def reset(self):
        self.flat_obs:      List[torch.Tensor] = []
        self.legal_actions: List[torch.Tensor] = []
        self.actions:       List[torch.Tensor] = []
        self.log_probs:     List[torch.Tensor] = []
        self.rewards:       List[float]         = []
        self.values:        List[torch.Tensor] = []
        self.dones:         List[bool]          = []
        self.all_hands:     List[torch.Tensor] = []   # (4, 52) 可选
        self.advantages:    Optional[torch.Tensor] = None
        self.returns:       Optional[torch.Tensor] = None

    def add(self, flat_obs, legal_actions, action, log_prob, reward, value, done,
            all_hands=None):
        self.flat_obs.append(torch.tensor(flat_obs,      dtype=torch.float32))
        self.legal_actions.append(torch.tensor(legal_actions, dtype=torch.float32))
        self.actions.append(torch.tensor(action,         dtype=torch.int64))
        self.log_probs.append(log_prob.detach().cpu() if hasattr(log_prob, 'detach')
                              else torch.tensor(log_prob))
        self.rewards.append(float(reward))
        self.values.append(value.detach().cpu()   if hasattr(value,    'detach')
                           else torch.tensor(value))
        self.dones.append(bool(done))
        if all_hands is not None:
            self.all_hands.append(torch.tensor(all_hands, dtype=torch.float32))

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float, gae_lambda: float
    ):
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae   = 0.0
        values_np  = np.array([v.item() for v in self.values], dtype=np.float32)

        for t in reversed(range(n)):
            next_val  = last_value if t == n - 1 else values_np[t + 1]
            next_done = self.dones[t + 1] if t < n - 1 else True
            delta     = (self.rewards[t] + gamma * next_val * (1.0 - float(next_done))
                         - values_np[t])
            last_gae  = delta + gamma * gae_lambda * (1.0 - float(next_done)) * last_gae
            advantages[t] = last_gae

        returns = advantages + values_np
        self.advantages = torch.tensor(advantages, dtype=torch.float32)
        self.returns    = torch.tensor(returns,    dtype=torch.float32)

    def __len__(self):
        return len(self.actions)

    def get_batches(self, batch_size: int):
        n       = len(self.actions)
        indices = np.random.permutation(n)
        device  = self.device

        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            batch = {
                'flat_obs':      torch.stack([self.flat_obs[i]      for i in idx]).to(device),
                'legal_actions': torch.stack([self.legal_actions[i] for i in idx]).to(device),
                'actions':       torch.stack([self.actions[i]       for i in idx]).to(device),
                'old_log_probs': torch.stack([self.log_probs[i]     for i in idx]).to(device),
                'advantages':    self.advantages[idx].to(device),
                'returns':       self.returns[idx].to(device),
            }
            if self.all_hands:
                batch['all_hands'] = torch.stack(
                    [self.all_hands[i] for i in idx]).to(device)
            yield batch


# ==============================================================================
# SubgameTrainer
# ==============================================================================

class SubgameTrainer:
    """
    通用子博弈训练器（新架构版）.

    外部接口:
        trainer = SubgameTrainer(env, config)
        trainer.run_bc_warmup()          # BC 预热（rule-based，~5k 局）
        trainer.run(num_rounds=10)       # IBR 交替训练
        trainer.evaluate_oracle(...)     # DDS oracle 评估
    """

    def __init__(self, env, config: SubgameConfig,
                 reward_stats: Optional[RunningStats] = None):
        self.env     = env
        self.config  = config
        self.device  = config.device
        self.active_players = config.active_players or list(range(NUM_PLAYERS))

        # ── HAPPO Agent ────────────────────────────────────────────────────
        mappo_cfg = MAPPOConfig(
            lr              = config.lr,
            gamma           = config.gamma,
            gae_lambda      = config.gae_lambda,
            clip_ratio      = config.clip_ratio,
            num_epochs      = config.num_epochs,
            batch_size      = config.batch_size,
            entropy_coef    = config.entropy_coef,
            value_coef      = config.value_coef,
            max_grad_norm   = config.max_grad_norm,
            device          = config.device,
            critic_lr_ratio = config.critic_lr_ratio,
            hidden_dim      = config.hidden_dim,
        )
        self.agent = MAPPOAgent(mappo_cfg)

        # ── Per-player flat rollout buffers ────────────────────────────────
        self.buffers: Dict[int, FlatRolloutBuffer] = {
            p: FlatRolloutBuffer(self.device) for p in range(NUM_PLAYERS)
        }

        # ── Belief Network（可选）──────────────────────────────────────────
        self.belief_net:     Optional[BeliefNetwork]     = None
        self.dual_info:      Optional[DualInfoComputer]  = None
        self.belief_optimizer = None

        if config.use_info_bonus:
            self.belief_net  = BeliefNetwork(hidden_dim=config.hidden_dim).to(self.device)
            self.dual_info   = DualInfoComputer(self.belief_net, beta=config.beta)
            self.belief_optimizer = torch.optim.Adam(
                self.belief_net.parameters(), lr=config.belief_lr)
            self.partner_stats  = RunningStats()
            self.opponent_stats = RunningStats()

        # ── reward 归一化（外部传入，跨 phase 持久）────────────────────────
        self.reward_stats: RunningStats = reward_stats or RunningStats()

        # ── FSP pool ───────────────────────────────────────────────────────
        self.fsp_pool = FSPPool(max_size=config.fsp_pool_size)

        # ── KL anchor（BC checkpoint，由外部 set_bc_anchor 设置）──────────
        self.bc_actors: Optional[Dict[int, MLPPolicyNetwork]] = None

        # ── 日志 ───────────────────────────────────────────────────────────
        self.log: List[dict] = []
        self._global_step = 0

    # ======================================================================
    # BC 预热（rule-based）
    # ======================================================================

    def run_bc_warmup(self, num_samples: int = None, num_epochs: int = None,
                      lr: float = None):
        """
        使用 rule-based 策略生成数据，做轻量 BC 预热.

        不依赖外部 WBridge5 数据集；仅用于 competitive 子博弈初始化。
        """
        from subgames.competitive_env import generate_rule_based_bc_data

        num_samples = num_samples or self.config.bc_warmup_samples
        num_epochs  = num_epochs  or self.config.bc_warmup_epochs
        lr          = lr          or self.config.bc_warmup_lr

        print(f"\n[BC Warmup] Generating {num_samples} rule-based samples...")
        data = generate_rule_based_bc_data(self.env, num_samples)

        if not data:
            print("[BC Warmup] No data generated, skipping.")
            return

        flat_obs_np = np.stack([d['flat_obs'] for d in data])  # (N, 301)
        actions_np  = np.array([d['action']   for d in data], dtype=np.int64)
        legal_np    = np.ones((len(data), NUM_BIDS), dtype=np.float32)  # BC 不限制

        flat_t   = torch.tensor(flat_obs_np, dtype=torch.float32)
        actions_t = torch.tensor(actions_np, dtype=torch.int64)
        legal_t  = torch.tensor(legal_np,    dtype=torch.float32)

        print(f"[BC Warmup] Training {num_epochs} epochs on {len(data)} samples...")

        for player in [NORTH, SOUTH]:  # 只训练 NS（训练方）
            actor = self.agent.get_actor(player)
            opt   = torch.optim.Adam(actor.parameters(), lr=lr)

            for epoch in range(num_epochs):
                idx  = np.random.permutation(len(data))
                loss_sum = 0.0
                n_batches = 0

                for start in range(0, len(data), self.config.batch_size):
                    b_idx = idx[start:start + self.config.batch_size]
                    b_flat   = flat_t[b_idx].to(self.device)
                    b_legal  = legal_t[b_idx].to(self.device)
                    b_act    = actions_t[b_idx].to(self.device)

                    logits = actor(b_flat, b_legal)
                    loss   = F.cross_entropy(logits, b_act)

                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), self.config.max_grad_norm)
                    opt.step()

                    loss_sum += loss.item()
                    n_batches += 1

                avg_loss = loss_sum / max(1, n_batches)

            print(f"  Player {'N' if player == NORTH else 'S'}: "
                  f"epoch {num_epochs} loss={avg_loss:.4f}")

        print("[BC Warmup] Done.")

    # ======================================================================
    # KL Anchor
    # ======================================================================

    def set_bc_anchor(self, agent_or_state_dict):
        """
        设置 BC 锚点（Stage 1 BC 结束时的快照）.

        Args:
            agent_or_state_dict: MAPPOAgent 或 state_dict dict
        """
        if isinstance(agent_or_state_dict, MAPPOAgent):
            src_agent = agent_or_state_dict
            n_state   = src_agent.model.actor_n.state_dict()
            s_state   = src_agent.model.actor_s.state_dict()
        else:
            sd = agent_or_state_dict
            def _extract(prefix):
                return {k[len(prefix)+1:]: v for k, v in sd.items()
                        if k.startswith(prefix + '.')}
            if any(k.startswith('actor_n.') for k in sd):
                n_state = _extract('actor_n')
                s_state = _extract('actor_s')
            else:
                # 旧格式或直接 state dict
                n_state = s_state = sd

        def _frozen_copy(state_dict):
            net = MLPPolicyNetwork(hidden_dim=self.config.hidden_dim).to(self.device)
            net.load_state_dict(state_dict)
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            return net

        self.bc_actors = {
            NORTH: _frozen_copy(n_state),
            SOUTH: _frozen_copy(s_state),
        }
        print("[KL Anchor] BC anchor set.")

    def _get_kl_lambda(self, round_idx: int) -> float:
        cfg      = self.config
        progress = min(1.0, round_idx / max(1, cfg.num_rounds - 1))
        return cfg.kl_lambda_start + (cfg.kl_lambda_end - cfg.kl_lambda_start) * progress

    # ======================================================================
    # FSP
    # ======================================================================

    def _maybe_add_to_fsp(self, round_idx: int):
        """每 fsp_add_interval 轮将当前 actor 存入 pool."""
        if (self.config.fsp_pool_size > 0
                and round_idx % self.config.fsp_add_interval == 0):
            self.fsp_pool.add(self.agent)
            print(f"  [FSP] Pool size: {len(self.fsp_pool)}")

    def _apply_fsp_opponent(self):
        """
        从 pool 中随机采样一个历史 checkpoint 作为 EW 对手.

        只在 rollout 时临时替换 EW actor；
        训练时恢复为当前 agent（EW 不训练，只是采样对手）。
        返回 sampled state_dict 或 None（pool 为空时用 self）。
        """
        if self.fsp_pool.is_empty():
            return None
        return self.fsp_pool.sample()

    # ======================================================================
    # Critic Warmup
    # ======================================================================

    def critic_warmup(self, num_deals: int = None, num_epochs: int = None):
        """大池子静态 buffer + 多 epoch Critic 预热（P51 设计）."""
        num_deals  = num_deals  or self.config.critic_prewarm_deals
        num_epochs = num_epochs or self.config.critic_prewarm_epochs
        conv_tol   = self.config.critic_prewarm_conv_tol

        print(f"[Critic Warmup] Collecting {num_deals} deals...")
        episodes = self._collect_episodes(num_deals, use_fsp_opponent=False)
        self._store_episodes(episodes)

        for player in self.active_players:
            buf = self.buffers[player]
            if len(buf) < 2:
                buf.reset()
                continue

            critic     = self.agent.get_critic(player)
            critic_opt = self.agent.get_critic_optimizer(player)

            # Bootstrap last value
            with torch.no_grad():
                fo = buf.flat_obs[-1].unsqueeze(0).to(self.device)
                ah = (buf.all_hands[-1].unsqueeze(0).to(self.device)
                      if buf.all_hands else None)
                last_val = critic(fo, ah).item()

            buf.compute_returns_and_advantages(
                last_val, self.agent.config.gamma, self.agent.config.gae_lambda)

            prev_loss = None
            for epoch in range(num_epochs):
                epoch_loss, n_batches = 0.0, 0
                for batch in buf.get_batches(self.agent.config.batch_size):
                    vals = critic(batch['flat_obs'], batch.get('all_hands'))
                    loss = F.mse_loss(vals, batch['returns'])
                    critic_opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), self.config.max_grad_norm)
                    critic_opt.step()
                    epoch_loss += loss.item()
                    n_batches  += 1
                epoch_loss /= max(1, n_batches)
                if prev_loss is not None and prev_loss > 1e-8:
                    if abs(epoch_loss - prev_loss) / prev_loss < conv_tol:
                        print(f"  [Critic Warmup] Player {player} converged @ epoch {epoch+1}")
                        break
                prev_loss = epoch_loss
            buf.reset()

    # ======================================================================
    # JIT Belief Burn-in
    # ======================================================================

    def jit_belief_burnin(self):
        """
        每次 N-phase 开始前对 Belief Net 做快速微调.

        采集 jit_burnin_deals 局 rollout，训练 jit_burnin_epochs epoch。
        防止 N 策略更新后 Belief Net OOD。
        """
        if self.belief_net is None or self.belief_optimizer is None:
            return

        print(f"  [JIT Burn-in] Collecting {self.config.jit_burnin_deals} deals...")
        episodes = self._collect_episodes(self.config.jit_burnin_deals,
                                          use_fsp_opponent=False)

        belief_data = []
        for ep in episodes:
            for step in ep:
                if 'belief_target' in step:
                    belief_data.append(step)

        if not belief_data:
            return

        self.belief_net.train()
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=self.belief_net.pos_weight.to(self.device))

        for epoch in range(self.config.jit_burnin_epochs):
            np.random.shuffle(belief_data)
            total_loss = 0.0
            n = 0
            for step in belief_data:
                oh  = torch.tensor(step['observer_hand'], dtype=torch.float32
                                   ).unsqueeze(0).to(self.device)
                h   = torch.tensor(step['history'],       dtype=torch.float32
                                   ).unsqueeze(0).to(self.device)
                op  = torch.tensor([step['observer_pos']], dtype=torch.long).to(self.device)
                tp  = torch.tensor([step['target_pos']],  dtype=torch.long).to(self.device)
                tgt = torch.tensor(step['belief_target'], dtype=torch.float32
                                   ).unsqueeze(0).to(self.device)

                logits = self.belief_net(oh, h, op, tp)
                loss   = criterion(logits, tgt)
                self.belief_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.belief_net.parameters(), self.config.max_grad_norm)
                self.belief_optimizer.step()
                total_loss += loss.item()
                n += 1

        print(f"  [JIT Burn-in] Done. loss={total_loss/max(1,n):.4f}")

    # ======================================================================
    # Episode Collection
    # ======================================================================

    def _collect_episodes(
        self,
        num_deals:         int,
        use_fsp_opponent:  bool = True,
        fsp_state_dict:    Optional[dict] = None,
    ) -> List[List[dict]]:
        """
        收集 num_deals 局 rollout，返回 episode list.

        每个 episode 是 step list，每个 step 含:
            flat_obs, legal_actions, action, log_prob, value, reward, done,
            all_hands, player, [belief_target, observer_hand, history, ...]
        """
        episodes   = []
        dealer     = self.env.dealer  # NORTH (固定)

        # FSP: 如果有对手 snapshot，临时加载到 EW actor
        fsp_sd     = fsp_state_dict
        if use_fsp_opponent and fsp_sd is None and not self.fsp_pool.is_empty():
            fsp_sd = self.fsp_pool.sample()

        for _ in range(num_deals):
            hands, dd_table = self.env.generate_deal()
            vul = (False, False)

            obs  = self.env.reset(hands, dd_table, vulnerability=vul)
            done = False
            ep   = []
            history_int = list(self.env.history_int)  # 含前缀

            while not done:
                player    = self.env.current_player
                all_hands = self.env._current_hands.copy()

                # 编码 flat obs
                flat_obs     = encode_obs_flat(obs, dealer, history_int)
                legal_actions = obs['legal_actions'].copy()

                # 选 actor: FSP 对手 or 当前 agent
                flat_t  = torch.tensor(flat_obs,      dtype=torch.float32
                                       ).unsqueeze(0).to(self.device)
                legal_t = torch.tensor(legal_actions, dtype=torch.float32
                                       ).unsqueeze(0).to(self.device)
                ah_t    = torch.tensor(all_hands,      dtype=torch.float32
                                       ).unsqueeze(0).to(self.device)

                is_opponent = (player in (EAST, WEST))
                if is_opponent and fsp_sd is not None:
                    # 临时加载 FSP snapshot 到 EW actor（只用于 get_action）
                    actor = self._get_fsp_actor(player, fsp_sd)
                else:
                    actor = self.agent.get_actor(player)

                critic = self.agent.get_critic(player)

                with torch.no_grad():
                    action, log_prob, _ = actor.get_action(flat_t, legal_t)
                    value               = critic(flat_t, ah_t)

                step = {
                    'flat_obs':     flat_obs,
                    'legal_actions': legal_actions,
                    'action':       action.item(),
                    'log_prob':     log_prob.squeeze(0),
                    'value':        value.squeeze(0),
                    'reward':       0.0,
                    'done':         False,
                    'all_hands':    all_hands,
                    'player':       player,
                }

                # ── Belief 数据记录 ──────────────────────────────────────
                if self.belief_net is not None:
                    if player in (NORTH, SOUTH):
                        # NS 决策步骤：记录完整 r_info 所需数据
                        partner = (player + 2) % 4
                        step.update({
                            'observer_hand':      all_hands[player],
                            'history':            self._encode_history(history_int),
                            'observer_pos':       player,
                            'target_pos':         partner,
                            'belief_target':      hand_to_belief_target(all_hands[partner]),
                            'history_int_before': history_int[:],
                        })
                    else:
                        # EW 决策步骤：记录 NS 视角对 EW 手牌推断的诊断数据
                        # 不参与 r_info 奖励，仅供 evaluate_ew_belief_update() 使用
                        step.update({
                            'ew_diagnostic':      True,
                            'observer_hand':      all_hands[NORTH],
                            'history':            self._encode_history(history_int),
                            'observer_pos':       NORTH,
                            'target_pos':         player,
                            'belief_target':      hand_to_belief_target(all_hands[player]),
                            'history_int_before': history_int[:],
                        })

                history_int.append(action.item())

                # 补充 history_int_after
                if self.belief_net is not None:
                    if player in (NORTH, SOUTH) or step.get('ew_diagnostic'):
                        step['history_int_after'] = history_int[:]
                obs, reward, done, info = self.env.step(action.item())

                step['reward'] = reward
                step['done']   = done
                ep.append(step)

            episodes.append(ep)

        return episodes

    def _get_fsp_actor(self, player: int, fsp_sd: dict) -> MLPPolicyNetwork:
        """
        获取 FSP snapshot 对应的 actor（临时对象，不影响 self.agent）.

        注: 简单实现：每次创建临时网络并加载权重。
        性能影响可接受（FSP actor 只用于推断，不需要梯度）。
        """
        role  = 'actor_n' if player == NORTH else 'actor_s'
        # EW 也映射到 actor_n / actor_s（EW 的 policy 从 pool 中来）
        if player == EAST:
            role = 'actor_n'
        elif player == WEST:
            role = 'actor_s'

        if role not in fsp_sd:
            return self.agent.get_actor(player)

        net = MLPPolicyNetwork(hidden_dim=self.config.hidden_dim).to(self.device)
        sd  = {k: v.to(self.device) for k, v in fsp_sd[role].items()}
        net.load_state_dict(sd)
        net.eval()
        return net

    def _store_episodes(self, episodes: List[List[dict]]):
        """将 episode list 存入各 player 的 buffer."""
        for ep in episodes:
            for step in ep:
                p = step['player']
                if p not in self.active_players:
                    continue
                buf = self.buffers[p]
                buf.add(
                    flat_obs      = step['flat_obs'],
                    legal_actions = step['legal_actions'],
                    action        = step['action'],
                    log_prob      = step['log_prob'],
                    reward        = step['reward'],
                    value         = step['value'],
                    done          = step['done'],
                    all_hands     = step.get('all_hands'),
                )

    def _encode_history(self, history_int: list) -> np.ndarray:
        """将整数历史列表编码为 (max_len, NUM_BIDS) one-hot（Belief Net 用）."""
        max_len = 60
        arr = np.zeros((max_len, NUM_BIDS), dtype=np.float32)
        for i, bid in enumerate(history_int[-max_len:]):
            arr[i, bid] = 1.0
        return arr

    # ======================================================================
    # r_info 计算
    # ======================================================================

    def _compute_info_bonus(self, episodes: List[List[dict]]) -> List[List[float]]:
        """
        为每个 episode 的每个 step 计算 r_info.

        r_info = I(bid; hand | partner) - β × I(bid; hand | opponent)
               ≈ [CE(q_before, hand) - CE(q_after, hand)]_partner
                 - β × [CE(q_before, hand) - CE(q_after, hand)]_opponent

        ReLU 截断：max(0, ·)，互信息 ≥ 0
        """
        if self.dual_info is None:
            return [[0.0] * len(ep) for ep in episodes]

        bonus_episodes = []

        for ep in episodes:
            bonuses = []
            for step in ep:
                if 'belief_target' not in step or step['player'] not in (NORTH, SOUTH):
                    bonuses.append(0.0)
                    continue

                player  = step['player']
                partner = (player + 2) % 4
                opp     = (player + 1) % 4  # 右侧对手

                h_before_enc = self._encode_history(step.get('history_int_before', []))
                h_after_enc  = self._encode_history(step.get('history_int_after',  []))

                oh  = torch.tensor(step['observer_hand'], dtype=torch.float32
                                   ).unsqueeze(0).to(self.device)
                hb  = torch.tensor(h_before_enc, dtype=torch.float32
                                   ).unsqueeze(0).to(self.device)
                ha  = torch.tensor(h_after_enc,  dtype=torch.float32
                                   ).unsqueeze(0).to(self.device)
                tgt = torch.tensor(step['belief_target'], dtype=torch.float32
                                   ).unsqueeze(0).to(self.device)

                op_t = torch.tensor([player],  dtype=torch.long).to(self.device)
                pp_t = torch.tensor([partner], dtype=torch.long).to(self.device)
                oo_t = torch.tensor([opp],     dtype=torch.long).to(self.device)

                with torch.no_grad():
                    # Partner 信息增益
                    b_before_partner = self.belief_net.get_probs(oh, hb, op_t, pp_t)
                    b_after_partner  = self.belief_net.get_probs(oh, ha, op_t, pp_t)
                    partner_gain = self.dual_info.compute_info_gain(
                        b_before_partner, b_after_partner, tgt)

                    # Opponent 信息泄露
                    b_before_opp = self.belief_net.get_probs(oh, hb, op_t, oo_t)
                    b_after_opp  = self.belief_net.get_probs(oh, ha, op_t, oo_t)
                    # opponent target: 对手的手牌特征
                    opp_tgt = torch.tensor(
                        hand_to_belief_target(step['all_hands'][opp]),
                        dtype=torch.float32).unsqueeze(0).to(self.device)
                    opponent_leak = self.dual_info.compute_info_gain(
                        b_before_opp, b_after_opp, opp_tgt)

                    bonus, _ = self.dual_info.compute_dual_info_bonus(
                        partner_gain, opponent_leak)

                bonuses.append(float(bonus.item()))

            bonus_episodes.append(bonuses)

        return bonus_episodes

    # ======================================================================
    # PPO Update
    # ======================================================================

    def _safe_update(self, player: int, round_idx: int) -> dict:
        """
        单 player 的 PPO update.

        包含:
        - GAE returns & advantages
        - Clipped policy loss + value loss + entropy bonus
        - KL anchor penalty
        - KL early stopping（epoch 级别）
        """
        buf   = self.buffers[player]
        if len(buf) < self.config.batch_size:
            buf.reset()
            return {}

        actor      = self.agent.get_actor(player)
        critic     = self.agent.get_critic(player)
        actor_opt  = self.agent.get_actor_optimizer(player)
        critic_opt = self.agent.get_critic_optimizer(player)
        kl_lambda  = self._get_kl_lambda(round_idx)

        # Bootstrap last value
        with torch.no_grad():
            fo  = buf.flat_obs[-1].unsqueeze(0).to(self.device)
            ah  = (buf.all_hands[-1].unsqueeze(0).to(self.device)
                   if buf.all_hands else None)
            last_val = critic(fo, ah).item()

        buf.compute_returns_and_advantages(
            last_val, self.agent.config.gamma, self.agent.config.gae_lambda)

        total_policy = total_value = total_entropy = total_kl = 0.0
        n_updates = 0

        for epoch in range(self.config.num_epochs):
            epoch_kl = 0.0
            n_batch_kl = 0

            for batch in buf.get_batches(self.agent.config.batch_size):
                b_flat   = batch['flat_obs']
                b_legal  = batch['legal_actions']
                b_act    = batch['actions']
                b_old_lp = batch['old_log_probs']
                b_ret    = batch['returns']
                b_adv    = batch['advantages']
                b_ah     = batch.get('all_hands')

                # Normalize advantage
                adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

                # Actor loss
                log_probs, entropy = actor.evaluate_actions(b_flat, b_legal, b_act)
                ratio = torch.exp(log_probs - b_old_lp)
                policy_loss = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1 - self.config.clip_ratio,
                                       1 + self.config.clip_ratio) * adv
                ).mean()

                # KL anchor
                kl_loss = torch.tensor(0.0, device=self.device)
                if self.bc_actors is not None and player in self.bc_actors and kl_lambda > 0:
                    bc_actor = self.bc_actors[player]
                    with torch.no_grad():
                        bc_logits = bc_actor(b_flat, b_legal)
                        bc_probs  = F.softmax(bc_logits, dim=-1).clamp(1e-8, 1.0)
                    cur_logits = actor(b_flat, b_legal)
                    cur_probs  = F.softmax(cur_logits, dim=-1).clamp(1e-8, 1.0)
                    kl_loss    = F.kl_div(
                        cur_probs.log(), bc_probs, reduction='batchmean')

                actor_loss = (policy_loss
                              - self.config.entropy_coef * entropy.mean()
                              + kl_lambda * kl_loss)

                actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), self.config.max_grad_norm)
                actor_opt.step()

                # Critic loss
                vals    = critic(b_flat, b_ah)
                old_v   = vals.detach()
                v_clip  = old_v + (vals - old_v).clamp(
                    -self.config.clip_ratio, self.config.clip_ratio)
                value_loss = torch.max(F.mse_loss(vals, b_ret),
                                       F.mse_loss(v_clip, b_ret))
                critic_opt.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), self.config.max_grad_norm)
                critic_opt.step()

                # Approx KL for early stopping
                with torch.no_grad():
                    approx_kl = ((b_old_lp - log_probs).mean()).item()
                epoch_kl   += approx_kl
                n_batch_kl += 1

                total_policy  += policy_loss.item()
                total_value   += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl      += kl_loss.item()
                n_updates     += 1

            # Epoch-level KL early stopping
            epoch_kl /= max(1, n_batch_kl)
            if (self.config.kl_early_stop_threshold > 0
                    and epoch_kl > self.config.kl_early_stop_threshold):
                break

        buf.reset()

        if n_updates == 0:
            return {}

        return {
            'policy_loss': total_policy  / n_updates,
            'value_loss':  total_value   / n_updates,
            'entropy':     total_entropy / n_updates,
            'kl_loss':     total_kl      / n_updates,
            'kl_lambda':   kl_lambda,
        }

    # ======================================================================
    # 主训练循环：IBR 交替训练
    # ======================================================================

    def run(self, num_rounds: int = None) -> List[dict]:
        """
        IBR 交替训练:

        Round k:
          S-phase:  训练 NS（S 先决策，N 后 rebid）
          N-phase:  训练 NS（N-phase JIT Belief Burn-in → 更新 N + S）
                    Agent B：在此阶段激活 r_info
          每 fsp_add_interval 轮将 actor snapshot 存入 FSP pool
        """
        num_rounds = num_rounds or self.config.num_rounds

        # 初始 Critic warmup
        print("\n[Trainer] Critic warmup...")
        self.critic_warmup()

        for rnd in range(num_rounds):
            print(f"\n══════ Round {rnd+1}/{num_rounds} ══════")

            # FSP snapshot
            self._maybe_add_to_fsp(rnd)
            fsp_sd = self._apply_fsp_opponent()

            # ── S-phase ────────────────────────────────────────────────────
            print("  [S-phase] Collecting episodes...")
            s_episodes = self._collect_episodes(
                self.config.steps_per_phase * self.config.deals_per_step,
                use_fsp_opponent=True, fsp_state_dict=fsp_sd)

            if self.config.use_info_bonus:
                bonuses = self._compute_info_bonus(s_episodes)
                for ep, bs in zip(s_episodes, bonuses):
                    for step, b in zip(ep, bs):
                        step['reward'] += b  # 逐步分配 r_info，beta 已在 compute_dual_info_bonus 内应用

            self._store_episodes(s_episodes)
            s_metrics = {}
            for p in self.active_players:
                m = self._safe_update(p, rnd)
                if m:
                    s_metrics[p] = m

            # ── N-phase: JIT Belief Burn-in → 收集 → 更新 ─────────────────
            if self.belief_net is not None:
                print("  [N-phase] JIT Belief Burn-in...")
                self.jit_belief_burnin()

            print("  [N-phase] Collecting episodes...")
            n_episodes = self._collect_episodes(
                self.config.steps_per_phase * self.config.deals_per_step,
                use_fsp_opponent=True, fsp_state_dict=fsp_sd)

            if self.config.use_info_bonus:
                bonuses = self._compute_info_bonus(n_episodes)
                for ep, bs in zip(n_episodes, bonuses):
                    for step, b in zip(ep, bs):
                        step['reward'] += b  # 逐步分配 r_info，beta 已在 compute_dual_info_bonus 内应用

            self._store_episodes(n_episodes)
            n_metrics = {}
            for p in self.active_players:
                m = self._safe_update(p, rnd)
                if m:
                    n_metrics[p] = m

            # 日志
            all_rewards = [
                step['reward']
                for ep in (s_episodes + n_episodes)
                for step in ep
                if step.get('done')
            ]
            mean_r = np.mean(all_rewards) if all_rewards else 0.0
            std_r  = np.std(all_rewards)  if all_rewards else 0.0

            log_entry = {
                'round':       rnd + 1,
                'mean_reward': mean_r,
                'std_reward':  std_r,
                's_metrics':   s_metrics,
                'n_metrics':   n_metrics,
                'fsp_pool_size': len(self.fsp_pool),
            }
            self.log.append(log_entry)
            self._print_log(log_entry)

        return self.log

    def _print_log(self, entry: dict):
        rnd  = entry['round']
        mr   = entry['mean_reward']
        sr   = entry['std_reward']
        fsp  = entry['fsp_pool_size']
        # 取 S 的 policy_loss / value_loss（典型 player）
        s_m  = entry['s_metrics'].get(SOUTH, {})
        pl   = s_m.get('policy_loss', 0)
        vl   = s_m.get('value_loss',  0)
        ent  = s_m.get('entropy',     0)
        kl   = s_m.get('kl_loss',     0)
        klam = s_m.get('kl_lambda',   0)
        print(f"  [Round {rnd}] r={mr:+.3f}±{sr:.3f} "
              f"pl={pl:.4f} vl={vl:.4f} ent={ent:.4f} "
              f"kl={kl:.5f}(λ={klam:.3f}) fsp={fsp}")

    # ======================================================================
    # 评估
    # ======================================================================

    def evaluate_oracle(self, num_deals: int = 1000) -> dict:
        """
        DDS oracle 评估（主要指标）.

        IMP regret = actual_imp - dds_optimal_imp  (≤ 0，越高越好)
        """
        from subgames.competitive_env import (
            dds_oracle_evaluate, make_agent_policy
        )
        policy = make_agent_policy(self.agent, deterministic=True)
        result = dds_oracle_evaluate(self.env, policy, num_deals)

        print(f"\n  [Oracle Eval] mean_regret={result['mean_regret']:+.3f} "
              f"± {result['std_regret']:.3f} IMP  "
              f"95% CI [{result['ci_lo']:+.3f}, {result['ci_hi']:+.3f}]")
        return result

    def evaluate_belief(self, num_deals: int = 50) -> dict:
        """评估 Belief Network 质量."""
        if self.belief_net is None:
            return {}

        self.belief_net.eval()
        all_probs, all_targets = [], []

        with torch.no_grad():
            for _ in range(num_deals):
                hands, dd_table = self.env.generate_deal()
                obs  = self.env.reset(hands, dd_table)
                done = False
                hist = list(self.env.history_int)

                while not done:
                    player   = self.env.current_player
                    flat_obs = encode_obs_flat(obs, NORTH, hist)
                    flat_t   = torch.tensor(flat_obs, dtype=torch.float32
                                            ).unsqueeze(0).to(self.device)
                    legal_t  = torch.tensor(obs['legal_actions'], dtype=torch.float32
                                            ).unsqueeze(0).to(self.device)
                    actor    = self.agent.get_actor(player)
                    action, _, _ = actor.get_action(flat_t, legal_t, deterministic=True)
                    hist.append(action.item())
                    obs, _, done, _ = self.env.step(action.item())

                h_enc = self._encode_history(hist)
                oh    = torch.tensor(hands[NORTH], dtype=torch.float32
                                     ).unsqueeze(0).to(self.device)
                h_t   = torch.tensor(h_enc, dtype=torch.float32
                                     ).unsqueeze(0).to(self.device)
                op    = torch.tensor([NORTH], dtype=torch.long).to(self.device)
                tp    = torch.tensor([SOUTH], dtype=torch.long).to(self.device)

                probs  = self.belief_net.get_probs(oh, h_t, op, tp)
                target = torch.tensor(
                    hand_to_belief_target(hands[SOUTH]), dtype=torch.float32
                ).unsqueeze(0).to(self.device)

                all_probs.append(probs)
                all_targets.append(target)

        self.belief_net.train()

        if not all_probs:
            return {}

        probs_cat   = torch.cat(all_probs,   dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        acc = belief_accuracy(probs_cat, targets_cat)
        print(f"  [BeliefNet] honor={acc['honor_acc']:.3f} "
              f"length={acc['length_acc']:.3f} overall={acc['overall_acc']:.3f}")
        return acc

    def evaluate_ew_belief_update(self, num_deals: int = 200) -> dict:
        """
        诊断指标：NS视角对EW手牌的推断更新量。

        在每一个EW叫品发生前后，用Belief Net计算NS对该EW手牌的
        交叉熵变化（信息增益）。这个量反映了EW叫品对NS推断的贡献：
            - 正值：EW的叫品帮助NS更好地推断EW手牌（EW在"暴露"）
            - 接近零：EW的叫品信息量低（比如在高阶Pass）

        这是论文RQ2（机制验证）的辅助证据：
        Agent B的β term是否有效惩罚了EW暴露，
        应该表现为B的opponent_leak比A更低。

        Returns:
            {
              'mean_ew_gain':    float,  # NS从EW叫品中平均获得的信息增益
              'std_ew_gain':     float,
              'per_action_type': dict,   # Pass/实质叫品/X分类统计
            }
        """
        if self.belief_net is None:
            return {}

        self.belief_net.eval()
        from env import BID_PASS, BID_DOUBLE, BID_1C

        gains_all   = []
        gains_pass  = []
        gains_real  = []
        gains_double= []

        with torch.no_grad():
            for _ in range(num_deals):
                hands, dd_table = self.env.generate_deal()
                obs  = self.env.reset(hands, dd_table)
                done = False
                hist = list(self.env.history_int)  # 含前缀

                while not done:
                    player  = self.env.current_player
                    flat_t  = torch.tensor(
                        encode_obs_flat(obs, NORTH, hist),
                        dtype=torch.float32).unsqueeze(0).to(self.device)
                    legal_t = torch.tensor(
                        obs['legal_actions'], dtype=torch.float32
                    ).unsqueeze(0).to(self.device)

                    actor  = self.agent.get_actor(player)
                    action, _, _ = actor.get_action(flat_t, legal_t, deterministic=True)
                    action_int = action.item()

                    # 只在 EW 步骤计算诊断
                    if player in (EAST, WEST):
                        h_before = self._encode_history(hist)
                        hist_after = hist + [action_int]
                        h_after  = self._encode_history(hist_after)

                        oh  = torch.tensor(hands[NORTH], dtype=torch.float32
                                           ).unsqueeze(0).to(self.device)
                        hb  = torch.tensor(h_before, dtype=torch.float32
                                           ).unsqueeze(0).to(self.device)
                        ha  = torch.tensor(h_after,  dtype=torch.float32
                                           ).unsqueeze(0).to(self.device)
                        op  = torch.tensor([NORTH],  dtype=torch.long).to(self.device)
                        tp  = torch.tensor([player], dtype=torch.long).to(self.device)
                        tgt = torch.tensor(
                            hand_to_belief_target(hands[player]),
                            dtype=torch.float32).unsqueeze(0).to(self.device)

                        b_before = self.belief_net.get_probs(oh, hb, op, tp)
                        b_after  = self.belief_net.get_probs(oh, ha, op, tp)
                        gain = self.dual_info.compute_info_gain(
                            b_before, b_after, tgt).item()

                        gains_all.append(gain)
                        if action_int == BID_PASS:
                            gains_pass.append(gain)
                        elif action_int == BID_DOUBLE or action_int == BID_DOUBLE + 1:
                            gains_double.append(gain)
                        else:
                            gains_real.append(gain)

                    hist.append(action_int)
                    obs, _, done, _ = self.env.step(action_int)

        self.belief_net.train()

        def _stats(lst):
            if not lst:
                return {'mean': 0.0, 'std': 0.0, 'n': 0}
            a = np.array(lst)
            return {'mean': float(a.mean()), 'std': float(a.std()), 'n': len(lst)}

        result = {
            'mean_ew_gain': float(np.mean(gains_all)) if gains_all else 0.0,
            'std_ew_gain':  float(np.std(gains_all))  if gains_all else 0.0,
            'per_action_type': {
                'pass':        _stats(gains_pass),
                'real_bid':    _stats(gains_real),
                'double':      _stats(gains_double),
            },
            'n_ew_steps': len(gains_all),
        }

        print(f"  [EW Belief Update] mean_gain={result['mean_ew_gain']:.4f} "
              f"(pass={result['per_action_type']['pass']['mean']:.4f} "
              f"real={result['per_action_type']['real_bid']['mean']:.4f} "
              f"X={result['per_action_type']['double']['mean']:.4f}) "
              f"n={result['n_ew_steps']}")
        return result
