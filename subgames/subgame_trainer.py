"""
Subgame Trainer  (新架构版)
============================

适配 301 维 MLP Actor（无 LSTM），HAPPO 独立 Critic，FSP pool，KL anchor。

核心变化（vs 旧版）:
1. collect_episodes 用 encode_obs_flat 生成 flat_obs (301,)，
   不再存 {'hand', 'history', 'position', 'vulnerability'} 字典。
   Buffer 只存 flat_obs + legal_actions + all_hands。
   P101: belief_conditioned 模式下 flat_obs = 397 维
   (301 base + 48 partner belief + 48 RHO belief)

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
    MLPPolicyNetwork, MLPValueNetwork, encode_obs_flat, OBS_DIM,
    BELIEF_OBS_DIM, BELIEF_FEAT_DIM, make_belief_features_prior,
    append_belief_features,
)
from algorithms.mappo import MAPPOAgent, MAPPOConfig
from utils.running_stats import RunningStats
from utils.hand_features import hand_to_belief_target, belief_accuracy, BELIEF_DIM
from utils.fsp_pool import FSPPool
from utils.imp import score_to_imp


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
    num_rounds:       int   = 20          # IBR 轮数（外层循环）
    steps_per_phase:  int   = 500         # 每 phase 采集步数（每步 deals_per_step 局）
    deals_per_step:   int   = 32          # 每步并行 rollout 局数
    accumulate_steps: int   = 4           # 累积 N 步数据后做 1 次 PPO update

    # ── 学习率 ──────────────────────────────────────────────────────────────
    lr:              float  = 1e-6        # Actor lr（Kita et al. 2024）
    critic_lr_ratio: float  = 10.0        # Critic lr = lr × critic_lr_ratio (P56: 5→10, dual-table IMP std≈7)
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
    kl_lambda_start:  float = 1.5         # P73
    kl_lambda_end:    float = 1.5         # P73: 暂不退火
    kl_anneal_frac:   float = 0.0         # P73: 0.0=固定不退火

    # ── KL Early Stopping ───────────────────────────────────────────────────
    kl_early_stop_threshold: float = 0.015

    # ── r_info ──────────────────────────────────────────────────────────────
    use_info_bonus:      bool  = False
    beta:                float = 0.05        # 内部β: r_info = I(partner) - β·I(opponent)
    info_reward_weight:  float = 0.2         # P87b: 占 IMP 方差的比例 (0.02→0.2)

    # ── FSP ─────────────────────────────────────────────────────────────────
    fsp_pool_size:    int   = 10          # Kita et al. 2024
    fsp_add_interval: int   = 2           # 每 N 轮将 actor 存入 pool

    # ── Belief Net Update ────────────────────────────────────────────────
    # P93: on-policy update (epochs=8, lr=5e-5) — caused catastrophic
    #   forgetting of pretrain foundation (val_loss 1.76→2.19 in Round 1).
    # P96: freeze_belief=True by default. With strong KL (λ=0.5),
    #   policy stays near SAYC and pretrain Belief Net remains valid.
    belief_update_epochs: int  = 3            # P95: only used if freeze_belief=False
    belief_update_lr:     float = 1e-5        # P95: only used if freeze_belief=False
    freeze_belief:        bool  = True        # P96: frozen by default

    # ── EWC for Belief Net (P97) ─────────────────────────────────────
    use_ewc:              bool  = False       # P97: EWC-protected on-policy update
    ewc_lambda:           float = 100.0       # P97: EWC penalty strength (Fisher normalized, mean penalty)
    ewc_fisher_samples:   int   = 5000        # Samples for Fisher computation

    # ── P98/P101: Belief-Conditioned Actor ─────────────────────────────
    belief_conditioned:   bool  = False       # P101: Actor input includes belief features (397 dim)
    # When True: obs_dim = 397 (301 base + 48 partner belief + 48 RHO belief)
    # Requires belief_net to be available (use_info_bonus=True or standalone belief)
    # Enables the Convention Card Protocol for Full Disclosure evaluation
    belief_warmup_rounds: int   = 2           # P98b: rounds using prior features instead of belief net
    # During warmup rounds, rollout uses make_belief_features_prior() for the 48-dim
    # belief input. This prevents garbage belief features from destabilising the actor
    # before the Belief Net has adapted to the RL trajectory distribution.

    # ── Critic Warmup ───────────────────────────────────────────────────────
    critic_prewarm_deals:  int   = 2048
    critic_prewarm_epochs: int   = 10
    critic_prewarm_conv_tol: float = 0.05

    # ── BC Warmup（rule-based）──────────────────────────────────────────────
    bc_warmup_samples: int  = 5000        # rule-based BC 样本数
    bc_warmup_epochs:  int  = 20          # BC 训练 epoch 数
    bc_warmup_lr:      float = 1e-4

    # ── 网络 ────────────────────────────────────────────────────────────────
    hidden_dim:       int   = 1024        # 4 × 1024 MLP

    # ── 运行控制 ────────────────────────────────────────────────────────────
    active_players:      Optional[List[int]] = None   # None = 全部四方
    eval_interval:       int   = 200
    log_interval:        int   = 50
    device:              str   = 'cpu'
    # ── 早停（基于 value_loss 平稳性）──────────────────────────────────────
    early_stop_patience:   int   = 8
    early_stop_vl_delta:   float = 0.15
    early_stop_enabled:    bool  = True


# ==============================================================================
# Belief Replay Buffer (P84)
# ==============================================================================

class BeliefReplayBuffer:
    """
    FIFO replay buffer for Belief Network training data.

    解决灾难性遗忘：JIT burn-in时从整个buffer采样训练，
    而不是只用当前轮的3000局数据。Buffer保留最近几轮的数据，
    自然淘汰过时样本。

    存储格式：5个numpy array，按行对齐。
    """

    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.oh:  Optional[np.ndarray] = None   # observer_hand (N, 52)
        self.h:   Optional[np.ndarray] = None   # history (N, max_len, NUM_BIDS)
        self.op:  Optional[np.ndarray] = None   # observer_pos (N,)
        self.tp:  Optional[np.ndarray] = None   # target_pos (N,)
        self.tgt: Optional[np.ndarray] = None   # belief_target (N, BELIEF_DIM)
        self.size = 0

    def add_batch(self, belief_data: List[dict]):
        """将一批belief数据加入buffer，超出容量则FIFO淘汰最老数据。"""
        if not belief_data:
            return

        oh  = np.stack([s['observer_hand'] for s in belief_data])
        h   = np.stack([s['history']       for s in belief_data])
        op  = np.array([s['observer_pos']  for s in belief_data])
        tp  = np.array([s['target_pos']    for s in belief_data])
        tgt = np.stack([s['belief_target'] for s in belief_data])

        if self.oh is None:
            # 首次添加
            self.oh, self.h, self.op, self.tp, self.tgt = oh, h, op, tp, tgt
        else:
            self.oh  = np.concatenate([self.oh,  oh],  axis=0)
            self.h   = np.concatenate([self.h,   h],   axis=0)
            self.op  = np.concatenate([self.op,  op],  axis=0)
            self.tp  = np.concatenate([self.tp,  tp],  axis=0)
            self.tgt = np.concatenate([self.tgt, tgt], axis=0)

        # FIFO截断
        if len(self.oh) > self.capacity:
            excess = len(self.oh) - self.capacity
            self.oh  = self.oh[excess:]
            self.h   = self.h[excess:]
            self.op  = self.op[excess:]
            self.tp  = self.tp[excess:]
            self.tgt = self.tgt[excess:]

        self.size = len(self.oh)

    def get_tensors(self):
        """返回全量数据的torch tensor（CPU）。"""
        return (
            torch.tensor(self.oh,  dtype=torch.float32),
            torch.tensor(self.h,   dtype=torch.float32),
            torch.tensor(self.op,  dtype=torch.long),
            torch.tensor(self.tp,  dtype=torch.long),
            torch.tensor(self.tgt, dtype=torch.float32),
        )

    def __len__(self):
        return self.size


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
            if len(idx) < 2:          # skip incomplete last batch (std() needs ≥2 samples)
                continue
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
        # P101: belief_conditioned mode uses 397-dim input for Actor & Critic
        _obs_dim = BELIEF_OBS_DIM if config.belief_conditioned else OBS_DIM
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
            obs_dim         = _obs_dim,
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

        # P101: belief_conditioned mode always needs a belief net
        _need_belief = config.use_info_bonus or config.belief_conditioned
        if _need_belief:
            self.belief_net  = BeliefNetwork(hidden_dim=config.hidden_dim).to(self.device)
            if config.use_info_bonus:
                self.dual_info = DualInfoComputer(self.belief_net, beta=config.beta)
            self.belief_optimizer = torch.optim.Adam(
                self.belief_net.parameters(), lr=config.belief_update_lr)
            self.partner_stats  = RunningStats()

        # ── reward 归一化（外部传入，跨 phase 持久）────────────────────────
        self.reward_stats: RunningStats = reward_stats or RunningStats()

        # ── FSP pool ───────────────────────────────────────────────────────
        self.fsp_pool = FSPPool(max_size=config.fsp_pool_size)

        # ── KL anchor（BC checkpoint，由外部 set_bc_anchor 设置）──────────
        self.bc_actors: Optional[Dict[int, MLPPolicyNetwork]] = None

        # ── FSP actor cache（避免每步 new 网络）──────────────────────────
        self._fsp_actor_cache: dict = {}  # role -> MLPPolicyNetwork
        self._fsp_cache_key: Optional[str] = None

        # ── 日志 ───────────────────────────────────────────────────────────
        self.log: List[dict] = []
        self._global_step = 0
        # ── 早停状态 ───────────────────────────────────────────────────────
        self._vl_history: List[float] = []
        self._fsp_seeded: bool = False

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

        # P101: pad with belief prior (96-dim) when belief_conditioned
        # During BC warmup, no belief net is trained yet → use uniform prior.
        # This ensures actor learns on 397-dim input from the start.
        if self.config.belief_conditioned:
            prior = make_belief_features_prior()  # (96,)
            prior_batch = np.tile(prior, (len(data), 1))  # (N, 96)
            flat_obs_np = np.concatenate([flat_obs_np, prior_batch], axis=1)  # (N, 397)

        flat_t   = torch.tensor(flat_obs_np, dtype=torch.float32)
        actions_t = torch.tensor(actions_np, dtype=torch.int64)
        legal_t  = torch.tensor(legal_np,    dtype=torch.float32)

        print(f"[BC Warmup] Training {num_epochs} epochs on {len(data)} samples...")

        for player in [NORTH, SOUTH, EAST, WEST]:  # 四方各自独立 actor
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

            name = {NORTH:'N', SOUTH:'S', EAST:'E', WEST:'W'}[player]
            print(f"  Player {name}: epoch {num_epochs} loss={avg_loss:.4f}")

        print("[BC Warmup] Done.")

    # ======================================================================
    # KL Anchor
    # ======================================================================

    def set_bc_anchor(self, agent_or_state_dict):
        """
        设置 BC 锚点（BC 结束时四方 actor 的快照）.
        必须在 BC warmup 结束后、RL 训练开始前调用。
        """
        def _frozen_copy(state_dict):
            _obs_dim = BELIEF_OBS_DIM if self.config.belief_conditioned else OBS_DIM
            net = MLPPolicyNetwork(obs_dim=_obs_dim,
                                   hidden_dim=self.config.hidden_dim).to(self.device)
            net.load_state_dict(state_dict)
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            return net

        if isinstance(agent_or_state_dict, MAPPOAgent):
            src = agent_or_state_dict.model
        else:
            # state_dict 格式：直接重建 agent 取不了，降级到当前 agent
            src = self.agent.model

        self.bc_actors = {
            NORTH: _frozen_copy(src.actor_n.state_dict()),
            SOUTH: _frozen_copy(src.actor_s.state_dict()),
            EAST:  _frozen_copy(src.actor_e.state_dict()),
            WEST:  _frozen_copy(src.actor_w.state_dict()),
        }
        print("[KL Anchor] BC anchor set for all 4 players.")

    def _get_kl_lambda(self, round_idx: int) -> float:
        cfg = self.config
        if cfg.kl_anneal_frac <= 0:
            # No annealing — fixed at start value
            return cfg.kl_lambda_start
        # Anneal over first kl_anneal_frac of rounds, then hold at end value
        anneal_rounds = max(1, int(cfg.num_rounds * cfg.kl_anneal_frac))
        progress = min(1.0, round_idx / anneal_rounds)
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

        half = num_deals // 2
        print(f"[Critic Warmup] Collecting {num_deals} deals (NS:{half} + EW:{half}, batch={self.config.deals_per_step})...")
        ns_eps = self._collect_episodes_batch(half, train_side='NS', fsp_sd=None,
                                               batch_size=self.config.deals_per_step,
                                               skip_dual_table=True)
        ew_eps = self._collect_episodes_batch(half, train_side='EW', fsp_sd=None,
                                               batch_size=self.config.deals_per_step,
                                               skip_dual_table=True)
        self._store_episodes(ns_eps)
        self._store_episodes(ew_eps)

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
    # Dual-table IMP reward (P55)
    # ======================================================================

    def _play_swapped_table_batch(
        self,
        deals: List[Tuple],   # list of (hands, dd_table, dealer, vulnerability)
        fsp_sd: Optional[dict],
    ) -> List[int]:
        """
        批量换桌 rollout (P56): 替代逐局串行的 _play_swapped_table.

        接受 N 个局面，用与 _collect_episodes_batch 相同的并行 env 机制
        同时 rollout 所有换桌局，全程 GPU 批量 forward，消除串行瓶颈。

        Returns: list of score_sw (int), 与 deals 一一对应。
        """
        from subgames.competitive_env import CompetitiveSubgameEnv, string_to_bid, FIXED_PREFIX

        if not deals:
            return []

        N = len(deals)

        # ── 预处理：换桌 hands/dd ───────────────────────────────────────────
        sw_hands_list = []
        sw_dd_list    = []
        dealers       = []
        vuls          = []
        orig_dd_list  = []

        for hands, dd_table, dealer, vul in deals:
            sw = np.zeros_like(hands)
            sw[0] = hands[1]; sw[1] = hands[0]
            sw[2] = hands[3]; sw[3] = hands[2]
            sw_dd = np.zeros_like(dd_table)
            sw_dd[:, 0] = dd_table[:, 1]; sw_dd[:, 1] = dd_table[:, 0]
            sw_dd[:, 2] = dd_table[:, 3]; sw_dd[:, 3] = dd_table[:, 2]
            sw_hands_list.append(sw)
            sw_dd_list.append(sw_dd)
            dealers.append(dealer)
            vuls.append(vul)
            orig_dd_list.append(sw_dd)

        BridgeBiddingEnv = __import__('env', fromlist=['BridgeBiddingEnv']).BridgeBiddingEnv

        # ── 初始化 N 个独立 inner env ────────────────────────────────────────
        inner_envs = [BridgeBiddingEnv(60) for _ in range(N)]
        slot_obs   = [None] * N
        slot_hist  = [None] * N
        slot_done  = [False] * N
        slot_dd    = orig_dd_list
        slot_vul   = vuls
        slot_dealer= dealers

        # 执行前缀并 reset
        for i in range(N):
            obs = inner_envs[i].reset(sw_hands_list[i], dealer=dealers[i],
                                      vulnerability=vuls[i])
            hist = []
            done = False
            for bid_str in FIXED_PREFIX:
                bid = string_to_bid(bid_str)
                hist.append(bid)
                obs, _, done, _ = inner_envs[i].step(bid)
                if done: break
            slot_obs[i]  = obs
            slot_hist[i] = hist
            slot_done[i] = done

        # 回填已在前缀阶段结束的局（极罕见）
        scores_sw = [None] * N

        # ── 批量 rollout ─────────────────────────────────────────────────────
        from collections import defaultdict

        while any(not d for d in slot_done):
            active = [i for i, d in enumerate(slot_done) if not d]

            groups = defaultdict(list)
            for i in active:
                player = inner_envs[i].state.current_player
                role   = {NORTH:'actor_n', EAST:'actor_e',
                          SOUTH:'actor_s', WEST:'actor_w'}[player]
                groups[role].append(i)

            actions_map = {}
            for role, slots in groups.items():
                if fsp_sd and role in fsp_sd:
                    ck = str(id(fsp_sd))
                    if self._fsp_cache_key != ck or role not in self._fsp_actor_cache:
                        _fsp_obs_dim = BELIEF_OBS_DIM if self.config.belief_conditioned else OBS_DIM
                        net = MLPPolicyNetwork(
                            obs_dim=_fsp_obs_dim,
                            hidden_dim=self.config.hidden_dim).to(self.device)
                        net.load_state_dict(
                            {k: v.to(self.device) for k, v in fsp_sd[role].items()})
                        net.eval()
                        self._fsp_actor_cache[role] = net
                        self._fsp_cache_key = ck
                    actor = self._fsp_actor_cache[role]
                else:
                    actor = getattr(self.agent.model, role)

                flat_batch  = np.stack([
                    encode_obs_flat(slot_obs[i], slot_dealer[i], slot_hist[i])
                    for i in slots])
                legal_batch = np.stack([slot_obs[i]['legal_actions'] for i in slots])

                # P98: append belief features
                if self.config.belief_conditioned and self.belief_net is not None:
                    _hands   = np.stack([inner_envs[i]._current_hands[inner_envs[i].current_player]
                                         for i in slots])
                    _hists   = [slot_hist[i] for i in slots]
                    _players = [inner_envs[i].current_player for i in slots]
                    bf = self._get_belief_features_batch(
                        _hands, _hists, _players, self.belief_net)
                    flat_batch = np.concatenate([flat_batch, bf], axis=1)

                flat_t  = torch.tensor(flat_batch,  dtype=torch.float32).to(self.device)
                legal_t = torch.tensor(legal_batch, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    actions, _, _ = actor.get_action(flat_t, legal_t, deterministic=False)

                for j, i in enumerate(slots):
                    actions_map[i] = actions[j].item()

            for i in active:
                action = actions_map[i]
                if not inner_envs[i]._is_valid_action(action):
                    action = BID_PASS
                slot_hist[i].append(action)
                obs_next, _, done, _ = inner_envs[i].step(action)
                slot_obs[i]  = obs_next
                slot_done[i] = done

                if done:
                    contract   = inner_envs[i].state.final_contract
                    scores_sw[i] = self.env._compute_score_ns(
                        contract, slot_dd[i], slot_vul[i])

        return scores_sw

    def _play_swapped_table(
        self,
        hands: np.ndarray,
        dd_table: np.ndarray,
        dealer: int,
        vulnerability: Tuple[bool, bool],
        fsp_sd: Optional[dict],
    ) -> int:
        """
        在换桌（N↔E, S↔W）上快速 rollout，返回 score_ns（桌1视角）.

        换桌语义:
            桌1 NS的手牌 → 桌2 EW位（seats 1,3）
            桌1 EW的手牌 → 桌2 NS位（seats 0,2）
        故桌2 score_ns（桌2视角NS）= 桌1 EW得分 = -桌1 NS得分（如果策略对称）。
        真正的双桌 IMP = score_to_imp(score_table1 - score_table2_as_table1_ns)
        其中 score_table2_as_table1_ns = -score_table2_ns_perspective。

        实现：直接 rollout 换桌局，返回换桌后的 score_ns（换桌视角），
        调用方计算 IMP = score_to_imp(score1 - (-score2_swapped))
                      = score_to_imp(score1 + score2_swapped)   ← 注意符号
        """
        from subgames.competitive_env import string_to_bid, FIXED_PREFIX

        # 换桌: opener/overcaller 阵营对调
        sw_hands    = np.zeros_like(hands)
        sw_hands[0] = hands[1]; sw_hands[1] = hands[0]
        sw_hands[2] = hands[3]; sw_hands[3] = hands[2]
        sw_dd       = np.zeros_like(dd_table)
        sw_dd[:, 0] = dd_table[:, 1]; sw_dd[:, 1] = dd_table[:, 0]
        sw_dd[:, 2] = dd_table[:, 3]; sw_dd[:, 3] = dd_table[:, 2]

        inner = __import__('env', fromlist=['BridgeBiddingEnv']).BridgeBiddingEnv(60)
        obs  = inner.reset(sw_hands, dealer=dealer, vulnerability=vulnerability)
        hist = []
        done = False

        for bid_str in FIXED_PREFIX:
            bid = string_to_bid(bid_str)
            hist.append(bid)
            obs, _, done, _ = inner.step(bid)
            if done:
                break

        while not done:
            player = inner.state.current_player
            role   = {NORTH:'actor_n', EAST:'actor_e',
                      SOUTH:'actor_s', WEST:'actor_w'}[player]

            # 换桌后，原来的 EW 现在坐 NS 位，用 FSP（对手）actor
            # 原来的 NS 现在坐 EW 位，用当前 agent actor
            # 但为了快速且无梯度，两边都用 inference-only 最新 actor
            if fsp_sd and role in fsp_sd:
                ck = str(id(fsp_sd))
                if self._fsp_cache_key != ck or role not in self._fsp_actor_cache:
                    _fsp_obs_dim = BELIEF_OBS_DIM if self.config.belief_conditioned else OBS_DIM
                    net = MLPPolicyNetwork(obs_dim=_fsp_obs_dim,
                                           hidden_dim=self.config.hidden_dim).to(self.device)
                    net.load_state_dict(
                        {k: v.to(self.device) for k, v in fsp_sd[role].items()})
                    net.eval()
                    self._fsp_actor_cache[role] = net
                    self._fsp_cache_key = ck
                actor = self._fsp_actor_cache[role]
            else:
                actor = getattr(self.agent.model, role)

            flat   = encode_obs_flat(obs, dealer, hist)
            # P98: append belief features
            if self.config.belief_conditioned and self.belief_net is not None:
                bf = self._get_belief_features_single(
                    inner._current_hands[player], hist, player, self.belief_net)
                flat = append_belief_features(flat, bf)
            flat_t = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(self.device)
            legal_t = torch.tensor(obs['legal_actions'], dtype=torch.float32
                                   ).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, _, _ = actor.get_action(flat_t, legal_t, deterministic=False)
            action_int = action.item()

            if not inner._is_valid_action(action_int):
                action_int = BID_PASS
            hist.append(action_int)
            obs, _, done, _ = inner.step(action_int)

        contract = inner.state.final_contract
        score_sw = self.env._compute_score_ns(contract, sw_dd, vulnerability)
        return score_sw

    # ======================================================================
    # Belief Net 独立预训练 (P55)
    # ======================================================================

    def pretrain_belief(self, num_rounds: int = 5, deals_per_round: int = 2000,
                        epochs_per_round: int = 5, max_epochs: int = 300):
        """
        Belief Network 独立预训练 (P55c).

        设计:
          1. 一次性收集 num_rounds × deals_per_round 局全量数据
          2. 训练到 early stopping (patience=15, tol=1e-4)
          3. ReduceLROnPlateau 自动降 LR (factor=0.5, patience=5)
          4. 90/10 train/val 分割

        max_epochs 默认 300：46k 样本约在 150-200 epoch 收敛。
        epochs_per_round 保留向后兼容，不再影响 max_epochs。
        """
        if self.belief_net is None:
            return

        total_deals = num_rounds * deals_per_round
        print(f"\n[Belief Pretrain] Collecting {total_deals} deals (1 pass)...")

        # ── 1. 一次性收集全量数据 ─────────────────────────────────────────────
        all_episodes = self._collect_episodes_batch(
            total_deals, train_side='NS',
            fsp_sd=None, batch_size=self.config.deals_per_step,
            skip_dual_table=True)

        belief_data = []
        for ep in all_episodes:
            for step in ep:
                if 'belief_target' in step and not step.get('ew_diagnostic'):
                    belief_data.append(step)

        if not belief_data:
            print("[Belief Pretrain] No belief data collected, skipping.")
            return

        N = len(belief_data)
        print(f"[Belief Pretrain] Dataset: {N} samples. Training to convergence...")

        oh_all  = torch.tensor(np.stack([s['observer_hand']  for s in belief_data]),
                               dtype=torch.float32)
        h_all   = torch.tensor(np.stack([s['history']        for s in belief_data]),
                               dtype=torch.float32)
        op_all  = torch.tensor([s['observer_pos'] for s in belief_data], dtype=torch.long)
        tp_all  = torch.tensor([s['target_pos']   for s in belief_data], dtype=torch.long)
        tgt_all = torch.tensor(np.stack([s['belief_target']  for s in belief_data]),
                               dtype=torch.float32)

        # 90/10 train/val split
        split    = int(N * 0.9)
        perm_all = np.random.permutation(N)
        tr_idx   = perm_all[:split]
        va_idx   = perm_all[split:]

        criterion = None  # P86: 不再需要独立criterion，用belief_net.compute_loss
        optimizer = torch.optim.Adam(
            self.belief_net.parameters(), lr=self.config.belief_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

        bs         = min(512, split)
        best_val   = float('inf')
        patience   = 15
        no_improve = 0

        # ── 2. 训练到收敛 ─────────────────────────────────────────────────────
        self.belief_net.train()
        for epoch in range(1, max_epochs + 1):
            # Train
            perm     = np.random.permutation(split)
            tr_loss  = 0.0; nb = 0
            for s in range(0, split, bs):
                idx    = tr_idx[perm[s:s+bs]]
                loss   = self.belief_net.compute_loss(
                    oh_all[idx].to(self.device), h_all[idx].to(self.device),
                    op_all[idx].to(self.device), tp_all[idx].to(self.device),
                    tgt_all[idx].to(self.device))
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(
                    self.belief_net.parameters(), self.config.max_grad_norm)
                optimizer.step()
                tr_loss += loss.item(); nb += 1
            tr_loss /= max(1, nb)

            # Validate
            self.belief_net.eval()
            with torch.no_grad():
                val_loss = self.belief_net.compute_loss(
                    oh_all[va_idx].to(self.device), h_all[va_idx].to(self.device),
                    op_all[va_idx].to(self.device), tp_all[va_idx].to(self.device),
                    tgt_all[va_idx].to(self.device)).item()
                probs    = self.belief_net.get_probs(
                    oh_all[va_idx].to(self.device), h_all[va_idx].to(self.device),
                    op_all[va_idx].to(self.device), tp_all[va_idx].to(self.device))
                acc      = belief_accuracy(probs, tgt_all[va_idx].to(self.device))
            self.belief_net.train()

            scheduler.step(val_loss)
            lr_now = optimizer.param_groups[0]['lr']

            print(f"  epoch {epoch:3d}  tr={tr_loss:.4f}  val={val_loss:.4f}  "
                  f"honor={acc['honor_acc']:.3f}  length={acc['length_acc']:.3f}  "
                  f"overall={acc['overall_acc']:.3f}  lr={lr_now:.2e}")

            # Early stopping
            if val_loss < best_val - 1e-4:
                best_val   = val_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  [Belief Pretrain] Early stop @ epoch {epoch} "
                          f"(no improve for {patience} epochs)")
                    break

        self.belief_net.train()
        print(f"[Belief Pretrain] Done. best_val_loss={best_val:.4f}")

        # ── P97: Compute Fisher Information Matrix for EWC ────────────
        if self.config.use_ewc:
            print(f"[Belief Pretrain] Computing Fisher for EWC "
                  f"(λ_ewc={self.config.ewc_lambda}, samples={self.config.ewc_fisher_samples})...")
            self.belief_net.compute_fisher(
                oh_all, h_all, op_all, tp_all, tgt_all,
                num_samples=self.config.ewc_fisher_samples,
            )

        # ── P97b: Save pretrain data for replay mixing ─────────────
        # Store a subsample of pretrain data to mix into on-policy updates.
        # This directly prevents catastrophic forgetting by keeping pretrain
        # loss in the training objective (data-level protection, not weight-level).
        replay_n = min(10000, N)
        replay_idx = np.random.permutation(N)[:replay_n]
        self._pretrain_replay = {
            'oh':  oh_all[replay_idx].clone(),
            'h':   h_all[replay_idx].clone(),
            'op':  op_all[replay_idx].clone(),
            'tp':  tp_all[replay_idx].clone(),
            'tgt': tgt_all[replay_idx].clone(),
        }
        print(f"[Belief Pretrain] Saved {replay_n} pretrain samples for replay mixing.")

    # ======================================================================
    # On-Policy Belief Update (P93)
    # ======================================================================

    def update_belief_on_policy(self, episodes: List[List[dict]]) -> Optional[float]:
        """
        P97b: Train Belief Net on mixed data: on-policy + pretrain replay.

        Key insight from P95/P96/P97:
        - P95: Pure on-policy (30万 samples, 3ep) destroys pretrain → val_loss 1.76→2.19
        - P96: Frozen belief → length 0.488→0.220, r_info becomes noise
        - P97: EWC alone fails — penalty too small relative to 30万 data gradient
        - P97b: Mix pretrain replay data into each batch (50/50 ratio).
          This keeps pretrain loss directly in the objective, preventing
          catastrophic forgetting at the data level rather than weight level.

        Optional: EWC penalty on top of replay for extra stability.

        Returns:
            final validation loss on ON-POLICY data, or None if no data
        """
        if self.belief_net is None or self.belief_optimizer is None:
            return None

        # ── Extract belief data from episodes ──
        belief_data = []
        for ep in episodes:
            for step in ep:
                if 'belief_target' in step and not step.get('ew_diagnostic'):
                    belief_data.append(step)

        if len(belief_data) < 100:
            return None

        oh  = torch.tensor(np.stack([s['observer_hand']  for s in belief_data]), dtype=torch.float32)
        h   = torch.tensor(np.stack([s['history']        for s in belief_data]), dtype=torch.float32)
        op  = torch.tensor(np.array([s['observer_pos']   for s in belief_data]), dtype=torch.long)
        tp  = torch.tensor(np.array([s['target_pos']     for s in belief_data]), dtype=torch.long)
        tgt = torch.tensor(np.stack([s['belief_target']  for s in belief_data]), dtype=torch.float32)

        N = len(belief_data)
        split = int(N * 0.9)
        perm = np.random.permutation(N)
        tr_idx = perm[:split]
        va_idx = perm[split:]

        # ── Check if pretrain replay data is available ──
        has_replay = hasattr(self, '_pretrain_replay') and self._pretrain_replay is not None
        if has_replay:
            rp = self._pretrain_replay
            rp_n = rp['oh'].size(0)

        self.belief_net.train()
        bs = min(512, split)
        half_bs = bs // 2  # half for on-policy, half for replay
        best_val = float('inf')
        no_improve = 0
        final_loss = 0.0

        for epoch in range(self.config.belief_update_epochs):
            ep_perm = np.random.permutation(split)
            rp_perm = np.random.permutation(rp_n) if has_replay else None
            tl = 0.0; tl_rp = 0.0; tl_ewc = 0.0; nb = 0
            rp_ptr = 0

            for s in range(0, split, bs):
                idx = tr_idx[ep_perm[s:s+bs]]

                # ── On-policy loss ──
                loss_op = self.belief_net.compute_loss(
                    oh[idx].to(self.device), h[idx].to(self.device),
                    op[idx].to(self.device), tp[idx].to(self.device),
                    tgt[idx].to(self.device))

                # ── Pretrain replay loss (P97b) ──
                loss_rp = torch.tensor(0.0, device=self.device)
                if has_replay:
                    # Sample a batch of pretrain data (same size as on-policy batch)
                    rp_size = min(len(idx), rp_n)
                    if rp_ptr + rp_size > rp_n:
                        rp_perm = np.random.permutation(rp_n)
                        rp_ptr = 0
                    rp_idx = rp_perm[rp_ptr:rp_ptr + rp_size]
                    rp_ptr += rp_size

                    loss_rp = self.belief_net.compute_loss(
                        rp['oh'][rp_idx].to(self.device),
                        rp['h'][rp_idx].to(self.device),
                        rp['op'][rp_idx].to(self.device),
                        rp['tp'][rp_idx].to(self.device),
                        rp['tgt'][rp_idx].to(self.device))

                # ── Combined loss: 80% on-policy + 20% replay (P97c) ──
                # On-policy dominant so belief tracks current policy,
                # replay minority prevents catastrophic forgetting of pretrain.
                loss = 0.8 * loss_op + 0.2 * loss_rp if has_replay else loss_op

                # ── Optional EWC penalty ──
                ewc_loss_val = 0.0
                if self.config.use_ewc and self.belief_net.has_ewc:
                    ewc_loss = self.belief_net.ewc_penalty()
                    loss = loss + (self.config.ewc_lambda / 2.0) * ewc_loss
                    ewc_loss_val = ewc_loss.item()

                self.belief_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.belief_net.parameters(), self.config.max_grad_norm)
                self.belief_optimizer.step()
                tl += loss_op.item(); tl_rp += loss_rp.item(); tl_ewc += ewc_loss_val; nb += 1
            train_loss = tl / max(1, nb)

            # Validation (on-policy data only — this is what matters for r_info)
            self.belief_net.eval()
            with torch.no_grad():
                val_loss = self.belief_net.compute_loss(
                    oh[va_idx].to(self.device), h[va_idx].to(self.device),
                    op[va_idx].to(self.device), tp[va_idx].to(self.device),
                    tgt[va_idx].to(self.device)).item()
            self.belief_net.train()

            final_loss = val_loss
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 2:  # early stop after 2 epochs no improvement
                    break

        rp_str = f" replay_loss={tl_rp/max(1,nb):.4f}" if has_replay else ""
        ewc_str = f" ewc_pen={tl_ewc/max(1,nb):.6f}" if (self.config.use_ewc and self.belief_net.has_ewc) else ""
        print(f"  [Belief Update] {len(belief_data)} samples, "
              f"{epoch+1} epochs, val_loss={final_loss:.4f}{rp_str}{ewc_str}")
        return final_loss

    # ======================================================================
    # Episode Collection
    # ======================================================================

    def _collect_episodes_batch(
        self,
        num_deals:        int,
        train_side:       str,
        fsp_sd:           Optional[dict] = None,
        batch_size:       int = 32,
        skip_dual_table:  bool = False,   # P55: skip swapped-table rollout (critic warmup)
        use_belief_prior: bool = False,   # P98b: use prior features instead of belief net
    ) -> List[List[dict]]:
        """
        批量 rollout：同时维护 batch_size 个 CompetitiveSubgameEnv 实例。
        每个时间步按 actor 分组批量 forward，显著提升 GPU 利用率。

        关键：每个 slot 使用独立的 CompetitiveSubgameEnv 实例，
        保证 _compute_terminal_reward() 正确计算 DDS oracle reward。

        train_side='NS': NS 用当前 agent，EW 用 frozen FSP
        train_side='EW': EW 用当前 agent，NS 用 frozen FSP
        """
        from subgames.competitive_env import CompetitiveSubgameEnv

        # ── 初始化 batch_size 个独立环境 ────────────────────────────────
        envs = [CompetitiveSubgameEnv.__new__(CompetitiveSubgameEnv)
                for _ in range(batch_size)]
        for e in envs:
            e.loader          = self.env.loader
            e.env             = __import__('env', fromlist=['BridgeBiddingEnv']).BridgeBiddingEnv(60)
            e.max_history_len = 60
            e.dealer          = NORTH
            e._sampled_dealer  = NORTH
            e._is_constrained_data = self.env._is_constrained_data
            e._filtered_deals      = self.env._filtered_deals
            e._current_hands = None; e._current_dd = None
            e._vulnerability = (False, False); e.history_int = []

        slot_obs    = [None] * batch_size
        slot_hist   = [None] * batch_size
        slot_dealer = [NORTH] * batch_size   # per-slot dealer (set on reset)
        slot_ep     = [[]   for _ in range(batch_size)]
        slot_done   = [True] * batch_size

        all_episodes: List[List[dict]] = []
        _pending_rewards: List[Tuple]   = []   # P76: single-table DDS IMP reward queue
        collected = 0

        def _reset(i):
            obs = envs[i].reset()
            slot_hist[i]   = list(envs[i].history_int)
            slot_dealer[i] = envs[i].dealer   # dealer chosen by generate_deal()
            slot_ep[i]     = []
            slot_done[i]   = False
            return obs

        slot_obs = [_reset(i) for i in range(batch_size)]

        while collected < num_deals:
            active = [i for i in range(batch_size) if not slot_done[i]]
            if not active:
                break

            # 按 actor 分组
            from collections import defaultdict
            groups = defaultdict(list)  # role_key -> [slot_idx]
            for i in active:
                player = envs[i].current_player
                is_train = ((train_side == 'NS' and player in (NORTH, SOUTH)) or
                            (train_side == 'EW' and player in (EAST,  WEST)))
                role = {NORTH:'actor_n', EAST:'actor_e',
                        SOUTH:'actor_s', WEST:'actor_w'}[player]
                key = f"train_{role}" if is_train else f"fsp_{role}"
                groups[key].append(i)

            actions_map = {}  # slot -> (action, log_prob, value, flat, legal, is_train)

            for key, slots in groups.items():
                is_train = key.startswith('train_')
                role = key.split('_', 1)[1]  # actor_n/s/e/w

                if is_train:
                    actor  = getattr(self.agent.model, role)
                    critic = getattr(self.agent.model,
                                     role.replace('actor','critic'))
                else:
                    if fsp_sd and role in fsp_sd:
                        ck = str(id(fsp_sd))
                        if self._fsp_cache_key != ck or role not in self._fsp_actor_cache:
                            _fsp_obs_dim = BELIEF_OBS_DIM if self.config.belief_conditioned else OBS_DIM
                            net = MLPPolicyNetwork(
                                obs_dim=_fsp_obs_dim,
                                hidden_dim=self.config.hidden_dim).to(self.device)
                            net.load_state_dict(
                                {k: v.to(self.device) for k, v in fsp_sd[role].items()})
                            net.eval()
                            self._fsp_actor_cache[role] = net
                            self._fsp_cache_key = ck
                        actor = self._fsp_actor_cache[role]
                    else:
                        actor = getattr(self.agent.model, role)
                    critic = getattr(self.agent.model,
                                     role.replace('actor','critic'))

                flat_batch  = np.stack([
                    encode_obs_flat(slot_obs[i], slot_dealer[i], slot_hist[i])
                    for i in slots])
                legal_batch = np.stack([slot_obs[i]['legal_actions'] for i in slots])
                ah_batch    = np.stack([envs[i]._current_hands for i in slots])

                # P98/P98b: append belief features if belief_conditioned
                if self.config.belief_conditioned:
                    if use_belief_prior or self.belief_net is None:
                        # P98b: warmup — use uninformative prior to avoid garbage input
                        bf = np.tile(make_belief_features_prior(), (len(slots), 1))
                    else:
                        _hands   = np.stack([envs[i]._current_hands[envs[i].current_player]
                                             for i in slots])
                        _hists   = [slot_hist[i] for i in slots]
                        _players = [envs[i].current_player for i in slots]
                        bf = self._get_belief_features_batch(
                            _hands, _hists, _players, self.belief_net)
                    flat_batch = np.concatenate([flat_batch, bf], axis=1)

                flat_t  = torch.tensor(flat_batch,  dtype=torch.float32).to(self.device)
                legal_t = torch.tensor(legal_batch, dtype=torch.float32).to(self.device)
                ah_t    = torch.tensor(ah_batch,    dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    actions, log_probs, _ = actor.get_action(flat_t, legal_t)
                    values = critic(flat_t, ah_t)

                for j, i in enumerate(slots):
                    actions_map[i] = (actions[j].item(), log_probs[j].cpu(),
                                      values[j].cpu(), flat_batch[j],
                                      legal_batch[j], is_train)

            # ── 执行动作 ────────────────────────────────────────────────
            for i in active:
                action, log_prob, value, flat_obs, legal_actions, is_train = actions_map[i]
                player    = envs[i].current_player
                all_hands = envs[i]._current_hands.copy()

                opener_seats_i = {slot_dealer[i], (slot_dealer[i] + 2) % 4}
                step = {
                    'flat_obs': flat_obs, 'legal_actions': legal_actions,
                    'action': action, 'log_prob': log_prob, 'value': value,
                    'reward': 0.0, 'done': False,
                    'all_hands': all_hands, 'player': player,
                    'is_training_side': is_train,
                    'is_opener': player in opener_seats_i,
                }

                # Belief 数据记录
                _dealer_i      = slot_dealer[i]
                opener_seats_i = {_dealer_i, (_dealer_i + 2) % 4}
                if self.belief_net is not None:
                    if player in opener_seats_i:
                        partner = (player + 2) % 4
                        step.update({
                            'observer_hand': all_hands[player],
                            'history':       self._encode_history(slot_hist[i]),
                            'observer_pos':  player, 'target_pos': partner,
                            'belief_target': hand_to_belief_target(all_hands[partner]),
                            'history_int_before': slot_hist[i][:],
                        })
                    else:
                        observer = _dealer_i
                        step.update({
                            'ew_diagnostic': True,
                            'observer_hand': all_hands[observer],
                            'history':       self._encode_history(slot_hist[i]),
                            'observer_pos':  observer, 'target_pos': player,
                            'belief_target': hand_to_belief_target(all_hands[player]),
                            'history_int_before': slot_hist[i][:],
                        })

                slot_hist[i].append(action)
                if self.belief_net is not None:
                    if player in opener_seats_i or step.get('ew_diagnostic'):
                        step['history_int_after'] = slot_hist[i][:]

                # env.step() gives oracle reward; we overwrite with dual-table IMP after batch
                obs_next, reward, done, info = envs[i].step(action)
                step['reward'] = reward   # placeholder, overwritten if not skip_dual_table
                step['done']   = done
                slot_ep[i].append(step)
                slot_obs[i] = obs_next

                if done:
                    all_episodes.append(slot_ep[i])
                    # P76: 单桌DDS IMP reward，不再做换桌rollout
                    if not skip_dual_table:
                        _pending_rewards.append((
                            len(all_episodes) - 1,
                            envs[i]._current_dd.copy(),
                            slot_dealer[i],
                            envs[i]._vulnerability,
                            envs[i].env.state.final_contract,
                        ))
                    collected += 1
                    slot_done[i] = True
                    if collected < num_deals:
                        slot_obs[i] = _reset(i)

        # ── P82: 每个player的最后一步都标记done=True并赋对应reward ────────
        # 旧逻辑只给env.step()返回done=True的那个step（最后行动者）赋reward，
        # 导致同一局中其他player的所有step reward=0且done=False，
        # GAE无法正确切分episode边界，且大部分player学不到任何信号。
        # 修复：遍历episode找到每个player最后出现的step，标记done+赋reward。
        if not skip_dual_table and _pending_rewards:
            for (ep_idx, dd, dealer, vul, contract) in _pending_rewards:
                score_ns    = self.env._compute_score_ns(contract, dd, vul)
                score_opt   = self.env._compute_dds_optimal_score_ns(dd, vul)
                imp_ns      = float(score_to_imp(score_ns  - score_opt))
                imp_ew      = float(score_to_imp(score_opt - score_ns))
                opener_seats = {dealer, (dealer + 2) % 4}

                # 找每个player最后出现的step index
                last_step_idx: Dict[int, int] = {}
                for s_idx, s in enumerate(all_episodes[ep_idx]):
                    last_step_idx[s['player']] = s_idx

                # 给每个player的最后一步标记done=True并赋reward
                for player, s_idx in last_step_idx.items():
                    s = all_episodes[ep_idx][s_idx]
                    s['done']   = True
                    s['reward'] = imp_ns if player in opener_seats else imp_ew

        return all_episodes[:num_deals]

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

        # FSP: 如果有对手 snapshot，临时加载到 EW actor
        fsp_sd     = fsp_state_dict
        if use_fsp_opponent and fsp_sd is None and not self.fsp_pool.is_empty():
            fsp_sd = self.fsp_pool.sample()

        for _ in range(num_deals):
            hands, dd_table = self.env.generate_deal()
            dealer = self.env.dealer   # set by generate_deal() via _sampled_dealer
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

                # P98: append belief features
                if self.config.belief_conditioned and self.belief_net is not None:
                    bf = self._get_belief_features_single(
                        all_hands[player], history_int, player, self.belief_net)
                    flat_obs = append_belief_features(flat_obs, bf)

                # 选 actor: FSP 对手 or 当前 agent
                flat_t  = torch.tensor(flat_obs,      dtype=torch.float32
                                       ).unsqueeze(0).to(self.device)
                legal_t = torch.tensor(legal_actions, dtype=torch.float32
                                       ).unsqueeze(0).to(self.device)
                ah_t    = torch.tensor(all_hands,      dtype=torch.float32
                                       ).unsqueeze(0).to(self.device)

                overcaller_seats = {(dealer + 1) % 4, (dealer + 3) % 4}
                is_opponent = (player in overcaller_seats)
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
                opener_seats     = {dealer, (dealer + 2) % 4}
                if self.belief_net is not None:
                    if player in opener_seats:
                        # 开叫方阵营决策步骤：记录完整 r_info 所需数据
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
                        # 争叫方决策步骤：记录开叫方视角对争叫方手牌推断的诊断数据
                        observer = dealer   # opener = observer for EW diagnostic
                        step.update({
                            'ew_diagnostic':      True,
                            'observer_hand':      all_hands[observer],
                            'history':            self._encode_history(history_int),
                            'observer_pos':       observer,
                            'target_pos':         player,
                            'belief_target':      hand_to_belief_target(all_hands[player]),
                            'history_int_before': history_int[:],
                        })

                history_int.append(action.item())

                # 补充 history_int_after
                if self.belief_net is not None:
                    if player in opener_seats or step.get('ew_diagnostic'):
                        step['history_int_after'] = history_int[:]
                obs, reward, done, info = self.env.step(action.item())

                step['reward'] = reward
                step['done']   = done
                ep.append(step)

            episodes.append(ep)

        return episodes

    def _get_fsp_actor(self, player: int, fsp_sd: dict) -> MLPPolicyNetwork:
        """FSP snapshot actor，缓存避免每步 new 网络."""
        role = {NORTH:'actor_n', EAST:'actor_e',
                SOUTH:'actor_s', WEST:'actor_w'}[player]
        if role not in fsp_sd:
            return self.agent.get_actor(player)
        cache_key = str(id(fsp_sd))
        if self._fsp_cache_key != cache_key or role not in self._fsp_actor_cache:
            _fsp_obs_dim = BELIEF_OBS_DIM if self.config.belief_conditioned else OBS_DIM
            net = MLPPolicyNetwork(obs_dim=_fsp_obs_dim,
                                   hidden_dim=self.config.hidden_dim).to(self.device)
            net.load_state_dict({k: v.to(self.device) for k, v in fsp_sd[role].items()})
            net.eval()
            self._fsp_actor_cache[role] = net
            self._fsp_cache_key = cache_key
        return self._fsp_actor_cache[role]

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
    # P98: Belief features for Actor input
    # ======================================================================

    def _get_belief_features_single(
        self,
        hand: np.ndarray,
        history_int: list,
        player: int,
        belief_net: Optional[BeliefNetwork] = None,
    ) -> np.ndarray:
        """
        P101: 获取单步 belief features (96,) 用于 actor 输入.

        查询 belief_net 两次:
          1. partner: "给定我的手牌和叫牌历史, partner 的手牌分布是什么?"
          2. RHO: "给定我的手牌和叫牌历史, 右手对手的手牌分布是什么?"

        返回 partner (48) + RHO (48) = 96 维.

        Args:
            hand:         当前玩家手牌 (52,)
            history_int:  叫牌历史 (整数列表)
            player:       当前玩家 seat (0-3)
            belief_net:   使用哪个 belief net (None=self.belief_net)

        Returns:
            belief_feats: (96,) float32 — [partner 48 | RHO 48]
        """
        bn = belief_net or self.belief_net
        if bn is None or not self.config.belief_conditioned:
            return make_belief_features_prior()

        partner = (player + 2) % 4
        rho     = (player - 1) % 4    # right-hand opponent (bid just before you)
        hist_enc = self._encode_history(history_int)

        with torch.no_grad():
            oh_t = torch.tensor(hand, dtype=torch.float32).unsqueeze(0).to(self.device)
            h_t  = torch.tensor(hist_enc, dtype=torch.float32).unsqueeze(0).to(self.device)
            op_t = torch.tensor([player],  dtype=torch.long).to(self.device)

            # Query partner
            tp_partner = torch.tensor([partner], dtype=torch.long).to(self.device)
            partner_probs = bn.get_probs(oh_t, h_t, op_t, tp_partner)  # (1, 48)

            # Query RHO
            tp_rho = torch.tensor([rho], dtype=torch.long).to(self.device)
            rho_probs = bn.get_probs(oh_t, h_t, op_t, tp_rho)          # (1, 48)

        return torch.cat([partner_probs, rho_probs], dim=-1).squeeze(0).cpu().numpy()

    def _get_belief_features_batch(
        self,
        hands: np.ndarray,
        history_ints: List[list],
        players: List[int],
        belief_net: Optional[BeliefNetwork] = None,
    ) -> np.ndarray:
        """
        P101: 批量获取 belief features (B, 96) 用于 actor 输入.

        对每个样本查询 belief_net 两次 (partner + RHO), 拼接返回 96 维.

        Args:
            hands:         (B, 52) 当前玩家手牌
            history_ints:  长度 B 的列表, 每个元素是叫牌历史
            players:       长度 B 的列表, 每个元素是当前玩家 seat
            belief_net:    使用哪个 belief net (None=self.belief_net)

        Returns:
            belief_feats: (B, 96) float32 — [partner 48 | RHO 48]
        """
        bn = belief_net or self.belief_net
        if bn is None or not self.config.belief_conditioned:
            return np.tile(make_belief_features_prior(), (len(players), 1))

        B = len(players)
        partners = [(p + 2) % 4 for p in players]
        rhos     = [(p - 1) % 4 for p in players]
        hist_encs = np.stack([self._encode_history(h) for h in history_ints])

        with torch.no_grad():
            oh_t = torch.tensor(hands, dtype=torch.float32).to(self.device)
            h_t  = torch.tensor(hist_encs, dtype=torch.float32).to(self.device)
            op_t = torch.tensor(players,  dtype=torch.long).to(self.device)

            # Query partner
            tp_partner = torch.tensor(partners, dtype=torch.long).to(self.device)
            partner_probs = bn.get_probs(oh_t, h_t, op_t, tp_partner)  # (B, 48)

            # Query RHO
            tp_rho = torch.tensor(rhos, dtype=torch.long).to(self.device)
            rho_probs = bn.get_probs(oh_t, h_t, op_t, tp_rho)          # (B, 48)

        return torch.cat([partner_probs, rho_probs], dim=-1).cpu().numpy()

    def _encode_for_actor(
        self,
        obs: dict,
        dealer: int,
        history_int: list,
        player: int,
        all_hands: Optional[np.ndarray] = None,
        belief_net: Optional[BeliefNetwork] = None,
    ) -> np.ndarray:
        """
        P101: 统一的 actor 输入编码.

        base mode (301-dim):         encode_obs_flat(obs, dealer, history_int)
        belief-conditioned (397-dim): base + partner_belief(48) + rho_belief(48)

        所有 policy closure 和 rollout 都应使用此方法。
        """
        flat = encode_obs_flat(obs, dealer, history_int)
        if self.config.belief_conditioned:
            hand = all_hands[player] if all_hands is not None else obs['hand']
            bf = self._get_belief_features_single(
                hand, history_int, player, belief_net or self.belief_net)
            flat = append_belief_features(flat, bf)
        return flat

    # ======================================================================
    # r_info 计算
    # ======================================================================

    def _compute_info_bonus(self, episodes: List[List[dict]]) -> List[List[float]]:
        """
        为每个 episode 的每个 step 计算 r_info，并归一化到 IMP 量纲。

        r_info_raw = I(bid; hand | partner) - β × I(bid; hand | opponent)
                   ≈ CE_reduction(belief_before, target) - CE_reduction(belief_after, target)

        P87b: 保留动态归一化（imp_std/rinfo_std 自动对齐量纲），
        w 的语义 = r_info 占 IMP 方差的比例。
        从 w=0.02（step_ir≈0.09，PPO 无法利用）提升到 w=0.2（step_ir≈1.4）。
        """
        if self.dual_info is None:
            return [[0.0] * len(ep) for ep in episodes]

        # P75 Fix A: 全批量化——从O(N×4次单样本forward)降到O(4次批量forward)
        valid_steps: List[dict] = []
        ep_step_idx: List[tuple] = []
        for ep_idx, ep in enumerate(episodes):
            for s_idx, step in enumerate(ep):
                if 'belief_target' in step and not step.get('ew_diagnostic'):
                    valid_steps.append(step)
                    ep_step_idx.append((ep_idx, s_idx))

        raw_ep_bonuses: List[List[float]] = [[0.0] * len(ep) for ep in episodes]
        if not valid_steps:
            return raw_ep_bonuses

        n = len(valid_steps)
        oh_arr   = np.stack([s['observer_hand'] for s in valid_steps])
        hb_arr   = np.stack([self._encode_history(s.get('history_int_before', [])) for s in valid_steps])
        ha_arr   = np.stack([self._encode_history(s.get('history_int_after',  [])) for s in valid_steps])
        tgt_arr  = np.stack([s['belief_target']  for s in valid_steps])
        op_arr   = np.array([s['player']          for s in valid_steps])
        pp_arr   = np.array([(s['player']+2)%4    for s in valid_steps])
        oo_arr   = np.array([(s['player']+1)%4    for s in valid_steps])
        opp_tgt_arr = np.stack([hand_to_belief_target(s['all_hands'][(s['player']+1)%4]) for s in valid_steps])

        oh_t      = torch.tensor(oh_arr,      dtype=torch.float32).to(self.device)
        hb_t      = torch.tensor(hb_arr,      dtype=torch.float32).to(self.device)
        ha_t      = torch.tensor(ha_arr,      dtype=torch.float32).to(self.device)
        tgt_t     = torch.tensor(tgt_arr,     dtype=torch.float32).to(self.device)
        op_t      = torch.tensor(op_arr,      dtype=torch.long).to(self.device)
        pp_t      = torch.tensor(pp_arr,      dtype=torch.long).to(self.device)
        oo_t      = torch.tensor(oo_arr,      dtype=torch.long).to(self.device)
        opp_tgt_t = torch.tensor(opp_tgt_arr, dtype=torch.float32).to(self.device)

        CHUNK = 2048
        bonuses_flat = np.zeros(n, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, n, CHUNK):
                sl = slice(start, start + CHUNK)
                b_before_p   = self.belief_net.get_probs(oh_t[sl], hb_t[sl], op_t[sl], pp_t[sl])
                b_after_p    = self.belief_net.get_probs(oh_t[sl], ha_t[sl], op_t[sl], pp_t[sl])
                partner_gain = self.dual_info.compute_info_gain(b_before_p, b_after_p, tgt_t[sl])
                b_before_o   = self.belief_net.get_probs(oh_t[sl], hb_t[sl], op_t[sl], oo_t[sl])
                b_after_o    = self.belief_net.get_probs(oh_t[sl], ha_t[sl], op_t[sl], oo_t[sl])
                opp_leak     = self.dual_info.compute_info_gain(b_before_o, b_after_o, opp_tgt_t[sl])
                bonus, _     = self.dual_info.compute_dual_info_bonus(partner_gain, opp_leak)
                bonuses_flat[start:start + CHUNK] = bonus.cpu().numpy()

        for i, (ep_idx, s_idx) in enumerate(ep_step_idx):
            raw_ep_bonuses[ep_idx][s_idx] = float(bonuses_flat[i])

        # ── 量纲归一化 (P73) + 外部权重 (P87b) ─────────────────────────────
        # 集中到最后一步（与IMP时序对齐），只做Scale不做Shift。
        # P87b: 恢复动态归一化（w 的语义 = r_info 占 IMP 方差的比例），
        # 仅将 w 从 0.02 提升到 0.2 使 step_ir 达到 ~1.5 IMP。
        ep_totals = [sum(b for b in bs) for bs in raw_ep_bonuses]
        for v in ep_totals:
            self.partner_stats.update(v)
        imp_std   = max(self.reward_stats.std, 1.0)
        rinfo_std = max(self.partner_stats.std, 1e-6)
        scale_factor = min(imp_std / rinfo_std, 1000.0) * self.config.info_reward_weight
        scaled_episodes = []
        for ep_bonuses, ep_total in zip(raw_ep_bonuses, ep_totals):
            scaled = [0.0] * len(ep_bonuses)
            scaled[-1] = ep_total * scale_factor
            scaled_episodes.append(scaled)
        return scaled_episodes

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

                # Normalize advantage — guard against degenerate batches
                adv_std = b_adv.std()
                if torch.isnan(adv_std) or adv_std < 1e-8:
                    continue   # skip this batch entirely
                adv = (b_adv - b_adv.mean()) / (adv_std + 1e-8)

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

                if torch.isnan(actor_loss):
                    continue   # skip batch — NaN loss would corrupt weights

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

    def run(self, num_rounds: int = None, sl_trainer: "SubgameTrainer" = None,
            h2h_deals: int = 500) -> List[dict]:
        """
        双桌对称训练 (P82):
          每轮对同一批牌做两次 rollout:
            桌1: agent=NS, FSP=EW → NS actors 学习
            桌2: agent=EW, FSP=NS → EW actors 学习
          Reward: 每桌独立 DDS regret = score_to_imp(score - dds_optimal)，可正可负。
          四方 actor 每轮都更新。
        """
        num_rounds = num_rounds or self.config.num_rounds
        n_deals    = self.config.steps_per_phase * self.config.deals_per_step
        batch_sz   = self.config.deals_per_step
        cfg = self.config
        print(f"[Config] kl_lambda_start={cfg.kl_lambda_start}  kl_lambda_end={cfg.kl_lambda_end}  kl_anneal_frac={cfg.kl_anneal_frac}")
        print(f"[Config] num_rounds={num_rounds}  deals_per_step={cfg.deals_per_step}  n_deals_per_phase={n_deals}  use_info_bonus={cfg.use_info_bonus}  beta={cfg.beta}  info_weight={cfg.info_reward_weight}")

        print("\n[Trainer] Critic warmup...")
        self.critic_warmup()

        # P74: FSP Pool用BC checkpoint作为pool[0]
        if not self._fsp_seeded and self.config.fsp_pool_size > 0:
            self.fsp_pool.add_permanent(self.agent)  # P90: SL baseline永不淘汰
            self._fsp_seeded = True
            print(f"  [FSP] Seeded pool with BC checkpoint as permanent (pool size: {len(self.fsp_pool)})")

        for rnd in range(num_rounds):
            print(f"\n══════ Round {rnd+1}/{num_rounds} ══════")

            # P98b: during belief warmup rounds, use prior features instead of belief net
            _bw = cfg.belief_conditioned and cfg.belief_warmup_rounds > 0 and rnd < cfg.belief_warmup_rounds
            if _bw:
                print(f"  [Belief] Warmup round {rnd+1}/{cfg.belief_warmup_rounds}: "
                      f"using prior features (Belief Net not yet adapted to RL dist)")

            self._maybe_add_to_fsp(rnd)
            fsp_sd = self._apply_fsp_opponent()
            _ir_vals: list = []; _bl_vals: list = []
            ns_metrics: dict = {}; ew_metrics: dict = {}

            # ── (P93: JIT burn-in removed — belief update moved to after PPO) ──

            # ── 桌1: agent=NS, FSP=EW ──────────────────────────────────
            print(f"  [Table1/NS] Collecting {n_deals} deals (batch={batch_sz})...")
            ns_eps = self._collect_episodes_batch(
                n_deals, train_side='NS', fsp_sd=fsp_sd, batch_size=batch_sz,
                use_belief_prior=_bw)

            # 收集NS方的DDS regret用于reward_stats和日志（每局一个值）
            raw_ns_vals = []
            for ep in ns_eps:
                for step in ep:
                    if step.get('done') and step['player'] in (NORTH, SOUTH):
                        v = float(step['reward'])
                        self.reward_stats.update(v)
                        raw_ns_vals.append(v)
                        break  # 同一局NS方reward相同，只取一次

            # r_info bonus（Agent B only, NS桌）
            if self.config.use_info_bonus:
                bonuses = self._compute_info_bonus(ns_eps)
                for ep, bs in zip(ns_eps, bonuses):
                    for step, b in zip(ep, bs):
                        step['reward'] += b
                        if b != 0.0: _ir_vals.append(b)

            # 存NS actors的数据并update
            for ep in ns_eps:
                for step in ep:
                    if step['player'] in (NORTH, SOUTH):
                        self.buffers[step['player']].add(
                            flat_obs=step['flat_obs'], legal_actions=step['legal_actions'],
                            action=step['action'], log_prob=step['log_prob'],
                            reward=step['reward'], value=step['value'],
                            done=step['done'], all_hands=step.get('all_hands'))
            for p in (NORTH, SOUTH):
                m = self._safe_update(p, rnd)
                if m: ns_metrics[p] = m

            # ── 桌2: agent=EW, FSP=NS ──────────────────────────────────
            print(f"  [Table2/EW] Collecting {n_deals} deals (batch={batch_sz})...")
            ew_eps = self._collect_episodes_batch(
                n_deals, train_side='EW', fsp_sd=fsp_sd, batch_size=batch_sz,
                use_belief_prior=_bw)

            # 收集EW方的DDS regret（每局一个值）
            raw_ew_vals = []
            for ep in ew_eps:
                for step in ep:
                    if step.get('done') and step['player'] in (EAST, WEST):
                        v = float(step['reward'])
                        self.reward_stats.update(v)
                        raw_ew_vals.append(v)
                        break  # 同一局EW方reward相同，只取一次

            # r_info bonus（Agent B only, EW桌）
            if self.config.use_info_bonus:
                ew_bonuses = self._compute_info_bonus(ew_eps)
                for ep, bs in zip(ew_eps, ew_bonuses):
                    for step, b in zip(ep, bs):
                        step['reward'] += b
                        if b != 0.0: _ir_vals.append(b)

            # 存EW actors的数据并update
            for ep in ew_eps:
                for step in ep:
                    if step['player'] in (EAST, WEST):
                        self.buffers[step['player']].add(
                            flat_obs=step['flat_obs'], legal_actions=step['legal_actions'],
                            action=step['action'], log_prob=step['log_prob'],
                            reward=step['reward'], value=step['value'],
                            done=step['done'], all_hands=step.get('all_hands'))
            for p in (EAST, WEST):
                m = self._safe_update(p, rnd)
                if m: ew_metrics[p] = m

            # ── Belief Update: frozen (P96) or on-policy (P93/P95) ─────────
            if self.belief_net is not None and not self.config.freeze_belief:
                bl = self.update_belief_on_policy(ns_eps + ew_eps)
                if bl is not None: _bl_vals.append(bl)

            # ── 日志 ────────────────────────────────────────────────────
            all_vals = raw_ns_vals + raw_ew_vals
            mean_r = float(np.mean(all_vals)) if all_vals else 0.0
            std_r  = float(np.std(all_vals))  if all_vals else 0.0
            mean_ns = float(np.mean(raw_ns_vals)) if raw_ns_vals else 0.0
            mean_ew = float(np.mean(raw_ew_vals)) if raw_ew_vals else 0.0

            log_entry = {
                'round': rnd+1, 'mean_reward': mean_r, 'std_reward': std_r,
                'mean_ns_regret': mean_ns, 'mean_ew_regret': mean_ew,
                'ns_metrics': ns_metrics, 'ew_metrics': ew_metrics,
                'fsp_pool_size': len(self.fsp_pool),
                'mean_ir':    float(np.mean(_ir_vals)) if _ir_vals else None,
                'belief_loss': float(np.mean(_bl_vals)) if _bl_vals else None,
                'imp_std_running': float(self.reward_stats.std),
            }
            self.log.append(log_entry)
            self._print_log(log_entry)

            # ── P91: Mini vs SL eval (every round, 500 deals) ────────
            if sl_trainer is not None:
                h2h_result = self.evaluate_head_to_head(
                    sl_trainer, num_deals=h2h_deals,
                    label_self="agent", label_other="SL")
                log_entry['vs_sl_imp'] = h2h_result['mean_imp']
                log_entry['vs_sl_p'] = h2h_result['p_value']

            # P74: 早停——基于value_loss平稳性
            if self.config.early_stop_enabled:
                all_vl = []
                for m in list(ns_metrics.values()) + list(ew_metrics.values()):
                    if 'value_loss' in m:
                        all_vl.append(m['value_loss'])
                if all_vl:
                    self._vl_history.append(float(np.mean(all_vl)))
                pat = self.config.early_stop_patience
                if len(self._vl_history) >= pat:
                    window = self._vl_history[-pat:]
                    vl_range = max(window) - min(window)
                    if vl_range < self.config.early_stop_vl_delta:
                        print(f"\n  [Early Stop] vl plateau (range={vl_range:.3f} < {self.config.early_stop_vl_delta} over {pat} rounds). Stopping at round {rnd+1}.")
                        break

        return self.log

    def _print_log(self, entry: dict):
        rnd = entry['round']
        fsp = entry['fsp_pool_size']
        mr  = entry['mean_reward']
        sr  = entry['std_reward']
        ns_r = entry.get('mean_ns_regret', 0.0)
        ew_r = entry.get('mean_ew_regret', 0.0)

        print(f"  [Round {rnd}] regret={mr:+.3f}±{sr:.3f}  (NS={ns_r:+.3f} EW={ew_r:+.3f})  fsp={fsp}")

        ns = entry.get('ns_metrics', {})
        ns_n = ns.get(NORTH, {}); ns_s = ns.get(SOUTH, {})
        if ns_n or ns_s:
            print(f"    NS │ N: pl={ns_n.get('policy_loss',0):+.4f} "
                  f"vl={ns_n.get('value_loss',0):.3f} "
                  f"ent={ns_n.get('entropy',0):.3f} │ "
                  f"S: pl={ns_s.get('policy_loss',0):+.4f} "
                  f"vl={ns_s.get('value_loss',0):.3f} "
                  f"ent={ns_s.get('entropy',0):.3f} "
                  f"kl={ns_s.get('kl_loss',0):.5f}(λ={ns_s.get('kl_lambda',0):.3f})")

        ew = entry.get('ew_metrics', {})
        ew_e = ew.get(EAST, {}); ew_w = ew.get(WEST, {})
        if ew_e or ew_w:
            print(f"    EW │ E: pl={ew_e.get('policy_loss',0):+.4f} "
                  f"vl={ew_e.get('value_loss',0):.3f} "
                  f"ent={ew_e.get('entropy',0):.3f} │ "
                  f"W: pl={ew_w.get('policy_loss',0):+.4f} "
                  f"vl={ew_w.get('value_loss',0):.3f} "
                  f"ent={ew_w.get('entropy',0):.3f} "
                  f"kl={ew_e.get('kl_loss',0):.5f}(λ={ew_e.get('kl_lambda',0):.3f})")

        ir = entry.get('mean_ir')
        bl = entry.get('belief_loss')
        if ir is not None:
            bl_str = f"{bl:.4f}" if bl is not None else "N/A"
            print(f"    r_info │ step_ir={ir:.4f}  belief_loss={bl_str}")

    # ======================================================================
    # 评估
    # ======================================================================

    def _mini_h2h_snapshot(self, num_deals: int = 100) -> float:
        """
        Mini head-to-head: 当前agent vs FSP pool最新checkpoint，双桌IMP。

        正值=当前agent比上一个版本自己强，是训练收敛的直接信号。
        负值=训练后退化。
        FSP pool为空时（训练初期）返回None。
        """
        from subgames.competitive_env import cross_evaluate

        if self.fsp_pool.is_empty():
            return None

        fsp_sd = self.fsp_pool.latest()   # 最新的FSP checkpoint作为基准对手
        env    = self.env
        device = self.device

        def _current_policy(obs, player, history_int):
            flat   = self._encode_for_actor(obs, env.dealer, history_int, player,
                                            env._current_hands)
            flat_t = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(device)
            legal  = torch.tensor(obs['legal_actions'], dtype=torch.float32).unsqueeze(0).to(device)
            actor  = self.agent.get_actor(player)
            with torch.no_grad():
                action, _, _ = actor.get_action(flat_t, legal, deterministic=True)
            return action.item()

        def _fsp_policy(obs, player, history_int):
            flat   = self._encode_for_actor(obs, env.dealer, history_int, player,
                                            env._current_hands)
            flat_t = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(device)
            legal  = torch.tensor(obs['legal_actions'], dtype=torch.float32).unsqueeze(0).to(device)
            role   = {0:'actor_n', 1:'actor_e', 2:'actor_s', 3:'actor_w'}[player]
            if role not in fsp_sd:
                actor = self.agent.get_actor(player)
            else:
                actor = self._get_fsp_actor(player, fsp_sd)
            with torch.no_grad():
                action, _, _ = actor.get_action(flat_t, legal, deterministic=True)
            return action.item()

        result = cross_evaluate(
            env,
            agent_a_ns_policy=_current_policy,
            agent_a_ew_policy=_current_policy,
            agent_b_ns_policy=_fsp_policy,
            agent_b_ew_policy=_fsp_policy,
            num_deals=num_deals,
        )
        return float(result.mean_imp)

    def _oracle_snapshot(self, num_deals: int = 100) -> float:
        """
        轻量 oracle 快照：估计当前策略的 IMP regret（仅供观察）.
        """
        from subgames.competitive_env import dds_oracle_evaluate

        agent    = self.agent
        env      = self.env
        device   = self.device

        def _policy(obs, player, history_int):
            flat   = self._encode_for_actor(obs, env.dealer, history_int, player,
                                            env._current_hands)
            flat_t = torch.tensor(flat, dtype=torch.float32
                                  ).unsqueeze(0).to(device)
            legal  = torch.tensor(obs['legal_actions'], dtype=torch.float32
                                  ).unsqueeze(0).to(device)
            actor  = agent.get_actor(player)
            with torch.no_grad():
                action, _, _ = actor.get_action(flat_t, legal, deterministic=True)
            return action.item()

        result = dds_oracle_evaluate(env, _policy, num_deals)
        return float(result['mean_regret'])

    def evaluate_oracle(self, num_deals: int = 1000) -> dict:
        """
        DDS oracle 评估（主要指标）.
        IMP regret = score_to_imp(score_ns − dds_optimal)  (≤ 0，越高越好)
        """
        from subgames.competitive_env import dds_oracle_evaluate

        agent = self.agent; env = self.env; device = self.device

        def _policy(obs, player, history_int):
            flat   = self._encode_for_actor(obs, env.dealer, history_int, player,
                                            env._current_hands)
            flat_t = torch.tensor(flat, dtype=torch.float32
                                  ).unsqueeze(0).to(device)
            legal  = torch.tensor(obs['legal_actions'], dtype=torch.float32
                                  ).unsqueeze(0).to(device)
            actor  = agent.get_actor(player)
            with torch.no_grad():
                action, _, _ = actor.get_action(flat_t, legal, deterministic=True)
            return action.item()

        result = dds_oracle_evaluate(env, _policy, num_deals)
        print(f"\n  [Oracle Eval] mean_regret={result['mean_regret']:+.3f} "
              f"± {result['std_regret']:.3f} IMP  "
              f"95% CI [{result['ci_lo']:+.3f}, {result['ci_hi']:+.3f}]")
        return result

    def evaluate_head_to_head(
        self,
        other_trainer: "SubgameTrainer",
        num_deals: int = 500,
        label_self: str = "A",
        label_other: str = "B",
        convention_sharing: bool = False,
    ) -> dict:
        """
        A vs B 直接对战评估.

        P98 Convention Card Protocol (convention_sharing=True):
          When self plays against other, each side's opponent gets access
          to the other's Belief Net — like reading the opponent's convention card.
          - self's NS uses other's Belief Net to understand other's EW bids
          - other's EW uses self's Belief Net to understand self's NS bids
          This implements Full Disclosure without requiring KL constraint.
        """
        from subgames.competitive_env import cross_evaluate

        env = self.env; device = self.device

        def _make_policy(trainer_, belief_net_for_opponents=None):
            """
            Create a policy closure for an agent.

            belief_net_for_opponents: if provided, this agent uses a DIFFERENT
              belief net (the opponent's) to understand opponent bids.
              In practice, in belief_conditioned mode:
              - For understanding partner bids: use own belief_net (co-trained)
              - For understanding opponent bids: use opponent's belief_net (convention card)

              Simplification for now: use own belief_net for all obs encoding.
              The convention card effect comes from the agent having been trained
              WITH belief-conditioned input, making it naturally responsive to
              belief quality. Full convention card sharing is a future extension.
            """
            bn = belief_net_for_opponents or trainer_.belief_net
            def _policy(obs, player, history_int):
                flat = trainer_._encode_for_actor(
                    obs, env.dealer, history_int, player,
                    env._current_hands, bn)
                flat_t = torch.tensor(flat, dtype=torch.float32
                                      ).unsqueeze(0).to(device)
                legal  = torch.tensor(obs['legal_actions'], dtype=torch.float32
                                      ).unsqueeze(0).to(device)
                actor  = trainer_.agent.get_actor(player)
                with torch.no_grad():
                    action, _, _ = actor.get_action(flat_t, legal, deterministic=True)
                return action.item()
            return _policy

        policy_self  = _make_policy(self)
        policy_other = _make_policy(other_trainer)

        result = cross_evaluate(
            env,
            agent_a_ns_policy=policy_self,
            agent_a_ew_policy=policy_self,
            agent_b_ns_policy=policy_other,
            agent_b_ew_policy=policy_other,
            num_deals=num_deals,
        )

        verdict = (f"✅ {label_self}>{label_other}" if result.mean_imp > 0
                   else f"❌ {label_other}>{label_self}" if result.mean_imp < 0
                   else "— tie")
        sig_str  = f"p={result.p_value:.3f} {'✅ sig' if result.significant else '(ns)'}"
        print(f"\n  [Head-to-Head] {label_self} vs {label_other}  "
              f"n={num_deals}  {sig_str}")
        print(f"    {label_self} mean IMP: {result.mean_imp:+.3f} ± {result.std_imp:.3f}  "
              f"win_rate={result.win_rate:.1%}  → {verdict}")
        return {
            'mean_imp':    result.mean_imp,
            'std_imp':     result.std_imp,
            'win_rate':    result.win_rate,
            'p_value':     result.p_value,
            'significant': result.significant,
            'n_deals':     result.n_deals,
        }

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
                hist   = list(self.env.history_int)
                dealer = self.env.dealer                    # dynamic after generate_deal
                opener = dealer                             # opener = dealer
                partner_of_opener = (dealer + 2) % 4       # target for belief

                while not done:
                    player   = self.env.current_player
                    flat_obs = self._encode_for_actor(
                        obs, dealer, hist, player, self.env._current_hands)
                    flat_t   = torch.tensor(flat_obs, dtype=torch.float32
                                            ).unsqueeze(0).to(self.device)
                    legal_t  = torch.tensor(obs['legal_actions'], dtype=torch.float32
                                            ).unsqueeze(0).to(self.device)
                    actor    = self.agent.get_actor(player)
                    action, _, _ = actor.get_action(flat_t, legal_t, deterministic=True)
                    hist.append(action.item())
                    obs, _, done, _ = self.env.step(action.item())

                h_enc = self._encode_history(hist)
                oh    = torch.tensor(hands[opener], dtype=torch.float32
                                     ).unsqueeze(0).to(self.device)
                h_t   = torch.tensor(h_enc, dtype=torch.float32
                                     ).unsqueeze(0).to(self.device)
                op    = torch.tensor([opener],            dtype=torch.long).to(self.device)
                tp    = torch.tensor([partner_of_opener], dtype=torch.long).to(self.device)

                probs  = self.belief_net.get_probs(oh, h_t, op, tp)
                target = torch.tensor(
                    hand_to_belief_target(hands[partner_of_opener]), dtype=torch.float32
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

    def evaluate_partner_info_gain(
        self,
        belief_net,          # external belief net (same for A and B → fair comparison)
        num_deals: int = 500,
    ) -> dict:
        """
        P97c diagnostic: measure partner inference gain per opener-side bid.

        For each opener-side bid (N or S in NS-training, E or W in EW-training),
        compute how much the bid reduces the belief net's uncertainty about the
        partner's hand. This is the partner_gain component of r_info.

        Using an EXTERNAL belief net (typically B's belief net) ensures fair
        comparison: same "judge" evaluates both A and B's communication quality.

        Returns:
            {
              'mean_partner_gain': float,
              'std_partner_gain':  float,
              'n_steps':           int,
              'per_position':      dict,  # N/S/E/W breakdown
            }
        """
        belief_net.eval()
        dual_info = DualInfoComputer(belief_net, beta=0.0)  # partner-only

        gains_all = []
        gains_by_pos = {0: [], 1: [], 2: [], 3: []}  # N, E, S, W

        with torch.no_grad():
            for _ in range(num_deals):
                hands, dd_table = self.env.generate_deal()
                obs  = self.env.reset(hands, dd_table)
                done = False
                hist   = list(self.env.history_int)
                dealer = self.env.dealer
                opener_seats = {dealer, (dealer + 2) % 4}

                while not done:
                    player = self.env.current_player
                    flat_obs = self._encode_for_actor(
                        obs, dealer, hist, player, self.env._current_hands)
                    flat_t   = torch.tensor(flat_obs, dtype=torch.float32
                                            ).unsqueeze(0).to(self.device)
                    legal_t  = torch.tensor(obs['legal_actions'], dtype=torch.float32
                                            ).unsqueeze(0).to(self.device)
                    actor    = self.agent.get_actor(player)
                    action, _, _ = actor.get_action(flat_t, legal_t, deterministic=True)
                    action_int = action.item()

                    # Compute partner gain for opener-side bids
                    if player in opener_seats:
                        partner = (player + 2) % 4
                        h_before = self._encode_history(hist)
                        hist_after = hist + [action_int]
                        h_after  = self._encode_history(hist_after)

                        oh  = torch.tensor(hands[player], dtype=torch.float32
                                           ).unsqueeze(0).to(self.device)
                        hb  = torch.tensor(h_before, dtype=torch.float32
                                           ).unsqueeze(0).to(self.device)
                        ha  = torch.tensor(h_after, dtype=torch.float32
                                           ).unsqueeze(0).to(self.device)
                        op  = torch.tensor([player],  dtype=torch.long).to(self.device)
                        tp  = torch.tensor([partner], dtype=torch.long).to(self.device)
                        tgt = torch.tensor(
                            hand_to_belief_target(hands[partner]),
                            dtype=torch.float32).unsqueeze(0).to(self.device)

                        b_before = belief_net.get_probs(oh, hb, op, tp)
                        b_after  = belief_net.get_probs(oh, ha, op, tp)
                        gain = dual_info.compute_info_gain(b_before, b_after, tgt).item()

                        gains_all.append(gain)
                        gains_by_pos[player].append(gain)

                    hist.append(action_int)
                    obs, _, done, _ = self.env.step(action_int)

        belief_net.train()

        def _stats(lst):
            if not lst:
                return {'mean': 0.0, 'std': 0.0, 'n': 0}
            a = np.array(lst)
            return {'mean': float(a.mean()), 'std': float(a.std()), 'n': len(lst)}

        pos_names = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
        per_pos = {pos_names[k]: _stats(v) for k, v in gains_by_pos.items()}

        result = {
            'mean_partner_gain': float(np.mean(gains_all)) if gains_all else 0.0,
            'std_partner_gain':  float(np.std(gains_all))  if gains_all else 0.0,
            'n_steps':           len(gains_all),
            'per_position':      per_pos,
        }

        print(f"  [Partner Info Gain] mean={result['mean_partner_gain']:.4f} "
              f"± {result['std_partner_gain']:.4f}  n={result['n_steps']}")
        for pos_name, stats in per_pos.items():
            if stats['n'] > 0:
                print(f"    {pos_name}: mean={stats['mean']:.4f} n={stats['n']}")
        return result

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
                hist   = list(self.env.history_int)  # 含前缀
                dealer = self.env.dealer
                opener = dealer
                overcaller_seats = {(dealer + 1) % 4, (dealer + 3) % 4}

                while not done:
                    player  = self.env.current_player
                    flat_t  = torch.tensor(
                        self._encode_for_actor(obs, dealer, hist, player,
                                               self.env._current_hands),
                        dtype=torch.float32).unsqueeze(0).to(self.device)
                    legal_t = torch.tensor(
                        obs['legal_actions'], dtype=torch.float32
                    ).unsqueeze(0).to(self.device)

                    actor  = self.agent.get_actor(player)
                    action, _, _ = actor.get_action(flat_t, legal_t, deterministic=True)
                    action_int = action.item()

                    # 只在争叫方（EW）步骤计算诊断
                    if player in overcaller_seats:
                        h_before = self._encode_history(hist)
                        hist_after = hist + [action_int]
                        h_after  = self._encode_history(hist_after)

                        oh  = torch.tensor(hands[opener], dtype=torch.float32
                                           ).unsqueeze(0).to(self.device)
                        hb  = torch.tensor(h_before, dtype=torch.float32
                                           ).unsqueeze(0).to(self.device)
                        ha  = torch.tensor(h_after,  dtype=torch.float32
                                           ).unsqueeze(0).to(self.device)
                        op  = torch.tensor([opener], dtype=torch.long).to(self.device)
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
