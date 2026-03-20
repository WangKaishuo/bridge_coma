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
    kl_lambda_start:  float = 1.5         # P73: 1.5（抑制早期entropy崩溃）
    kl_lambda_end:    float = 1.5         # P73: 暂不退火（与start相同），等vl稳定后再开启
    kl_anneal_frac:   float = 0.0         # P73: 0.0=固定不退火（1.0=全程退火）

    # ── KL Early Stopping ───────────────────────────────────────────────────
    kl_early_stop_threshold: float = 0.015

    # ── r_info ──────────────────────────────────────────────────────────────
    use_info_bonus:   bool  = False
    beta:             float = 0.05        # README: β=0.05 主配置

    # ── FSP ─────────────────────────────────────────────────────────────────
    fsp_pool_size:    int   = 10          # Kita et al. 2024
    fsp_add_interval: int   = 2           # 每 N 轮将 actor 存入 pool

    # ── JIT Belief Burn-in ──────────────────────────────────────────────────
    jit_burnin_deals:  int  = 3000        # P56: 1000→3000，追赶 actor 策略变化速度
    jit_burnin_epochs: int  = 3           # 快速微调 epoch 数

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
    oracle_snap_interval: int  = 1     # 每轮都做 oracle 快照（DDS 已预计算，几乎零开销）
    oracle_snap_deals:   int   = 500   # 500 局，误差 ±0.3 IMP，趋势清晰
    device:              str   = 'cpu'


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

        # ── FSP actor cache（避免每步 new 网络）──────────────────────────
        self._fsp_actor_cache: dict = {}  # role -> MLPPolicyNetwork
        self._fsp_cache_key: Optional[str] = None

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
            net = MLPPolicyNetwork(hidden_dim=self.config.hidden_dim).to(self.device)
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
                        net = MLPPolicyNetwork(
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
                    net = MLPPolicyNetwork(hidden_dim=self.config.hidden_dim).to(self.device)
                    net.load_state_dict(
                        {k: v.to(self.device) for k, v in fsp_sd[role].items()})
                    net.eval()
                    self._fsp_actor_cache[role] = net
                    self._fsp_cache_key = ck
                actor = self._fsp_actor_cache[role]
            else:
                actor = getattr(self.agent.model, role)

            flat   = encode_obs_flat(obs, dealer, hist)
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

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=self.belief_net.pos_weight.to(self.device))
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
                logits = self.belief_net(
                    oh_all[idx].to(self.device), h_all[idx].to(self.device),
                    op_all[idx].to(self.device), tp_all[idx].to(self.device))
                loss   = criterion(logits, tgt_all[idx].to(self.device))
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(
                    self.belief_net.parameters(), self.config.max_grad_norm)
                optimizer.step()
                tr_loss += loss.item(); nb += 1
            tr_loss /= max(1, nb)

            # Validate
            self.belief_net.eval()
            with torch.no_grad():
                vlogits = self.belief_net(
                    oh_all[va_idx].to(self.device), h_all[va_idx].to(self.device),
                    op_all[va_idx].to(self.device), tp_all[va_idx].to(self.device))
                val_loss = criterion(vlogits, tgt_all[va_idx].to(self.device)).item()
                probs    = torch.sigmoid(vlogits)
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
            return None

        self.belief_net.train()
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=self.belief_net.pos_weight.to(self.device))

        # 预先批量化，避免逐样本 forward
        oh_all  = torch.tensor(np.stack([s['observer_hand'] for s in belief_data]), dtype=torch.float32)
        h_all   = torch.tensor(np.stack([s['history']       for s in belief_data]), dtype=torch.float32)
        op_all  = torch.tensor([s['observer_pos'] for s in belief_data], dtype=torch.long)
        tp_all  = torch.tensor([s['target_pos']   for s in belief_data], dtype=torch.long)
        tgt_all = torch.tensor(np.stack([s['belief_target'] for s in belief_data]), dtype=torch.float32)
        N = len(belief_data); bs = min(256, N); final_loss = 0.0

        for epoch in range(self.config.jit_burnin_epochs):
            perm = np.random.permutation(N)
            tl = 0.0; nb = 0
            for s in range(0, N, bs):
                idx = perm[s:s+bs]
                logits = self.belief_net(oh_all[idx].to(self.device), h_all[idx].to(self.device),
                                         op_all[idx].to(self.device), tp_all[idx].to(self.device))
                loss = criterion(logits, tgt_all[idx].to(self.device))
                self.belief_optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.belief_net.parameters(), self.config.max_grad_norm)
                self.belief_optimizer.step()
                tl += loss.item(); nb += 1
            final_loss = tl / max(1, nb)

        print(f"  [JIT Burn-in] Done. loss={final_loss:.4f} samples={N}")
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
        _pending_swaps: List[Tuple]   = []   # P56: batch swap queue
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
                            net = MLPPolicyNetwork(
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

                step = {
                    'flat_obs': flat_obs, 'legal_actions': legal_actions,
                    'action': action, 'log_prob': log_prob, 'value': value,
                    'reward': 0.0, 'done': False,
                    'all_hands': all_hands, 'player': player,
                    'is_training_side': is_train,
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
                    # stash deal info for batch swap rollout
                    if not skip_dual_table:
                        _pending_swaps.append((
                            len(all_episodes) - 1,      # ep index
                            envs[i]._current_hands.copy(),
                            envs[i]._current_dd.copy(),
                            slot_dealer[i],
                            envs[i]._vulnerability,
                            envs[i].env.state.final_contract,
                        ))
                    collected += 1
                    slot_done[i] = True
                    if collected < num_deals:
                        slot_obs[i] = _reset(i)

        # ── P56: 批量换桌 rollout，一次性计算所有双桌 IMP ─────────────────
        if not skip_dual_table and _pending_swaps:
            deals_for_swap = [
                (hands, dd, dealer, vul)
                for _, hands, dd, dealer, vul, _ in _pending_swaps
            ]
            scores_sw = self._play_swapped_table_batch(deals_for_swap, fsp_sd)

            for (ep_idx, hands, dd, dealer, vul, contract), score2_sw in \
                    zip(_pending_swaps, scores_sw):
                score1    = self.env._compute_score_ns(contract, dd, vul)
                imp_ns    = float(score_to_imp(score1 - (-score2_sw)))
                opener_seats = {dealer, (dealer + 2) % 4}
                for s in all_episodes[ep_idx]:
                    if s['done']:
                        s['reward'] = imp_ns if s['player'] in opener_seats else -imp_ns

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
            net = MLPPolicyNetwork(hidden_dim=self.config.hidden_dim).to(self.device)
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
    # r_info 计算
    # ======================================================================

    def _compute_info_bonus(self, episodes: List[List[dict]]) -> List[List[float]]:
        """
        为每个 episode 的每个 step 计算 r_info，并归一化到 IMP 量纲。

        r_info_raw = I(bid; hand | partner) - β × I(bid; hand | opponent)
                   ≈ CE_reduction(belief_before, target) - CE_reduction(belief_after, target)

        量纲问题 (P55):
            raw per-step gain ≈ 0.001, IMP per-episode ≈ 6 std
            → 先把 per-episode 的 r_info_raw 之和用 RunningStats 归一化，
              得到 r_info_norm ~ N(0,1)，
              再乘以 imp_scale（IMP reward 的运行标准差），
              最后均摊回每个 step。
              这样 β 的物理意义是：r_info 占 IMP 奖励的比例。
        """
        if self.dual_info is None:
            return [[0.0] * len(ep) for ep in episodes]

        raw_ep_bonuses: List[List[float]] = []

        for ep in episodes:
            bonuses = []
            for step in ep:
                if 'belief_target' not in step or step.get('ew_diagnostic'):
                    bonuses.append(0.0)
                    continue

                player  = step['player']
                partner = (player + 2) % 4
                opp     = (player + 1) % 4

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
                    b_before_p = self.belief_net.get_probs(oh, hb, op_t, pp_t)
                    b_after_p  = self.belief_net.get_probs(oh, ha, op_t, pp_t)
                    partner_gain = self.dual_info.compute_info_gain(
                        b_before_p, b_after_p, tgt)

                    b_before_o = self.belief_net.get_probs(oh, hb, op_t, oo_t)
                    b_after_o  = self.belief_net.get_probs(oh, ha, op_t, oo_t)
                    opp_tgt = torch.tensor(
                        hand_to_belief_target(step['all_hands'][opp]),
                        dtype=torch.float32).unsqueeze(0).to(self.device)
                    opponent_leak = self.dual_info.compute_info_gain(
                        b_before_o, b_after_o, opp_tgt)

                    bonus, _ = self.dual_info.compute_dual_info_bonus(
                        partner_gain, opponent_leak)

                bonuses.append(float(bonus.item()))
            raw_ep_bonuses.append(bonuses)

        # ── 量纲归一化 (P73) ────────────────────────────────────────────────
        # P73 修复：集中到最后一步（与IMP时序对齐），只做Scale不做Shift。
        #
        # 旧方案（P55）问题：
        #   - r_info 均摊到每步 → Dense reward；IMP 只在终步 → Sparse reward。
        #   - GAE 累加 dense r_info 导致中间步 return 膨胀4-8倍，Critic无法拟合。
        #
        # 新方案：
        #   - per-episode r_info 之和集中赋给 done=True 的那一步。
        #   - 只做 scale（std对齐IMP），不减mean（保留信息增益的非负性质）。
        #   - scale_factor 上限1000，防止早期 rinfo_std≈0 时除以极小数爆炸。

        ep_totals = [sum(b for b in bs) for bs in raw_ep_bonuses]

        # 更新运行统计（只用于计算scale，不做shift）
        for v in ep_totals:
            self.partner_stats.update(v)

        imp_std   = max(self.reward_stats.std, 1.0)
        rinfo_std = max(self.partner_stats.std, 1e-6)
        scale_factor = min(imp_std / rinfo_std, 1000.0)   # 上限防爆炸

        # 集中到最后一步（done=True的位置），其余步归零
        scaled_episodes = []
        for ep_bonuses, ep_total in zip(raw_ep_bonuses, ep_totals):
            scaled = [0.0] * len(ep_bonuses)
            scaled[-1] = ep_total * scale_factor   # 最后一步接收全部bonus
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

    def run(self, num_rounds: int = None) -> List[dict]:
        """
        双桌 IBR 训练 (P55):
          每局 done 时跑换桌 rollout，reward = 双桌 IMP（非 oracle regret）
          开叫方阵营 +IMP，争叫方阵营 -IMP
          Agent B: r_info 归一化到 IMP 量纲后叠加
        """
        num_rounds = num_rounds or self.config.num_rounds
        n_deals    = self.config.steps_per_phase * self.config.deals_per_step
        batch_sz   = self.config.deals_per_step

        print("\n[Trainer] Critic warmup...")
        self.critic_warmup()

        for rnd in range(num_rounds):
            print(f"\n══════ Round {rnd+1}/{num_rounds} ══════")
            self._maybe_add_to_fsp(rnd)
            fsp_sd = self._apply_fsp_opponent()
            _ir_vals: list = []; _bl_vals: list = []
            ns_metrics: dict = {}; ew_metrics: dict = {}

            # ── 桌1: 训练 NS ────────────────────────────────────────────
            print(f"  [Table1/NS] Collecting {n_deals} deals (batch={batch_sz})...")
            ns_eps = self._collect_episodes_batch(
                n_deals, train_side='NS', fsp_sd=fsp_sd, batch_size=batch_sz)

            # 更新 IMP reward 运行统计（供 r_info 归一化用）
            # 必须在 r_info 叠加之前采集，保证 reward_stats 是纯 IMP 分布
            raw_imp_vals = []
            for ep in ns_eps:
                for step in ep:
                    if step.get('done') and step['player'] in (NORTH, SOUTH):
                        v = float(step['reward'])
                        self.reward_stats.update(v)
                        raw_imp_vals.append(v)

            if self.config.use_info_bonus:
                bonuses = self._compute_info_bonus(ns_eps)
                for ep, bs in zip(ns_eps, bonuses):
                    for step, b in zip(ep, bs):
                        step['reward'] += b
                        if b != 0.0: _ir_vals.append(b)

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

            # ── JIT Belief Burn-in（Agent B only）────────────────────────
            if self.belief_net is not None:
                print("  [JIT Belief Burn-in]...")
                bl = self.jit_belief_burnin()
                if bl is not None: _bl_vals.append(bl)

            # ── 桌2: 训练 EW ────────────────────────────────────────────
            print(f"  [Table2/EW] Collecting {n_deals} deals (batch={batch_sz})...")
            ew_eps = self._collect_episodes_batch(
                n_deals, train_side='EW', fsp_sd=fsp_sd, batch_size=batch_sz)

            # P73 Fix 2: EW 的 IMP stats 更新（reward_stats 供 r_info scale 用）
            for ep in ew_eps:
                for step in ep:
                    if step.get('done') and step['player'] in (EAST, WEST):
                        self.reward_stats.update(float(step['reward']))

            # P73 Fix 2: EW 也叠加 r_info（之前只有 NS 有，导致对称性破坏）
            if self.config.use_info_bonus:
                ew_bonuses = self._compute_info_bonus(ew_eps)
                for ep, bs in zip(ew_eps, ew_bonuses):
                    for step, b in zip(ep, bs):
                        step['reward'] += b
                        if b != 0.0: _ir_vals.append(b)

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

            # ── Oracle 快照（每 oracle_snap_interval 轮，仅观察）────────────
            oracle_snap = None
            snap_iv = self.config.oracle_snap_interval
            if snap_iv > 0 and (rnd + 1) % snap_iv == 0:
                oracle_snap = self._oracle_snapshot(self.config.oracle_snap_deals)

            # ── 日志 ────────────────────────────────────────────────────
            mean_r = float(np.mean(raw_imp_vals)) if raw_imp_vals else 0.0
            std_r  = float(np.std(raw_imp_vals))  if raw_imp_vals else 0.0

            log_entry = {
                'round': rnd+1, 'mean_reward': mean_r, 'std_reward': std_r,
                'ns_metrics': ns_metrics, 'ew_metrics': ew_metrics,
                'fsp_pool_size': len(self.fsp_pool),
                'mean_ir':    float(np.mean(_ir_vals)) if _ir_vals else None,
                'belief_loss': float(np.mean(_bl_vals)) if _bl_vals else None,
                'imp_std_running': float(self.reward_stats.std),
                'oracle_regret_snap': oracle_snap,
            }
            self.log.append(log_entry)
            self._print_log(log_entry)

        return self.log

    def _print_log(self, entry: dict):
        rnd = entry['round']
        fsp = entry['fsp_pool_size']
        mr  = entry['mean_reward']
        sr  = entry['std_reward']

        # oracle regret snapshot（仅观察，每 oracle_snap_interval 轮计算一次）
        snap = entry.get('oracle_regret_snap')
        snap_str = f"  oracle≈{snap:+.2f}" if snap is not None else ""
        print(f"  [Round {rnd}] imp={mr:+.3f}±{sr:.3f}{snap_str}  fsp={fsp}")

        ns = entry.get('ns_metrics', entry.get('s_metrics', {}))
        ns_n = ns.get(NORTH, {}); ns_s = ns.get(SOUTH, {})
        if ns_n or ns_s:
            print(f"    NS │ N: pl={ns_n.get('policy_loss',0):+.4f} "
                  f"vl={ns_n.get('value_loss',0):.3f} "
                  f"ent={ns_n.get('entropy',0):.3f} │ "
                  f"S: pl={ns_s.get('policy_loss',0):+.4f} "
                  f"vl={ns_s.get('value_loss',0):.3f} "
                  f"ent={ns_s.get('entropy',0):.3f} "
                  f"kl={ns_s.get('kl_loss',0):.5f}(λ={ns_s.get('kl_lambda',0):.3f})")

        ew = entry.get('ew_metrics', entry.get('n_metrics', {}))
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
            # ep_equiv: episode-level r_info 等效 IMP（≈ step-level × ~8 steps/ep）
            ep_equiv = abs(ir) * 8
            print(f"    r_info │ step_ir={ir:.4f}  ep_equiv≈{ep_equiv:.3f} IMP  "
                  f"belief_loss={bl_str}")

    # ======================================================================
    # 评估
    # ======================================================================

    def _oracle_snapshot(self, num_deals: int = 100) -> float:
        """
        轻量 oracle 快照：估计当前策略的 IMP regret（仅供观察）.
        """
        from subgames.competitive_env import dds_oracle_evaluate

        agent    = self.agent
        env      = self.env
        device   = self.device

        def _policy(obs, player, history_int):
            flat   = encode_obs_flat(obs, env.dealer, history_int)
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
            flat   = encode_obs_flat(obs, env.dealer, history_int)
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
    ) -> dict:
        """A vs B 直接对战评估."""
        from subgames.competitive_env import cross_evaluate

        env = self.env; device = self.device

        def _make_policy(agent_):
            def _policy(obs, player, history_int):
                flat   = encode_obs_flat(obs, env.dealer, history_int)
                flat_t = torch.tensor(flat, dtype=torch.float32
                                      ).unsqueeze(0).to(device)
                legal  = torch.tensor(obs['legal_actions'], dtype=torch.float32
                                      ).unsqueeze(0).to(device)
                actor  = agent_.get_actor(player)
                with torch.no_grad():
                    action, _, _ = actor.get_action(flat_t, legal, deterministic=True)
                return action.item()
            return _policy

        policy_self  = _make_policy(self.agent)
        policy_other = _make_policy(other_trainer.agent)

        result = cross_evaluate(
            env,
            agent_a_ns_policy=policy_self,
            agent_a_ew_policy=policy_self,
            agent_b_ns_policy=policy_other,
            agent_b_ew_policy=policy_other,
            num_deals=num_deals,
        )

        verdict = "✅ A>B" if result.mean_imp > 0 else ("❌ B>A" if result.mean_imp < 0 else "— tie")
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
                    flat_obs = encode_obs_flat(obs, dealer, hist)
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
                        encode_obs_flat(obs, dealer, hist),
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
