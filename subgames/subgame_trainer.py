"""
Subgame Trainer
===============

通用子博弈训练框架.

关键设计:
- active_players: 只训练指定玩家 (e.g., [SOUTH] for Stage 1)
- accumulate_steps: 累积 N 步数据后做 1 次 PPO update
- safe_update: 完全自主的 PPO update (不调 agent.update()),
  有稳健 advantage 归一化 (防 std=0 NaN)
- 可选 Dual-Info Bonus (BeliefNetwork + DualInfoComputer)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from env import NUM_PLAYERS, NUM_BIDS, BID_PASS, NORTH, EAST, SOUTH, WEST
from networks import ActorCritic, BeliefNetwork
from networks.belief_net import DualInfoComputer
from algorithms.ippo import PPOConfig, RolloutBuffer
from algorithms.mappo import MAPPOAgent, MAPPOConfig, MAPPORolloutBuffer
from utils.running_stats import RunningStats
from utils.hand_features import (
    hand_to_belief_target, batch_hand_to_belief_target, belief_accuracy
)


@dataclass
class SubgameConfig:
    """子博弈训练配置."""
    # Training
    num_steps: int = 5000
    deals_per_step: int = 32
    accumulate_steps: int = 4         # collect N steps before 1 PPO update
    lr: float = 1e-4
    belief_lr: float = 1e-4          # Stage 2 在线微调用低 lr, 防止 256-sample batch 震荡覆盖预训练

    # Info bonus
    use_info_bonus: bool = False
    beta: float = 0.05   # info bonus 权重; 0.05-0.1 = "微风"; 0.5 会盖过 IMP 信号
    lambda_start: float = 0.5
    lambda_end: float = 0.1
    belief_warmup_steps: int = 500

    # Network
    hand_dim: int = 256
    history_dim: int = 256
    hidden_dim: int = 256

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    num_epochs: int = 4
    batch_size: int = 64
    entropy_coef: float = 0.01
    entropy_coef_start: float = 0.05   # 探索期高 entropy (防止早期坍缩)
    entropy_coef_end: float = 0.01     # 收敛期低 entropy
    entropy_anneal_frac: float = 0.5   # 前 50% 步数做退火
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # KL Anchor: 限制 RL 策略偏离 BC 策略的程度
    # loss += kl_lambda * KL(pi_current || pi_bc)
    # kl_lambda 从 kl_lambda_start 线性退火到 kl_lambda_end
    # 设 0.0 = 关闭 (向后兼容)
    kl_lambda_start: float = 0.0
    kl_lambda_end: float = 0.0
    kl_anneal_frac: float = 1.0        # KL 退火覆盖的训练比例

    # Single-step (Contextual Bandit) mode
    # 当 episode 中每个 player 仅做 1-2 步决策时启用.
    # 用 batch-mean baseline 替代 GAE, 断开 Critic 梯度.
    # Competitive 子博弈 (multi-step) 走标准 GAE, 此处设 False.
    single_step: bool = False

    # Which players does the agent control?
    active_players: Optional[List[int]] = None

    # Minimum buffer size before PPO update
    min_buffer_size: int = 4

    # Eval
    eval_interval: int = 500
    log_interval: int = 50

    # Device
    device: str = 'cpu'


class SubgameTrainer:
    """通用子博弈训练器."""

    def __init__(self, env, config: SubgameConfig):
        self.env = env
        self.config = config
        self.device = config.device
        self.active_players = config.active_players or list(range(NUM_PLAYERS))

        mappo_config = MAPPOConfig(
            hand_dim=config.hand_dim,
            history_dim=config.history_dim,
            hidden_dim=config.hidden_dim,
            lr=config.lr,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_ratio=config.clip_ratio,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            entropy_coef=config.entropy_coef,
            value_coef=config.value_coef,
            max_grad_norm=config.max_grad_norm,
            device=config.device,
        )
        self.agent = MAPPOAgent(mappo_config)

        # Belief network (optional)
        self.belief_net = None
        self.dual_info = None
        self.belief_optimizer = None

        if config.use_info_bonus:
            self.belief_net = BeliefNetwork(
                hand_dim=config.hand_dim,
                history_dim=config.history_dim,
                hidden_dim=config.hidden_dim,
            ).to(self.device)
            self.dual_info = DualInfoComputer(self.belief_net, beta=config.beta)
            self.belief_optimizer = torch.optim.Adam(
                self.belief_net.parameters(), lr=config.belief_lr
            )
            self.partner_stats = RunningStats()
            self.opponent_stats = RunningStats()

        # BC anchor model (frozen reference, 用于 KL penalty)
        # 由外部调用者通过 set_bc_anchor() 设置
        self.bc_model = None

        self.log = []
        self._current_step = 0  # 当前训练步数, 用于 entropy annealing

        # Reward 归一化: 在线 running stats, 将 IMP regret 归一化为 ~N(0,1)
        # 与全环境训练方式一致 (普适化设计)
        self.reward_stats = RunningStats()

    def critic_warmup_step(self, num_deals: int = 256) -> float:
        """
        Critic 预热: 用当前 Actor rollout 数据做一步 MSE 监督.

        关键设计:
        - target = 当前策略在环境里实际拿到的 final_reward (非 DDS optimal).
          这是 V(s) 的无偏估计. 若用 DDS reward 做 target,
          Critic 会系统性高估当前策略, Stage 2 的 advantage 全为负值,
          策略梯度惩罚所有动作 → 坍缩.
        - 只更新 Critic (agent.critic_optimizer), 不更新 Actor.
        - 复用 agent.critic_optimizer (持久化, 保留动量状态).
        - 返回 value_loss (float) 供外部监控收敛.
        """
        agent = self.agent
        device = self.device

        # 1. Rollout: 用当前 Actor 策略收集 episodes
        episodes = self.collect_episodes(num_deals)

        # 2. 构造 (obs, all_hands, target_value) 对
        obs_list, hands_list, target_list = [], [], []
        for ep in episodes:
            final_r = float(ep['final_reward'])
            for player, traj in ep['player_trajectories'].items():
                target_r = final_r if player % 2 == 0 else -final_r
                for step in traj:
                    obs_list.append(step['obs'])
                    hands_list.append(step.get('_all_hands'))
                    target_list.append(target_r)

        if not obs_list:
            return 0.0

        def stack_obs(obs_list):
            keys = obs_list[0].keys()
            return {k: torch.stack([
                torch.tensor(o[k], dtype=torch.float32) for o in obs_list
            ]).to(device) for k in keys}

        obs_batch = stack_obs(obs_list)
        targets = torch.tensor(target_list, dtype=torch.float32).to(device)

        hands_batch = None
        if hands_list[0] is not None:
            hands_batch = torch.stack([
                torch.tensor(h, dtype=torch.float32) for h in hands_list
            ]).to(device)

        # 3. 用持久化 critic_optimizer 做 mini-batch MSE 更新
        # 复用 optimizer 保留动量, 不要每次新建 (否则动量清零, warmup 效率低)
        critic_optimizer = agent.critic_optimizer
        batch_size = min(256, len(obs_list))
        idx = torch.randperm(len(obs_list))
        total_loss = 0.0
        num_batches = 0

        for start in range(0, len(obs_list), batch_size):
            b_idx = idx[start:start + batch_size]
            b_obs = {k: v[b_idx] for k, v in obs_batch.items()}
            b_targets = targets[b_idx]
            b_hands = hands_batch[b_idx] if hands_batch is not None else None

            values = agent.model.critic(b_obs, b_hands).squeeze(-1)
            loss = F.mse_loss(values, b_targets)

            critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                agent.model.critic.parameters(), self.config.max_grad_norm)
            critic_optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def set_bc_anchor(self, state_dict: dict):
        """
        设置 BC 锚点模型 (Stage 1 结束时的快照).

        调用后, safe_update() 会在 loss 中加入 KL 惩罚项:
          loss += kl_lambda * KL(pi_current || pi_bc)

        只需要 Actor 部分计算参考 logits, 不需要 Critic.
        从完整 state_dict 中提取 actor.* 前缀的参数即可,
        避免 centralized_critic 结构不匹配的问题.
        """
        from networks.policy_net import PolicyNetwork
        self.bc_model = PolicyNetwork(
            hand_dim=self.config.hand_dim,
            history_dim=self.config.history_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)
        # 从完整 ActorCritic state_dict 中提取 actor.* 参数
        actor_state = {
            k[len('actor.'):]: v
            for k, v in state_dict.items()
            if k.startswith('actor.')
        }
        self.bc_model.load_state_dict(actor_state)
        self.bc_model.eval()
        for p in self.bc_model.parameters():
            p.requires_grad_(False)

    def get_lambda(self, step: int) -> float:
        cfg = self.config
        if step < cfg.belief_warmup_steps:
            return 0.0
        progress = min(1.0, (step - cfg.belief_warmup_steps) /
                       max(1, cfg.num_steps - cfg.belief_warmup_steps))
        return cfg.lambda_start + (cfg.lambda_end - cfg.lambda_start) * progress

    def get_entropy_coef(self, step: int) -> float:
        """
        Entropy coefficient annealing.

        前 entropy_anneal_frac 的步数: 从 entropy_coef_start 线性退到 entropy_coef_end.
        之后: 固定为 entropy_coef_end.

        防止 PPO 在小动作空间 (Stayman 7 choices) 中过早坍缩到单一动作.
        """
        cfg = self.config
        anneal_steps = int(cfg.num_steps * cfg.entropy_anneal_frac)
        if step >= anneal_steps:
            return cfg.entropy_coef_end
        progress = step / max(1, anneal_steps)
        return cfg.entropy_coef_start + (cfg.entropy_coef_end - cfg.entropy_coef_start) * progress

    def get_kl_lambda(self, step: int) -> float:
        """
        KL 锚定系数退火 (全局基准系数).

        位置自适应 KL 的基准值, 由 compute_context_kl_weights() 在
        mini-batch 级别进一步按叫牌阶数加权.
        从 kl_lambda_start 线性退到 kl_lambda_end.
        """
        cfg = self.config
        if cfg.kl_lambda_start == 0.0:
            return 0.0
        anneal_steps = int(cfg.num_steps * cfg.kl_anneal_frac)
        if step >= anneal_steps:
            return cfg.kl_lambda_end
        progress = step / max(1, anneal_steps)
        return cfg.kl_lambda_start + (cfg.kl_lambda_end - cfg.kl_lambda_start) * progress

    @staticmethod
    def compute_context_kl_weights(history_obs: torch.Tensor) -> torch.Tensor:
        """
        位置自适应 KL 权重 — 基于叫牌序列的当前最高阶数 (context level).

        设计原则 (Gemini / 桥牌知识):
          1阶: 系统基石 (开叫/应叫), 偏离 BC 后同伴完全迷失 → 权重 1.5
          2阶: 常规应叫续叫                                    → 权重 1.0
          3阶: 逼叫序列, 仍需人类协定                          → 权重 0.5
          4阶: 进局决策, 开始允许 RL 自主探索                  → 权重 0.25
          5阶+: 高阶争叫/满贯, 完全交给 DDS 算力博弈           → 权重 0.1

        关键: 权重由状态 (history) 决定, 而非由 agent 选择的动作决定.
        若用动作决定权重, agent 会通过跳叫高阶来规避 KL 惩罚 (gradient exploit).

        Args:
            history_obs: (B, max_len, NUM_BIDS) one-hot 叫牌历史张量
                         来自 batch['obs']['history']

        Returns:
            weights: (B,) 每个样本的 KL 权重, 值域 [0.1, 1.5]
        """
        B = history_obs.shape[0]
        device = history_obs.device

        # BID_1C = 3 (index), 对应1阶最低叫品
        # 实质叫品 index ∈ [3, 37] (BID_1C 到 7NT)
        # one-hot 的列索引即 bid index
        # bid level = (bid_index - 3) // 5 + 1  for bid_index in [3, 37]

        # 找到每个样本历史中最高实质叫品的 level
        # history_obs: (B, T, 38) → argmax over bid dim: (B, T)
        bid_indices = history_obs.argmax(dim=-1)  # (B, T)

        # 只考虑实质叫品 (index >= 3, 即 BID_1C+)
        is_real_bid = (bid_indices >= 3)  # (B, T)

        # 对每个样本, 找最大实质叫品 index
        # 将非实质叫品位置置 0
        real_bid_indices = bid_indices * is_real_bid.long()  # (B, T)
        max_bid_idx = real_bid_indices.max(dim=-1).values    # (B,)

        # 计算 context level: (bid_index - 3) // 5 + 1
        # 若没有任何实质叫品 (全 Pass 前缀), level = 1
        context_level = torch.where(
            max_bid_idx >= 3,
            (max_bid_idx - 3) // 5 + 1,
            torch.ones(B, dtype=torch.long, device=device)
        ).float()  # (B,)

        # 映射到权重 (指数级衰减, 契合桥牌特性)
        weights = torch.ones(B, device=device)
        weights = torch.where(context_level <= 1, torch.full_like(weights, 1.5), weights)
        weights = torch.where(context_level == 2, torch.full_like(weights, 1.0), weights)
        weights = torch.where(context_level == 3, torch.full_like(weights, 0.5),  weights)
        weights = torch.where(context_level == 4, torch.full_like(weights, 0.25), weights)
        weights = torch.where(context_level >= 5, torch.full_like(weights, 0.1),  weights)

        return weights

    # ====================================================================
    # Rollout collection
    # ====================================================================

    def collect_episodes(self, num_deals: int) -> List[Dict]:
        """
        采样 episode.

        env.step() 可能内部自动执行规则玩家 (e.g., north_rule in Stayman).
        trainer 只记录 active_players 中由 agent 做出的决策.
        """
        episodes = []

        for _ in range(num_deals):
            hands, dd_table = self.env.generate_deal()
            obs = self.env.reset(hands, dd_table)

            player_trajs = {p: [] for p in self.active_players}
            done = False

            while not done:
                player = self.env.current_player

                # 检查是否是 agent 决策的 player
                if player not in self.active_players:
                    # 非 active player: 用 agent 当前策略 (deterministic)
                    # 而非随机, 确保冻结的 player 保持 BC 学到的行为
                    all_hands = self.env._current_hands
                    action, _ = self.agent.get_action(
                        obs, all_hands=all_hands, deterministic=True)
                    obs, reward, done, info = self.env.step(action)
                    continue

                history_before = self.env.history.copy()

                all_hands = self.env._current_hands
                action, extra = self.agent.get_action(obs, all_hands=all_hands)
                extra['_all_hands'] = all_hands

                obs_next, reward, done, info = self.env.step(action)
                history_after = self.env.history.copy()

                step_data = {
                    'obs': obs,
                    'action': action,
                    'reward': reward,
                    'done': done,
                    'player': player,
                    'hands': all_hands.copy(),
                    'history_before': history_before,
                    'history_after': history_after,
                    **extra,
                }
                player_trajs[player].append(step_data)

                obs = obs_next

            # Backfill terminal reward
            final_reward = reward
            for p, traj in player_trajs.items():
                if traj:
                    r = final_reward if p % 2 == 0 else -final_reward
                    traj[-1]['reward'] = float(r)

            episodes.append({
                'player_trajectories': player_trajs,
                'hands': hands,
                'dd_table': dd_table,
                'final_reward': final_reward,
            })

        return episodes

    def store_episodes(self, episodes: List[Dict]):
        """存入 agent buffer, 只存 active players.

        如果启用 info_bonus, 对 N (NORTH) 的 terminal reward 叠加
        ReLU 截断后的 info gain:
            r_total = r_IMP + β * max(0, I(bid;hand|partner) - β2*I(bid;hand|opp))
        β 通常设为较小值 (0.05-0.1), 让信息奖励是"微风"而非"飓风".
        """
        for ep in episodes:
            # 预计算 info bonus (只在 use_info_bonus 且 N 是 active player 时)
            info_bonus_by_player = {}
            if (self.config.use_info_bonus and self.belief_net is not None
                    and NORTH in self.active_players):
                hands = ep['hands']
                for player, traj in ep['player_trajectories'].items():
                    if player % 2 != 0:   # 只有 NS 方 (偶数 = N/S)
                        continue
                    if player not in self.active_players:
                        continue
                    total_gain = 0.0
                    partner = (player + 2) % 4
                    opponent = (player + 1) % 4
                    self.belief_net.eval()
                    with torch.no_grad():
                        for step in traj:
                            hb = self._encode_history(step['history_before'])
                            ha = self._encode_history(step['history_after'])
                            pg = self._compute_single_info_gain(
                                hands[partner], hb, ha, partner, player, hands[player])
                            ol = self._compute_single_info_gain(
                                hands[opponent], hb, ha, opponent, player, hands[player])
                            # ReLU 已在 _compute_single_info_gain 内执行
                            total_gain += pg - self.config.beta * ol
                    self.belief_net.train()
                    info_bonus_by_player[player] = total_gain

            for player, traj in ep['player_trajectories'].items():
                if player not in self.active_players:
                    continue
                for i, step in enumerate(traj):
                    reward = step['reward']
                    is_terminal = (i == len(traj) - 1)
                    if is_terminal and player in info_bonus_by_player:
                        reward = reward + info_bonus_by_player[player]
                    self.agent.store_transition(
                        player,
                        step['obs'],
                        step['action'],
                        step['log_prob'],
                        reward,
                        step['value'],
                        step['done'],
                        all_hands=step.get('_all_hands'),
                    )

    # ====================================================================
    # Safe PPO Update (replaces agent.update())
    # ====================================================================

    def safe_update(self) -> Dict[str, float]:
        """
        稳健的 PPO update.

        完全替代 agent.update(), 增加:
        - 只更新 active_players
        - 稳健 advantage 归一化 (防 std=0 NaN)
        - Entropy coefficient annealing (防止小动作空间早期坍缩)
        - single_step 模式: 用 batch-mean baseline 替代 GAE,
          断开 Critic 梯度, 防止 Value Loss 爆炸污染共享编码器
        - KL Anchor (可选): loss += kl_lambda * KL(pi_current || pi_bc)
          防止 RL 摧毁 BC 策略 (如退化为无脑 4M)
        """
        agent = self.agent
        min_size = self.config.min_buffer_size
        ent_coef = self.get_entropy_coef(self._current_step)
        kl_lambda = self.get_kl_lambda(self._current_step)
        is_single = self.config.single_step
        use_kl = (kl_lambda > 0.0 and self.bc_model is not None)

        total_loss = total_policy = total_value = total_entropy = num_updates = 0
        total_kl = 0.0

        for player in range(NUM_PLAYERS):
            buffer = agent.buffers[player]

            if player not in self.active_players or len(buffer.actions) < min_size:
                buffer.reset()
                continue

            # ============================================================
            # 偷梁换柱: single_step → batch-mean baseline 替代 GAE
            # ============================================================
            if is_single:
                # 1-step episode: Q(s,a) = r(s,a), 无需 bootstrapping
                rewards = torch.tensor(buffer.rewards, dtype=torch.float32)
                buffer.returns = rewards
                # batch-mean baseline: advantage = reward - mean(reward)
                baseline = rewards.mean()
                advantages = rewards - baseline
                # 标准化 (跨整个 buffer, 而非 mini-batch 内再做一次)
                adv_std = advantages.std()
                if torch.isfinite(adv_std) and adv_std > 1e-6:
                    advantages = advantages / (adv_std + 1e-8)
                buffer.advantages = advantages
            else:
                # 标准 GAE (Competitive 子博弈等 multi-step 场景)
                with torch.no_grad():
                    last_obs = {k: v.unsqueeze(0).to(self.device)
                                for k, v in buffer.observations[-1].items()}
                    last_hands = (buffer.all_hands[-1].unsqueeze(0).to(self.device)
                                  if buffer.all_hands else None)
                    last_value = agent.model.critic(last_obs, last_hands).item()

                buffer.compute_returns_and_advantages(
                    last_value, agent.config.gamma, agent.config.gae_lambda
                )

            for _ in range(agent.config.num_epochs):
                for batch in buffer.get_batches(agent.config.batch_size):
                    log_probs, entropy = agent.model.actor.evaluate_actions(
                        batch['obs'], batch['actions']
                    )

                    ratio = torch.exp(log_probs - batch['old_log_probs'])

                    # Advantage: single_step 已在 buffer 级别标准化;
                    # multi-step 仍需 mini-batch 级别标准化
                    adv = batch['advantages']
                    if not is_single:
                        if adv.numel() > 1:
                            adv_std = adv.std()
                            if torch.isfinite(adv_std) and adv_std > 1e-6:
                                adv = (adv - adv.mean()) / (adv_std + 1e-8)
                            else:
                                adv = adv - adv.mean()

                    policy_loss = -torch.min(
                        ratio * adv,
                        torch.clamp(ratio,
                                    1 - agent.config.clip_ratio,
                                    1 + agent.config.clip_ratio) * adv
                    ).mean()

                    # ====================================================
                    # KL Anchor: KL(pi_current || pi_bc)
                    # 目的: 防止 RL 摧毁 BC 学到的策略
                    #
                    # 手写 KL, 不用 F.kl_div:
                    # F.kl_div(input, target) 中 target 被当作常数,
                    # 梯度只流向 input. 我们的 input=bc_log_probs (冻结),
                    # target=curr_probs (需要优化) → 梯度为零, 锚定完全失效.
                    #
                    # 正确写法: KL(curr||bc) = sum(P_curr * (logP_curr - logP_bc))
                    # bc_logits 在 no_grad 里 (bc_model 冻结);
                    # curr_logits 在 no_grad 外 → 梯度正确流回 actor.
                    # ====================================================
                    kl_loss = torch.tensor(0.0, device=self.device)
                    if use_kl:
                        MASK_VAL = -1e9

                        # bc 参考分布: 冻结, 不需要梯度
                        with torch.no_grad():
                            bc_logits = self.bc_model(batch['obs'])
                            if 'legal_actions' in batch['obs']:
                                illegal = batch['obs']['legal_actions'] < 0.5
                                bc_logits = bc_logits.masked_fill(illegal, MASK_VAL)
                            bc_log_probs = F.log_softmax(bc_logits, dim=-1)

                        # curr 分布: 在计算图内, 梯度流回 actor
                        curr_logits_kl = agent.model.actor(batch['obs'])
                        if 'legal_actions' in batch['obs']:
                            illegal = batch['obs']['legal_actions'] < 0.5
                            curr_logits_kl = curr_logits_kl.masked_fill(illegal, MASK_VAL)
                        curr_log_probs_kl = F.log_softmax(curr_logits_kl, dim=-1)
                        curr_probs_kl = curr_log_probs_kl.exp()

                        # KL(curr || bc) = Σ P_curr * (log P_curr - log P_bc), shape (B,)
                        kl_per_sample = (curr_probs_kl * (curr_log_probs_kl - bc_log_probs)
                                         ).sum(dim=-1)

                        # 位置自适应权重: 基于叫牌历史的当前最高阶数 (context level)
                        # 权重由状态决定, 非动作决定 (防 gradient exploit)
                        if 'history' in batch['obs']:
                            ctx_weights = self.compute_context_kl_weights(
                                batch['obs']['history'].to(self.device))
                            kl_loss = (kl_per_sample * ctx_weights).mean()
                        else:
                            kl_loss = kl_per_sample.mean()

                        if not torch.isfinite(kl_loss):
                            kl_loss = torch.tensor(0.0, device=self.device)

                    # Loss: single_step → 断开 Critic 梯度 (value_coef=0)
                    if is_single:
                        loss = (policy_loss
                                - ent_coef * entropy.mean()
                                + kl_lambda * kl_loss)
                        agent.actor_optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            agent.model.actor.parameters(), agent.config.max_grad_norm)
                        agent.actor_optimizer.step()
                        value_loss_val = 0.0
                    else:
                        # Actor update (含 KL penalty)
                        actor_loss = (policy_loss
                                      - ent_coef * entropy.mean()
                                      + kl_lambda * kl_loss)
                        agent.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            agent.model.actor.parameters(), agent.config.max_grad_norm)
                        agent.actor_optimizer.step()

                        # Critic update: 独立 optimizer + value clipping
                        # value clipping 防止单步 Critic 更新幅度过大 (PPO2 风格)
                        values = agent.model.critic(
                            batch['obs'], batch.get('all_hands'))
                        old_values = values.detach()
                        v_clipped = old_values + (values - old_values).clamp(
                            -agent.config.clip_ratio, agent.config.clip_ratio)
                        value_loss = torch.max(
                            F.mse_loss(values, batch['returns']),
                            F.mse_loss(v_clipped, batch['returns'])
                        )
                        agent.critic_optimizer.zero_grad()
                        value_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            agent.model.critic.parameters(), agent.config.max_grad_norm)
                        agent.critic_optimizer.step()
                        value_loss_val = value_loss.item()

                    # 统计 (single_step 下 loss=actor_loss, multi-step 下分开记)
                    policy_loss_val = policy_loss.item()
                    total_loss += policy_loss_val + value_loss_val
                    total_policy += policy_loss_val
                    total_value += value_loss_val
                    total_entropy += entropy.mean().item()
                    total_kl += kl_loss.item()
                    num_updates += 1

            buffer.reset()

        if num_updates == 0:
            return {}

        return {
            'loss': total_loss / num_updates,
            'policy_loss': total_policy / num_updates,
            'value_loss': total_value / num_updates,
            'entropy': total_entropy / num_updates,
            'entropy_coef': ent_coef,
            'kl_loss': total_kl / num_updates,
            'kl_lambda': kl_lambda,
        }

    # ====================================================================
    # Belief training
    # ====================================================================

    def train_belief_step(self, episodes: List[Dict]) -> float:
        if self.belief_net is None:
            return 0.0

        belief_data = self._extract_belief_data(episodes)
        if not belief_data:
            return 0.0

        total_loss = 0.0
        n = 0

        for batch in self._iter_belief_batches(belief_data, batch_size=256):
            loss = self.belief_net.compute_loss(
                batch['observer_hand'], batch['history'],
                batch['observer_pos'], batch['target_pos'],
                batch['target_hand'],
            )
            self.belief_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.belief_net.parameters(), 1.0)
            self.belief_optimizer.step()
            total_loss += loss.item()
            n += 1

        return total_loss / max(1, n)

    def jit_belief_burnin(self, num_deals: int = 2000, epochs: int = 3,
                          lr: float = 1e-3):
        """
        JIT (Just-In-Time) Belief Burn-in.

        在每轮 N-phase 开始前调用, 用当前策略的 rollout 快速更新 Belief Net,
        使其与 N 的最新叫牌语言保持同步.

        设计原则:
        - 冻结 N 和 S (不更新 PPO), 只跑 rollout 采集数据
        - 用较大的 lr (1e-3) 猛训 3-5 个 epoch, 快速追上语义漂移
        - 完成后 lr 恢复到正常 belief_lr (1e-4)

        Args:
            num_deals:  用于 burn-in 的对局数 (建议 1000-3000)
            epochs:     训练轮数 (建议 3-5)
            lr:         burn-in 专用 lr (比正常 belief_lr 大 10x)
        """
        if self.belief_net is None or self.belief_optimizer is None:
            return

        # 1. 采集当前策略的 rollout (不更新任何参数)
        episodes = self.collect_episodes(num_deals)
        belief_data = self._extract_belief_data(episodes)
        if not belief_data:
            return

        # 2. 临时调大 lr
        for pg in self.belief_optimizer.param_groups:
            pg['lr'] = lr

        # 3. 多 epoch 猛训 Belief Net
        total_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch in self._iter_belief_batches(belief_data, batch_size=256):
                loss = self.belief_net.compute_loss(
                    batch['observer_hand'], batch['history'],
                    batch['observer_pos'], batch['target_pos'],
                    batch['target_hand'],
                )
                self.belief_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.belief_net.parameters(), 1.0)
                self.belief_optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            total_loss = epoch_loss / max(1, n_batches)

        # 4. 恢复正常 lr
        normal_lr = getattr(self.config, 'belief_lr', 1e-4)
        for pg in self.belief_optimizer.param_groups:
            pg['lr'] = normal_lr

        print(f"    [JIT Belief Burn-in] {num_deals} deals × {epochs} epochs "
              f"→ belief_loss={total_loss:.4f}")

    def _extract_belief_data(self, episodes: List[Dict]) -> List[Dict]:
        data = []
        for ep in episodes:
            hands = ep['hands']
            for player, traj in ep['player_trajectories'].items():
                for step in traj:
                    hist = step['history_after']
                    for target in range(NUM_PLAYERS):
                        if target == player:
                            continue
                        data.append({
                            'observer_hand': hands[player],
                            'history': self._encode_history(hist),
                            'observer_pos': player,
                            'target_pos': target,
                            # 48 维二值特征，替代原始 52 维 one-hot
                            'target_hand': hand_to_belief_target(hands[target]),
                        })
        return data

    def _encode_history(self, history: List[int]) -> np.ndarray:
        max_len = self.env.max_history_len
        encoded = np.zeros((max_len, NUM_BIDS), dtype=np.float32)
        for i, bid in enumerate(history[-max_len:]):
            encoded[i, bid] = 1.0
        return encoded

    def _iter_belief_batches(self, data: List[Dict], batch_size: int):
        indices = np.random.permutation(len(data))
        for start in range(0, len(data), batch_size):
            batch_idx = indices[start:start + batch_size]
            if len(batch_idx) < 2:
                continue
            yield {
                'observer_hand': torch.tensor(
                    np.array([data[i]['observer_hand'] for i in batch_idx]),
                    dtype=torch.float32).to(self.device),
                'history': torch.tensor(
                    np.array([data[i]['history'] for i in batch_idx]),
                    dtype=torch.float32).to(self.device),
                'observer_pos': torch.tensor(
                    [data[i]['observer_pos'] for i in batch_idx],
                    dtype=torch.long).to(self.device),
                'target_pos': torch.tensor(
                    [data[i]['target_pos'] for i in batch_idx],
                    dtype=torch.long).to(self.device),
                'target_hand': torch.tensor(
                    np.array([data[i]['target_hand'] for i in batch_idx]),
                    dtype=torch.float32).to(self.device),
            }

    # ====================================================================
    # Info bonus monitoring
    # ====================================================================

    def compute_info_bonus_for_episodes(self, episodes: List[Dict]) -> Dict[str, float]:
        if self.belief_net is None:
            return {'partner_gain': 0, 'opponent_leak': 0, 'info_ratio': 1.0}

        partner_gains = []
        opponent_leaks = []

        self.belief_net.eval()
        with torch.no_grad():
            for ep in episodes:
                hands = ep['hands']
                for player, traj in ep['player_trajectories'].items():
                    if player % 2 == 1:
                        continue
                    partner = (player + 2) % 4
                    opponent = (player + 1) % 4
                    for step in traj:
                        h_before = self._encode_history(step['history_before'])
                        h_after = self._encode_history(step['history_after'])
                        pg = self._compute_single_info_gain(
                            hands[partner], h_before, h_after,
                            partner, player, hands[player])
                        partner_gains.append(pg)
                        ol = self._compute_single_info_gain(
                            hands[opponent], h_before, h_after,
                            opponent, player, hands[player])
                        opponent_leaks.append(ol)
        self.belief_net.train()

        pg_mean = np.mean(partner_gains) if partner_gains else 0.0
        ol_mean = np.mean(opponent_leaks) if opponent_leaks else 1e-8
        ratio = pg_mean / (abs(ol_mean) + 1e-8)

        return {
            'partner_gain': float(pg_mean),
            'opponent_leak': float(ol_mean),
            'info_ratio': float(ratio),
        }

    def _compute_single_info_gain(self, observer_hand, history_before,
                                   history_after, observer_pos, target_pos,
                                   target_hand) -> float:
        oh = torch.tensor(observer_hand, dtype=torch.float32).unsqueeze(0).to(self.device)
        hb = torch.tensor(history_before, dtype=torch.float32).unsqueeze(0).to(self.device)
        ha = torch.tensor(history_after, dtype=torch.float32).unsqueeze(0).to(self.device)
        op = torch.tensor([observer_pos], dtype=torch.long).to(self.device)
        tp = torch.tensor([target_pos], dtype=torch.long).to(self.device)
        th = torch.tensor(
            hand_to_belief_target(target_hand), dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        # 用 get_probs (sigmoid 概率), 而非 forward (logits)
        # BCE 需要 [0,1] 的概率输入, logits 会产生 NaN
        belief_before = self.belief_net.get_probs(oh, hb, op, tp).clamp(1e-7, 1 - 1e-7)
        belief_after  = self.belief_net.get_probs(oh, ha, op, tp).clamp(1e-7, 1 - 1e-7)

        ce_before = F.binary_cross_entropy(belief_before, th, reduction="none").mean(dim=-1)
        ce_after  = F.binary_cross_entropy(belief_after,  th, reduction="none").mean(dim=-1)

        # 互信息在数学上 >= 0; 负值是 Belief Net 滞后产生的估算误差
        # ReLU 截断: 不惩罚探索, 只奖励成功传递信息
        gain = float((ce_before - ce_after).item())
        return max(0.0, gain)

    # ====================================================================
    # Main training loop
    # ====================================================================

    def train(self) -> List[Dict]:
        cfg = self.config
        active_str = ','.join(str(p) for p in self.active_players)
        eff_deals = cfg.deals_per_step * cfg.accumulate_steps
        mode_str = "single_step (batch-mean baseline)" if cfg.single_step else "GAE+Critic"
        print(f"SubgameTrainer: {cfg.num_steps} steps, "
              f"mode={mode_str}, "
              f"info_bonus={cfg.use_info_bonus}, beta={cfg.beta}, "
              f"active_players=[{active_str}], "
              f"accumulate={cfg.accumulate_steps}, "
              f"effective_deals/update={eff_deals}")

        all_episodes_window = []
        latest_update_stats = {}  # 缓存最新 PPO update 指标

        for step in range(1, cfg.num_steps + 1):
            self._current_step = step
            episodes = self.collect_episodes(cfg.deals_per_step)
            all_episodes_window.extend(episodes)

            belief_loss = 0.0
            if cfg.use_info_bonus:
                belief_loss = self.train_belief_step(episodes)

            self.store_episodes(episodes)

            # PPO update every accumulate_steps
            if step % cfg.accumulate_steps == 0:
                stats = self.safe_update()
                if stats:
                    latest_update_stats = stats

            # Logging
            if step % cfg.log_interval == 0:
                rewards = [ep['final_reward'] for ep in all_episodes_window]
                rw_arr = np.array(rewards) if rewards else np.array([0.0])
                info_metrics = self.compute_info_bonus_for_episodes(episodes) \
                    if cfg.use_info_bonus else {}

                entry = {
                    'step': step,
                    'mean_reward': float(rw_arr.mean()),
                    'std_reward': float(rw_arr.std()),
                    'p25_reward': float(np.percentile(rw_arr, 25)),
                    'p75_reward': float(np.percentile(rw_arr, 75)),
                    'belief_loss': float(belief_loss),
                    'lambda': self.get_lambda(step),
                    **info_metrics,
                    **latest_update_stats,
                }
                self.log.append(entry)

                if step % cfg.eval_interval == 0:
                    # 紧凑的多指标日志行
                    ent_str = f"ent={entry.get('entropy', 0):.2f}"
                    ec_str = f"ec={entry.get('entropy_coef', 0):.3f}"
                    vl_str = f"vl={entry.get('value_loss', 0):.3f}"
                    pl_str = f"pl={entry.get('policy_loss', 0):.3f}"
                    p25 = entry['p25_reward']
                    p75 = entry['p75_reward']

                    line = (f"  [Step {step}/{cfg.num_steps}] "
                            f"r={entry['mean_reward']:+.2f}±{entry['std_reward']:.2f} "
                            f"[p25={p25:.2f} p75={p75:.2f}] "
                            f"{ent_str} {ec_str} {vl_str} {pl_str}")

                    # KL anchor 监控
                    kl_val = entry.get('kl_loss', 0)
                    kl_lam = entry.get('kl_lambda', 0)
                    if kl_lam > 0:
                        line += f" kl={kl_val:.4f}(λ={kl_lam:.3f})"

                    if cfg.use_info_bonus:
                        line += (f" bl={belief_loss:.4f}"
                                 f" ir={entry.get('info_ratio', 'N/A')}")

                    print(line)

                all_episodes_window = []

        return self.log

    def evaluate_belief_accuracy(self, num_deals: int = 50) -> float:
        """
        评估 Belief Network 质量。

        指标 (替代旧版 top13_hit_rate):
            top8_acc    — 前8张 (AKQJT987) 归属准确率
            shape_acc   — 牌型阶梯 (has_4+/5+/6+/7+) 准确率
            overall_acc — 全部48维准确率
            (见 utils/hand_features.py)

        Returns:
            overall_acc: float, 作为主指标与旧接口保持兼容
        """
        if self.belief_net is None:
            return 0.0

        self.belief_net.eval()
        all_probs, all_targets = [], []

        with torch.no_grad():
            for _ in range(num_deals):
                hands, dd_table = self.env.generate_deal()
                obs = self.env.reset(hands, dd_table)
                done = False
                while not done:
                    all_hands = self.env._current_hands
                    action, _ = self.agent.get_action(
                        obs, all_hands=all_hands, deterministic=True)
                    obs, _, done, _ = self.env.step(action)

                history = self._encode_history(self.env.history)
                oh = torch.tensor(hands[NORTH], dtype=torch.float32).unsqueeze(0).to(self.device)
                h  = torch.tensor(history,      dtype=torch.float32).unsqueeze(0).to(self.device)
                op = torch.tensor([NORTH],       dtype=torch.long).to(self.device)
                tp = torch.tensor([SOUTH],       dtype=torch.long).to(self.device)

                probs  = self.belief_net.get_probs(oh, h, op, tp)
                # 目标转为 48 维特征
                target = torch.tensor(
                    hand_to_belief_target(hands[SOUTH]), dtype=torch.float32
                ).unsqueeze(0).to(self.device)

                all_probs.append(probs)
                all_targets.append(target)

        self.belief_net.train()

        if not all_probs:
            return 0.0

        probs_cat   = torch.cat(all_probs,   dim=0)   # (N, 48)
        targets_cat = torch.cat(all_targets, dim=0)   # (N, 48)
        metrics = belief_accuracy(probs_cat, targets_cat)
        # 打印详细指标供监控
        print(f"  [BeliefNet] top8_acc={metrics['top8_acc']:.3f}  "
              f"shape_acc={metrics['shape_acc']:.3f}  "
              f"overall_acc={metrics['overall_acc']:.3f}")
        return metrics['overall_acc']


# ============================================================================
# Head-to-Head Evaluator
# ============================================================================

class HeadToHeadEvaluator:
    """
    双桌 IMP 对比评估框架 (普适化).

    原理: 同一副牌在两桌同时打, Agent A 和 Agent B 分别担任 NS 方,
    EW 方用相同的固定策略 (规则/frozen policy).
    双桌 IMP 差 = B_score - A_score, 消除牌力方差.

    这是真实团队赛 (Team Match) 的评估方式, 是论文核心对比指标.
    与子博弈类型无关, 只要 env 有 generate_deal() 和 reset() 接口即可.

    用法:
        evaluator = HeadToHeadEvaluator(env_factory, trainer_a, trainer_b)
        result = evaluator.evaluate(num_deals=500)
        print(f"B vs A: {result['mean_imp_diff']:+.2f} IMP")
    """

    def __init__(self, env_factory, trainer_a: SubgameTrainer,
                 trainer_b: SubgameTrainer, device: str = 'cpu'):
        """
        Args:
            env_factory: 无参数可调用对象, 返回一个新的子博弈 env 实例.
                         每次 evaluate() 用同一副牌在两个独立 env 里运行.
            trainer_a: Agent A (对照组, 无 r_info)
            trainer_b: Agent B (实验组, 有 r_info)
            device: torch device
        """
        self.env_factory = env_factory
        self.trainer_a = trainer_a
        self.trainer_b = trainer_b
        self.device = device

    def _run_one_agent_on_env(self, env, trainer,
                              hands: np.ndarray, dd_table: np.ndarray) -> float:
        """
        用指定 trainer 在已有 env 实例上打一副牌.
        返回 terminal reward (raw IMP regret, ≤ 0).
        使用 deterministic policy (argmax).
        """
        obs = env.reset(hands, dd_table)
        done = False
        reward = 0.0

        while not done:
            all_hands = env._current_hands
            action, _ = trainer.agent.get_action(
                obs, all_hands=all_hands, deterministic=True)
            obs, reward, done, _ = env.step(action)

        return float(reward)

    def evaluate(self, num_deals: int = 500) -> dict:
        """
        双桌对比评估.

        对每副牌:
          - A 打一遍, 得到 imp_a (terminal reward = raw IMP regret)
          - B 打同一副牌, 得到 imp_b
          - diff = imp_b - imp_a (正数 = B 更好)

        复用同一对 env 实例, 避免每副牌重建 env 触发大量初始化 print.
        """
        # 创建一次 env, 整个评估过程复用
        env_a = self.env_factory()
        env_b = self.env_factory()

        imp_a_list, imp_b_list, diff_list = [], [], []

        for _ in range(num_deals):
            hands, dd_table = env_a.generate_deal()

            imp_a = self._run_one_agent_on_env(env_a, self.trainer_a, hands, dd_table)
            imp_b = self._run_one_agent_on_env(env_b, self.trainer_b, hands, dd_table)
            diff  = imp_b - imp_a

            imp_a_list.append(imp_a)
            imp_b_list.append(imp_b)
            diff_list.append(diff)

        imp_a_arr  = np.array(imp_a_list)
        imp_b_arr  = np.array(imp_b_list)
        diff_arr   = np.array(diff_list)

        win_rate_b = float((diff_arr > 0).mean())
        tie_rate   = float((diff_arr == 0).mean())

        return {
            'mean_imp_a':    float(imp_a_arr.mean()),
            'std_imp_a':     float(imp_a_arr.std()),
            'mean_imp_b':    float(imp_b_arr.mean()),
            'std_imp_b':     float(imp_b_arr.std()),
            'mean_imp_diff': float(diff_arr.mean()),
            'std_imp_diff':  float(diff_arr.std()),
            'win_rate_b':    win_rate_b,
            'tie_rate':      tie_rate,
            'lose_rate_b':   float((diff_arr < 0).mean()),
            'n_deals':       num_deals,
        }

    def print_summary(self, result: dict, label_a: str = 'A (MAPPO)',
                      label_b: str = 'B (MAPPO+r_info)'):
        """打印对比摘要."""
        print(f"\n  ── Head-to-Head: {label_b} vs {label_a} ──")
        print(f"  {label_a:25s}: {result['mean_imp_a']:+.2f} ± {result['std_imp_a']:.2f} IMP")
        print(f"  {label_b:25s}: {result['mean_imp_b']:+.2f} ± {result['std_imp_b']:.2f} IMP")
        print(f"  Δ (B − A):               {result['mean_imp_diff']:+.2f} ± {result['std_imp_diff']:.2f} IMP")
        print(f"  Win rate (B > A):         {result['win_rate_b']:.1%}")
        print(f"  Tie rate (B = A):         {result['tie_rate']:.1%}")
        print(f"  Lose rate (B < A):        {result['lose_rate_b']:.1%}")
        print(f"  Deals evaluated:          {result['n_deals']}")
        verdict = "✅ B WINS" if result['mean_imp_diff'] > 0 else "❌ A WINS / TIE"
        print(f"  Verdict:                  {verdict}")
