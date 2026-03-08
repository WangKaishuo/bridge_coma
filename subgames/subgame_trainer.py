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


@dataclass
class SubgameConfig:
    """子博弈训练配置."""
    # Training
    num_steps: int = 5000
    deals_per_step: int = 32
    accumulate_steps: int = 4         # collect N steps before 1 PPO update
    lr: float = 1e-4
    belief_lr: float = 1e-3

    # Info bonus
    use_info_bonus: bool = False
    beta: float = 0.5
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

        self.log = []
        self._current_step = 0  # 当前训练步数, 用于 entropy annealing

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
        """存入 agent buffer, 只存 active players."""
        for ep in episodes:
            for player, traj in ep['player_trajectories'].items():
                if player not in self.active_players:
                    continue
                for step in traj:
                    self.agent.store_transition(
                        player,
                        step['obs'],
                        step['action'],
                        step['log_prob'],
                        step['reward'],
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
        """
        agent = self.agent
        min_size = self.config.min_buffer_size
        ent_coef = self.get_entropy_coef(self._current_step)
        is_single = self.config.single_step

        total_loss = total_policy = total_value = total_entropy = num_updates = 0

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

                    # Loss: single_step → 断开 Critic 梯度 (value_coef=0)
                    if is_single:
                        loss = policy_loss - ent_coef * entropy.mean()
                        value_loss_val = 0.0
                    else:
                        values = agent.model.critic(
                            batch['obs'], batch.get('all_hands'))
                        value_loss = F.mse_loss(values, batch['returns'])
                        loss = (policy_loss
                                + agent.config.value_coef * value_loss
                                - ent_coef * entropy.mean())
                        value_loss_val = value_loss.item()

                    agent.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.model.parameters(), agent.config.max_grad_norm
                    )
                    agent.optimizer.step()

                    total_loss += loss.item()
                    total_policy += policy_loss.item()
                    total_value += value_loss_val
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
            'entropy_coef': ent_coef,
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
                            'target_hand': hands[target],
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
        th = torch.tensor(target_hand, dtype=torch.float32).unsqueeze(0).to(self.device)

        belief_before = self.belief_net(oh, hb, op, tp).clamp(1e-7, 1 - 1e-7)
        belief_after = self.belief_net(oh, ha, op, tp).clamp(1e-7, 1 - 1e-7)

        ce_before = F.binary_cross_entropy(belief_before, th, reduction='none').sum(dim=-1)
        ce_after = F.binary_cross_entropy(belief_after, th, reduction='none').sum(dim=-1)

        return float((ce_before - ce_after).item())

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

                    if cfg.use_info_bonus:
                        line += (f" bl={belief_loss:.4f}"
                                 f" ir={entry.get('info_ratio', 'N/A')}")

                    print(line)

                all_episodes_window = []

        return self.log

    def evaluate_belief_accuracy(self, num_deals: int = 50) -> float:
        if self.belief_net is None:
            return 0.0

        self.belief_net.eval()
        correct = total = 0

        with torch.no_grad():
            for _ in range(num_deals):
                hands, dd_table = self.env.generate_deal()
                obs = self.env.reset(hands, dd_table)
                done = False
                while not done:
                    legal = obs['legal_actions']
                    action = np.random.choice(np.where(legal > 0.5)[0])
                    obs, _, done, _ = self.env.step(action)

                history = self._encode_history(self.env.history)
                oh = torch.tensor(hands[NORTH], dtype=torch.float32).unsqueeze(0).to(self.device)
                h = torch.tensor(history, dtype=torch.float32).unsqueeze(0).to(self.device)
                op = torch.tensor([NORTH], dtype=torch.long).to(self.device)
                tp = torch.tensor([SOUTH], dtype=torch.long).to(self.device)

                belief = self.belief_net(oh, h, op, tp)
                target = torch.tensor(hands[SOUTH], dtype=torch.float32).unsqueeze(0).to(self.device)

                pred = (belief > 0.5).float()
                correct += (pred == target).sum().item()
                total += 52

        self.belief_net.train()
        return correct / max(1, total)
