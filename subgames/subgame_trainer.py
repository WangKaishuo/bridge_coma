"""
Subgame Trainer
===============

通用子博弈训练框架, 支持:
- Stayman (纯合作, 单桌 IMP)
- Competitive (合作-对抗, self-play)

支持可选的 Dual-Info Bonus:
- use_info_bonus=False → 纯 MAPPO (control)
- use_info_bonus=True, beta=0.0 → partner-only
- use_info_bonus=True, beta=0.5 → dual-info
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
    num_steps: int = 5000           # 总训练步数 (iterations)
    deals_per_step: int = 8         # 每步采样的牌副数
    lr: float = 3e-4
    belief_lr: float = 1e-3

    # Info bonus
    use_info_bonus: bool = False
    beta: float = 0.5               # opponent penalty weight
    lambda_start: float = 0.5       # info bonus 初始权重
    lambda_end: float = 0.1         # info bonus 最终权重 (退火)
    belief_warmup_steps: int = 500  # 前 N 步只训练 Belief (不用 info bonus)

    # Network
    hand_dim: int = 256
    history_dim: int = 256
    hidden_dim: int = 256

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    num_epochs: int = 4
    batch_size: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Eval
    eval_interval: int = 200
    eval_deals: int = 100
    log_interval: int = 50

    # Device
    device: str = 'cpu'


class SubgameTrainer:
    """
    通用子博弈训练器.

    同时管理:
    - MAPPOAgent (policy + value)
    - BeliefNetwork (推断手牌)
    - DualInfoComputer (r_info)
    """

    def __init__(self, env, config: SubgameConfig):
        """
        Args:
            env: StaymanSubgameEnv or CompetitiveSubgameEnv
            config: SubgameConfig
        """
        self.env = env
        self.config = config
        self.device = config.device

        # MAPPO agent
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

        # Belief network (optional, for info bonus)
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

            # Running stats for normalization
            self.partner_stats = RunningStats()
            self.opponent_stats = RunningStats()

        # Training log
        self.log = []

    def get_lambda(self, step: int) -> float:
        """Info bonus 权重退火: 线性从 lambda_start → lambda_end."""
        cfg = self.config
        if step < cfg.belief_warmup_steps:
            return 0.0  # warmup 阶段不用 info bonus
        progress = min(1.0, (step - cfg.belief_warmup_steps) /
                       max(1, cfg.num_steps - cfg.belief_warmup_steps))
        return cfg.lambda_start + (cfg.lambda_end - cfg.lambda_start) * progress

    # ====================================================================
    # Rollout collection
    # ====================================================================

    def collect_episodes(self, num_deals: int) -> List[Dict]:
        """
        在子博弈环境中采样 episode.

        每个 episode 记录:
        - 标准 MAPPO 轨迹 (obs, action, reward, ...)
        - belief 数据: 每步的 history_before/after (用于 info bonus)
        """
        episodes = []

        for _ in range(num_deals):
            hands, dd_table = self.env.generate_deal()
            obs = self.env.reset(hands, dd_table)

            player_trajs = {p: [] for p in range(NUM_PLAYERS)}
            done = False

            while not done:
                player = self.env.current_player

                # 保存 history before action
                history_before = self.env.history.copy()

                # Get action from agent
                all_hands = self.env._current_hands
                action, extra = self.agent.get_action(obs, all_hands=all_hands)
                extra['_all_hands'] = all_hands

                obs_next, reward, done, info = self.env.step(action)

                # 保存 history after action
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

            # 终局: 回填 reward 到最后一步
            final_reward = info.get('imp', reward)
            for p, traj in player_trajs.items():
                if traj:
                    # NS 得正分, EW 得负分
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
        """将 episode 数据存入 agent buffer."""
        for ep in episodes:
            for player, traj in ep['player_trajectories'].items():
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
    # Belief training
    # ====================================================================

    def train_belief_step(self, episodes: List[Dict]) -> float:
        """
        一步 Belief Network 训练.

        从 episode 中提取 (observer_hand, history, target_hand) 数据.
        """
        if self.belief_net is None:
            return 0.0

        belief_data = self._extract_belief_data(episodes)
        if not belief_data:
            return 0.0

        # Batch training
        total_loss = 0.0
        n = 0

        for batch in self._iter_belief_batches(belief_data, batch_size=256):
            loss = self.belief_net.compute_loss(
                batch['observer_hand'],
                batch['history'],
                batch['observer_pos'],
                batch['target_pos'],
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
        """从 episode 提取 belief 训练数据."""
        data = []
        for ep in episodes:
            hands = ep['hands']  # (4, 52)
            for player, traj in ep['player_trajectories'].items():
                for step in traj:
                    hist = step['history_after']
                    # observer → target pairs (all other players)
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
        """将 history (list of bid ints) 编码为 one-hot 序列."""
        max_len = self.env.max_history_len
        encoded = np.zeros((max_len, NUM_BIDS), dtype=np.float32)
        for i, bid in enumerate(history[-max_len:]):
            encoded[i, bid] = 1.0
        return encoded

    def _iter_belief_batches(self, data: List[Dict], batch_size: int):
        """迭代 belief 训练 batch."""
        indices = np.random.permutation(len(data))

        for start in range(0, len(data), batch_size):
            batch_idx = indices[start:start + batch_size]
            if len(batch_idx) < 2:
                continue

            yield {
                'observer_hand': torch.tensor(
                    np.array([data[i]['observer_hand'] for i in batch_idx]),
                    dtype=torch.float32
                ).to(self.device),
                'history': torch.tensor(
                    np.array([data[i]['history'] for i in batch_idx]),
                    dtype=torch.float32
                ).to(self.device),
                'observer_pos': torch.tensor(
                    [data[i]['observer_pos'] for i in batch_idx],
                    dtype=torch.long
                ).to(self.device),
                'target_pos': torch.tensor(
                    [data[i]['target_pos'] for i in batch_idx],
                    dtype=torch.long
                ).to(self.device),
                'target_hand': torch.tensor(
                    np.array([data[i]['target_hand'] for i in batch_idx]),
                    dtype=torch.float32
                ).to(self.device),
            }

    # ====================================================================
    # Compute info bonus (stub — actual integration with advantage)
    # ====================================================================

    def compute_info_bonus_for_episodes(self, episodes: List[Dict]) -> Dict[str, float]:
        """
        计算 episode 中每步的 info bonus 指标 (用于监控, 不直接改 advantage).

        Returns:
            metrics dict: partner_gain_mean, opponent_leak_mean, info_ratio
        """
        if self.belief_net is None:
            return {'partner_gain': 0, 'opponent_leak': 0, 'info_ratio': 1.0}

        partner_gains = []
        opponent_leaks = []

        self.belief_net.eval()
        with torch.no_grad():
            for ep in episodes:
                hands = ep['hands']
                for player, traj in ep['player_trajectories'].items():
                    if player % 2 == 1:  # 只看 NS
                        continue
                    partner = (player + 2) % 4
                    opponent = (player + 1) % 4

                    for step in traj:
                        h_before = self._encode_history(step['history_before'])
                        h_after = self._encode_history(step['history_after'])

                        # Partner's info gain
                        pg = self._compute_single_info_gain(
                            hands[partner], h_before, h_after,
                            partner, player, hands[player]
                        )
                        partner_gains.append(pg)

                        # Opponent's info leak
                        ol = self._compute_single_info_gain(
                            hands[opponent], h_before, h_after,
                            opponent, player, hands[player]
                        )
                        opponent_leaks.append(ol)

        self.belief_net.train()

        pg_mean = np.mean(partner_gains) if partner_gains else 0.0
        ol_mean = np.mean(opponent_leaks) if opponent_leaks else 1e-8
        ratio = pg_mean / (ol_mean + 1e-8)

        return {
            'partner_gain': pg_mean,
            'opponent_leak': ol_mean,
            'info_ratio': ratio,
        }

    def _compute_single_info_gain(
        self,
        observer_hand: np.ndarray,
        history_before: np.ndarray,
        history_after: np.ndarray,
        observer_pos: int,
        target_pos: int,
        target_hand: np.ndarray,
    ) -> float:
        """计算单个观察者的信息增益."""
        oh = torch.tensor(observer_hand, dtype=torch.float32).unsqueeze(0).to(self.device)
        hb = torch.tensor(history_before, dtype=torch.float32).unsqueeze(0).to(self.device)
        ha = torch.tensor(history_after, dtype=torch.float32).unsqueeze(0).to(self.device)
        op = torch.tensor([observer_pos], dtype=torch.long).to(self.device)
        tp = torch.tensor([target_pos], dtype=torch.long).to(self.device)
        th = torch.tensor(target_hand, dtype=torch.float32).unsqueeze(0).to(self.device)

        belief_before = self.belief_net(oh, hb, op, tp)
        belief_after = self.belief_net(oh, ha, op, tp)

        ce_before = F.binary_cross_entropy(belief_before, th, reduction='none').sum(dim=-1)
        ce_after = F.binary_cross_entropy(belief_after, th, reduction='none').sum(dim=-1)

        return float((ce_before - ce_after).item())

    # ====================================================================
    # Main training loop
    # ====================================================================

    def train(self) -> List[Dict]:
        """
        主训练循环.

        Returns:
            training log (list of dicts)
        """
        cfg = self.config
        print(f"SubgameTrainer: {cfg.num_steps} steps, "
              f"info_bonus={cfg.use_info_bonus}, beta={cfg.beta}")

        for step in range(1, cfg.num_steps + 1):
            # 1. Collect episodes
            episodes = self.collect_episodes(cfg.deals_per_step)

            # 2. Train belief (if using info bonus)
            belief_loss = 0.0
            if cfg.use_info_bonus:
                belief_loss = self.train_belief_step(episodes)

            # 3. Store to MAPPO buffer and update
            self.store_episodes(episodes)
            update_stats = self.agent.update()

            # 4. Logging
            if step % cfg.log_interval == 0:
                rewards = [ep['final_reward'] for ep in episodes]
                info_metrics = self.compute_info_bonus_for_episodes(episodes) \
                    if cfg.use_info_bonus else {}

                entry = {
                    'step': step,
                    'mean_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'belief_loss': belief_loss,
                    'lambda': self.get_lambda(step),
                    **info_metrics,
                    **(update_stats or {}),
                }
                self.log.append(entry)

                if step % cfg.eval_interval == 0:
                    print(f"  [Step {step}/{cfg.num_steps}] "
                          f"reward={entry['mean_reward']:+.2f} "
                          f"belief_loss={belief_loss:.4f} "
                          f"info_ratio={entry.get('info_ratio', 'N/A')}")

        return self.log

    def evaluate_belief_accuracy(self, num_deals: int = 50) -> float:
        """
        评估 Belief Network 的准确率.

        准确率 = 对 target 手中确实有的牌, belief > 0.5 的比例.
        """
        if self.belief_net is None:
            return 0.0

        self.belief_net.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for _ in range(num_deals):
                hands, dd_table = self.env.generate_deal()
                # 简单测试: N 看完整 history 后预测 S 的手牌
                obs = self.env.reset(hands, dd_table)
                # 用随机策略走几步
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

                belief = self.belief_net(oh, h, op, tp)  # (1, 52)
                target = torch.tensor(hands[SOUTH], dtype=torch.float32).unsqueeze(0).to(self.device)

                pred = (belief > 0.5).float()
                correct += (pred == target).sum().item()
                total += 52

        self.belief_net.train()
        return correct / max(1, total)
