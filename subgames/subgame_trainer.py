"""
Subgame Trainer
===============

通用子博弈训练框架, 支持:
- Stayman (纯合作, 单桌 IMP) — 只训练 NS
- Competitive (合作-对抗, self-play) — 训练所有 4 方

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
    num_steps: int = 5000
    deals_per_step: int = 32          # 32 deals × ~4 bids/player × 4 players ≈ 512 transitions
    lr: float = 3e-4
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
    batch_size: int = 64              # smaller batch → more gradient steps per update
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Which players does the agent control?
    # None = all 4 (Competitive), [0,2] = NS only (Stayman)
    active_players: Optional[List[int]] = None

    # Minimum buffer size before PPO update (prevents NaN from std of 1 element)
    min_buffer_size: int = 4

    # Eval
    eval_interval: int = 200
    eval_deals: int = 100
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

    def get_lambda(self, step: int) -> float:
        cfg = self.config
        if step < cfg.belief_warmup_steps:
            return 0.0
        progress = min(1.0, (step - cfg.belief_warmup_steps) /
                       max(1, cfg.num_steps - cfg.belief_warmup_steps))
        return cfg.lambda_start + (cfg.lambda_end - cfg.lambda_start) * progress

    # ====================================================================
    # Rollout collection
    # ====================================================================

    def collect_episodes(self, num_deals: int) -> List[Dict]:
        """采样 episode, 只记录 active_players 的轨迹."""
        episodes = []

        for _ in range(num_deals):
            hands, dd_table = self.env.generate_deal()
            obs = self.env.reset(hands, dd_table)

            player_trajs = {p: [] for p in self.active_players}
            done = False

            while not done:
                player = self.env.current_player
                history_before = self.env.history.copy()

                all_hands = self.env._current_hands
                action, extra = self.agent.get_action(obs, all_hands=all_hands)
                extra['_all_hands'] = all_hands

                obs_next, reward, done, info = self.env.step(action)
                history_after = self.env.history.copy()

                if player in self.active_players:
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

            final_reward = info.get('imp', reward)
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
        """将 episode 数据存入 agent buffer. 只存 active players."""
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

    def safe_update(self) -> Dict[str, float]:
        """
        安全的 PPO update.

        1. 清空非 active players 的 buffer
        2. 跳过 action 数量 < min_buffer_size 的 buffer (防止 std() NaN)
        """
        min_size = self.config.min_buffer_size

        for p in range(NUM_PLAYERS):
            buf = self.agent.buffers[p]
            if p not in self.active_players or len(buf.actions) < min_size:
                buf.reset()

        return self.agent.update()

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
                            partner, player, hands[player]
                        )
                        partner_gains.append(pg)

                        ol = self._compute_single_info_gain(
                            hands[opponent], h_before, h_after,
                            opponent, player, hands[player]
                        )
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
        print(f"SubgameTrainer: {cfg.num_steps} steps, "
              f"info_bonus={cfg.use_info_bonus}, beta={cfg.beta}, "
              f"active_players=[{active_str}]")

        for step in range(1, cfg.num_steps + 1):
            episodes = self.collect_episodes(cfg.deals_per_step)

            belief_loss = 0.0
            if cfg.use_info_bonus:
                belief_loss = self.train_belief_step(episodes)

            self.store_episodes(episodes)
            update_stats = self.safe_update()

            if step % cfg.log_interval == 0:
                rewards = [ep['final_reward'] for ep in episodes]
                info_metrics = self.compute_info_bonus_for_episodes(episodes) \
                    if cfg.use_info_bonus else {}

                entry = {
                    'step': step,
                    'mean_reward': float(np.mean(rewards)),
                    'std_reward': float(np.std(rewards)),
                    'belief_loss': float(belief_loss),
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
        if self.belief_net is None:
            return 0.0

        self.belief_net.eval()
        correct = 0
        total = 0

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
