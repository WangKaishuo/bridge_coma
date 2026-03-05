#!/usr/bin/env python3
"""
Training Script
===============

训练流程：
1. 采样 num_deals 副牌 × 4 dealer = batch_episodes 个 episode
2. 每个 episode 打双桌，用 IMP 作为终局 reward（NS +IMP, EW -IMP）
3. 将轨迹存入 agent buffer
4. 每 update_interval 个 episode 做一次 PPO update
5. 定期评估（双桌 IMP）并保存 checkpoint
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from env import BridgeBiddingEnv, DualTableEnv, NUM_PLAYERS
from env.dual_table_env import make_random_policy
from utils.scoring import calculate_score
from algorithms import IPPOAgent, PPOConfig, MAPPOAgent, MAPPOConfig


class Trainer:
    """训练器"""

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        data_path = config['data_path']
        if not Path(data_path).exists():
            raise FileNotFoundError(f"DDS data not found: {data_path}")

        self.dual_env = DualTableEnv(data_path=data_path)

        # 创建 agent
        algorithm = config.get('algorithm', 'mappo')
        net_kwargs = dict(
            hand_dim=256, history_dim=256, hidden_dim=256, device=self.device
        )
        if algorithm == 'ippo':
            self.agent = IPPOAgent(PPOConfig(**net_kwargs))
        else:
            self.agent = MAPPOAgent(MAPPOConfig(**net_kwargs))

        self.is_mappo = isinstance(self.agent, MAPPOAgent)

        # 训练参数
        self.deals_per_collect = config.get('deals_per_collect', 4)  # 每次采样的牌副数
        self.rotate_dealer = config.get('rotate_dealer', True)       # 一副牌 ×4 dealer
        # rotate=True 时: 4 deals × 4 dealers = 16 episodes per collect

        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 统计
        self.total_episodes = 0
        self.total_updates = 0

    # =========================================================================
    # 训练核心
    # =========================================================================

    def collect_and_store(self) -> dict:
        """
        采样一批 episode，存入 agent buffer。

        Returns:
            batch 统计信息
        """
        def policy_fn(obs):
            """训练用 policy，返回 (action, extras)"""
            hands = self.dual_env.env.state.hands
            if self.is_mappo:
                action, extra = self.agent.get_action(obs, all_hands=hands)
            else:
                action, extra = self.agent.get_action(obs)
            # 额外记录 all_hands 以便 MAPPO store
            extra['_all_hands'] = hands
            return action, extra

        episodes = self.dual_env.collect_episodes(
            policy_fn=policy_fn,
            num_deals=self.deals_per_collect,
            rotate_dealer=self.rotate_dealer,
        )

        # 存入 agent buffer
        imps = []
        bid_lengths = []
        contracts = []

        for ep in episodes:
            imp_ns = ep['imp_ns']
            imps.append(imp_ns)
            contract = ep['contract']
            contracts.append(str(contract) if contract else 'Pass-out')

            for player in range(NUM_PLAYERS):
                traj = ep['player_trajectories'][player]
                bid_lengths.append(len(traj))
                for step in traj:
                    if self.is_mappo:
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
                    else:
                        self.agent.store_transition(
                            player,
                            step['obs'],
                            step['action'],
                            step['log_prob'],
                            step['reward'],
                            step['value'],
                            step['done'],
                        )

        self.total_episodes += len(episodes)

        return {
            'num_episodes': len(episodes),
            'mean_imp': np.mean(imps),
            'std_imp': np.std(imps),
            'mean_bid_length': np.mean(bid_lengths) if bid_lengths else 0,
            'pass_out_rate': sum(1 for c in contracts if c == 'Pass-out') / len(contracts),
        }

    def evaluate(self, num_deals: int = 50) -> dict:
        """
        评估：用双桌 IMP，确定性策略。

        Args:
            num_deals: 评估的牌副数
        """
        def eval_policy(obs):
            hands = self.dual_env.env.state.hands
            if self.is_mappo:
                action, _ = self.agent.get_action(obs, all_hands=hands, deterministic=True)
            else:
                action, _ = self.agent.get_action(obs, deterministic=True)
            return action

        imps = []
        scores = []
        contracts = []

        for _ in range(num_deals):
            result = self.dual_env.play_deal(eval_policy)
            imps.append(result.imp_ns)
            scores.append(result.score_1)
            if result.contract_1:
                contracts.append(str(result.contract_1))
            else:
                contracts.append('Pass-out')

        pass_out_rate = sum(1 for c in contracts if c == 'Pass-out') / max(1, len(contracts))

        return {
            'mean_imp': np.mean(imps) if imps else 0,
            'std_imp': np.std(imps) if imps else 0,
            'mean_score': np.mean(scores) if scores else 0,
            'pass_out_rate': pass_out_rate,
            'num_deals': num_deals,
        }

    # =========================================================================
    # 主训练循环
    # =========================================================================

    def train(
        self,
        num_iterations: int,
        eval_interval: int = 50,
        save_interval: int = 200,
        eval_deals: int = 50,
    ):
        """
        主训练循环。

        每个 iteration:
          1. collect deals_per_collect 副牌（×4 dealer = 16 episodes）
          2. 做一次 PPO update

        Args:
            num_iterations: 总迭代次数
            eval_interval: 每隔多少次迭代评估一次
            save_interval: 每隔多少次迭代保存一次
            eval_deals: 评估时用多少副牌
        """
        algorithm = self.config.get('algorithm', 'mappo')
        eps_per_iter = self.deals_per_collect * (4 if self.rotate_dealer else 1)

        print(f"{'=' * 60}")
        print(f"Training: {algorithm.upper()}")
        print(f"Device: {self.device}")
        print(f"Deals/iter: {self.deals_per_collect}, Rotate dealer: {self.rotate_dealer}")
        print(f"Episodes/iter: {eps_per_iter}")
        print(f"Total iterations: {num_iterations}")
        print(f"{'=' * 60}")

        for it in tqdm(range(1, num_iterations + 1), desc="Training"):
            # 1. 收集 episode
            collect_stats = self.collect_and_store()

            # 2. PPO update
            update_stats = self.agent.update()
            if update_stats:
                self.total_updates += 1

            # 3. 日志
            if it % eval_interval == 0:
                eval_stats = self.evaluate(eval_deals)
                print(
                    f"\n[Iter {it}] "
                    f"Episodes: {self.total_episodes} | "
                    f"Updates: {self.total_updates}"
                )
                print(
                    f"  Train: IMP={collect_stats['mean_imp']:+.2f}±{collect_stats['std_imp']:.2f}, "
                    f"BidLen={collect_stats['mean_bid_length']:.1f}, "
                    f"PassOut={collect_stats['pass_out_rate']:.1%}"
                )
                print(
                    f"  Eval:  IMP={eval_stats['mean_imp']:+.2f}±{eval_stats['std_imp']:.2f}, "
                    f"Score={eval_stats['mean_score']:.0f}, "
                    f"PassOut={eval_stats['pass_out_rate']:.1%}"
                )
                if update_stats:
                    print(
                        f"  Loss:  total={update_stats.get('loss', 0):.4f}, "
                        f"policy={update_stats.get('policy_loss', 0):.4f}, "
                        f"value={update_stats.get('value_loss', 0):.4f}, "
                        f"entropy={update_stats.get('entropy', 0):.4f}"
                    )

            # 4. 保存
            if it % save_interval == 0:
                path = str(self.checkpoint_dir / f'ckpt_iter{it}.pt')
                self.agent.save(path)
                print(f"  Saved: {path}")

        # 最终保存
        final_path = str(self.checkpoint_dir / 'final.pt')
        self.agent.save(final_path)
        print(f"\nTraining complete. Final model: {final_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Bridge-COMA Training")
    p.add_argument('--algorithm', default='mappo', choices=['ippo', 'mappo'])
    p.add_argument('--data_path', required=True, help='DDS data path (file or directory)')
    p.add_argument('--num_iterations', type=int, default=2500,
                   help='Training iterations (default: 2500, ~40k episodes with rotate)')
    p.add_argument('--deals_per_collect', type=int, default=4,
                   help='Deals sampled per iteration (default: 4, ×4 dealer = 16 episodes)')
    p.add_argument('--no_rotate', action='store_true',
                   help='Disable dealer rotation (1 dealer per deal instead of 4)')
    p.add_argument('--eval_interval', type=int, default=50)
    p.add_argument('--save_interval', type=int, default=200)
    p.add_argument('--eval_deals', type=int, default=50)
    p.add_argument('--device', default=None, help='Device (default: auto)')
    p.add_argument('--checkpoint_dir', default='checkpoints')
    args = p.parse_args()

    config = {
        'algorithm': args.algorithm,
        'data_path': args.data_path,
        'deals_per_collect': args.deals_per_collect,
        'rotate_dealer': not args.no_rotate,
        'checkpoint_dir': args.checkpoint_dir,
    }
    if args.device:
        config['device'] = args.device

    trainer = Trainer(config)
    trainer.train(
        num_iterations=args.num_iterations,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        eval_deals=args.eval_deals,
    )


if __name__ == "__main__":
    main()
