"""
Dual Table Environment
======================

双桌 IMP 计算环境（使用预生成 DDS 数据）

功能：
- 双桌 self-play IMP 计算
- 训练轨迹收集（双桌 IMP 作为 reward）
- 一副牌 ×4 dealer 轮转
- 局况（vulnerability）随机化
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field

from env.bridge_bidding_env import BridgeBiddingEnv, NUM_PLAYERS, BID_PASS
from utils.scoring import Contract, calculate_score
from utils.dds_data import create_loader
from utils.imp import score_to_imp


# 四种局况组合
VULNERABILITY_COMBOS = [
    (False, False),  # None Vul
    (True, False),   # NS Vul
    (False, True),   # EW Vul
    (True, True),    # Both Vul
]


@dataclass
class DualTableResult:
    """双桌结果"""
    hands: np.ndarray
    dd_table: np.ndarray
    dealer: int = 0
    vulnerability: Tuple[bool, bool] = (False, False)

    contract_1: Optional[Contract] = None
    score_1: int = 0
    history_1: List[int] = field(default_factory=list)

    contract_2: Optional[Contract] = None
    score_2: int = 0
    history_2: List[int] = field(default_factory=list)

    imp_ns: int = 0

    @property
    def imp_ew(self) -> int:
        return -self.imp_ns


class DualTableEnv:
    """双桌 IMP 环境"""

    def __init__(self, data_path: str, max_history_len: int = 60):
        """
        Args:
            data_path: DDS 数据路径（文件或目录）
            max_history_len: 历史长度
        """
        self.env = BridgeBiddingEnv(max_history_len)
        self.loader = create_loader(data_path)

    def sample_deal(self) -> Tuple[np.ndarray, np.ndarray]:
        """采样一副牌"""
        return self.loader.sample_one()

    # =========================================================================
    # 评估接口
    # =========================================================================

    def play_deal(
        self,
        policy_fn: Callable[[Dict], int],
        hands: Optional[np.ndarray] = None,
        dd_table: Optional[np.ndarray] = None,
        dealer: int = 0,
        vulnerability: Tuple[bool, bool] = (False, False),
    ) -> DualTableResult:
        """
        进行双桌比赛（评估用）

        policy_fn 签名: obs -> action_int
        """
        if hands is None or dd_table is None:
            hands, dd_table = self.sample_deal()

        # 桌 1: 正常位置
        contract_1, score_1, history_1 = self._play_single_table(
            policy_fn, hands, dd_table, dealer, vulnerability
        )

        # 桌 2: N↔E, S↔W 互换
        swapped_hands, swapped_dd = self._swap(hands, dd_table)
        contract_2, score_2, history_2 = self._play_single_table(
            policy_fn, swapped_hands, swapped_dd, dealer, vulnerability
        )

        return DualTableResult(
            hands=hands, dd_table=dd_table,
            dealer=dealer, vulnerability=vulnerability,
            contract_1=contract_1, score_1=score_1, history_1=history_1,
            contract_2=contract_2, score_2=score_2, history_2=history_2,
            imp_ns=score_to_imp(score_1 - score_2),
        )

    # =========================================================================
    # 训练轨迹收集（核心接口）
    # =========================================================================

    def collect_episodes(
        self,
        policy_fn: Callable[[Dict], Tuple[int, Dict]],
        num_deals: int,
        rotate_dealer: bool = True,
    ) -> List[Dict]:
        """
        收集训练轨迹，使用双桌 IMP 作为 reward。

        一副牌的流程：
        1. 采样一副牌 (hands, dd_table)
        2. 随机 vulnerability
        3. 对每个 dealer (0-3 or 单个):
           a. 桌 1 正常位置叫牌 → score_1 (NS 视角)
           b. 桌 2 互换位置叫牌 → score_2 (NS 视角)
           c. IMP = score_to_imp(score_1 - score_2)
           d. 记录桌 1 的轨迹，用 IMP 作为终局 reward
              NS 玩家得 +IMP, EW 玩家得 -IMP

        Args:
            policy_fn: obs -> (action_int, extras_dict)
                       extras_dict 需包含 'log_prob' 和 'value'
            num_deals: 采样的牌副数
            rotate_dealer: True 则一副牌用 4 次（dealer=0,1,2,3），False 则随机 1 次

        Returns:
            List[Dict]，每个 Dict 是一个 episode，包含:
                'player_trajectories': {player_id: [step_dict, ...]}
                'hands', 'dd_table', 'dealer', 'vulnerability'
                'contract', 'score', 'imp_ns'
        """
        episodes = []

        for _ in range(num_deals):
            hands, dd_table = self.sample_deal()
            vulnerability = VULNERABILITY_COMBOS[np.random.randint(4)]

            dealers = range(4) if rotate_dealer else [np.random.randint(4)]

            for dealer in dealers:
                episode = self._collect_one_episode(
                    policy_fn, hands, dd_table, dealer, vulnerability
                )
                episodes.append(episode)

        return episodes

    def _collect_one_episode(
        self,
        policy_fn: Callable[[Dict], Tuple[int, Dict]],
        hands: np.ndarray,
        dd_table: np.ndarray,
        dealer: int,
        vulnerability: Tuple[bool, bool],
    ) -> Dict:
        """
        收集一个 episode 的轨迹（双桌 IMP）。

        桌 1 收集详细轨迹（用于训练），桌 2 只取最终分数（用于算 IMP）。
        """
        # ---- 桌 1: 收集轨迹 ----
        obs = self.env.reset(hands, dealer, vulnerability)
        player_trajs = {p: [] for p in range(NUM_PLAYERS)}
        done = False

        while not done:
            player = self.env.state.current_player
            action, extra = policy_fn(obs)
            next_obs, _, done, info = self.env.step(action)

            # 中间步 reward=0，终局步后面回填
            player_trajs[player].append({
                'obs': obs,
                'action': action,
                'reward': 0.0,
                'done': done,
                **extra,
            })
            obs = next_obs

        contract_1 = self.env.state.final_contract
        score_1 = self._compute_score(contract_1, dd_table, vulnerability)

        # ---- 桌 2: 只要最终分数 ----
        swapped_hands, swapped_dd = self._swap(hands, dd_table)

        def eval_policy(obs_):
            action_, extra_ = policy_fn(obs_)
            return action_

        _, score_2, _ = self._play_single_table(
            eval_policy, swapped_hands, swapped_dd, dealer, vulnerability
        )

        # ---- 计算 IMP 并回填 reward ----
        imp_ns = score_to_imp(score_1 - score_2)

        # 终局 reward 分配: NS 得 +IMP, EW 得 -IMP
        for player, traj in player_trajs.items():
            if traj:  # 该玩家至少叫过一次
                reward = imp_ns if player % 2 == 0 else -imp_ns
                traj[-1]['reward'] = float(reward)

        return {
            'player_trajectories': player_trajs,
            'hands': hands,
            'dd_table': dd_table,
            'dealer': dealer,
            'vulnerability': vulnerability,
            'contract': contract_1,
            'score': score_1,
            'imp_ns': imp_ns,
        }

    # =========================================================================
    # 内部方法
    # =========================================================================

    @staticmethod
    def _swap(
        hands: np.ndarray, dd_table: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        位置互换: N↔E, S↔W

        self-play 语义: 桌 1 的 NS 牌在桌 2 由 EW 位拿到，反之亦然。
        """
        swapped_hands = np.zeros_like(hands)
        swapped_hands[0] = hands[1]
        swapped_hands[1] = hands[0]
        swapped_hands[2] = hands[3]
        swapped_hands[3] = hands[2]

        swapped_dd = np.zeros_like(dd_table)
        swapped_dd[:, 0] = dd_table[:, 1]
        swapped_dd[:, 1] = dd_table[:, 0]
        swapped_dd[:, 2] = dd_table[:, 3]
        swapped_dd[:, 3] = dd_table[:, 2]

        return swapped_hands, swapped_dd

    def _play_single_table(
        self,
        policy_fn: Callable[[Dict], int],
        hands: np.ndarray,
        dd_table: np.ndarray,
        dealer: int,
        vulnerability: Tuple[bool, bool],
    ) -> Tuple[Optional[Contract], int, List[int]]:
        """进行单桌叫牌，返回 (contract, score_ns, history)"""
        obs = self.env.reset(hands, dealer, vulnerability)
        history = []
        done = False

        while not done:
            action = policy_fn(obs)
            obs, _, done, info = self.env.step(action)
            history.append(info['bid'])

        contract = self.env.state.final_contract
        score = self._compute_score(contract, dd_table, vulnerability)
        return contract, score, history

    @staticmethod
    def _compute_score(
        contract: Optional[Contract],
        dd_table: np.ndarray,
        vulnerability: Tuple[bool, bool],
    ) -> int:
        """计算得分（NS 视角）"""
        if contract is None:
            return 0

        tricks = int(dd_table[contract.suit, contract.declarer])
        declarer_vul = vulnerability[contract.declarer % 2]
        score = calculate_score(contract, tricks, declarer_vul)

        # 转换为 NS 视角
        if contract.declarer % 2 == 1:  # EW 庄家
            score = -score

        return score


def make_random_policy():
    """创建随机策略（评估用，签名: obs -> action_int）"""
    def policy(obs):
        legal = obs['legal_actions']
        return np.random.choice(np.where(legal > 0.5)[0])
    return policy
