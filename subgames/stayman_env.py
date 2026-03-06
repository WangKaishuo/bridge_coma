"""
Stayman Subgame Environment
============================

纯合作子博弈, 固定前缀 1NT - Pass - 2C - Pass.

特点:
- EW 全部自动 Pass, 只有 NS 决策
- 无 BC 预热 (动作空间小 ~6 种)
- 单桌评估: actual score vs DDS optimal → IMP
- 约束发牌: N 15-17 HCP 均型, S 8+ HCP 有 4 张高花
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from env import (
    BridgeBiddingEnv, NUM_BIDS, NUM_PLAYERS,
    BID_PASS, BID_1C,
    bid_to_string, string_to_bid,
    NORTH, EAST, SOUTH, WEST,
)
from utils.scoring import Contract, calculate_score
from utils.imp import score_to_imp
from utils.dds_data import create_loader
from subgames.action_mask import (
    count_hcp, is_balanced, count_suit_length, suit_lengths,
)


class StaymanSubgameEnv:
    """
    Stayman 子博弈环境.

    固定前缀: 1NT - Pass - 2C - Pass
    学习目标: 后续叫牌找到 4-4 高花配合
    评估: 单桌 actual vs DDS optimal → IMP
    """

    FIXED_PREFIX = ["1NT", "Pass", "2C", "Pass"]

    def __init__(self, data_path: str, max_history_len: int = 60):
        self.loader = create_loader(data_path)
        self.env = BridgeBiddingEnv(max_history_len)
        self.max_history_len = max_history_len

        # 检查数据是否已经是约束数据 (所有牌都符合约束)
        # 如果是, 就直接用; 如果不是, 进行预筛选
        self._is_constrained_data = self._check_if_constrained()
        if not self._is_constrained_data:
            self._filtered_deals = []
            self._prefetch(min_deals=500, max_attempts=50000)
        else:
            self._filtered_deals = None
            print(f"StaymanSubgameEnv: using pre-generated constrained data "
                  f"({len(self.loader)} samples)")

    def _check_if_constrained(self, sample_size: int = 20) -> bool:
        """检查数据是否全部符合约束 (抽样检查)."""
        passed = 0
        for _ in range(sample_size):
            hands, _ = self.loader.sample_one()
            if self._satisfies_constraints(hands):
                passed += 1
        # 如果 90%+ 符合, 认为是约束数据
        return passed >= sample_size * 0.9

    def _prefetch(self, min_deals: int, max_attempts: int):
        """预筛选符合约束的牌."""
        attempts = 0
        while len(self._filtered_deals) < min_deals and attempts < max_attempts:
            attempts += 1
            hands, dd_table = self.loader.sample_one()
            if self._satisfies_constraints(hands):
                self._filtered_deals.append((hands, dd_table))

        rate = len(self._filtered_deals) / max(1, attempts)
        print(f"StaymanSubgameEnv: prefetched {len(self._filtered_deals)} deals "
              f"from {attempts} attempts ({rate:.1%} acceptance)")

    def _satisfies_constraints(self, hands: np.ndarray) -> bool:
        """检查牌是否符合 Stayman 约束."""
        return (self._satisfies_opener(hands[NORTH]) and
                self._satisfies_responder(hands[SOUTH]))

    @staticmethod
    def _satisfies_opener(hand: np.ndarray) -> bool:
        """N: 15-17 HCP, 均型 (无单缺)."""
        hcp = count_hcp(hand)
        return 15 <= hcp <= 17 and is_balanced(hand)

    @staticmethod
    def _satisfies_responder(hand: np.ndarray) -> bool:
        """S: 8+ HCP, 至少有一门 4 张高花."""
        hcp = count_hcp(hand)
        if hcp < 8:
            return False
        h = count_suit_length(hand, 2)
        s = count_suit_length(hand, 3)
        return h >= 4 or s >= 4

    def generate_deal(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成一副符合约束的牌.

        如果用预生成约束数据, 直接 sample;
        否则从预筛选缓存或实时筛选获取。
        """
        if self._is_constrained_data:
            return self.loader.sample_one()

        if self._filtered_deals:
            idx = np.random.randint(len(self._filtered_deals))
            return self._filtered_deals[idx]

        # fallback: 实时采样
        for _ in range(10000):
            hands, dd_table = self.loader.sample_one()
            if self._satisfies_constraints(hands):
                return hands, dd_table

        raise RuntimeError("Cannot generate a valid Stayman deal after 10000 attempts")

    def reset(self, hands: Optional[np.ndarray] = None,
              dd_table: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        重置环境, 自动执行固定前缀.

        Returns:
            obs: 前缀执行完后, 下一个决策者 (N) 的观测
        """
        if hands is None or dd_table is None:
            hands, dd_table = self.generate_deal()

        self._current_hands = hands
        self._current_dd = dd_table

        # N is dealer, N opens 1NT
        obs = self.env.reset(hands, dealer=NORTH, vulnerability=(False, False))

        # 执行固定前缀
        for bid_str in self.FIXED_PREFIX:
            bid = string_to_bid(bid_str)
            obs, _, done, _ = self.env.step(bid)
            if done:
                break

        return obs

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        执行动作. NS 由 agent 决策, EW 自动 Pass.

        Returns:
            (obs, reward, done, info)
            reward: 叫牌结束时 = single_table_imp, 中间步 = 0
        """
        # 当前玩家的行动
        obs, _, done, info = self.env.step(action)

        if done:
            reward = self._compute_terminal_reward()
            info['imp'] = reward
            return obs, reward, done, info

        # 如果下一个决策者是 EW, 自动 Pass
        while not done and self.env.state.current_player in (EAST, WEST):
            obs, _, done, _ = self.env.step(BID_PASS)
            if done:
                reward = self._compute_terminal_reward()
                info['imp'] = reward
                return obs, reward, done, info

        return obs, 0.0, done, info

    def _compute_terminal_reward(self) -> float:
        """
        单桌评估: actual score vs DDS optimal → IMP.

        IMP 在此表示 agent 叫品质量与理论最优的差距:
        0 = 完美, 负 = 较差, 极少为正 (因为 DDS 是上界)
        """
        contract = self.env.state.final_contract
        dd_table = self._current_dd
        hands = self._current_hands

        actual_score = self._compute_score(contract, dd_table, is_ns=True)
        optimal = self._get_optimal_contract_ns(dd_table)
        optimal_score = self._compute_score(optimal, dd_table, is_ns=True)

        return float(score_to_imp(actual_score - optimal_score))

    def _compute_score(self, contract: Optional[Contract],
                       dd_table: np.ndarray, is_ns: bool) -> int:
        """计算 NS 视角得分."""
        if contract is None:
            return 0

        tricks = int(dd_table[contract.suit, contract.declarer])
        vul = False  # Stayman 子博弈默认无局
        score = calculate_score(contract, tricks, vul)

        if contract.declarer % 2 == 1:  # EW 庄家
            score = -score

        return score

    @staticmethod
    def _get_optimal_contract_ns(dd_table: np.ndarray) -> Optional[Contract]:
        """
        从 DD table 找 NS 的最优定约.

        遍历所有 NS 可打的定约 (N/S 为庄家),
        选得分最高的。
        """
        best_score = 0  # Pass-out = 0
        best_contract = None

        for declarer in [NORTH, SOUTH]:
            for suit in range(5):  # C, D, H, S, NT
                tricks = int(dd_table[suit, declarer])
                for level in range(1, 8):
                    required = 6 + level
                    if tricks >= required:
                        c = Contract(level=level, suit=suit, doubled=0,
                                     declarer=declarer)
                        score = calculate_score(c, tricks, vulnerable=False)
                        if score > best_score:
                            best_score = score
                            best_contract = c

        return best_contract

    @property
    def current_player(self) -> int:
        return self.env.state.current_player

    @property
    def history(self) -> List[int]:
        return self.env.state.history.copy()
