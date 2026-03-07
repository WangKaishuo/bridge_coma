"""
Stayman Subgame Environment
============================

纯合作子博弈, 固定前缀 1NT - Pass - 2C - Pass.

支持两种模式:
- north_rule=True:  N 用规则策略 (Stage 1 训练 S 用)
- north_rule=False: N 由 agent 决策 (Stage 2 联合微调用)
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


# ============================================================================
# N's rule strategy for Stayman
# ============================================================================

def north_stayman_rule(hands: np.ndarray, history: list) -> int:
    """
    N 的 Stayman 规则策略.

    1NT-Pass-2C-Pass 后:
      Round 1 (回应 2C): 有 4+H → 2H, 有 4+S → 2S, 都没有 → 2D
      Round 3 (回应邀请): 17 HCP → 接受 (3NT/4H/4S), 15-16 → Pass

    Args:
        hands: (4, 52) — 需要 hands[NORTH]
        history: 当前叫牌历史 (list of bid ints)
    Returns:
        bid index
    """
    n_hand = hands[NORTH]
    prefix_len = 4  # 1NT Pass 2C Pass
    rounds_after = len(history) - prefix_len

    if rounds_after == 0:
        # Round 1: N responds to 2C
        h = count_suit_length(n_hand, 2)  # hearts
        s = count_suit_length(n_hand, 3)  # spades
        if h >= 4:
            return string_to_bid("2H")
        elif s >= 4:
            return string_to_bid("2S")
        else:
            return string_to_bid("2D")

    elif rounds_after == 4:
        # Round 3: N decides on S's invitation (3H/3S/2NT)
        hcp = count_hcp(n_hand)
        s_bid = history[-2]  # S 的上一个叫品 (跳过 E 的 pass)

        if hcp >= 17:
            # 接受邀请
            s_bid_str = bid_to_string(s_bid)
            if s_bid_str in ("3H",):
                return string_to_bid("4H")
            elif s_bid_str in ("3S",):
                return string_to_bid("4S")
            elif s_bid_str in ("2NT",):
                return string_to_bid("3NT")
            else:
                return string_to_bid("3NT")
        else:
            # 拒绝邀请
            return BID_PASS

    # 其他轮次: pass
    return BID_PASS


class StaymanSubgameEnv:
    """
    Stayman 子博弈环境.

    Args:
        data_path: DDS 数据路径
        north_rule: True = N 用规则策略, False = N 由 agent 决策
    """

    FIXED_PREFIX = ["1NT", "Pass", "2C", "Pass"]

    def __init__(self, data_path: str, north_rule: bool = True,
                 max_history_len: int = 60):
        self.loader = create_loader(data_path)
        self.env = BridgeBiddingEnv(max_history_len)
        self.max_history_len = max_history_len
        self.north_rule = north_rule

        self._is_constrained_data = self._check_if_constrained()
        if not self._is_constrained_data:
            self._filtered_deals = []
            self._prefetch(min_deals=500, max_attempts=50000)
        else:
            self._filtered_deals = None
            print(f"StaymanSubgameEnv: using pre-generated constrained data "
                  f"({len(self.loader)} samples), north_rule={north_rule}")

    def _check_if_constrained(self, sample_size: int = 20) -> bool:
        passed = 0
        for _ in range(sample_size):
            hands, _ = self.loader.sample_one()
            if self._satisfies_constraints(hands):
                passed += 1
        return passed >= sample_size * 0.9

    def _satisfies_constraints(self, hands: np.ndarray) -> bool:
        return (self._satisfies_opener(hands[NORTH]) and
                self._satisfies_responder(hands[SOUTH]))

    @staticmethod
    def _satisfies_opener(hand: np.ndarray) -> bool:
        hcp = count_hcp(hand)
        return 15 <= hcp <= 17 and is_balanced(hand)

    @staticmethod
    def _satisfies_responder(hand: np.ndarray) -> bool:
        hcp = count_hcp(hand)
        if hcp < 8:
            return False
        h = count_suit_length(hand, 2)
        s = count_suit_length(hand, 3)
        return h >= 4 or s >= 4

    def _prefetch(self, min_deals: int, max_attempts: int):
        attempts = 0
        while len(self._filtered_deals) < min_deals and attempts < max_attempts:
            attempts += 1
            hands, dd_table = self.loader.sample_one()
            if self._satisfies_constraints(hands):
                self._filtered_deals.append((hands, dd_table))
        rate = len(self._filtered_deals) / max(1, attempts)
        print(f"StaymanSubgameEnv: prefetched {len(self._filtered_deals)} deals "
              f"({rate:.1%} acceptance)")

    def generate_deal(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._is_constrained_data:
            return self.loader.sample_one()
        if self._filtered_deals:
            idx = np.random.randint(len(self._filtered_deals))
            return self._filtered_deals[idx]
        for _ in range(10000):
            hands, dd_table = self.loader.sample_one()
            if self._satisfies_constraints(hands):
                return hands, dd_table
        raise RuntimeError("Cannot generate a valid Stayman deal")

    def reset(self, hands: Optional[np.ndarray] = None,
              dd_table: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """重置环境, 执行固定前缀, 返回下一个 agent 决策者的 obs."""
        if hands is None or dd_table is None:
            hands, dd_table = self.generate_deal()

        self._current_hands = hands
        self._current_dd = dd_table

        obs = self.env.reset(hands, dealer=NORTH, vulnerability=(False, False))

        # 执行固定前缀
        for bid_str in self.FIXED_PREFIX:
            bid = string_to_bid(bid_str)
            obs, _, done, _ = self.env.step(bid)
            if done:
                break

        # 自动执行 N 的规则叫品 + EW pass, 直到轮到 agent 决策的 player
        obs = self._auto_play_non_agent(obs, done=False)
        return obs

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        执行 agent 的动作, 然后自动执行规则 player + EW pass.

        Returns:
            (obs, reward, done, info)
        """
        obs, _, done, info = self.env.step(action)

        if done:
            reward = self._compute_terminal_reward()
            info['imp'] = self._compute_eval_imp()
            info['scaled_score'] = reward
            return obs, reward, done, info

        # 自动执行 N 规则 + EW pass, 直到轮到 agent
        obs = self._auto_play_non_agent(obs, done=False)
        done_after = self.env.state.history and self._check_env_done()

        if done_after:
            reward = self._compute_terminal_reward()
            info['imp'] = self._compute_eval_imp()
            info['scaled_score'] = reward
            # 需要重新获取最终 obs
            obs = self.env._get_observation()
            return obs, reward, True, info

        # 中间步
        if not done_after:
            obs['legal_actions'] = self._get_stayman_mask()

        return obs, 0.0, False, info

    def _auto_play_non_agent(self, obs: dict, done: bool) -> dict:
        """
        自动执行非 agent 控制的玩家:
        - EW 永远 auto-pass
        - north_rule=True 时, N 也用规则策略

        返回下一个需要 agent 决策的 player 的 obs.
        """
        while not done:
            player = self.env.state.current_player

            if player in (EAST, WEST):
                obs, _, done, _ = self.env.step(BID_PASS)
            elif player == NORTH and self.north_rule:
                bid = north_stayman_rule(self._current_hands, self.env.state.history)
                # 确保合法
                if not self.env._is_valid_action(bid):
                    bid = BID_PASS
                obs, _, done, _ = self.env.step(bid)
            else:
                # 轮到 agent 决策的 player, 停下来
                break

        if not done:
            obs['legal_actions'] = self._get_stayman_mask()

        return obs

    def _check_env_done(self) -> bool:
        """检查 env 是否结束."""
        return self.env._check_done()

    def _get_stayman_mask(self) -> np.ndarray:
        """Stayman 专用 action mask."""
        mask = np.zeros(NUM_BIDS, dtype=np.float32)
        history = self.env.state.history
        prefix_len = len(self.FIXED_PREFIX)
        rounds_after = len(history) - prefix_len

        if rounds_after == 0:
            # N responds to 2C (only if north_rule=False)
            mask[string_to_bid("2D")] = 1.0
            mask[string_to_bid("2H")] = 1.0
            mask[string_to_bid("2S")] = 1.0
        elif rounds_after == 2:
            # S rebids
            mask[BID_PASS] = 1.0
            mask[string_to_bid("2NT")] = 1.0
            mask[string_to_bid("3NT")] = 1.0
            mask[string_to_bid("3H")] = 1.0
            mask[string_to_bid("3S")] = 1.0
            mask[string_to_bid("4H")] = 1.0
            mask[string_to_bid("4S")] = 1.0
        elif rounds_after == 4:
            # N decides on invitation
            mask[BID_PASS] = 1.0
            mask[string_to_bid("3NT")] = 1.0
            mask[string_to_bid("4H")] = 1.0
            mask[string_to_bid("4S")] = 1.0
        else:
            mask[BID_PASS] = 1.0

        env_legal = self.env._get_legal_actions()
        mask = mask * env_legal

        if mask.sum() < 0.5:
            mask[BID_PASS] = 1.0

        return mask

    # Shifted IMP reward 参数
    # -13 IMP 对应有局小满贯奖分 (750 分差 → 13 IMP), 覆盖 >95% Stayman 牌局
    IMP_FLOOR = -13.0
    IMP_EPSILON = 0.01  # 极端负值仍有微弱梯度, 避免零梯度死区

    def _compute_terminal_reward(self) -> float:
        """
        训练 reward: Shifted IMP → [ε, 1.0]

        r = clamp((IMP_diff - IMP_FLOOR) / (-IMP_FLOOR), ε, 1.0)

        数学含义: IMP regret 的线性补集.
          IMP_diff =  0 → r = 1.0  (完美匹配 DDS 最优)
          IMP_diff = -6 → r ≈ 0.54 (漏局, 中等惩罚)
          IMP_diff = -13 → r = ε   (极大失误, 接近零但保留梯度)
          IMP_diff < -13 → r = ε   (超极端, soft clamp)
        """
        imp_diff = self._compute_imp_diff()
        r = (imp_diff - self.IMP_FLOOR) / (-self.IMP_FLOOR)
        r = max(r, self.IMP_EPSILON)
        r = min(r, 1.0)
        return float(r)

    def _compute_imp_diff(self) -> float:
        """计算 actual vs DDS optimal 的 IMP 差值 (≤ 0)."""
        contract = self.env.state.final_contract
        dd_table = self._current_dd
        actual_score = self._compute_score(contract, dd_table, is_ns=True)
        optimal = self._get_optimal_contract_ns(dd_table)
        optimal_score = self._compute_score(optimal, dd_table, is_ns=True)
        return float(score_to_imp(actual_score - optimal_score))

    def _compute_eval_imp(self) -> float:
        """评估用 IMP: actual vs DDS optimal."""
        return self._compute_imp_diff()

    def _compute_score(self, contract: Optional[Contract],
                       dd_table: np.ndarray, is_ns: bool) -> int:
        if contract is None:
            return 0
        tricks = int(dd_table[contract.suit, contract.declarer])
        vul = False
        score = calculate_score(contract, tricks, vul)
        if contract.declarer % 2 == 1:
            score = -score
        return score

    @staticmethod
    def _get_optimal_contract_ns(dd_table: np.ndarray) -> Optional[Contract]:
        best_score = 0
        best_contract = None
        for declarer in [NORTH, SOUTH]:
            for suit in range(5):
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
