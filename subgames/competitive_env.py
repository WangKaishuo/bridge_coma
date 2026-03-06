"""
Competitive Subgame Environment
================================

合作-对抗子博弈, 固定前缀 1H - 1S.

特点:
- 四方都参与决策, 竞争激烈
- 1S 争叫 (非跳叫阻叫 2S), 双方牌力接近
- 需要 BC 预热
- 双桌 IMP 交叉对抗评估
- 支持 play_mixed: NS/EW 由不同 agent 控制
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

from env import (
    BridgeBiddingEnv, NUM_BIDS, NUM_PLAYERS,
    BID_PASS, BID_DOUBLE, BID_1C,
    bid_to_string, string_to_bid,
    NORTH, EAST, SOUTH, WEST,
)
from utils.scoring import Contract, calculate_score
from utils.imp import score_to_imp
from utils.dds_data import create_loader
from subgames.action_mask import (
    count_hcp, count_suit_length, suit_lengths,
)


class CompetitiveSubgameEnv:
    """
    竞叫子博弈环境.

    固定前缀: 1H (N) - 1S (E)
    约束: N 5+H 12-21 HCP,  E 5+S 8-16 HCP
    """

    FIXED_PREFIX = ["1H", "1S"]

    def __init__(self, data_path: str, max_history_len: int = 60):
        self.loader = create_loader(data_path)
        self.env = BridgeBiddingEnv(max_history_len)
        self.max_history_len = max_history_len

        self._is_constrained_data = self._check_if_constrained()
        if not self._is_constrained_data:
            self._filtered_deals = []
            self._prefetch(min_deals=500, max_attempts=50000)
        else:
            self._filtered_deals = None
            print(f"CompetitiveSubgameEnv: using pre-generated constrained data "
                  f"({len(self.loader)} samples)")

    def _check_if_constrained(self, sample_size: int = 20) -> bool:
        passed = 0
        for _ in range(sample_size):
            hands, _ = self.loader.sample_one()
            if self._satisfies_constraints(hands):
                passed += 1
        return passed >= sample_size * 0.9

    def _prefetch(self, min_deals: int, max_attempts: int):
        attempts = 0
        while len(self._filtered_deals) < min_deals and attempts < max_attempts:
            attempts += 1
            hands, dd_table = self.loader.sample_one()
            if self._satisfies_constraints(hands):
                self._filtered_deals.append((hands, dd_table))

        rate = len(self._filtered_deals) / max(1, attempts)
        print(f"CompetitiveSubgameEnv: prefetched {len(self._filtered_deals)} deals "
              f"from {attempts} attempts ({rate:.1%} acceptance)")

    def _satisfies_constraints(self, hands: np.ndarray) -> bool:
        return (self._satisfies_opener(hands[NORTH]) and
                self._satisfies_overcaller(hands[EAST]))

    @staticmethod
    def _satisfies_opener(hand: np.ndarray) -> bool:
        """N: 5+ hearts, 12-21 HCP."""
        hcp = count_hcp(hand)
        h = count_suit_length(hand, 2)
        return 12 <= hcp <= 21 and h >= 5

    @staticmethod
    def _satisfies_overcaller(hand: np.ndarray) -> bool:
        """E: 5+ spades, 8-16 HCP."""
        hcp = count_hcp(hand)
        s = count_suit_length(hand, 3)
        return 8 <= hcp <= 16 and s >= 5

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

        raise RuntimeError("Cannot generate valid competitive deal")

    def reset(self, hands: Optional[np.ndarray] = None,
              dd_table: Optional[np.ndarray] = None,
              vulnerability: Tuple[bool, bool] = (False, False)) -> Dict[str, np.ndarray]:
        """
        重置环境, 执行固定前缀 [1H, 1S].

        Returns:
            obs: 前缀后下一决策者 (S) 的观测
        """
        if hands is None or dd_table is None:
            hands, dd_table = self.generate_deal()

        self._current_hands = hands
        self._current_dd = dd_table
        self._vulnerability = vulnerability

        obs = self.env.reset(hands, dealer=NORTH, vulnerability=vulnerability)

        # 执行固定前缀
        for bid_str in self.FIXED_PREFIX:
            bid = string_to_bid(bid_str)
            obs, _, done, _ = self.env.step(bid)
            if done:
                break

        return obs

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """执行动作. 四方都参与决策."""
        obs, _, done, info = self.env.step(action)

        reward = 0.0
        if done:
            reward = self._compute_terminal_reward()
            info['imp'] = reward

        return obs, reward, done, info

    def _compute_terminal_reward(self) -> float:
        """
        双桌 IMP (NS 视角), 用于 self-play 训练.

        桌 1 已经打完 (当前 env 状态).
        桌 2: 同一副牌, NS↔EW 互换位置, 用当前 agent 策略再叫一次.
        IMP = score_to_imp(score_1 - score_2)
        """
        contract_1 = self.env.state.final_contract
        score_1 = self._compute_score_ns(contract_1, self._current_dd, self._vulnerability)

        # 桌 2: 互换位置叫牌
        score_2 = self._play_swapped_table()

        return float(score_to_imp(score_1 - score_2))

    def _play_swapped_table(self) -> int:
        """
        桌 2: NS↔EW 互换位置, 随机策略快速叫牌, 返回 NS 视角得分.
        """
        hands = self._current_hands
        dd_table = self._current_dd

        # 互换: N↔E, S↔W
        swapped_hands = np.zeros_like(hands)
        swapped_hands[0] = hands[1]  # new N = old E
        swapped_hands[1] = hands[0]  # new E = old N
        swapped_hands[2] = hands[3]  # new S = old W
        swapped_hands[3] = hands[2]  # new W = old S

        swapped_dd = np.zeros_like(dd_table)
        swapped_dd[:, 0] = dd_table[:, 1]
        swapped_dd[:, 1] = dd_table[:, 0]
        swapped_dd[:, 2] = dd_table[:, 3]
        swapped_dd[:, 3] = dd_table[:, 2]

        # 用一个临时 env 快速叫牌 (随机策略, 只需要最终分数)
        from env import BridgeBiddingEnv
        tmp_env = BridgeBiddingEnv(self.max_history_len)
        obs = tmp_env.reset(swapped_hands, dealer=NORTH, vulnerability=self._vulnerability)

        # 执行固定前缀 (互换后可能不符合约束, 但没关系 — 只需要对称评估)
        prefix_ok = True
        for bid_str in self.FIXED_PREFIX:
            bid = string_to_bid(bid_str)
            if not tmp_env._is_valid_action(bid):
                prefix_ok = False
                break
            obs, _, done, _ = tmp_env.step(bid)
            if done:
                break

        if not prefix_ok or done:
            # 前缀无法执行, 用随机策略从头叫
            obs = tmp_env.reset(swapped_hands, dealer=NORTH, vulnerability=self._vulnerability)
            done = False

        # 随机策略走完
        while not done:
            legal = obs['legal_actions']
            action = np.random.choice(np.where(legal > 0.5)[0])
            obs, _, done, _ = tmp_env.step(action)

        contract_2 = tmp_env.state.final_contract
        return self._compute_score_ns(contract_2, swapped_dd, self._vulnerability)

    def _compute_score_ns(self, contract: Optional[Contract],
                          dd_table: np.ndarray,
                          vulnerability: Tuple[bool, bool]) -> int:
        if contract is None:
            return 0
        tricks = int(dd_table[contract.suit, contract.declarer])
        vul = vulnerability[contract.declarer % 2]
        score = calculate_score(contract, tricks, vul)
        if contract.declarer % 2 == 1:
            score = -score
        return score

    @property
    def current_player(self) -> int:
        return self.env.state.current_player

    @property
    def history(self) -> List[int]:
        return self.env.state.history.copy()

    # ====================================================================
    # Mixed play (for cross-evaluation)
    # ====================================================================

    def play_mixed(
        self,
        hands: np.ndarray,
        dd_table: np.ndarray,
        ns_policy: Callable[[Dict], int],
        ew_policy: Callable[[Dict], int],
        vulnerability: Tuple[bool, bool] = (False, False),
    ) -> Tuple[Optional[Contract], int, List[int]]:
        """
        混合对抗: NS/EW 由不同策略控制.

        Args:
            ns_policy: obs -> action_int (for N, S)
            ew_policy: obs -> action_int (for E, W)

        Returns:
            (contract, score_ns, history_ints)
        """
        obs = self.env.reset(hands, dealer=NORTH, vulnerability=vulnerability)

        # 执行固定前缀
        for bid_str in self.FIXED_PREFIX:
            bid = string_to_bid(bid_str)
            obs, _, done, _ = self.env.step(bid)
            if done:
                break

        history = list(self.env.state.history)

        while not done:
            player = self.env.state.current_player
            if player % 2 == 0:  # NS
                action = ns_policy(obs)
            else:  # EW
                action = ew_policy(obs)

            obs, _, done, _ = self.env.step(action)

        contract = self.env.state.final_contract
        score = self._compute_score_ns(contract, dd_table, vulnerability)
        history = list(self.env.state.history)

        return contract, score, history


# ====================================================================
# Cross-evaluation (dual-table IMP)
# ====================================================================

@dataclass
class CrossEvalResult:
    """交叉对抗结果."""
    mean_imp: float
    std_imp: float
    win_rate: float     # agent_a 赢的比例
    p_value: float
    significant: bool   # p < 0.05
    n_deals: int
    all_imps: List[float]


def cross_evaluate(
    env: CompetitiveSubgameEnv,
    agent_a_ns_policy: Callable[[Dict], int],
    agent_a_ew_policy: Callable[[Dict], int],
    agent_b_ns_policy: Callable[[Dict], int],
    agent_b_ew_policy: Callable[[Dict], int],
    num_deals: int = 500,
) -> CrossEvalResult:
    """
    交叉对抗评估 (双桌 IMP).

    桌 1: A=NS, B=EW → score_1 (NS 视角)
    桌 2: B=NS, A=EW → score_2 (NS 视角)
    IMP = score_to_imp(score_1 - score_2)  (A 视角: 正=A赢)

    Args:
        agent_a_ns/ew_policy: agent A 分别打 NS/EW 时的策略
        agent_b_ns/ew_policy: agent B 分别打 NS/EW 时的策略
    """
    from scipy.stats import ttest_1samp

    imps = []

    for _ in range(num_deals):
        hands, dd_table = env.generate_deal()
        vul = [(False, False), (True, False), (False, True), (True, True)][np.random.randint(4)]

        # 桌 1: A=NS, B=EW
        _, score_1, _ = env.play_mixed(
            hands, dd_table,
            ns_policy=agent_a_ns_policy,
            ew_policy=agent_b_ew_policy,
            vulnerability=vul,
        )

        # 桌 2: B=NS, A=EW
        _, score_2, _ = env.play_mixed(
            hands, dd_table,
            ns_policy=agent_b_ns_policy,
            ew_policy=agent_a_ew_policy,
            vulnerability=vul,
        )

        imp = score_to_imp(score_1 - score_2)
        imps.append(float(imp))

    imps_arr = np.array(imps)
    t_stat, p_val = ttest_1samp(imps_arr, 0)

    return CrossEvalResult(
        mean_imp=float(imps_arr.mean()),
        std_imp=float(imps_arr.std()),
        win_rate=float((imps_arr > 0).mean()),
        p_value=float(p_val),
        significant=bool(p_val < 0.05),
        n_deals=num_deals,
        all_imps=imps,
    )


def make_agent_policy(agent, deterministic: bool = True) -> Callable[[Dict], int]:
    """
    从 agent 创建 policy function (obs -> action_int).

    agent 需要有 model.actor, device 属性.
    """
    import torch

    def policy(obs: Dict[str, np.ndarray]) -> int:
        obs_t = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(agent.device)
                 for k, v in obs.items()}
        with torch.no_grad():
            action, _, _ = agent.model.actor.get_action(obs_t, deterministic=deterministic)
        return action.item()

    return policy
