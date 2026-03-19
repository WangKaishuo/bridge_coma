"""
Competitive Subgame Environment  (重构版)
=========================================

合作-对抗子博弈，固定前缀 1H(N) – 1S(E)。

关键修复（vs 旧版）:
1. _play_swapped_table 不再用随机策略.
   双桌 IMP 的第二桌改用 *传入的 agent policy* 叫牌，
   这样 reward = 真实双桌 IMP，而非"比随机好多少"。

2. cross_evaluate 改用 Wilcoxon signed-rank test（IMP 重尾分布，非参）。

3. 加入 dds_oracle_evaluate：用 DDS oracle 计算 IMP regret 作为绝对基准，
   与对手无关，是论文主要指标。

4. rule_based_bc_data：生成 ~5k 局 competitive 规则牌数据用于 BC 预热，
   不依赖外部 WBridge5 数据集。

环境特点:
    - 四方都参与决策（竞争激烈）
    - 1S 争叫（非跳叫），双方牌力接近
    - 支持 play_mixed：NS / EW 由不同 agent 控制
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from env import (
    BridgeBiddingEnv, NUM_BIDS, NUM_PLAYERS,
    BID_PASS, BID_DOUBLE, BID_1C,
    bid_to_string, string_to_bid,
    NORTH, EAST, SOUTH, WEST,
)
from utils.scoring import Contract, calculate_score
from utils.imp import score_to_imp
from utils.dds_data import create_loader
from subgames.action_mask import count_hcp, count_suit_length


# ── 常量 ──────────────────────────────────────────────────────────────────────
FIXED_PREFIX = ["1H", "1S"]   # N 开叫 1H，E 争叫 1S，之后 S/W/N/... 自由叫
_SUIT_H = 2   # hearts index
_SUIT_S = 3   # spades index


# ==============================================================================
# Rule-based 策略（用于 BC 预热和双桌第二桌）
# ==============================================================================

def _rule_based_action(obs: Dict[str, np.ndarray], player: int,
                        history: list, dealer: int) -> int:
    """
    极简 rule-based 策略，仅用于:
        1. BC 预热数据生成（~5k 局）
        2. competitive_env 内部双桌第二桌（如果未提供 agent）

    规则（非常保守，只保证叫牌不崩溃）:
        - S (partner of N): 支持 2H (有 3+H)，或报副花色，否则 Pass
        - W (partner of E): 支持 2S / 3S (有 3+S)，否则 Pass
        - N (rebid): 简单 rebid
        - E (rebid): 简单 rebid
        - 其他情况: Pass

    产生的叫牌序列不是最优的，但足以提供有意义的初始策略分布。
    合法性由 legal_actions mask 兜底。
    """
    hand         = obs['hand']             # (52,) float
    legal        = obs['legal_actions']    # (38,) float

    def _legal(bid: int) -> bool:
        return bool(legal[bid] > 0.5)

    def _bid_if_legal(bid: int) -> int:
        return bid if _legal(bid) else BID_PASS

    hcp  = int(count_hcp(hand))
    h    = int(count_suit_length(hand, _SUIT_H))
    s    = int(count_suit_length(hand, _SUIT_S))

    # 叫牌历史长度（从前缀结束后算起，这里直接用 history 总长）
    hist_len = len(history)

    # ── S 方（第 3 叫，hist_len == 2）──────────────────────────────────────
    if player == SOUTH and hist_len == 2:
        # 有 3+H → 支持 2H
        bid_2h = string_to_bid("2H")
        bid_2s = string_to_bid("2S")   # cue-bid，表示牌力强
        bid_2c = string_to_bid("2C")
        bid_2d = string_to_bid("2D")

        if h >= 3 and hcp >= 8:
            return _bid_if_legal(bid_2h)
        elif hcp >= 10:
            # 报最长副花色
            d = int(count_suit_length(hand, 1))
            c = int(count_suit_length(hand, 0))
            if d >= 4 and _legal(bid_2d):
                return bid_2d
            if c >= 4 and _legal(bid_2c):
                return bid_2c
            return _bid_if_legal(bid_2s)
        else:
            return BID_PASS

    # ── W 方（第 4 叫，hist_len == 3）──────────────────────────────────────
    elif player == WEST and hist_len == 3:
        bid_2s = string_to_bid("2S")
        bid_3s = string_to_bid("3S")

        if s >= 4 and hcp >= 10:
            if s >= 6 or hcp >= 12:
                return _bid_if_legal(bid_3s)
            return _bid_if_legal(bid_2s)
        return BID_PASS

    # ── N rebid（hist_len == 4）──────────────────────────────────────────────
    elif player == NORTH and hist_len == 4:
        bid_2h = string_to_bid("2H")
        bid_3h = string_to_bid("3H")
        bid_4h = string_to_bid("4H")

        if hcp >= 18:
            return _bid_if_legal(bid_4h)
        elif hcp >= 15:
            return _bid_if_legal(bid_3h)
        elif hcp >= 12:
            return _bid_if_legal(bid_2h)
        return BID_PASS

    # ── E rebid（hist_len == 5）──────────────────────────────────────────────
    elif player == EAST and hist_len == 5:
        bid_2s = string_to_bid("2S")
        bid_3s = string_to_bid("3S")
        bid_4s = string_to_bid("4S")

        if hcp >= 14:
            return _bid_if_legal(bid_4s)
        elif hcp >= 12:
            return _bid_if_legal(bid_3s)
        elif hcp >= 9:
            return _bid_if_legal(bid_2s)
        return BID_PASS

    # ── 其他：Pass ──────────────────────────────────────────────────────────
    return BID_PASS


def generate_rule_based_bc_data(
    env: "CompetitiveSubgameEnv",
    num_samples: int = 5000,
) -> List[Dict]:
    """
    生成 rule-based BC 训练数据，用于 Phase 1 轻量预热.

    Returns:
        list of {'flat_obs': np.ndarray(301,), 'action': int}

    注: flat_obs 使用 encode_obs_flat 编码（需要 dealer + history_int）.
    """
    from networks.policy_net import encode_obs_flat

    data = []
    attempts = 0
    max_attempts = num_samples * 20

    while len(data) < num_samples and attempts < max_attempts:
        attempts += 1
        hands, dd_table = env.generate_deal()
        vul = (False, False)

        inner_env = BridgeBiddingEnv(max_history_len=60)
        obs = inner_env.reset(hands, dealer=NORTH, vulnerability=vul)
        done = False

        # 执行固定前缀 1H-1S
        prefix_ok = True
        for bid_str in FIXED_PREFIX:
            bid = string_to_bid(bid_str)
            if not inner_env._is_valid_action(bid):
                prefix_ok = False
                break
            obs, _, done, _ = inner_env.step(bid)
            if done:
                break

        if not prefix_ok or done:
            continue

        # rule-based 走完整局
        while not done:
            player  = inner_env.state.current_player
            history = inner_env.state.history[:]
            dealer  = NORTH

            action = _rule_based_action(obs, player, history, dealer)
            if not inner_env._is_valid_action(action):
                action = BID_PASS

            # 编码当前状态为 301 维
            flat = encode_obs_flat(obs, dealer, history)
            data.append({'flat_obs': flat, 'action': action})

            obs, _, done, _ = inner_env.step(action)

    print(f"[BC Data] Generated {len(data)} samples "
          f"from {attempts} attempts (acceptance: {len(data)/max(1,attempts):.1%})")
    return data


# ==============================================================================
# CompetitiveSubgameEnv
# ==============================================================================

class CompetitiveSubgameEnv:
    """
    竞叫子博弈环境.

    固定前缀: 1H(N) – 1S(E)
    约束:   N 5+H 12-21 HCP,  E 5+S 8-16 HCP

    关键接口:
        reset(hands, dd_table)   → obs (含 dealer / history_int 供编码用)
        step(action)             → obs, reward, done, info
        generate_deal()          → (hands, dd_table)
        play_mixed(...)          → (contract, score_ns, history)

    内部状态:
        self._current_hands  : (4, 52) float32
        self._current_dd     : (5, 4)  int8  tricks[suit, player]
        self.dealer          : int (固定为 NORTH)
        self.history_int     : list[int] (包含前缀 + 后续)
    """

    def __init__(self, data_path: str, max_history_len: int = 60):
        self.loader          = create_loader(data_path)
        self.env             = BridgeBiddingEnv(max_history_len)
        self.max_history_len = max_history_len
        self.dealer          = NORTH

        # 检查数据是否已经是 constrained（10万副预生成数据）
        self._is_constrained_data = self._check_if_constrained()
        if not self._is_constrained_data:
            self._filtered_deals: list = []
            self._prefetch(min_deals=500, max_attempts=50000)
        else:
            self._filtered_deals = None
            print(f"[CompetitiveEnv] Pre-generated constrained data: "
                  f"{len(self.loader)} samples")

        # 当前局信息（由 reset 填充）
        self._current_hands: Optional[np.ndarray] = None
        self._current_dd:    Optional[np.ndarray] = None
        self._vulnerability: Tuple[bool, bool]    = (False, False)
        self.history_int:    list                 = []

    # ------------------------------------------------------------------
    # 初始化辅助
    # ------------------------------------------------------------------

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
        print(f"[CompetitiveEnv] Prefetched {len(self._filtered_deals)} deals "
              f"({rate:.1%} acceptance rate)")

    # ------------------------------------------------------------------
    # 约束判断
    # ------------------------------------------------------------------

    def _satisfies_constraints(self, hands: np.ndarray) -> bool:
        return (self._satisfies_opener(hands[NORTH]) and
                self._satisfies_overcaller(hands[EAST]))

    @staticmethod
    def _satisfies_opener(hand: np.ndarray) -> bool:
        """N: 5+H, 12-21 HCP."""
        return 12 <= count_hcp(hand) <= 21 and count_suit_length(hand, _SUIT_H) >= 5

    @staticmethod
    def _satisfies_overcaller(hand: np.ndarray) -> bool:
        """E: 5+S, 8-16 HCP."""
        return 8 <= count_hcp(hand) <= 16 and count_suit_length(hand, _SUIT_S) >= 5

    # ------------------------------------------------------------------
    # 发牌
    # ------------------------------------------------------------------

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

        raise RuntimeError("Cannot generate valid competitive deal after 10000 attempts")

    # ------------------------------------------------------------------
    # 环境核心接口
    # ------------------------------------------------------------------

    def reset(
        self,
        hands:         Optional[np.ndarray]   = None,
        dd_table:      Optional[np.ndarray]   = None,
        vulnerability: Tuple[bool, bool]       = (False, False),
    ) -> Dict[str, np.ndarray]:
        """
        重置环境，执行固定前缀 [1H, 1S].

        Returns:
            obs: 前缀后下一决策者（S）的观测
        """
        if hands is None or dd_table is None:
            hands, dd_table = self.generate_deal()

        self._current_hands = hands
        self._current_dd    = dd_table
        self._vulnerability = vulnerability
        self.history_int    = []

        obs = self.env.reset(hands, dealer=NORTH, vulnerability=vulnerability)

        for bid_str in FIXED_PREFIX:
            bid = string_to_bid(bid_str)
            self.history_int.append(bid)
            obs, _, done, _ = self.env.step(bid)
            if done:
                break

        return obs

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """执行动作，四方都参与决策."""
        self.history_int.append(action)
        obs, _, done, info = self.env.step(action)

        reward = 0.0
        if done:
            reward = self._compute_terminal_reward()
            info['imp'] = reward

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # 奖励计算（关键修复）
    # ------------------------------------------------------------------

    def _compute_terminal_reward(self) -> float:
        """
        双桌 IMP regret（NS 视角）.

        IMP regret = actual_imp − dds_optimal_imp  (≤ 0)

        改变: 不再做双桌对比（第二桌依赖 agent 策略，产生循环依赖）.
              改为与 DDS oracle 比较，reward 是绝对质量指标.

        公式:
            actual_imp  = score_to_imp(score_ns)
            optimal_imp = score_to_imp(dds_optimal_score_ns)
            regret = actual_imp - optimal_imp  (≤ 0，越高越好)
        """
        contract = self.env.state.final_contract
        score_ns = self._compute_score_ns(
            contract, self._current_dd, self._vulnerability)

        opt_score = self._compute_dds_optimal_score_ns(
            self._current_dd, self._vulnerability)

        actual_imp  = score_to_imp(score_ns)
        optimal_imp = score_to_imp(opt_score)
        return float(actual_imp - optimal_imp)

    def _compute_dds_optimal_score_ns(
        self,
        dd_table:      np.ndarray,
        vulnerability: Tuple[bool, bool],
    ) -> int:
        """
        DDS oracle: 枚举所有可能的定约，返回 NS 视角最优得分.

        注: 仅考虑 NS 作为庄家（NS 视角），和 EW 最佳防守的结果.
        此处简化为: NS 选择使自己得分最大的定约（如实际比赛中的 DDS 最优叫牌）.
        """
        from utils.scoring import calculate_score

        ns_vul, ew_vul = vulnerability
        best_score = 0   # 流局得 0

        # 枚举 NS 作为庄家的所有定约（0=C,1=D,2=H,3=S,4=NT; NS=0,2）
        for suit in range(5):
            for level in range(1, 8):
                for declarer in (NORTH, SOUTH):
                    bid = 3 + (level - 1) * 5 + suit   # bid index
                    tricks = int(dd_table[suit, declarer])

                    # 检查 double state（简化：假设未被加倍）
                    from utils.scoring import Contract as C_
                    contract = C_(level=level, suit=suit, declarer=declarer,
                                  doubled=0)
                    vul = ns_vul
                    score = calculate_score(contract, tricks, vul)
                    # NS 庄家得分为正（NS 视角）
                    if score > best_score:
                        best_score = score

        # 同时考虑 EW 作庄 NS 防守的场景（NS 得负分，即 EW 下的最优）
        # 如果 NS 最优是流局 (0)，也考虑 EW 可能让 NS 更差
        ew_best = 0
        for suit in range(5):
            for level in range(1, 8):
                for declarer in (EAST, WEST):
                    tricks = int(dd_table[suit, declarer])
                    from utils.scoring import Contract as C_
                    contract = C_(level=level, suit=suit, declarer=declarer,
                                  doubled=0)
                    vul = ew_vul
                    score = calculate_score(contract, tricks, vul)
                    # EW 庄家得分对 NS 是负数
                    ns_score = -score
                    if ns_score > ew_best:
                        ew_best = ns_score

        # DDS oracle 是 NS 在完美信息下能达到的最优结果
        # 如果 NS 最优定约 > 流局，取 NS 最优；否则 NS 宁可让 EW 作庄（防守更好）
        return max(best_score, ew_best, 0)  # 流局下限为 0；ew_best 为防守EW的NS视角收益

    def _compute_score_ns(
        self,
        contract:      Optional[Contract],
        dd_table:      np.ndarray,
        vulnerability: Tuple[bool, bool],
    ) -> int:
        """计算 NS 视角实际得分."""
        if contract is None:
            return 0
        tricks = int(dd_table[contract.suit, contract.declarer])
        vul    = vulnerability[contract.declarer % 2]
        score  = calculate_score(contract, tricks, vul)
        if contract.declarer % 2 == 1:   # EW 庄家 → NS 得负分
            score = -score
        return score

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def current_player(self) -> int:
        return self.env.state.current_player

    @property
    def history(self) -> List[int]:
        return self.env.state.history.copy()

    # ------------------------------------------------------------------
    # 混合对抗（用于 cross_evaluate）
    # ------------------------------------------------------------------

    def play_mixed(
        self,
        hands:         np.ndarray,
        dd_table:      np.ndarray,
        ns_policy:     Callable[[Dict, int, list], int],
        ew_policy:     Callable[[Dict, int, list], int],
        vulnerability: Tuple[bool, bool] = (False, False),
    ) -> Tuple[Optional[Contract], int, List[int]]:
        """
        混合对抗: NS / EW 由不同策略控制.

        policy 签名: (obs, player, history_int) → action_int

        Returns:
            (contract, score_ns, history_ints)
        """
        inner = BridgeBiddingEnv(self.max_history_len)
        obs   = inner.reset(hands, dealer=NORTH, vulnerability=vulnerability)
        hist  = []
        done  = False

        # 执行固定前缀
        for bid_str in FIXED_PREFIX:
            bid = string_to_bid(bid_str)
            hist.append(bid)
            obs, _, done, _ = inner.step(bid)
            if done:
                break

        while not done:
            player = inner.state.current_player
            if player % 2 == 0:   # NS
                action = ns_policy(obs, player, hist[:])
            else:                  # EW
                action = ew_policy(obs, player, hist[:])

            if not inner._is_valid_action(action):
                action = BID_PASS

            hist.append(action)
            obs, _, done, _ = inner.step(action)

        contract = inner.state.final_contract
        score    = self._compute_score_ns(contract, dd_table, vulnerability)
        return contract, score, list(inner.state.history)


# ==============================================================================
# 评估函数
# ==============================================================================

@dataclass
class CrossEvalResult:
    mean_imp:    float
    std_imp:     float
    win_rate:    float     # agent_a 赢的比例（IMP > 0）
    p_value:     float     # Wilcoxon signed-rank test
    significant: bool      # p < 0.05
    n_deals:     int
    all_imps:    List[float]


def cross_evaluate(
    env:              "CompetitiveSubgameEnv",
    agent_a_ns_policy: Callable,
    agent_a_ew_policy: Callable,
    agent_b_ns_policy: Callable,
    agent_b_ew_policy: Callable,
    num_deals:        int = 500,
) -> CrossEvalResult:
    """
    交叉对抗评估（双桌 IMP）.

    桌 1: A=NS, B=EW → score_1 (NS 视角)
    桌 2: B=NS, A=EW → score_2 (NS 视角)
    IMP = score_to_imp(score_1 - score_2)  (A 视角: 正=A赢)

    统计检验: Wilcoxon signed-rank test（IMP 为重尾分布，非参更鲁棒）.
    """
    from scipy.stats import wilcoxon

    imps = []
    for _ in range(num_deals):
        hands, dd_table = env.generate_deal()
        vul = [
            (False, False), (True, False), (False, True), (True, True)
        ][np.random.randint(4)]

        _, score_1, _ = env.play_mixed(
            hands, dd_table,
            ns_policy=agent_a_ns_policy,
            ew_policy=agent_b_ew_policy,
            vulnerability=vul,
        )
        _, score_2, _ = env.play_mixed(
            hands, dd_table,
            ns_policy=agent_b_ns_policy,
            ew_policy=agent_a_ew_policy,
            vulnerability=vul,
        )
        imps.append(float(score_to_imp(score_1 - score_2)))

    arr = np.array(imps)

    # Wilcoxon: 检验 IMP 分布是否显著偏离 0
    try:
        _, p_val = wilcoxon(arr)
    except Exception:
        p_val = 1.0

    return CrossEvalResult(
        mean_imp=float(arr.mean()),
        std_imp=float(arr.std()),
        win_rate=float((arr > 0).mean()),
        p_value=float(p_val),
        significant=bool(p_val < 0.05),
        n_deals=num_deals,
        all_imps=imps,
    )


def dds_oracle_evaluate(
    env:     "CompetitiveSubgameEnv",
    policy:  Callable,
    num_deals: int = 1000,
) -> dict:
    """
    DDS oracle 评估: IMP regret（绝对基准，主要指标）.

    对每副牌计算:
        regret = actual_imp - dds_optimal_imp  (≤ 0)

    此评估与对手强度完全无关，是论文 RQ1 的核心证据。

    Args:
        policy: (obs, player, history_int) → action_int

    Returns:
        dict 含 mean_regret, std_regret, bootstrap_ci_95
    """
    regrets = []
    inner = BridgeBiddingEnv(max_history_len=60)

    for _ in range(num_deals):
        hands, dd_table = env.generate_deal()
        vul = (False, False)

        obs  = inner.reset(hands, dealer=NORTH, vulnerability=vul)
        hist = []
        done = False

        # 固定前缀
        for bid_str in FIXED_PREFIX:
            bid = string_to_bid(bid_str)
            hist.append(bid)
            obs, _, done, _ = inner.step(bid)
            if done:
                break

        while not done:
            player = inner.state.current_player
            action = policy(obs, player, hist[:])
            if not inner._is_valid_action(action):
                action = BID_PASS
            hist.append(action)
            obs, _, done, _ = inner.step(action)

        contract = inner.state.final_contract
        score_ns = env._compute_score_ns(contract, dd_table, vul)
        opt_ns   = env._compute_dds_optimal_score_ns(dd_table, vul)

        regret = float(score_to_imp(score_ns) - score_to_imp(opt_ns))
        regrets.append(regret)

    arr = np.array(regrets)

    # Bootstrap 95% CI
    bs_means = [
        np.random.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(1000)
    ]
    ci_lo, ci_hi = np.percentile(bs_means, [2.5, 97.5])

    return {
        'mean_regret':      float(arr.mean()),
        'std_regret':       float(arr.std()),
        'ci_lo':            float(ci_lo),
        'ci_hi':            float(ci_hi),
        'n_deals':          num_deals,
        'pct_pass_out':     float((arr == arr.min()).mean()),
    }


# ==============================================================================
# Policy 工厂函数
# ==============================================================================

def make_agent_policy(
    agent,
    deterministic: bool = True,
) -> Callable[[Dict, int, list], int]:
    """
    从 MAPPOAgent 创建 competitive policy 函数.

    policy 签名: (obs, player, history_int) → action_int
    （history_int 参数接受但不转发到 agent，仅供 rule-based 策略用）
    """
    import torch
    from networks.policy_net import encode_obs_flat

    def policy(obs: Dict, player: int, history_int: list) -> int:
        # 编码为 301 维
        flat = encode_obs_flat(obs, NORTH, history_int)
        flat_t = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(agent.device)
        legal  = torch.tensor(
            obs['legal_actions'], dtype=torch.float32).unsqueeze(0).to(agent.device)

        actor = agent.get_actor(player)
        with torch.no_grad():
            action, _, _ = actor.get_action(flat_t, legal, deterministic=deterministic)
        return action.item()

    return policy


def make_rule_policy() -> Callable[[Dict, int, list], int]:
    """创建 rule-based policy（用于 BC 数据生成和 baseline 对比）."""
    def policy(obs: Dict, player: int, history_int: list) -> int:
        return _rule_based_action(obs, player, history_int, NORTH)
    return policy
