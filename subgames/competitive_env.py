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

    dealer 由 env.generate_deal() 动态决定（dealer rotation 已支持）.
    """
    from networks.policy_net import encode_obs_flat

    data = []
    attempts = 0
    max_attempts = num_samples * 20

    while len(data) < num_samples and attempts < max_attempts:
        attempts += 1
        hands, dd_table = env.generate_deal()
        dealer = env._sampled_dealer
        vul = (False, False)

        inner_env = BridgeBiddingEnv(max_history_len=60)
        obs = inner_env.reset(hands, dealer=dealer, vulnerability=vul)
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
        self._sampled_dealer: int = NORTH  # set by generate_deal(), used by reset()

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
            # Use NORTH as reference dealer for the check (pre-generated data is N-opener)
            if self._satisfies_constraints(hands, dealer=NORTH):
                passed += 1
        return passed >= sample_size * 0.9

    def _prefetch(self, min_deals: int, max_attempts: int):
        attempts = 0
        rng = np.random.default_rng(42)
        while len(self._filtered_deals) < min_deals and attempts < max_attempts:
            attempts += 1
            hands, dd_table = self.loader.sample_one()
            # Check all 4 rotations — store as (hands, dd_table, dealer)
            for dealer in range(NUM_PLAYERS):
                if self._satisfies_constraints(hands, dealer=dealer):
                    self._filtered_deals.append((hands, dd_table, dealer))
                    break  # one rotation per deal to avoid bias
        rate = len(self._filtered_deals) / max(1, attempts)
        print(f"[CompetitiveEnv] Prefetched {len(self._filtered_deals)} deals "
              f"({rate:.1%} acceptance rate)")

    # ------------------------------------------------------------------
    # 约束判断
    # ------------------------------------------------------------------

    def _satisfies_constraints(self, hands: np.ndarray,
                               dealer: int = NORTH) -> bool:
        opener_seat     = dealer
        overcaller_seat = (dealer + 1) % NUM_PLAYERS
        return (self._satisfies_opener(hands[opener_seat]) and
                self._satisfies_overcaller(hands[overcaller_seat]))

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
        """
        返回 (hands, dd_table)，同时设置 self._sampled_dealer 供 reset() 使用.

        对于预生成的约束数据（dealer=NORTH 固定），从 4 个旋转中随机选一个，
        并把 hands 按 dealer 轮换，使约束始终对应 (dealer, dealer+1)。
        对于 prefetch 数据，dealer 已经在 _prefetch 时确定。
        """
        if self._is_constrained_data:
            # Pre-generated data: N=opener, E=overcaller. Rotate to random dealer.
            hands, dd_table = self.loader.sample_one()
            rotation = np.random.randint(NUM_PLAYERS)
            hands    = np.roll(hands, -rotation, axis=0)     # shift player seats
            dd_table = np.roll(dd_table, -rotation, axis=1)  # shift declarer axis
            self._sampled_dealer = rotation
            return hands, dd_table

        if self._filtered_deals:
            idx = np.random.randint(len(self._filtered_deals))
            entry = self._filtered_deals[idx]
            if len(entry) == 3:
                hands, dd_table, dealer = entry
            else:
                hands, dd_table = entry; dealer = NORTH
            self._sampled_dealer = dealer
            return hands, dd_table

        # Fallback: search with random dealer rotation
        for _ in range(10000):
            hands, dd_table = self.loader.sample_one()
            dealer = np.random.randint(NUM_PLAYERS)
            rotated = np.roll(hands, -dealer, axis=0)
            if self._satisfies_constraints(rotated, dealer=NORTH):
                self._sampled_dealer = dealer
                return np.roll(hands, -dealer, axis=0), np.roll(dd_table, -dealer, axis=1)

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

        dealer 由 generate_deal() 的最后一次调用决定（self._sampled_dealer）。
        如果 hands/dd_table 是外部传入的，dealer 默认为 NORTH（保持兼容）。

        Returns:
            obs: 前缀后下一决策者（dealer+2, i.e. partner of opener）的观测
        """
        if hands is None or dd_table is None:
            hands, dd_table = self.generate_deal()
            dealer = self._sampled_dealer
        else:
            dealer = NORTH   # external caller: assume N-opener convention

        self._current_hands = hands
        self._current_dd    = dd_table
        self._vulnerability = vulnerability
        self.dealer         = dealer
        self.history_int    = []

        obs = self.env.reset(hands, dealer=dealer, vulnerability=vulnerability)

        # Fixed prefix: opener (dealer) bids 1H; overcaller (dealer+1) bids 1S
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
        IMP regret（NS 视角，越高越好，≤ 0）.

            regret = score_to_imp(score_ns − optimal_score_ns)

        注意: IMP 是非线性的，必须先做差再转换。
        """
        contract = self.env.state.final_contract
        score_ns = self._compute_score_ns(
            contract, self._current_dd, self._vulnerability)

        opt_score = self._compute_dds_optimal_score_ns(
            self._current_dd, self._vulnerability)

        return float(score_to_imp(score_ns - opt_score))

    def _compute_dds_optimal_score_ns(
        self,
        dd_table:      np.ndarray,
        vulnerability: Tuple[bool, bool],
    ) -> int:
        """
        DDS oracle: NS 视角的双明手博弈均衡得分（P77修复）.

        正确语义：双明手均衡是双方都知道所有手牌时的博弈平衡点——
        任何一方再叫牌都只会得到更差的结果，因此停叫。
        1H-1S之后流局不可能发生。

        正确算法：对每个花色，只考虑能叫成的定约：
            - NS在该花色能叫成的最高级数 → NS视角正分
            - EW在该花色能叫成的最高级数 → NS视角负分
            - 两者取对NS更好的结果
        跨所有花色取全局最大值。

        错误的旧逻辑：
            - best_score=0预设流局可选（竞争叫牌中不成立）
            - EW作庄分支全pass
            - 误把EW宕牌的负分（NS视角正数）当作可选结果
              （EW永远不会自愿叫宕牌的定约）
        """
        from utils.scoring import Contract as C_

        ns_vul, ew_vul = vulnerability
        best_score = None

        for suit in range(5):
            # NS 在该花色能叫成的最高级数（叫成才有正分）
            ns_best = None
            for level in range(7, 0, -1):
                for declarer in (NORTH, SOUTH):
                    tricks = int(dd_table[suit, declarer])
                    score  = calculate_score(
                        C_(level=level, suit=suit, declarer=declarer, doubled=0),
                        tricks, ns_vul)
                    if score > 0:
                        if ns_best is None or score > ns_best:
                            ns_best = score

            # EW 在该花色能叫成的最高级数（NS视角取负）
            # EW会选对自己最有利的定约（叫最高能成的级数），
            # 对NS而言这是最坏情况（NS视角取min）
            ew_best_ns_view = None
            for level in range(7, 0, -1):
                for declarer in (EAST, WEST):
                    tricks   = int(dd_table[suit, declarer])
                    ew_score = calculate_score(
                        C_(level=level, suit=suit, declarer=declarer, doubled=0),
                        tricks, ew_vul)
                    if ew_score > 0:
                        ns_view = -ew_score
                        # EW叫最高能成的级数对NS最不利：取NS视角最小值
                        if ew_best_ns_view is None or ns_view < ew_best_ns_view:
                            ew_best_ns_view = ns_view

            # 该花色对NS最好的结果（NS打 vs 让EW打，取较优者）
            candidates = [x for x in (ns_best, ew_best_ns_view) if x is not None]
            if candidates:
                suit_best = max(candidates)
                if best_score is None or suit_best > best_score:
                    best_score = suit_best

        return best_score if best_score is not None else 0

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
        dealer:        Optional[int] = None,
    ) -> Tuple[Optional[Contract], int, List[int]]:
        """
        混合对抗: NS / EW 由不同策略控制.

        opener_seats  = {dealer, (dealer+2)%4}  (开叫方阵营)
        overcall_seats= {(dealer+1)%4, (dealer+3)%4}

        NS/EW 语义保持: 开叫方阵营 = NS policy, 争叫方阵营 = EW policy.
        """
        dealer = dealer if dealer is not None else self.dealer
        self.dealer = dealer  # P93 fix: policy closures read env.dealer for encode_obs_flat
        opener_seats = {dealer, (dealer + 2) % NUM_PLAYERS}

        inner = BridgeBiddingEnv(self.max_history_len)
        obs   = inner.reset(hands, dealer=dealer, vulnerability=vulnerability)
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
            if player in opener_seats:
                action = ns_policy(obs, player, hist[:])
            else:
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
    env:               "CompetitiveSubgameEnv",
    agent_a_ns_policy: Callable,
    agent_a_ew_policy: Callable,
    agent_b_ns_policy: Callable,
    agent_b_ew_policy: Callable,
    num_deals:         int = 500,
) -> CrossEvalResult:
    """
    交叉对抗评估（双桌 IMP，A 视角）.

    桌1: A=开叫方阵营, B=争叫方阵营 → score_1
    桌2: B=开叫方阵营, A=争叫方阵营 → score_2
    IMP = score_to_imp(score_1 - score_2)  (A 视角: 正=A赢)

    统计检验: Wilcoxon signed-rank（IMP 重尾非参）.
    dealer 轮换：每局 generate_deal() 后从 env._sampled_dealer 读取。
    """
    from scipy.stats import wilcoxon

    imps = []
    for _ in range(num_deals):
        hands, dd_table = env.generate_deal()
        dealer = env._sampled_dealer
        vul = [(False, False), (True, False),
               (False, True),  (True, True)][np.random.randint(4)]

        _, score_1, _ = env.play_mixed(
            hands, dd_table,
            ns_policy=agent_a_ns_policy,
            ew_policy=agent_b_ew_policy,
            vulnerability=vul, dealer=dealer)
        _, score_2, _ = env.play_mixed(
            hands, dd_table,
            ns_policy=agent_b_ns_policy,
            ew_policy=agent_a_ew_policy,
            vulnerability=vul, dealer=dealer)
        imps.append(float(score_to_imp(score_1 - score_2)))

    arr = np.array(imps)
    try:
        _, p_val = wilcoxon(arr)
    except Exception:
        p_val = 1.0

    return CrossEvalResult(
        mean_imp=float(arr.mean()), std_imp=float(arr.std()),
        win_rate=float((arr > 0).mean()), p_value=float(p_val),
        significant=bool(p_val < 0.05), n_deals=num_deals, all_imps=imps,
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
        dealer = env._sampled_dealer
        env.dealer = dealer  # P93 fix: policy closures read env.dealer for encode_obs_flat
        vul = (False, False)

        obs  = inner.reset(hands, dealer=dealer, vulnerability=vul)
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

        regret = float(score_to_imp(score_ns - opt_ns))
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
    dealer: int = NORTH,
) -> Callable[[Dict, int, list], int]:
    """
    从 MAPPOAgent 创建 competitive policy 函数.

    policy 签名: (obs, player, history_int) → action_int

    dealer: 当前局的 dealer seat（由 env.dealer 在 generate_deal() 后读取）。
    encode_obs_flat 需要 dealer 来编码相对位置。
    """
    import torch
    from networks.policy_net import encode_obs_flat

    def policy(obs: Dict, player: int, history_int: list) -> int:
        flat = encode_obs_flat(obs, dealer, history_int)
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
