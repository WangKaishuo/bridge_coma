"""
Competitive Subgame Environment  (P54 重构版)
=============================================

合作-对抗子博弈，固定前缀 1H(opener) – 1S(overcaller)。

P54 修复:
1. Dealer 轮换: generate_deal() 随机选 rotation，reset() 用 _sampled_dealer。
   约束判断、play_mixed、dds_oracle_evaluate 全部 dealer-aware。
2. score_to_imp 修复: regret = score_to_imp(score_ns − opt_score)。
   原来是 IMP(a) − IMP(b)，非线性导致量级失真。
3. _compute_dds_optimal_score_ns: 清理死代码，只保留正确的 NS-as-declarer 枚举。
4. make_agent_policy: 接受 dealer 参数，encode_obs_flat 用正确的 dealer。
5. play_mixed: opener_seats 按 dealer 推算，不再硬编码 player%2==0。
6. generate_rule_based_bc_data: dealer 从 env._sampled_dealer 动态读取。

环境特点:
    - 四方都参与决策（竞争激烈）
    - 固定前缀 1H-1S（开叫方/争叫方由 dealer 决定）
    - 支持 play_mixed：开叫方阵营 / 争叫方阵营 由不同 agent 控制
    - dds_oracle_evaluate：绝对基准，论文 RQ1 核心指标
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
FIXED_PREFIX = ["1H", "1S"]   # opener 叫 1H，overcaller 叫 1S，之后自由叫
_SUIT_H = 2   # hearts index
_SUIT_S = 3   # spades index


# ==============================================================================
# Rule-based 策略（BC fallback / 数据生成）
# ==============================================================================

def _rule_based_action(obs: Dict[str, np.ndarray], player: int,
                        history: list, dealer: int) -> int:
    """
    极简 rule-based 策略（用于 BC fallback 数据生成）.

    规则相对于绝对座位（NORTH/EAST/SOUTH/WEST）。
    dealer rotation 后，规则仍适用——因为 generate_deal() 已把
    hands 按 rotation 重排，使 opener 始终在 hands[dealer]。
    """
    hand  = obs['hand']
    legal = obs['legal_actions']

    def _legal(bid: int) -> bool:
        return bool(legal[bid] > 0.5)

    def _bid_if_legal(bid: int) -> int:
        return bid if _legal(bid) else BID_PASS

    hcp = int(count_hcp(hand))
    h   = int(count_suit_length(hand, _SUIT_H))
    s   = int(count_suit_length(hand, _SUIT_S))

    hist_len = len(history)

    # opener partner (dealer+2) — 第 3 叫
    opener_partner = (dealer + 2) % NUM_PLAYERS
    if player == opener_partner and hist_len == 2:
        bid_2h = string_to_bid("2H")
        bid_2s = string_to_bid("2S")
        bid_2c = string_to_bid("2C")
        bid_2d = string_to_bid("2D")
        if h >= 3 and hcp >= 8:
            return _bid_if_legal(bid_2h)
        elif hcp >= 10:
            d = int(count_suit_length(hand, 1))
            c = int(count_suit_length(hand, 0))
            if d >= 4 and _legal(bid_2d): return bid_2d
            if c >= 4 and _legal(bid_2c): return bid_2c
            return _bid_if_legal(bid_2s)
        return BID_PASS

    # overcaller partner (dealer+3) — 第 4 叫
    overcaller_partner = (dealer + 3) % NUM_PLAYERS
    if player == overcaller_partner and hist_len == 3:
        bid_2s = string_to_bid("2S")
        bid_3s = string_to_bid("3S")
        if s >= 4 and hcp >= 10:
            return _bid_if_legal(bid_3s if (s >= 6 or hcp >= 12) else bid_2s)
        return BID_PASS

    # opener rebid — 第 5 叫
    if player == dealer and hist_len == 4:
        bid_2h = string_to_bid("2H")
        bid_3h = string_to_bid("3H")
        bid_4h = string_to_bid("4H")
        if hcp >= 18:   return _bid_if_legal(bid_4h)
        elif hcp >= 15: return _bid_if_legal(bid_3h)
        elif hcp >= 12: return _bid_if_legal(bid_2h)
        return BID_PASS

    # overcaller rebid — 第 6 叫
    overcaller = (dealer + 1) % NUM_PLAYERS
    if player == overcaller and hist_len == 5:
        bid_2s = string_to_bid("2S")
        bid_3s = string_to_bid("3S")
        bid_4s = string_to_bid("4S")
        if hcp >= 14:   return _bid_if_legal(bid_4s)
        elif hcp >= 12: return _bid_if_legal(bid_3s)
        elif hcp >= 9:  return _bid_if_legal(bid_2s)
        return BID_PASS

    return BID_PASS


def generate_rule_based_bc_data(
    env: "CompetitiveSubgameEnv",
    num_samples: int = 5000,
) -> List[Dict]:
    """
    生成 rule-based BC 训练数据（SL checkpoint 不可用时的 fallback）.

    dealer 由 env.generate_deal() → env._sampled_dealer 动态决定。
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

        while not done:
            player  = inner_env.state.current_player
            history = inner_env.state.history[:]
            action  = _rule_based_action(obs, player, history, dealer)
            if not inner_env._is_valid_action(action):
                action = BID_PASS
            flat = encode_obs_flat(obs, dealer, history)
            data.append({'flat_obs': flat, 'action': action})
            obs, _, done, _ = inner_env.step(action)

    print(f"[BC Data] Generated {len(data)} samples "
          f"from {attempts} attempts ({len(data)/max(1,attempts):.1%})")
    return data


# ==============================================================================
# CompetitiveSubgameEnv
# ==============================================================================

class CompetitiveSubgameEnv:
    """
    竞叫子博弈环境（P54）.

    固定前缀: opener(dealer) 叫 1H，overcaller(dealer+1) 叫 1S。
    约束:     opener 5+H 12-21 HCP，overcaller 5+S 8-16 HCP。
    Dealer:   每局由 generate_deal() 随机选取（0-3），存入 _sampled_dealer。

    核心接口:
        reset()                  → obs
        step(action)             → obs, reward, done, info
        generate_deal()          → (hands, dd_table)  [设置 _sampled_dealer]
        play_mixed(...)          → (contract, score_ns, history)
    """

    def __init__(self, data_path: str, max_history_len: int = 60):
        self.loader           = create_loader(data_path)
        self.env              = BridgeBiddingEnv(max_history_len)
        self.max_history_len  = max_history_len
        self.dealer           = NORTH
        self._sampled_dealer  = NORTH   # written by generate_deal(), read by reset()

        # 判断是否为预生成约束数据
        self._is_constrained_data = self._check_if_constrained()
        if not self._is_constrained_data:
            self._filtered_deals: list = []
            self._prefetch(min_deals=500, max_attempts=50000)
        else:
            self._filtered_deals = None
            print(f"[CompetitiveEnv] Pre-generated constrained data: "
                  f"{len(self.loader)} samples")

        self._current_hands: Optional[np.ndarray] = None
        self._current_dd:    Optional[np.ndarray] = None
        self._vulnerability: Tuple[bool, bool]    = (False, False)
        self.history_int:    list                 = []

    # ------------------------------------------------------------------
    # 初始化辅助
    # ------------------------------------------------------------------

    def _check_if_constrained(self, sample_size: int = 20) -> bool:
        """预生成数据：N=opener，E=overcaller（dealer=NORTH 约定）."""
        passed = 0
        for _ in range(sample_size):
            hands, _ = self.loader.sample_one()
            if self._satisfies_constraints(hands, dealer=NORTH):
                passed += 1
        return passed >= sample_size * 0.9

    def _prefetch(self, min_deals: int, max_attempts: int):
        """从通用数据筛选满足约束的局，存 (hands, dd_table, dealer) 三元组."""
        attempts = 0
        while len(self._filtered_deals) < min_deals and attempts < max_attempts:
            attempts += 1
            hands, dd_table = self.loader.sample_one()
            for dealer in range(NUM_PLAYERS):
                if self._satisfies_constraints(hands, dealer=dealer):
                    self._filtered_deals.append((hands, dd_table, dealer))
                    break
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
        """opener: 5+H, 12-21 HCP."""
        return 12 <= count_hcp(hand) <= 21 and count_suit_length(hand, _SUIT_H) >= 5

    @staticmethod
    def _satisfies_overcaller(hand: np.ndarray) -> bool:
        """overcaller: 5+S, 8-16 HCP."""
        return 8 <= count_hcp(hand) <= 16 and count_suit_length(hand, _SUIT_S) >= 5

    # ------------------------------------------------------------------
    # 发牌（dealer rotation）
    # ------------------------------------------------------------------

    def generate_deal(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回 (hands, dd_table)，同时将本局 dealer 写入 self._sampled_dealer。

        预生成约束数据（N=opener）: 随机 roll，使任意座位都可能是 opener。
        prefetch 数据: dealer 已在 _prefetch 时确定。
        通用数据 fallback: 随机选 rotation 并验证约束。
        """
        if self._is_constrained_data:
            hands, dd_table = self.loader.sample_one()
            rotation = np.random.randint(NUM_PLAYERS)
            # np.roll 沿 axis=0 移动 player seats，axis=1 移动 declarer axis
            hands    = np.roll(hands,    -rotation, axis=0)
            dd_table = np.roll(dd_table, -rotation, axis=1)
            self._sampled_dealer = rotation
            return hands, dd_table

        if self._filtered_deals:
            idx   = np.random.randint(len(self._filtered_deals))
            entry = self._filtered_deals[idx]
            if len(entry) == 3:
                hands, dd_table, dealer = entry
            else:
                hands, dd_table = entry; dealer = NORTH
            self._sampled_dealer = dealer
            return hands, dd_table

        # fallback: 随机 rotation 搜索
        for _ in range(10000):
            hands, dd_table = self.loader.sample_one()
            dealer = int(np.random.randint(NUM_PLAYERS))
            rotated = np.roll(hands, -dealer, axis=0)
            if self._satisfies_constraints(rotated, dealer=NORTH):
                self._sampled_dealer = dealer
                return (np.roll(hands,    -dealer, axis=0),
                        np.roll(dd_table, -dealer, axis=1))

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

        dealer 由最近一次 generate_deal() 写入的 _sampled_dealer 决定。
        外部传入 hands/dd_table 时 dealer 默认 NORTH（兼容旧调用）。
        """
        if hands is None or dd_table is None:
            hands, dd_table = self.generate_deal()
            dealer = self._sampled_dealer
        else:
            dealer = NORTH   # external caller: N-opener convention

        self._current_hands = hands
        self._current_dd    = dd_table
        self._vulnerability = vulnerability
        self.dealer         = dealer
        self.history_int    = []

        obs = self.env.reset(hands, dealer=dealer, vulnerability=vulnerability)

        # Fixed prefix: opener(dealer) bids 1H; overcaller(dealer+1) bids 1S
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
    # 奖励计算
    # ------------------------------------------------------------------

    def _compute_terminal_reward(self) -> float:
        """
        IMP regret（NS/开叫方视角，越高越好，≤ 0）.

            regret = score_to_imp(score_ns − optimal_score_ns)

        P54 修复: IMP 是非线性的，必须先做差再转换。
        原来的 IMP(a) − IMP(b) 在大分差时误差显著。
        """
        contract = self.env.state.final_contract
        score_ns  = self._compute_score_ns(
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
        DDS oracle: 开叫方阵营（NS）视角的最优得分.

        枚举所有 NS-as-declarer 定约，取 NS 得分最大值。
        流局 = 0（始终可选，作为下限）。

        注: 仅考虑未加倍（doubled=0），保守估计。
            EW 作庄时 NS 得负分，而流局已保证 ≥ 0，故无需枚举 EW-declarer。
        """
        from utils.scoring import Contract as C_

        ns_vul    = vulnerability[0]
        best_score = 0   # 流局下限

        for suit in range(5):
            for level in range(1, 8):
                for declarer in (NORTH, SOUTH):
                    tricks = int(dd_table[suit, declarer])
                    contract = C_(level=level, suit=suit,
                                  declarer=declarer, doubled=0)
                    score = calculate_score(contract, tricks, ns_vul)
                    if score > best_score:
                        best_score = score

        return best_score

    def _compute_score_ns(
        self,
        contract:      Optional[Contract],
        dd_table:      np.ndarray,
        vulnerability: Tuple[bool, bool],
    ) -> int:
        """
        计算 NS 视角实际得分.

        vulnerability = (ns_vul, ew_vul)
        declarer % 2 == 0 → NS 庄 → 用 ns_vul，得分为正
        declarer % 2 == 1 → EW 庄 → 用 ew_vul，得分取负
        """
        if contract is None:
            return 0
        tricks = int(dd_table[contract.suit, contract.declarer])
        vul    = vulnerability[contract.declarer % 2]
        score  = calculate_score(contract, tricks, vul)
        if contract.declarer % 2 == 1:   # EW 庄 → NS 视角为负
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
    # 混合对抗（cross_evaluate 用）
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
        混合对抗: 开叫方阵营用 ns_policy，争叫方阵营用 ew_policy.

        opener_seats  = {dealer, (dealer+2)%4}
        overcall_seats= {(dealer+1)%4, (dealer+3)%4}
        """
        dealer       = dealer if dealer is not None else self.dealer
        opener_seats = {dealer, (dealer + 2) % NUM_PLAYERS}

        inner = BridgeBiddingEnv(self.max_history_len)
        obs   = inner.reset(hands, dealer=dealer, vulnerability=vulnerability)
        hist  = []
        done  = False

        for bid_str in FIXED_PREFIX:
            bid = string_to_bid(bid_str)
            hist.append(bid)
            obs, _, done, _ = inner.step(bid)
            if done:
                break

        while not done:
            player = inner.state.current_player
            action = (ns_policy(obs, player, hist[:])
                      if player in opener_seats
                      else ew_policy(obs, player, hist[:]))
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
    win_rate:    float
    p_value:     float
    significant: bool
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
    交叉对抗评估（双桌 IMP）.

    桌1: A=opener-side, B=overcall-side → score_1
    桌2: B=opener-side, A=overcall-side → score_2
    IMP = score_to_imp(score_1 − score_2)  (A 视角: 正=A赢)
    统计检验: Wilcoxon signed-rank（IMP 重尾非参）.
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
            ns_policy=agent_a_ns_policy, ew_policy=agent_b_ew_policy,
            vulnerability=vul, dealer=dealer)
        _, score_2, _ = env.play_mixed(
            hands, dd_table,
            ns_policy=agent_b_ns_policy, ew_policy=agent_a_ew_policy,
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
    env:       "CompetitiveSubgameEnv",
    policy:    Callable,
    num_deals: int = 1000,
) -> dict:
    """
    DDS oracle 评估: IMP regret（绝对基准，论文 RQ1 核心）.

        regret = score_to_imp(score_ns − dds_optimal_score_ns)  (≤ 0)

    P54 修复: 先做分差再转换 IMP（原来两次分别转换再做差，非线性误差）。
    policy 签名: (obs, player, history_int) → action_int
    """
    regrets = []
    inner   = BridgeBiddingEnv(max_history_len=60)

    for _ in range(num_deals):
        hands, dd_table = env.generate_deal()
        dealer = env._sampled_dealer
        vul    = (False, False)

        obs  = inner.reset(hands, dealer=dealer, vulnerability=vul)
        hist = []
        done = False

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

        regrets.append(float(score_to_imp(score_ns - opt_ns)))

    arr = np.array(regrets)
    bs_means = [np.random.choice(arr, size=len(arr), replace=True).mean()
                for _ in range(1000)]
    ci_lo, ci_hi = np.percentile(bs_means, [2.5, 97.5])

    return {
        'mean_regret': float(arr.mean()),
        'std_regret':  float(arr.std()),
        'ci_lo':       float(ci_lo),
        'ci_hi':       float(ci_hi),
        'n_deals':     num_deals,
        'pct_pass_out': float((arr == arr.min()).mean()),
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

    dealer 参数用于 encode_obs_flat（相对位置编码）。
    注: dealer rotation 后，应在每局 generate_deal() 后重新创建 policy，
        或使用 dds_oracle_evaluate 内联 dealer 的做法。
    """
    import torch
    from networks.policy_net import encode_obs_flat

    def policy(obs: Dict, player: int, history_int: list) -> int:
        flat   = encode_obs_flat(obs, dealer, history_int)
        flat_t = torch.tensor(flat, dtype=torch.float32
                              ).unsqueeze(0).to(agent.device)
        legal  = torch.tensor(obs['legal_actions'], dtype=torch.float32
                              ).unsqueeze(0).to(agent.device)
        actor  = agent.get_actor(player)
        with torch.no_grad():
            action, _, _ = actor.get_action(flat_t, legal,
                                            deterministic=deterministic)
        return action.item()

    return policy


def make_dynamic_agent_policy(
    agent,
    env: "CompetitiveSubgameEnv",
    deterministic: bool = True,
) -> Callable[[Dict, int, list], int]:
    """
    dealer-rotation 感知版 policy 工厂.

    每次调用时从 env.dealer 读取当前局的 dealer（由 generate_deal() 写入），
    因此 encode_obs_flat 始终使用正确的 dealer，无需每局重新创建 policy。
    用于 evaluate_oracle 等需要跨多局评估的场合。
    """
    import torch
    from networks.policy_net import encode_obs_flat

    def policy(obs: Dict, player: int, history_int: list) -> int:
        flat   = encode_obs_flat(obs, env.dealer, history_int)
        flat_t = torch.tensor(flat, dtype=torch.float32
                              ).unsqueeze(0).to(agent.device)
        legal  = torch.tensor(obs['legal_actions'], dtype=torch.float32
                              ).unsqueeze(0).to(agent.device)
        actor  = agent.get_actor(player)
        with torch.no_grad():
            action, _, _ = actor.get_action(flat_t, legal,
                                            deterministic=deterministic)
        return action.item()

    return policy


def make_rule_policy(dealer: int = NORTH) -> Callable[[Dict, int, list], int]:
    """Rule-based policy（用于 baseline 对比）."""
    def policy(obs: Dict, player: int, history_int: list) -> int:
        return _rule_based_action(obs, player, history_int, dealer)
    return policy
