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


def south_stayman_rule(hands: np.ndarray, history: list) -> int:
    """
    S 的 Stayman 规则策略.

    1NT-Pass-2C-Pass-{N回应}-Pass 后, S 根据 N 的回应 + 自身手牌决策:

    N 叫 2H (有 4+H):
      S 有 4+H:  10+ HCP → 4H, 8-9 HCP → 3H (邀请)
      S 无 4H:   10+ HCP → 3NT, 8-9 HCP → 2NT (邀请)
    N 叫 2S (有 4+S):
      S 有 4+S:  10+ HCP → 4S, 8-9 HCP → 3S (邀请)
      S 无 4S:   10+ HCP → 3NT, 8-9 HCP → 2NT (邀请)
    N 叫 2D (否认高花):
      10+ HCP → 3NT, 8-9 HCP → 2NT (邀请)

    Args:
        hands: (4, 52) — 需要 hands[SOUTH]
        history: 当前叫牌历史 (list of bid ints)
    Returns:
        bid index
    """
    s_hand = hands[SOUTH]
    hcp = count_hcp(s_hand)
    s_h = count_suit_length(s_hand, 2)  # hearts
    s_s = count_suit_length(s_hand, 3)  # spades

    prefix_len = 4  # 1NT Pass 2C Pass
    rounds_after = len(history) - prefix_len

    if rounds_after != 2:
        # 不是 S 的 Round 2 决策点, Pass
        return BID_PASS

    # N 的回应是 history[-2] (跳过 E 的 Pass)
    n_response = history[-2]
    n_bid_str = bid_to_string(n_response)

    game_values = hcp >= 10  # 10+ HCP → 成局力
    invite = (8 <= hcp <= 9)  # 8-9 HCP → 邀请力

    if n_bid_str == "2♥" or n_bid_str == "2H":
        # N 有 4+H
        if s_h >= 4:
            # 找到配合
            return string_to_bid("4H") if game_values else string_to_bid("3H")
        else:
            # 无配合, 根据点力选 NT
            return string_to_bid("3NT") if game_values else string_to_bid("2NT")

    elif n_bid_str == "2♠" or n_bid_str == "2S":
        # N 有 4+S
        if s_s >= 4:
            # 找到配合
            return string_to_bid("4S") if game_values else string_to_bid("3S")
        else:
            # 无配合, 根据点力选 NT
            return string_to_bid("3NT") if game_values else string_to_bid("2NT")

    else:
        # N 叫 2D, 否认高花
        return string_to_bid("3NT") if game_values else string_to_bid("2NT")


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

        # 中间步: 使用桥牌规则合法性 mask
        if not done_after:
            obs['legal_actions'] = self.env._get_legal_actions()

        return obs, 0.0, False, info

    def _auto_play_non_agent(self, obs: dict, done: bool) -> dict:
        """
        自动执行非 agent 控制的玩家:
        - EW 永远 auto-pass
        - north_rule=True 时, N 也用规则策略

        返回下一个需要 agent 决策的 player 的 obs.
        legal_actions 直接使用桥牌规则合法性 mask (普适化, 无硬编码限制).
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

        # 使用桥牌规则合法性 mask, 而非硬编码的 Stayman 专用 mask
        # 这确保 agent 面对完整的 38-bid 动作空间, 与全环境一致
        if not done:
            obs['legal_actions'] = self.env._get_legal_actions()

        return obs

    def _check_env_done(self) -> bool:
        """检查 env 是否结束."""
        return self.env._check_done()

    def _get_stayman_mask(self) -> np.ndarray:
        """
        [Deprecated] Stayman 专用硬编码 action mask.

        已被桥牌规则合法性 mask (env._get_legal_actions()) 取代.
        保留此函数仅供向后兼容, 训练/评估代码不应调用.
        """
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

    def _compute_terminal_reward(self) -> float:
        """
        训练 reward: 直接返回 IMP regret (actual - DDS optimal).

        值域: (-∞, 0], 通常在 [-13, 0] 范围内.
        归一化由 SubgameTrainer 的 RunningStats 在线处理,
        确保 reward 分布 ~ N(0,1), 与全环境训练方式完全一致.

        DDS baseline 使用完整 7 阶无限制最优 (max_level=7),
        让 agent 面对真实的机会成本, 而非人为压低的天花板.
        """
        return float(self._compute_imp_diff())

    def _compute_imp_diff(self) -> float:
        """
        计算 actual vs DDS optimal (无级别限制) 的 IMP 差值 (≤ 0).
        """
        contract = self.env.state.final_contract
        dd_table = self._current_dd
        actual_score = self._compute_score(contract, dd_table, is_ns=True)
        # max_level=7: 无限制, 反映真实机会成本
        optimal = self._get_optimal_contract_ns(dd_table, max_level=7)
        optimal_score = self._compute_score(optimal, dd_table, is_ns=True)
        return float(score_to_imp(actual_score - optimal_score))

    def _compute_eval_imp(self) -> float:
        """评估用 IMP: actual vs DDS optimal (无级别限制)."""
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
    def _get_optimal_contract_ns(dd_table: np.ndarray,
                                 max_level: int = 7) -> Optional[Contract]:
        """
        计算 NS 方的 DDS 最优定约.

        Args:
            dd_table: (5, 4) DDS 赢墩表
            max_level: 最高阶数限制 (默认 7=无限制,
                       设为 4 时与 Stayman action mask 对齐)
        """
        best_score = 0
        best_contract = None
        for declarer in [NORTH, SOUTH]:
            for suit in range(5):
                tricks = int(dd_table[suit, declarer])
                for level in range(1, max_level + 1):
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


# ============================================================================
# BC Dataset for Stayman
# ============================================================================

def create_bc_dataset_for_stayman(
    data_path: str,
    num_samples: int = 10000,
    max_history_len: int = 60,
    players: str = 'south',
) -> list:
    """
    为 Stayman 子博弈生成 BC 训练数据.

    对每副牌, 模拟完整 Stayman 序列 (N 规则 + S 规则),
    采集指定玩家的所有决策: (obs, target_action) 对.

    采集点:
      N Round1: 回应 2C → 2D / 2H / 2S
      S Round2: 再叫   → 邀请 (2NT/3H/3S) 或 直接进局 (3NT/4H/4S)
      N Round3: 应邀   → Pass (拒绝) / 3NT / 4H / 4S (接受)
                只在 S 叫邀请时才存在

    Args:
        data_path:      DDS 数据路径
        num_samples:    目标样本数 (建议 ≥ 15000 确保稀有情形覆盖)
        max_history_len:历史编码长度
        players:        'south'=只采S, 'north'=只采N, 'both'=采N+S

    Returns:
        list of {'obs': dict, 'action': int}
    """
    loader = create_loader(data_path)
    env = BridgeBiddingEnv(max_history_len)
    data = []
    collect_north = players in ('north', 'both')
    collect_south = players in ('south', 'both')

    # 统计计数器 (用于摘要)
    n_r3_reached = 0
    n_r3_pass = 0
    n_r3_accept = 0
    s_action_counts = {}

    max_attempts = num_samples * 20   # 安全上限 (接受率~30%, 有足够余量)
    attempts = 0

    while len(data) < num_samples and attempts < max_attempts:
        attempts += 1
        hands, dd_table = loader.sample_one()

        # 验证约束
        n_hand, s_hand = hands[NORTH], hands[SOUTH]
        n_hcp = count_hcp(n_hand)
        s_hcp = count_hcp(s_hand)
        if not (15 <= n_hcp <= 17 and is_balanced(n_hand)):
            continue
        s_h = count_suit_length(s_hand, 2)
        s_s = count_suit_length(s_hand, 3)
        if not (s_hcp >= 8 and (s_h >= 4 or s_s >= 4)):
            continue

        # 模拟叫牌: 固定前缀 1NT - Pass - 2C - Pass
        obs = env.reset(hands, dealer=NORTH, vulnerability=(False, False))
        done = False
        for bid_str in ["1NT", "Pass", "2C", "Pass"]:
            obs, _, done, _ = env.step(string_to_bid(bid_str))
            if done: break
        if done:
            continue

        # ── N Round 1: 回应 2C ────────────────────────────────────────────
        n_target = north_stayman_rule(hands, env.state.history)
        if not env._is_valid_action(n_target):
            n_target = BID_PASS

        if collect_north:
            n_mask = np.zeros(NUM_BIDS, dtype=np.float32)
            for b in ["2D", "2H", "2S"]:
                n_mask[string_to_bid(b)] = 1.0
            n_mask *= env._get_legal_actions()
            if n_mask.sum() < 0.5:
                n_mask[BID_PASS] = 1.0
            if n_mask[n_target] > 0.5:
                n_obs = {k: v.copy() for k, v in obs.items()}
                n_obs['legal_actions'] = n_mask
                data.append({'obs': n_obs, 'action': int(n_target)})

        obs, _, done, _ = env.step(n_target)
        if done: continue
        obs, _, done, _ = env.step(BID_PASS)   # E Pass
        if done: continue

        # ── S Round 2: 邀请或直接进局 ────────────────────────────────────
        s_target = south_stayman_rule(hands, env.state.history)
        s_action_counts[bid_to_string(s_target)] = \
            s_action_counts.get(bid_to_string(s_target), 0) + 1

        if collect_south:
            s_mask = np.zeros(NUM_BIDS, dtype=np.float32)
            for b in ["Pass", "2NT", "3NT", "3H", "3S", "4H", "4S"]:
                s_mask[string_to_bid(b)] = 1.0
            s_mask *= env._get_legal_actions()
            if s_mask.sum() < 0.5:
                s_mask[BID_PASS] = 1.0
            if s_mask[s_target] > 0.5:
                s_obs = {k: v.copy() for k, v in obs.items()}
                s_obs['legal_actions'] = s_mask
                data.append({'obs': s_obs, 'action': int(s_target)})

        obs, _, done, _ = env.step(s_target)
        if done: continue
        obs, _, done, _ = env.step(BID_PASS)   # E Pass
        if done:
            # S 直接进局 → 游戏结束, 无需 N 应答
            continue

        # ── N Round 3: 接受或拒绝邀请 ────────────────────────────────────
        # 只有 S 叫了邀请 (2NT/3H/3S) 时才能到达这里
        n_r3_reached += 1
        n_target2 = north_stayman_rule(hands, env.state.history)
        if not env._is_valid_action(n_target2):
            n_target2 = BID_PASS

        if n_target2 == BID_PASS:
            n_r3_pass += 1
        else:
            n_r3_accept += 1

        if collect_north:
            n_mask2 = np.zeros(NUM_BIDS, dtype=np.float32)
            for b in ["Pass", "3NT", "4H", "4S"]:
                n_mask2[string_to_bid(b)] = 1.0
            n_mask2 *= env._get_legal_actions()
            if n_mask2.sum() < 0.5:
                n_mask2[BID_PASS] = 1.0
            if n_mask2[n_target2] > 0.5:
                n_obs2 = {k: v.copy() for k, v in obs.items()}
                n_obs2['legal_actions'] = n_mask2
                data.append({'obs': n_obs2, 'action': int(n_target2)})

    # ── 摘要 ──────────────────────────────────────────────────────────────
    n_count = sum(1 for d in data
                  if d['obs']['position'].argmax() == NORTH) if collect_north else 0
    s_count = sum(1 for d in data
                  if d['obs']['position'].argmax() == SOUTH) if collect_south else 0

    # N Round1 vs Round3 细分
    r1_bids = {string_to_bid(b) for b in ["2D", "2H", "2S"]}
    n_r1_count = sum(1 for d in data
                     if d['obs']['position'].argmax() == NORTH
                     and d['action'] in r1_bids) if collect_north else 0
    n_r3_count = n_count - n_r1_count

    print(f"Stayman BC dataset: {len(data)} samples  "
          f"(N={n_count} [R1={n_r1_count} R3={n_r3_count}], S={s_count})")
    if collect_north and n_r3_reached > 0:
        print(f"  N Round3: reached={n_r3_reached}, "
              f"Pass={n_r3_pass} ({n_r3_pass/n_r3_reached:.0%}), "
              f"Accept={n_r3_accept} ({n_r3_accept/n_r3_reached:.0%})")
    if s_action_counts:
        invite = sum(v for k, v in s_action_counts.items()
                     if k in ("2NT", "3♥", "3H", "3♠", "3S"))
        game   = sum(v for k, v in s_action_counts.items()
                     if k in ("3NT", "4♥", "4H", "4♠", "4S"))
        total_s = invite + game
        print(f"  S Round2: invite={invite} ({invite/max(1,total_s):.0%}), "
              f"game={game} ({game/max(1,total_s):.0%})")
    return data
    loader = create_loader(data_path)
    env = BridgeBiddingEnv(max_history_len)
    data = []
    collect_north = players in ('north', 'both')
    collect_south = players in ('south', 'both')

    # ── debug counters ────────────────────────────────────────────────────
    dbg = {
        'deals_tried': 0,
        'n_hcp': {15: 0, 16: 0, 17: 0},
        's_hcp': {8: 0, 9: 0, 10: 0, 'other': 0},
        's_action': {},          # S 叫了什么
        'n_r1_action': {},       # N Round1 叫了什么
        'n_r3_reached': 0,       # 到达 N Round3 的次数
        'n_r3_action': {},       # N Round3 叫了什么 (Pass / 4H / 4S / 3NT)
        'n_r3_hcp': {15: 0, 16: 0, 17: 0},  # N Round3 时 N 的 HCP
    }
    # ─────────────────────────────────────────────────────────────────────

    for _ in range(num_samples * 10):   # 扩大上限，确保能采满
        if len(data) >= num_samples:
            break

        hands, dd_table = loader.sample_one()
        dbg['deals_tried'] += 1

        # 验证约束
        n_hand, s_hand = hands[NORTH], hands[SOUTH]
        n_hcp = count_hcp(n_hand)
        s_hcp = count_hcp(s_hand)
        if not (15 <= n_hcp <= 17 and is_balanced(n_hand)):
            continue
        s_h = count_suit_length(s_hand, 2)
        s_s = count_suit_length(s_hand, 3)
        if not (s_hcp >= 8 and (s_h >= 4 or s_s >= 4)):
            continue

        # debug: 记录 HCP 分布
        dbg['n_hcp'][n_hcp] = dbg['n_hcp'].get(n_hcp, 0) + 1
        k = s_hcp if s_hcp in (8, 9, 10) else 'other'
        dbg['s_hcp'][k] = dbg['s_hcp'].get(k, 0) + 1

        # 模拟叫牌
        obs = env.reset(hands, dealer=NORTH, vulnerability=(False, False))

        # 固定前缀: 1NT - Pass - 2C - Pass
        done = False
        for bid_str in ["1NT", "Pass", "2C", "Pass"]:
            bid = string_to_bid(bid_str)
            obs, _, done, _ = env.step(bid)
            if done:
                break
        if done:
            continue

        # --- N Round 1: 回应 2C (2D/2H/2S) ---
        n_target = north_stayman_rule(hands, env.state.history)
        if not env._is_valid_action(n_target):
            n_target = BID_PASS

        dbg['n_r1_action'][bid_to_string(n_target)] = \
            dbg['n_r1_action'].get(bid_to_string(n_target), 0) + 1

        if collect_north:
            n_mask = np.zeros(NUM_BIDS, dtype=np.float32)
            n_mask[string_to_bid("2D")] = 1.0
            n_mask[string_to_bid("2H")] = 1.0
            n_mask[string_to_bid("2S")] = 1.0
            env_legal = env._get_legal_actions()
            n_mask = n_mask * env_legal
            if n_mask.sum() < 0.5:
                n_mask[BID_PASS] = 1.0

            if n_mask[n_target] > 0.5:
                n_obs = {k: v.copy() for k, v in obs.items()}
                n_obs['legal_actions'] = n_mask
                data.append({'obs': n_obs, 'action': int(n_target)})

        # 执行 N 的叫品
        obs, _, done, _ = env.step(n_target)
        if done:
            continue

        # E Pass
        obs, _, done, _ = env.step(BID_PASS)
        if done:
            continue

        # --- S Round 2: 再叫 ---
        s_target = south_stayman_rule(hands, env.state.history)
        dbg['s_action'][bid_to_string(s_target)] = \
            dbg['s_action'].get(bid_to_string(s_target), 0) + 1

        if collect_south:
            s_mask = np.zeros(NUM_BIDS, dtype=np.float32)
            s_mask[BID_PASS] = 1.0
            s_mask[string_to_bid("2NT")] = 1.0
            s_mask[string_to_bid("3NT")] = 1.0
            s_mask[string_to_bid("3H")] = 1.0
            s_mask[string_to_bid("3S")] = 1.0
            s_mask[string_to_bid("4H")] = 1.0
            s_mask[string_to_bid("4S")] = 1.0
            env_legal = env._get_legal_actions()
            s_mask = s_mask * env_legal
            if s_mask.sum() < 0.5:
                s_mask[BID_PASS] = 1.0

            if s_mask[s_target] > 0.5:
                s_obs = {k: v.copy() for k, v in obs.items()}
                s_obs['legal_actions'] = s_mask
                data.append({'obs': s_obs, 'action': int(s_target)})

        # 执行 S 的叫品
        obs, _, done, _ = env.step(s_target)
        if done:
            continue

        # E Pass
        obs, _, done, _ = env.step(BID_PASS)
        if done:
            # S 直接叫局 (4H/4S/3NT), 游戏结束, 不需要 N 应答
            continue

        # --- N Round 3: 接受或拒绝邀请 ---
        dbg['n_r3_reached'] += 1
        dbg['n_r3_hcp'][n_hcp] = dbg['n_r3_hcp'].get(n_hcp, 0) + 1

        if collect_north:
            n_target2 = north_stayman_rule(hands, env.state.history)
            if not env._is_valid_action(n_target2):
                n_target2 = BID_PASS

            dbg['n_r3_action'][bid_to_string(n_target2)] = \
                dbg['n_r3_action'].get(bid_to_string(n_target2), 0) + 1

            n_mask2 = np.zeros(NUM_BIDS, dtype=np.float32)
            n_mask2[BID_PASS] = 1.0
            n_mask2[string_to_bid("3NT")] = 1.0
            n_mask2[string_to_bid("4H")] = 1.0
            n_mask2[string_to_bid("4S")] = 1.0
            env_legal2 = env._get_legal_actions()
            n_mask2 = n_mask2 * env_legal2
            if n_mask2.sum() < 0.5:
                n_mask2[BID_PASS] = 1.0

            if n_mask2[n_target2] > 0.5:
                n_obs2 = {k: v.copy() for k, v in obs.items()}
                n_obs2['legal_actions'] = n_mask2
                data.append({'obs': n_obs2, 'action': int(n_target2)})

    # ── debug report ──────────────────────────────────────────────────────
    total_deals = sum(dbg['n_hcp'].values())
    print(f"\n[BC Debug] Deals sampled: {total_deals} (tried {dbg['deals_tried']})")

    print(f"\n  N HCP distribution:")
    for v in (15, 16, 17):
        n = dbg['n_hcp'].get(v, 0)
        print(f"    {v} HCP: {n:5d} ({n/max(1,total_deals):.1%})")

    print(f"\n  S HCP distribution:")
    for v in (8, 9, 10, 'other'):
        n = dbg['s_hcp'].get(v, 0)
        print(f"    {str(v):5s} HCP: {n:5d} ({n/max(1,total_deals):.1%})")

    print(f"\n  N Round1 actions (2D/2H/2S):")
    for bid, n in sorted(dbg['n_r1_action'].items(), key=lambda x: -x[1]):
        print(f"    {bid:6s}: {n:5d} ({n/max(1,total_deals):.1%})")

    print(f"\n  S Round2 actions:")
    for bid, n in sorted(dbg['s_action'].items(), key=lambda x: -x[1]):
        print(f"    {bid:6s}: {n:5d} ({n/max(1,total_deals):.1%})")

    print(f"\n  N Round3 reached: {dbg['n_r3_reached']} "
          f"({dbg['n_r3_reached']/max(1,total_deals):.1%} of deals)")
    if dbg['n_r3_reached'] > 0:
        print(f"  N Round3 HCP when reached:")
        for v in (15, 16, 17):
            n = dbg['n_r3_hcp'].get(v, 0)
            print(f"    {v} HCP: {n:5d} ({n/max(1,dbg['n_r3_reached']):.1%})")
        print(f"  N Round3 actions (Pass=拒绝 / 4H,4S,3NT=接受):")
        for bid, n in sorted(dbg['n_r3_action'].items(), key=lambda x: -x[1]):
            print(f"    {bid:6s}: {n:5d} ({n/max(1,dbg['n_r3_reached']):.1%})")

    n_count = sum(1 for d in data if d['obs']['position'].argmax() == NORTH) if collect_north else 0
    s_count = sum(1 for d in data if d['obs']['position'].argmax() == SOUTH) if collect_south else 0
    print(f"\n  BC samples breakdown: total={len(data)}  N={n_count}  S={s_count}")
    if collect_north and n_count > 0:
        n_r1 = sum(1 for d in data
                   if d['obs']['position'].argmax() == NORTH
                   and d['action'] in [string_to_bid("2D"),
                                       string_to_bid("2H"),
                                       string_to_bid("2S")])
        n_r3 = n_count - n_r1
        print(f"    N Round1 samples: {n_r1}  N Round3 samples: {n_r3}")
        r3_pass = sum(1 for d in data
                      if d['obs']['position'].argmax() == NORTH
                      and d['action'] == BID_PASS)
        r3_accept = n_r3 - r3_pass
        print(f"    N Round3: Pass(拒绝)={r3_pass}  Accept(接受)={r3_accept}")

    print(f"Stayman BC dataset: {len(data)} samples (N={n_count}, S={s_count})")
    return data
    """
    为 Stayman 子博弈生成 BC 训练数据.

    对每副牌, 模拟完整 Stayman 序列 (N 规则 + S 规则),
    采集指定玩家的决策: (obs, target_action) 对.

    Args:
        data_path: DDS 数据路径
        num_samples: 目标样本数
        max_history_len: 历史编码长度
        players: 'south' = 只采 S, 'north' = 只采 N, 'both' = 采 N+S

    Returns:
        list of {'obs': dict, 'action': int}
    """
    loader = create_loader(data_path)
    env = BridgeBiddingEnv(max_history_len)
    data = []
    collect_north = players in ('north', 'both')
    collect_south = players in ('south', 'both')

    for _ in range(num_samples * 2):
        if len(data) >= num_samples:
            break

        hands, dd_table = loader.sample_one()

        # 验证约束
        n_hand, s_hand = hands[NORTH], hands[SOUTH]
        n_hcp = count_hcp(n_hand)
        s_hcp = count_hcp(s_hand)
        if not (15 <= n_hcp <= 17 and is_balanced(n_hand)):
            continue
        s_h = count_suit_length(s_hand, 2)
        s_s = count_suit_length(s_hand, 3)
        if not (s_hcp >= 8 and (s_h >= 4 or s_s >= 4)):
            continue

        # 模拟叫牌
        obs = env.reset(hands, dealer=NORTH, vulnerability=(False, False))

        # 固定前缀: 1NT - Pass - 2C - Pass
        done = False
        for bid_str in ["1NT", "Pass", "2C", "Pass"]:
            bid = string_to_bid(bid_str)
            obs, _, done, _ = env.step(bid)
            if done:
                break
        if done:
            continue

        # --- N Round 1: 回应 2C (2D/2H/2S) ---
        n_target = north_stayman_rule(hands, env.state.history)
        if not env._is_valid_action(n_target):
            n_target = BID_PASS

        if collect_north:
            # N 的 Stayman mask: 只允许 2D/2H/2S
            n_mask = np.zeros(NUM_BIDS, dtype=np.float32)
            n_mask[string_to_bid("2D")] = 1.0
            n_mask[string_to_bid("2H")] = 1.0
            n_mask[string_to_bid("2S")] = 1.0
            env_legal = env._get_legal_actions()
            n_mask = n_mask * env_legal
            if n_mask.sum() < 0.5:
                n_mask[BID_PASS] = 1.0

            if n_mask[n_target] > 0.5:
                n_obs = {k: v.copy() for k, v in obs.items()}
                n_obs['legal_actions'] = n_mask
                data.append({'obs': n_obs, 'action': int(n_target)})

        # 执行 N 的叫品
        obs, _, done, _ = env.step(n_target)
        if done:
            continue

        # E Pass
        obs, _, done, _ = env.step(BID_PASS)
        if done:
            continue

        # --- S Round 2: 再叫 ---
        if collect_south:
            s_target = south_stayman_rule(hands, env.state.history)

            s_mask = np.zeros(NUM_BIDS, dtype=np.float32)
            s_mask[BID_PASS] = 1.0
            s_mask[string_to_bid("2NT")] = 1.0
            s_mask[string_to_bid("3NT")] = 1.0
            s_mask[string_to_bid("3H")] = 1.0
            s_mask[string_to_bid("3S")] = 1.0
            s_mask[string_to_bid("4H")] = 1.0
            s_mask[string_to_bid("4S")] = 1.0
            env_legal = env._get_legal_actions()
            s_mask = s_mask * env_legal
            if s_mask.sum() < 0.5:
                s_mask[BID_PASS] = 1.0

            if s_mask[s_target] > 0.5:
                s_obs = {k: v.copy() for k, v in obs.items()}
                s_obs['legal_actions'] = s_mask
                data.append({'obs': s_obs, 'action': int(s_target)})

        # 执行 S 的叫品
        obs, _, done, _ = env.step(s_target)
        if done:
            continue

        # E Pass
        obs, _, done, _ = env.step(BID_PASS)
        if done:
            # S 直接叫局 (4H/4S/3NT), 游戏结束, 不需要 N 应答
            continue

        # --- N Round 3: 接受或拒绝邀请 ---
        # 只有 S 叫了邀请 (3H/3S/2NT) 时才能走到这里
        if collect_north:
            n_target2 = north_stayman_rule(hands, env.state.history)
            if not env._is_valid_action(n_target2):
                n_target2 = BID_PASS

            n_mask2 = np.zeros(NUM_BIDS, dtype=np.float32)
            n_mask2[BID_PASS] = 1.0              # 拒绝邀请
            n_mask2[string_to_bid("3NT")] = 1.0  # 接受无将邀请
            n_mask2[string_to_bid("4H")] = 1.0   # 接受红桃邀请
            n_mask2[string_to_bid("4S")] = 1.0   # 接受黑桃邀请
            env_legal2 = env._get_legal_actions()
            n_mask2 = n_mask2 * env_legal2
            if n_mask2.sum() < 0.5:
                n_mask2[BID_PASS] = 1.0

            if n_mask2[n_target2] > 0.5:
                n_obs2 = {k: v.copy() for k, v in obs.items()}
                n_obs2['legal_actions'] = n_mask2
                data.append({'obs': n_obs2, 'action': int(n_target2)})

    n_count = sum(1 for d in data if d['obs']['position'].argmax() == NORTH) if collect_north else 0
    s_count = sum(1 for d in data if d['obs']['position'].argmax() == SOUTH) if collect_south else 0
    print(f"Stayman BC dataset: {len(data)} samples "
          f"(N={n_count}, S={s_count})")
    return data
