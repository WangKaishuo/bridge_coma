"""
Bridge Bidding Environment
==========================

桥牌叫牌环境 (Dec-POMDP)

修复：
- 结束条件判断（四家全Pass流局 / 有实质叫品后三家Pass）
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from utils.scoring import Contract

# 常量
NUM_PLAYERS = 4
NUM_SUITS = 5      # C, D, H, S, NT
NUM_LEVELS = 7     # 1-7
NUM_BIDS = 38      # Pass + Double + Redouble + 35 bids

# 叫品索引
BID_PASS = 0
BID_DOUBLE = 1
BID_REDOUBLE = 2
BID_1C = 3  # 1C-7NT: 3-37

# 位置
NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3


def bid_to_string(bid: int) -> str:
    """叫品索引转字符串"""
    if bid == BID_PASS:
        return "Pass"
    elif bid == BID_DOUBLE:
        return "X"
    elif bid == BID_REDOUBLE:
        return "XX"
    else:
        level = (bid - BID_1C) // 5 + 1
        suit = (bid - BID_1C) % 5
        suit_names = ['♣', '♦', '♥', '♠', 'NT']
        return f"{level}{suit_names[suit]}"


def string_to_bid(s: str) -> int:
    """字符串转叫品索引"""
    s = s.upper().strip()
    if s in ('PASS', 'P'):
        return BID_PASS
    elif s in ('X', 'DBL', 'DOUBLE'):
        return BID_DOUBLE
    elif s in ('XX', 'RDBL', 'REDOUBLE'):
        return BID_REDOUBLE
    else:
        suit_map = {'C': 0, '♣': 0, 'D': 1, '♦': 1, 'H': 2, '♥': 2, 
                    'S': 3, '♠': 3, 'N': 4, 'NT': 4}
        level = int(s[0])
        suit_str = s[1:].replace('NT', 'N')
        suit = suit_map.get(suit_str, suit_map.get(suit_str[0]))
        return BID_1C + (level - 1) * 5 + suit


@dataclass
class BiddingState:
    """叫牌状态"""
    hands: np.ndarray                          # (4, 52) one-hot
    dealer: int = 0                            # 发牌人
    vulnerability: Tuple[bool, bool] = (False, False)  # (NS_vul, EW_vul)
    history: List[int] = field(default_factory=list)   # 叫牌历史
    current_player: int = 0
    final_contract: Optional['Contract'] = None
    
    def __post_init__(self):
        self.current_player = self.dealer


class BridgeBiddingEnv:
    """
    桥牌叫牌环境
    
    观测空间：
        - hand: (52,) 自己的手牌
        - history: (max_len, 38) 叫牌历史 one-hot
        - legal_actions: (38,) 合法动作 mask
        - position: (4,) 位置 one-hot
        - vulnerability: (2,) 局况
    
    动作空间：
        - 0: Pass
        - 1: Double
        - 2: Redouble
        - 3-37: 1C-7NT
    """
    
    def __init__(self, max_history_len: int = 60):
        self.max_history_len = max_history_len
        self.state: Optional[BiddingState] = None
    
    def reset(
        self,
        hands: Optional[np.ndarray] = None,
        dealer: int = 0,
        vulnerability: Tuple[bool, bool] = (False, False)
    ) -> Dict[str, np.ndarray]:
        """重置环境"""
        if hands is None:
            hands = self._deal_random()
        
        self.state = BiddingState(
            hands=hands.astype(np.float32),
            dealer=dealer,
            vulnerability=vulnerability
        )
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """执行动作"""
        assert self.state is not None
        assert self._is_valid_action(action), f"Invalid action: {action}"
        
        self.state.history.append(action)
        
        done = self._check_done()
        reward = 0.0
        
        if done:
            self.state.final_contract = self._get_final_contract()
        else:
            self.state.current_player = (self.state.current_player + 1) % 4
        
        obs = self._get_observation()
        info = {'bid': action, 'bid_str': bid_to_string(action)}
        
        return obs, reward, done, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """获取当前玩家的观测"""
        player = self.state.current_player
        
        # 手牌
        hand = self.state.hands[player].copy()
        
        # 叫牌历史 one-hot
        history = np.zeros((self.max_history_len, NUM_BIDS), dtype=np.float32)
        for i, bid in enumerate(self.state.history[-self.max_history_len:]):
            history[i, bid] = 1.0
        
        # 合法动作
        legal_actions = self._get_legal_actions()
        
        # 位置
        position = np.zeros(4, dtype=np.float32)
        position[player] = 1.0
        
        # 局况
        vulnerability = np.array([
            float(self.state.vulnerability[0]),  # NS
            float(self.state.vulnerability[1])   # EW
        ], dtype=np.float32)
        
        return {
            'hand': hand,
            'history': history,
            'legal_actions': legal_actions,
            'position': position,
            'vulnerability': vulnerability,
        }
    
    def _get_legal_actions(self) -> np.ndarray:
        """获取合法动作 mask

        Double/Redouble 状态机（桥牌规则）:
          每个实质叫品之后，后缀的加倍状态只有三种：
            UNDOUBLED  → 对手可以 Double
            DOUBLED    → 我方可以 Redouble；对手不能再 Double
            REDOUBLED  → 双方均不能再 X 或 XX

          任何新的实质叫品都重置状态为 UNDOUBLED。
          用单一变量 double_state 遍历后缀，避免两个独立 boolean 逻辑割裂的 bug。
        """
        legal = np.zeros(NUM_BIDS, dtype=np.float32)
        legal[BID_PASS] = 1.0  # Pass 总是合法

        history = self.state.history

        # 找到最高实质叫品及其位置
        highest_bid = None
        last_real_bid_idx = -1
        for i, bid in enumerate(history):
            if bid >= BID_1C:
                highest_bid = bid
                last_real_bid_idx = i

        # 新叫品必须高于当前最高叫品
        min_bid = BID_1C if highest_bid is None else highest_bid + 1
        for bid in range(min_bid, NUM_BIDS):
            legal[bid] = 1.0

        if last_real_bid_idx < 0:
            return legal  # 尚无实质叫品，X/XX 均不合法

        # 统一状态机：扫描最后实质叫品之后的后缀
        # double_state: 0=undoubled, 1=doubled, 2=redoubled
        double_state = 0
        last_real_bidder = (self.state.dealer + last_real_bid_idx) % 4

        for i, bid in enumerate(history[last_real_bid_idx + 1:]):
            if bid == BID_DOUBLE:
                double_state = 1
            elif bid == BID_REDOUBLE:
                double_state = 2
            # 新实质叫品不会出现在这里（last_real_bid_idx 已是最后一个）

        current = self.state.current_player

        # Double 合法条件：undoubled + 最后实质叫品是对手叫的
        if double_state == 0 and last_real_bidder % 2 != current % 2:
            legal[BID_DOUBLE] = 1.0

        # Redouble 合法条件：doubled + 最后实质叫品是我方叫的
        if double_state == 1 and last_real_bidder % 2 == current % 2:
            legal[BID_REDOUBLE] = 1.0

        return legal
    
    def _is_valid_action(self, action: int) -> bool:
        """检查动作是否合法"""
        legal = self._get_legal_actions()
        return legal[action] > 0.5
    
    def _check_done(self) -> bool:
        """
        检查叫牌是否结束
        
        结束条件：
        1. 四家全 Pass（流局）
        2. 有实质叫品后，三家连续 Pass
        """
        history = self.state.history
        n = len(history)
        
        if n < 4:
            return False
        
        # 检查是否有实质叫品
        has_real_bid = any(b >= BID_1C for b in history)
        
        # 统计末尾连续 Pass
        consecutive_passes = 0
        for bid in reversed(history):
            if bid == BID_PASS:
                consecutive_passes += 1
            else:
                break
        
        # 情况 1: 四家全 Pass（流局）
        if n == 4 and consecutive_passes == 4:
            return True
        
        # 情况 2: 有实质叫品后三家连续 Pass
        if has_real_bid and consecutive_passes >= 3:
            return True
        
        return False
    
    def _get_final_contract(self) -> Optional[Contract]:
        """获取最终定约"""
        history = self.state.history
        
        # 检查是否流局
        if all(b == BID_PASS for b in history):
            return None
        
        # 找最后的实质叫品
        last_bid = None
        last_bid_idx = -1
        for i, bid in enumerate(history):
            if bid >= BID_1C:
                last_bid = bid
                last_bid_idx = i
        
        if last_bid is None:
            return None
        
        # 解析叫品
        level = (last_bid - BID_1C) // 5 + 1
        suit = (last_bid - BID_1C) % 5
        
        # 检查加倍状态
        doubled = 0
        for bid in history[last_bid_idx + 1:]:
            if bid == BID_DOUBLE:
                doubled = 1
            elif bid == BID_REDOUBLE:
                doubled = 2
        
        # 确定庄家：同阵营中最先叫该花色的人
        last_bidder = (self.state.dealer + last_bid_idx) % 4
        team = last_bidder % 2  # 0=NS, 1=EW
        
        declarer = last_bidder
        for i, bid in enumerate(history):
            if bid >= BID_1C:
                bid_suit = (bid - BID_1C) % 5
                bidder = (self.state.dealer + i) % 4
                if bid_suit == suit and bidder % 2 == team:
                    declarer = bidder
                    break
        
        return Contract(level=level, suit=suit, doubled=doubled, declarer=declarer)
    
    def _deal_random(self) -> np.ndarray:
        """随机发牌"""
        deck = np.arange(52)
        np.random.shuffle(deck)
        hands = np.zeros((4, 52), dtype=np.float32)
        for i, card in enumerate(deck):
            hands[i // 13, card] = 1.0
        return hands
    
    def render(self):
        """打印当前状态"""
        if self.state is None:
            print("Environment not initialized")
            return
        
        suit_symbols = ['♣', '♦', '♥', '♠']
        rank_chars = "23456789TJQKA"
        player_names = ['North', 'East', 'South', 'West']
        
        print("\n" + "=" * 50)
        for p in range(4):
            hand_str = []
            for suit in range(3, -1, -1):
                cards = []
                for rank in range(12, -1, -1):
                    if self.state.hands[p, suit * 13 + rank] > 0.5:
                        cards.append(rank_chars[rank])
                hand_str.append(f"{suit_symbols[suit]}{''.join(cards)}")
            print(f"{player_names[p]:6s}: {' '.join(hand_str)}")
        
        print("-" * 50)
        print("Bidding:", ' '.join(bid_to_string(b) for b in self.state.history))
        print(f"Current: {player_names[self.state.current_player]}")
        print("=" * 50)
