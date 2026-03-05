"""
Scoring - Bridge Score Calculation
==================================

Single Source of Truth for all scoring logic.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Contract:
    """定约"""
    level: int          # 1-7
    suit: int           # 0=C, 1=D, 2=H, 3=S, 4=NT
    doubled: int        # 0=normal, 1=doubled, 2=redoubled
    declarer: int       # 0=N, 1=E, 2=S, 3=W
    
    @property
    def required_tricks(self) -> int:
        return 6 + self.level
    
    def __str__(self) -> str:
        suit_names = ['♣', '♦', '♥', '♠', 'NT']
        player_names = ['N', 'E', 'S', 'W']
        doubled_str = ['', 'X', 'XX'][self.doubled]
        return f"{self.level}{suit_names[self.suit]}{doubled_str} by {player_names[self.declarer]}"


def calculate_score(contract: Contract, tricks: int, vulnerable: bool) -> int:
    """
    计算定约得分（庄家视角）
    
    Args:
        contract: 定约
        tricks: 实际赢得的墩数 (0-13)
        vulnerable: 庄家是否有局
    
    Returns:
        得分（正数=做成，负数=宕）
    """
    required = contract.required_tricks
    result = tricks - required
    
    if result >= 0:
        return _score_made(contract, result, vulnerable)
    else:
        return _score_down(contract, -result, vulnerable)


def _score_made(contract: Contract, overtricks: int, vulnerable: bool) -> int:
    """做成定约的得分"""
    level = contract.level
    suit = contract.suit
    doubled = contract.doubled
    
    # 基本墩分
    if suit <= 1:  # Minor (C, D)
        trick_value = 20
    else:  # Major (H, S) or NT
        trick_value = 30
    
    # 定约分
    if suit == 4:  # NT: 首墩 40，之后 30
        contract_points = 40 + (level - 1) * 30
    else:
        contract_points = level * trick_value
    
    # 加倍/再加倍
    if doubled == 1:
        contract_points *= 2
    elif doubled == 2:
        contract_points *= 4
    
    # 奖分
    bonus = 0
    
    # 成局奖分
    if contract_points >= 100:
        bonus += 500 if vulnerable else 300
    else:
        bonus += 50  # 部分定约
    
    # 加倍成功奖分
    if doubled == 1:
        bonus += 50
    elif doubled == 2:
        bonus += 100
    
    # 满贯奖分
    if level == 6:  # 小满贯
        bonus += 750 if vulnerable else 500
    elif level == 7:  # 大满贯
        bonus += 1500 if vulnerable else 1000
    
    # 超墩分
    if doubled == 0:
        overtrick_value = trick_value
    elif doubled == 1:
        overtrick_value = 200 if vulnerable else 100
    else:  # redoubled
        overtrick_value = 400 if vulnerable else 200
    
    return contract_points + bonus + overtricks * overtrick_value


def _score_down(contract: Contract, undertricks: int, vulnerable: bool) -> int:
    """宕掉的罚分（返回负数）"""
    doubled = contract.doubled
    
    if doubled == 0:
        # 未加倍
        penalty_per_trick = 100 if vulnerable else 50
        return -undertricks * penalty_per_trick
    
    # 加倍/再加倍的罚分表
    if vulnerable:
        # 有局：200, 300, 300, 300, ...
        penalties = [200, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300]
    else:
        # 无局：100, 200, 200, 300, 300, 300, ...
        penalties = [100, 200, 200, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300]
    
    total = sum(penalties[:undertricks])
    
    # 再加倍翻倍
    if doubled == 2:
        total *= 2
    
    return -total
