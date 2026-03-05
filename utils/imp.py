"""
IMP Calculator
==============

International Match Points conversion.
"""

# 标准 IMP 转换表
_IMP_TABLE = [
    (20, 0), (50, 1), (90, 2), (130, 3), (170, 4),
    (220, 5), (270, 6), (320, 7), (370, 8), (430, 9),
    (500, 10), (600, 11), (750, 12), (900, 13),
    (1100, 14), (1300, 15), (1500, 16), (1750, 17),
    (2000, 18), (2250, 19), (2500, 20), (3000, 21),
    (3500, 22), (4000, 23), (float('inf'), 24)
]


def score_to_imp(score_diff: int) -> int:
    """
    将分差转换为 IMP
    
    Args:
        score_diff: 分差（正数=己方领先）
    
    Returns:
        IMP（正数=己方赢）
    """
    abs_diff = abs(score_diff)
    sign = 1 if score_diff >= 0 else -1
    
    for threshold, imp in _IMP_TABLE:
        if abs_diff < threshold:
            return sign * imp
    
    return sign * 24


def imp_to_vp(imp_diff: int, boards: int = 16) -> float:
    """
    将 IMP 转换为 VP (Victory Points)
    
    使用 WBF 连续 VP 量表
    
    Args:
        imp_diff: IMP 差
        boards: 副数
    
    Returns:
        VP (0-20 scale, 10-10 为平局)
    """
    # 简化的线性近似
    # 实际 WBF 量表更复杂
    ratio = imp_diff / (boards * 0.5)  # 标准化
    vp = 10 + ratio * 5
    return max(0, min(20, vp))
