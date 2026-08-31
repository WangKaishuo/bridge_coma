"""International Match Point conversion utilities."""

_IMP_TABLE = [
    (20, 0), (50, 1), (90, 2), (130, 3), (170, 4), (220, 5),
    (270, 6), (320, 7), (370, 8), (430, 9), (500, 10), (600, 11),
    (750, 12), (900, 13), (1100, 14), (1300, 15), (1500, 16),
    (1750, 17), (2000, 18), (2250, 19), (2500, 20), (3000, 21),
    (3500, 22), (4000, 23), (float("inf"), 24),
]


def score_to_imp(score_diff: int) -> int:
    """Convert a signed duplicate-score difference to IMPs."""
    sign = 1 if score_diff >= 0 else -1
    for threshold, imp in _IMP_TABLE:
        if abs(score_diff) < threshold:
            return sign * imp
    return sign * 24


def imp_to_vp(imp_diff: int, boards: int = 16) -> float:
    """Return a simple linear 0-20 VP approximation."""
    ratio = imp_diff / (boards * 0.5)
    return max(0, min(20, 10 + ratio * 5))
