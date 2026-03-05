"""
Running Statistics
==================

在线计算均值和方差（Welford 算法）
用于归一化 Dual-Info Bonus
"""

import numpy as np


class RunningStats:
    """
    Welford's online algorithm for computing mean and variance.
    
    用于训练过程中动态归一化 info bonus
    """
    
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
    
    def update(self, x: float):
        """更新统计量"""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
    
    def update_batch(self, xs: np.ndarray):
        """批量更新"""
        for x in xs.flatten():
            self.update(float(x))
    
    @property
    def variance(self) -> float:
        if self.n < 2:
            return 1.0
        return self.M2 / (self.n - 1)
    
    @property
    def std(self) -> float:
        return np.sqrt(self.variance)
    
    def normalize(self, x: float) -> float:
        """Z-score 归一化"""
        return (x - self.mean) / (self.std + 1e-8)
    
    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0


class EMAStats:
    """
    Exponential Moving Average statistics.
    
    比 Welford 更适合非平稳环境
    """
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.mean = 0.0
        self.var = 1.0
        self.initialized = False
    
    def update(self, x: float):
        if not self.initialized:
            self.mean = x
            self.var = 1.0
            self.initialized = True
        else:
            delta = x - self.mean
            self.mean += self.alpha * delta
            self.var = (1 - self.alpha) * (self.var + self.alpha * delta * delta)
    
    @property
    def std(self) -> float:
        return np.sqrt(self.var)
    
    def normalize(self, x: float) -> float:
        return (x - self.mean) / (self.std + 1e-8)
