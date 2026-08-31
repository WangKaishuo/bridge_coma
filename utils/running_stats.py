"""Online statistics used to normalize auxiliary rewards."""

import numpy as np


class RunningStats:
    """Welford online mean and variance."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (value - self.mean)

    def update_batch(self, values: np.ndarray) -> None:
        for value in values.flatten():
            self.update(float(value))

    @property
    def variance(self) -> float:
        return 1.0 if self.n < 2 else self.M2 / (self.n - 1)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    def normalize(self, value: float) -> float:
        return (value - self.mean) / (self.std + 1e-8)

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0


class EMAStats:
    """Exponential moving statistics for non-stationary signals."""

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.mean = 0.0
        self.var = 1.0
        self.initialized = False

    def update(self, value: float) -> None:
        if not self.initialized:
            self.mean = value
            self.var = 1.0
            self.initialized = True
            return
        delta = value - self.mean
        self.mean += self.alpha * delta
        self.var = (1 - self.alpha) * (self.var + self.alpha * delta * delta)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var))

    def normalize(self, value: float) -> float:
        return (value - self.mean) / (self.std + 1e-8)
