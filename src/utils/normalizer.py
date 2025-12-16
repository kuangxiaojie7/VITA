from __future__ import annotations

import numpy as np
import torch


class RunningMeanStd:
    """Track running mean and variance for normalization."""

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x) -> None:
        arr = np.asarray(x, dtype=np.float64)
        if arr.size == 0:
            return
        batch_mean = arr.mean()
        batch_var = arr.var()
        batch_count = arr.size
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta * delta * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = float(new_mean)
        self.var = float(max(new_var, 1e-8))
        self.count = float(tot_count)

    def _std(self) -> float:
        return float(np.sqrt(self.var) + 1e-8)

    def normalize(self, x: torch.Tensor, *, use_mean: bool = True) -> torch.Tensor:
        std = self._std()
        if use_mean:
            mean = float(self.mean)
            return (x - mean) / std
        return x / std

    def denormalize(self, x: torch.Tensor, *, use_mean: bool = True) -> torch.Tensor:
        std = self._std()
        if use_mean:
            mean = float(self.mean)
            return x * std + mean
        return x * std
