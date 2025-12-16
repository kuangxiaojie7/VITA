from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    """Track running mean and variance for normalization, saved with the model."""

    def __init__(self, epsilon: float = 1e-4) -> None:
        super().__init__()
        # Buffers ensure stats are saved/loaded with the model.
        self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("var", torch.tensor(1.0))
        self.register_buffer("count", torch.tensor(epsilon))

    def update(self, x) -> None:
        arr = np.asarray(x, dtype=np.float64)
        if arr.size == 0:
            return
        batch_mean = arr.mean()
        batch_var = arr.var()
        batch_count = arr.size
        current_mean = float(self.mean.item())
        current_var = float(self.var.item())
        current_count = float(self.count.item())
        delta = batch_mean - current_mean
        tot_count = current_count + batch_count
        new_mean = current_mean + delta * batch_count / tot_count
        m_a = current_var * current_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta * delta * current_count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean.data.fill_(new_mean)
        self.var.data.fill_(max(new_var, 1e-8))
        self.count.data.fill_(tot_count)

    def _std(self) -> float:
        return float(np.sqrt(self.var.item()) + 1e-8)

    def normalize(self, x: torch.Tensor, *, use_mean: bool = True) -> torch.Tensor:
        std = self._std()
        if use_mean:
            mean = float(self.mean.item())
            return (x - mean) / std
        return x / std

    def denormalize(self, x: torch.Tensor, *, use_mean: bool = True) -> torch.Tensor:
        std = self._std()
        if use_mean:
            mean = float(self.mean.item())
            return x * std + mean
        return x * std
