from __future__ import annotations

import torch
import torch.nn as nn


class GatedResidualBlock(nn.Module):
    """Gated residual fusion between self feature and communication feature."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, self_feat: torch.Tensor, comm_feat: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([self_feat, comm_feat], dim=-1)))
        return self_feat + gate * comm_feat
