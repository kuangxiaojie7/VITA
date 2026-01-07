from __future__ import annotations

import torch
import torch.nn as nn


class GatedResidualBlock(nn.Module):
    """Gated residual fusion between self feature and communication feature."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -5.0)

    def forward(
        self,
        self_feat: torch.Tensor,
        comm_feat: torch.Tensor,
        enabled: bool = True,
        strength: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        if (not enabled) or (float(strength) <= 0.0):
            return self_feat
        if not torch.is_tensor(strength):
            strength = torch.as_tensor(strength, device=self_feat.device, dtype=self_feat.dtype)
        strength = strength.clamp(min=0.0, max=1.0)
        gate_input = torch.cat([self_feat, comm_feat], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input)) * strength
        return self_feat + gate * comm_feat
