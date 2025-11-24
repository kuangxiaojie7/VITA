from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    """MLP + GRU encoder for local observation sequences."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(
        self,
        seq: torch.Tensor,
        hidden: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            seq: [B, T, input_dim]
            hidden: optional initial hidden state [1, B, hidden_dim]
            masks: optional masks [B, 1] to reset hidden when episodes end
        """
        if seq.dim() == 2:
            seq = seq.unsqueeze(1)
        B, T, D = seq.shape
        x = self.mlp(seq.view(B * T, D)).view(B, T, -1)
        if hidden is None:
            hidden = torch.zeros(1, B, self.gru.hidden_size, device=seq.device)
        if masks is not None:
            masks = masks.view(1, B, 1)
            hidden = hidden * masks
        out, next_hidden = self.gru(x, hidden)
        return out[:, -1], next_hidden
