from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrustPredictor(nn.Module):
    """Self-supervised module predicting neighbor actions and deriving trust masks."""

    def __init__(self, hidden_dim: int, action_dim: int, gamma: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.gamma = gamma

    def forward(
        self,
        neighbor_feat: torch.Tensor,
        neighbor_next_action: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            neighbor_feat: [B, K, hidden_dim]
            neighbor_next_action: optional [B, K, action_dim] one-hot labels
        Returns:
            pred_actions: predicted logits [B, K, action_dim]
            trust_mask: [B, K, 1] trust scores in [0, 1]
        """
        logits = self.net(neighbor_feat)
        if neighbor_next_action is None:
            trust = torch.ones(neighbor_feat.size(0), neighbor_feat.size(1), 1, device=neighbor_feat.device)
        else:
            mse = F.mse_loss(logits, neighbor_next_action, reduction="none").mean(dim=-1, keepdim=True)
            trust = torch.exp(-self.gamma * mse)
        return logits, trust
