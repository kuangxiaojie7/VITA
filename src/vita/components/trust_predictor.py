from __future__ import annotations

import torch
import torch.nn as nn


class TrustPredictor(nn.Module):
    """Learned trust gate for neighbor messages.

    In clean environments there may be no "liars", but messages can still be redundant or irrelevant.
    This module outputs a per-neighbor trust score in [0, 1] conditioned on the receiver (self) and
    the sender (neighbor) features, so it can learn relevance-based sparse communication.
    """

    def __init__(self, hidden_dim: int, gamma: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gamma = float(gamma)

    def forward(
        self,
        self_feat: torch.Tensor,
        neighbor_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            self_feat: [B, hidden_dim]
            neighbor_feat: [B, K, hidden_dim]
        Returns:
            trust: [B, K, 1] trust scores in [0, 1]
        """
        if self_feat.dim() != 2:
            raise ValueError(f"Expected self_feat to have shape [B, H], got {tuple(self_feat.shape)}")
        if neighbor_feat.dim() != 3:
            raise ValueError(f"Expected neighbor_feat to have shape [B, K, H], got {tuple(neighbor_feat.shape)}")

        self_rep = self_feat.unsqueeze(1).expand(-1, neighbor_feat.size(1), -1)
        x = torch.cat([self_rep, neighbor_feat, torch.abs(self_rep - neighbor_feat)], dim=-1)
        logits = self.net(x)
        trust = torch.sigmoid(logits)
        if self.gamma != 1.0:
            trust = trust.clamp(min=1e-6, max=1.0).pow(self.gamma)
        return trust
