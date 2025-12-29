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
            pred_actions: predicted action probabilities [B, K, action_dim]
            trust_mask: [B, K, 1] trust scores in [0, 1]
        """
        logits = self.net(neighbor_feat)
        probs = torch.softmax(logits, dim=-1)
        if neighbor_next_action is None:
            trust = torch.ones(neighbor_feat.size(0), neighbor_feat.size(1), 1, device=neighbor_feat.device)
        else:
            has_label = neighbor_next_action.sum(dim=-1, keepdim=True) > 1e-6
            p_true = (probs * neighbor_next_action).sum(dim=-1, keepdim=True).clamp_min(1e-6)
            trust_raw = p_true.pow(self.gamma)
            trust = torch.where(has_label, trust_raw, torch.ones_like(trust_raw))
        return probs, trust
