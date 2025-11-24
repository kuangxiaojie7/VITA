from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class VIBGATLayer(nn.Module):
    """Variational bottleneck + trust-aware attention aggregation."""

    def __init__(self, hidden_dim: int, latent_dim: int, kl_beta: float):
        super().__init__()
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)
        self.query_proj = nn.Linear(hidden_dim, latent_dim)
        self.key_proj = nn.Linear(latent_dim, latent_dim)
        self.value_proj = nn.Linear(latent_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, hidden_dim)
        self.kl_beta = kl_beta

    def forward(
        self,
        self_feat: torch.Tensor,
        neighbor_feat: torch.Tensor,
        trust_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            self_feat: [B, hidden_dim]
            neighbor_feat: [B, K, hidden_dim]
            trust_mask: [B, K, 1]
        """
        mu = self.to_mu(neighbor_feat)
        logvar = self.to_logvar(neighbor_feat)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        query = self.query_proj(self_feat).unsqueeze(1)
        keys = self.key_proj(z)
        values = self.value_proj(z)

        attn_logits = torch.matmul(query, keys.transpose(-2, -1)).squeeze(1) / (keys.size(-1) ** 0.5)
        attn_logits = attn_logits + torch.log(trust_mask.squeeze(-1) + 1e-6)
        attn_weights = torch.softmax(attn_logits, dim=-1).unsqueeze(-2)
        context = torch.matmul(attn_weights, values).squeeze(-2)
        comm_feat = self.out_proj(context)

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        return comm_feat, self.kl_beta * kl
