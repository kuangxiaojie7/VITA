from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.distributions import Categorical
import torch.nn.functional as F

from .components import FeatureEncoder, TrustPredictor, VIBGATLayer, GatedResidualBlock


@dataclass
class VITAAgentConfig:
    obs_dim: int
    state_dim: int
    action_dim: int
    hidden_dim: int = 128
    latent_dim: int = 64
    trust_gamma: float = 1.0
    kl_beta: float = 1e-3
    trust_lambda: float = 0.1
    max_neighbors: int = 5
    comm_dropout: float = 0.1
    enable_trust: bool = True
    enable_kl: bool = True


class VITAAgent(torch.nn.Module):
    def __init__(self, cfg: VITAAgentConfig):
        super().__init__()
        self.cfg = cfg
        self.actor_encoder = FeatureEncoder(cfg.obs_dim, cfg.hidden_dim)
        self.critic_encoder = FeatureEncoder(cfg.state_dim, cfg.hidden_dim)
        self.trust_predictor = TrustPredictor(cfg.hidden_dim, cfg.action_dim, cfg.trust_gamma)
        self.vib_gat = VIBGATLayer(cfg.hidden_dim, cfg.latent_dim, cfg.kl_beta)
        self.residual = GatedResidualBlock(cfg.hidden_dim)
        self.neighbor_norm = torch.nn.LayerNorm(cfg.hidden_dim)
        self.comm_dropout = torch.nn.Dropout(cfg.comm_dropout)
        self.critic_mlp = torch.nn.Sequential(
            torch.nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.policy_head = torch.nn.Linear(cfg.hidden_dim, cfg.action_dim)
        self.value_head = torch.nn.Linear(cfg.hidden_dim, 1)

    @property
    def rnn_hidden_dim(self) -> int:
        return self.cfg.hidden_dim

    def _encode_neighbors(self, neighbor_seq: torch.Tensor) -> torch.Tensor:
        # neighbor_seq: [B, K, T, obs_dim]
        B, K, T, D = neighbor_seq.shape
        flat = neighbor_seq.view(B * K, T, D)
        feat, _ = self.actor_encoder(flat, None, None)
        feat = self.neighbor_norm(feat)
        return feat.view(B, K, -1)

    def _mask_logits(self, logits: torch.Tensor, avail_actions: torch.Tensor | None) -> torch.Tensor:
        if avail_actions is None:
            return logits
        mask = (avail_actions < 0.5)
        all_masked = mask.all(dim=-1, keepdim=True)
        if all_masked.any():
            mask = mask & (~all_masked)
        return logits.masked_fill(mask, -1e9)

    def act(
        self,
        obs_seq: torch.Tensor,
        state: torch.Tensor,
        neighbor_seq: torch.Tensor,
        neighbor_next_actions: torch.Tensor | None,
        rnn_states_actor: torch.Tensor,
        rnn_states_critic: torch.Tensor,
        masks: torch.Tensor,
        avail_actions: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        self_feat, next_actor = self.actor_encoder(obs_seq, rnn_states_actor.unsqueeze(0), masks)
        neighbor_feat = self._encode_neighbors(neighbor_seq)
        if self.cfg.enable_trust:
            _, trust_mask = self.trust_predictor(neighbor_feat, neighbor_next_actions)
            trust_mask = trust_mask.detach()
        else:
            trust_mask = torch.ones(neighbor_feat.size(0), neighbor_feat.size(1), 1, device=neighbor_feat.device)
        comm_feat, kl_loss = self.vib_gat(self_feat, neighbor_feat, trust_mask)
        if not self.cfg.enable_kl:
            kl_loss = torch.zeros(1, device=self_feat.device)
        comm_feat = self.comm_dropout(comm_feat)
        fused = self.residual(self_feat, comm_feat)
        logits = self.policy_head(fused)
        logits = self._mask_logits(logits, avail_actions)
        dist = Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            actions = dist.sample().unsqueeze(-1)
        log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)

        critic_feat, next_critic = self.critic_encoder(state.unsqueeze(1), rnn_states_critic.unsqueeze(0), masks)
        critic_feat = self.critic_mlp(critic_feat)
        values = self.value_head(critic_feat)

        return {
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "entropy": entropy,
            "next_actor_state": next_actor.squeeze(0),
            "next_critic_state": next_critic.squeeze(0),
            "kl_loss": kl_loss,
        }

    def evaluate_actions(
        self,
        obs_seq: torch.Tensor,
        state: torch.Tensor,
        neighbor_seq: torch.Tensor,
        neighbor_next_actions: torch.Tensor,
        actions: torch.Tensor,
        rnn_states_actor: torch.Tensor,
        rnn_states_critic: torch.Tensor,
        masks: torch.Tensor,
        avail_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self_feat, _ = self.actor_encoder(obs_seq, rnn_states_actor.unsqueeze(0), masks)
        neighbor_feat = self._encode_neighbors(neighbor_seq)
        if self.cfg.enable_trust:
            pred_actions, trust_mask = self.trust_predictor(neighbor_feat, neighbor_next_actions)
            trust_mask = trust_mask.detach()
        else:
            trust_mask = torch.ones(neighbor_feat.size(0), neighbor_feat.size(1), 1, device=neighbor_feat.device)
            pred_actions = neighbor_next_actions
        comm_feat, kl_loss = self.vib_gat(self_feat, neighbor_feat, trust_mask)
        if not self.cfg.enable_kl:
            kl_loss = torch.zeros(1, device=self_feat.device)
        comm_feat = self.comm_dropout(comm_feat)
        fused = self.residual(self_feat, comm_feat)
        logits = self.policy_head(fused)
        logits = self._mask_logits(logits, avail_actions)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)

        critic_feat, _ = self.critic_encoder(state.unsqueeze(1), rnn_states_critic.unsqueeze(0), masks)
        critic_feat = self.critic_mlp(critic_feat)
        values = self.value_head(critic_feat)
        if self.cfg.enable_trust:
            trust_loss = F.mse_loss(pred_actions, neighbor_next_actions)
        else:
            trust_loss = torch.zeros(1, device=obs_seq.device)

        return {
            "log_probs": log_probs,
            "entropy": entropy,
            "values": values,
            "kl_loss": kl_loss,
            "trust_loss": trust_loss,
        }

    def get_values(
        self,
        state: torch.Tensor,
        rnn_states_critic: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        critic_feat, next_state = self.critic_encoder(state.unsqueeze(1), rnn_states_critic.unsqueeze(0), masks)
        critic_feat = self.critic_mlp(critic_feat)
        values = self.value_head(critic_feat)
        return values, next_state.squeeze(0)
