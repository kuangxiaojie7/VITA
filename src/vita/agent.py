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


class VITAAgent(torch.nn.Module):
    def __init__(self, cfg: VITAAgentConfig):
        super().__init__()
        self.cfg = cfg
        self.actor_encoder = FeatureEncoder(cfg.obs_dim, cfg.hidden_dim)
        self.critic_encoder = FeatureEncoder(cfg.state_dim, cfg.hidden_dim)
        self.trust_predictor = TrustPredictor(cfg.hidden_dim, cfg.action_dim, cfg.trust_gamma)
        self.vib_gat = VIBGATLayer(cfg.hidden_dim, cfg.latent_dim, cfg.kl_beta)
        self.residual = GatedResidualBlock(cfg.hidden_dim)
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
        return feat.view(B, K, -1)

    def act(
        self,
        obs_seq: torch.Tensor,
        state: torch.Tensor,
        neighbor_seq: torch.Tensor,
        neighbor_next_actions: torch.Tensor | None,
        rnn_states_actor: torch.Tensor,
        rnn_states_critic: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        self_feat, next_actor = self.actor_encoder(obs_seq, rnn_states_actor.unsqueeze(0), masks)
        neighbor_feat = self._encode_neighbors(neighbor_seq)
        _, trust_mask = self.trust_predictor(neighbor_feat, neighbor_next_actions)
        comm_feat, kl_loss = self.vib_gat(self_feat, neighbor_feat, trust_mask)
        fused = self.residual(self_feat, comm_feat)
        logits = self.policy_head(fused)
        dist = Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            actions = dist.sample().unsqueeze(-1)
        log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)

        critic_feat, next_critic = self.critic_encoder(state.unsqueeze(1), rnn_states_critic.unsqueeze(0), masks)
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
    ) -> Dict[str, torch.Tensor]:
        self_feat, _ = self.actor_encoder(obs_seq, rnn_states_actor.unsqueeze(0), masks)
        neighbor_feat = self._encode_neighbors(neighbor_seq)
        pred_actions, trust_mask = self.trust_predictor(neighbor_feat, neighbor_next_actions)
        comm_feat, kl_loss = self.vib_gat(self_feat, neighbor_feat, trust_mask)
        fused = self.residual(self_feat, comm_feat)
        logits = self.policy_head(fused)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)

        critic_feat, _ = self.critic_encoder(state.unsqueeze(1), rnn_states_critic.unsqueeze(0), masks)
        values = self.value_head(critic_feat)
        trust_loss = F.mse_loss(pred_actions, neighbor_next_actions)

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
        values = self.value_head(critic_feat)
        return values, next_state.squeeze(0)
