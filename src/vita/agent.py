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
    trust_threshold: float = 0.0
    attn_bias_coef: float = 1.0


class VITAAgent(torch.nn.Module):
    def __init__(self, cfg: VITAAgentConfig):
        super().__init__()
        self.cfg = cfg
        self.actor_encoder = FeatureEncoder(cfg.obs_dim, cfg.hidden_dim)
        self.critic_encoder = FeatureEncoder(cfg.state_dim, cfg.hidden_dim)
        self.comm_encoder = FeatureEncoder(cfg.obs_dim, cfg.hidden_dim)
        self.trust_predictor = TrustPredictor(cfg.hidden_dim, cfg.trust_gamma)
        self.vib_gat = VIBGATLayer(cfg.hidden_dim, cfg.latent_dim, cfg.kl_beta, cfg.attn_bias_coef)
        self.residual = GatedResidualBlock(cfg.hidden_dim)
        self.neighbor_norm = torch.nn.LayerNorm(cfg.hidden_dim)
        self.comm_dropout = torch.nn.Dropout(cfg.comm_dropout)
        critic_hidden = max(cfg.hidden_dim, 256)
        self.critic_mlp = torch.nn.Sequential(
            torch.nn.Linear(cfg.hidden_dim, critic_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(critic_hidden, critic_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(critic_hidden, cfg.hidden_dim),
        )
        self.policy_head = torch.nn.Linear(cfg.hidden_dim, cfg.action_dim)
        self.value_head = torch.nn.Linear(cfg.hidden_dim, 1)
        self.comm_enabled = True
        self.comm_strength = 0.0
        self.trust_strength = 0.0

    def set_comm_enabled(self, enabled: bool) -> None:
        self.comm_enabled = enabled
        if not enabled:
            self.comm_strength = 0.0

    def set_comm_strength(self, strength: float) -> None:
        strength = float(max(0.0, min(1.0, strength)))
        self.comm_strength = strength

    def set_trust_strength(self, strength: float) -> None:
        if not self.cfg.enable_trust:
            self.trust_strength = 0.0
            return
        strength = float(max(0.0, min(1.0, strength)))
        self.trust_strength = strength

    def set_trust_active(self, active: bool) -> None:
        self.set_trust_strength(1.0 if active else 0.0)

    @property
    def rnn_hidden_dim(self) -> int:
        return self.cfg.hidden_dim

    def _encode_neighbors(self, neighbor_seq: torch.Tensor) -> torch.Tensor:
        # neighbor_seq: [B, K, T, obs_dim]
        B, K, T, D = neighbor_seq.shape
        flat = neighbor_seq.view(B * K, T, D)
        feat, _ = self.comm_encoder(flat, None, None)
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
        neighbor_mask: torch.Tensor | None,
        neighbor_next_actions: torch.Tensor | None,
        rnn_states_actor: torch.Tensor,
        rnn_states_critic: torch.Tensor,
        masks: torch.Tensor,
        avail_actions: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        self_feat, next_actor = self.actor_encoder(obs_seq, rnn_states_actor.unsqueeze(0), masks)
        if (not self.comm_enabled) or (self.comm_strength <= 0.0):
            comm_feat = torch.zeros_like(self_feat)
            kl_loss = torch.zeros(1, device=self_feat.device)
            kl_raw = torch.zeros(1, device=self_feat.device)
        else:
            neighbor_feat = self._encode_neighbors(neighbor_seq)
            if neighbor_mask is None:
                neighbor_mask = torch.ones(neighbor_feat.size(0), neighbor_feat.size(1), 1, device=neighbor_feat.device)
            comm_mask = neighbor_mask.float()
            use_trust = self.cfg.enable_trust and (self.trust_strength > 0.0)
            if use_trust:
                trust_scores_raw = self.trust_predictor(self_feat, neighbor_feat)
                trust_scores = (1.0 - float(self.trust_strength)) + float(self.trust_strength) * trust_scores_raw
            else:
                trust_scores_raw = None
                trust_scores = torch.ones_like(comm_mask)
            if use_trust and self.cfg.trust_threshold > 0.0:
                threshold = float(self.cfg.trust_threshold) * float(self.comm_strength)
                trust_gate_hard = (trust_scores >= threshold).float()
                trust_gate = trust_gate_hard + trust_scores - trust_scores.detach()
                comm_mask = comm_mask * trust_gate
            neighbor_feat = neighbor_feat * comm_mask
            trust_scores = trust_scores * comm_mask + (1e-6 * (1.0 - comm_mask))
            comm_feat, kl_loss, kl_raw = self.vib_gat(
                self_feat,
                neighbor_feat,
                trust_scores,
                comm_mask,
                alive_mask=comm_mask,
                deterministic=deterministic,
            )
            if not self.cfg.enable_kl:
                kl_loss = torch.zeros(1, device=self_feat.device)
            comm_feat = self.comm_dropout(comm_feat)
        fused = self.residual(self_feat, comm_feat, self.comm_enabled, self.comm_strength)
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
            "kl_raw": kl_raw,
        }

    def evaluate_actions(
        self,
        obs_seq: torch.Tensor,
        state: torch.Tensor,
        neighbor_seq: torch.Tensor,
        neighbor_mask: torch.Tensor | None,
        neighbor_next_actions: torch.Tensor,
        actions: torch.Tensor,
        rnn_states_actor: torch.Tensor,
        rnn_states_critic: torch.Tensor,
        masks: torch.Tensor,
        avail_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self_feat, next_actor = self.actor_encoder(obs_seq, rnn_states_actor.unsqueeze(0), masks)
        if (not self.comm_enabled) or (self.comm_strength <= 0.0):
            comm_feat = torch.zeros_like(self_feat)
            kl_loss = torch.zeros(1, device=self_feat.device)
            trust_loss = torch.zeros(1, device=obs_seq.device)
            kl_raw = torch.zeros(1, device=self_feat.device)
            trust_score_mean = torch.zeros(1, device=obs_seq.device)
            trust_score_p10 = torch.zeros(1, device=obs_seq.device)
            trust_score_p50 = torch.zeros(1, device=obs_seq.device)
            trust_score_p90 = torch.zeros(1, device=obs_seq.device)
            trust_gate_ratio = torch.zeros(1, device=obs_seq.device)
            comm_valid_neighbors = torch.zeros(1, device=obs_seq.device)
            comm_kept_neighbors = torch.zeros(1, device=obs_seq.device)
        else:
            neighbor_feat = self._encode_neighbors(neighbor_seq)
            if neighbor_mask is None:
                neighbor_mask = torch.ones(neighbor_feat.size(0), neighbor_feat.size(1), 1, device=neighbor_feat.device)
            comm_mask = neighbor_mask.float()
            use_trust = self.cfg.enable_trust and (self.trust_strength > 0.0)
            if use_trust:
                trust_scores_raw = self.trust_predictor(self_feat, neighbor_feat)
                trust_scores = (1.0 - float(self.trust_strength)) + float(self.trust_strength) * trust_scores_raw
            else:
                trust_scores = torch.ones_like(comm_mask)
                trust_scores_raw = None
            if use_trust and self.cfg.trust_threshold > 0.0:
                threshold = float(self.cfg.trust_threshold) * float(self.comm_strength)
                trust_gate_hard = (trust_scores >= threshold).float()
                trust_gate = trust_gate_hard + trust_scores - trust_scores.detach()
                comm_mask = comm_mask * trust_gate
            neighbor_feat = neighbor_feat * comm_mask
            trust_scores = trust_scores * comm_mask + (1e-6 * (1.0 - comm_mask))
            comm_feat, kl_loss, kl_raw = self.vib_gat(
                self_feat,
                neighbor_feat,
                trust_scores,
                comm_mask,
                alive_mask=comm_mask,
                deterministic=False,
            )
            if not self.cfg.enable_kl:
                kl_loss = torch.zeros(1, device=self_feat.device)
            comm_feat = self.comm_dropout(comm_feat)
            if use_trust:
                valid = (neighbor_mask > 0.5).float()
                denom = valid.sum().clamp_min(1.0)
                trust_loss = (trust_scores_raw * valid).sum() / denom
            else:
                trust_loss = torch.zeros(1, device=obs_seq.device)

            # Trust diagnostics (computed on valid neighbor slots before comm dropout).
            valid_mask = (neighbor_mask > 0.5).squeeze(-1)
            comm_valid_neighbors = valid_mask.float().sum(dim=-1).mean()
            comm_kept_neighbors = (comm_mask > 0.5).squeeze(-1).float().sum(dim=-1).mean()
            trust_values = (trust_scores_raw if trust_scores_raw is not None else trust_scores).squeeze(-1)[valid_mask]
            if trust_values.numel() == 0:
                trust_score_mean = torch.zeros(1, device=obs_seq.device)
                trust_score_p10 = torch.zeros(1, device=obs_seq.device)
                trust_score_p50 = torch.zeros(1, device=obs_seq.device)
                trust_score_p90 = torch.zeros(1, device=obs_seq.device)
                trust_gate_ratio = torch.zeros(1, device=obs_seq.device)
            else:
                trust_score_mean = trust_values.mean()
                q = torch.tensor([0.1, 0.5, 0.9], device=trust_values.device, dtype=trust_values.dtype)
                qv = torch.quantile(trust_values, q)
                trust_score_p10 = qv[0]
                trust_score_p50 = qv[1]
                trust_score_p90 = qv[2]
                kept = (comm_mask > 0.5).squeeze(-1)[valid_mask].float().sum()
                trust_gate_ratio = kept / valid_mask.float().sum().clamp_min(1.0)
        fused = self.residual(self_feat, comm_feat, self.comm_enabled, self.comm_strength)
        logits = self.policy_head(fused)
        logits = self._mask_logits(logits, avail_actions)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)

        critic_feat, next_critic = self.critic_encoder(state.unsqueeze(1), rnn_states_critic.unsqueeze(0), masks)
        critic_feat = self.critic_mlp(critic_feat)
        values = self.value_head(critic_feat)
        return {
            "log_probs": log_probs,
            "entropy": entropy,
            "values": values,
            "kl_loss": kl_loss,
            "kl_raw": kl_raw,
            "trust_loss": trust_loss,
            "trust_score_mean": trust_score_mean,
            "trust_score_p10": trust_score_p10,
            "trust_score_p50": trust_score_p50,
            "trust_score_p90": trust_score_p90,
            "trust_gate_ratio": trust_gate_ratio,
            "comm_valid_neighbors": comm_valid_neighbors,
            "comm_kept_neighbors": comm_kept_neighbors,
            "comm_strength": torch.tensor(float(self.comm_strength), device=obs_seq.device),
            "comm_enabled": torch.tensor(float(self.comm_enabled), device=obs_seq.device),
            "next_actor_state": next_actor.squeeze(0),
            "next_critic_state": next_critic.squeeze(0),
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
