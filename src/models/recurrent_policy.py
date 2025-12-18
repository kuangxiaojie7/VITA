from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass
class PolicyConfig:
    obs_dim: int
    state_dim: int
    action_dim: int
    hidden_dim: int = 128
    activation: str = "relu"


def _make_mlp(input_dim: int, hidden_dim: int, activation: str) -> nn.Sequential:
    act_cls = {"relu": nn.ReLU, "tanh": nn.Tanh}.get(activation, nn.ReLU)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        act_cls(),
        nn.Linear(hidden_dim, hidden_dim),
        act_cls(),
    )


class RecurrentMAPPOPolicy(nn.Module):
    """Recurrent actor-critic shared across agents."""

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.actor_base = _make_mlp(cfg.obs_dim, cfg.hidden_dim, cfg.activation)
        self.critic_base = _make_mlp(cfg.state_dim, cfg.hidden_dim, cfg.activation)
        self.gru_actor = nn.GRUCell(cfg.hidden_dim, cfg.hidden_dim)
        self.gru_critic = nn.GRUCell(cfg.hidden_dim, cfg.hidden_dim)
        self.actor_head = nn.Linear(cfg.hidden_dim, cfg.action_dim)
        self.value_head = nn.Linear(cfg.hidden_dim, 1)

    @property
    def rnn_hidden_dim(self) -> int:
        return self.cfg.hidden_dim

    def _actor_forward(
        self, obs: torch.Tensor, rnn_states: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.actor_base(obs)
        rnn_states = rnn_states * masks
        hidden = self.gru_actor(x, rnn_states)
        logits = self.actor_head(hidden)
        return logits, hidden

    def _critic_forward(
        self, state: torch.Tensor, rnn_states: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.critic_base(state)
        rnn_states = rnn_states * masks
        hidden = self.gru_critic(x, rnn_states)
        value = self.value_head(hidden)
        return value, hidden

    def get_initial_states(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.cfg.hidden_dim)

    def _mask_logits(self, logits: torch.Tensor, avail_actions: torch.Tensor) -> torch.Tensor:
        if avail_actions is None:
            return logits
        mask = (avail_actions < 0.5)
        all_masked = mask.all(dim=-1, keepdim=True)
        if all_masked.any():
            mask = mask & (~all_masked)
        return logits.masked_fill(mask, -1e9)

    def act(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        rnn_states_actor: torch.Tensor,
        rnn_states_critic: torch.Tensor,
        masks: torch.Tensor,
        avail_actions: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, next_actor = self._actor_forward(obs, rnn_states_actor, masks)
        logits = self._mask_logits(logits, avail_actions)
        dist = Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            actions = dist.sample().unsqueeze(-1)
        log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        values, next_critic = self._critic_forward(state, rnn_states_critic, masks)
        return actions, log_probs, values, entropy, next_actor, next_critic

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        rnn_states_actor: torch.Tensor,
        rnn_states_critic: torch.Tensor,
        masks: torch.Tensor,
        avail_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, _ = self._actor_forward(obs, rnn_states_actor, masks)
        logits = self._mask_logits(logits, avail_actions)
        dist = Categorical(logits=logits)
        action_log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        dist_entropy = dist.entropy().unsqueeze(-1)
        values, _ = self._critic_forward(state, rnn_states_critic, masks)
        return action_log_probs, dist_entropy, values

    def evaluate_actions_sequence(
        self,
        obs_seq: torch.Tensor,
        state_seq: torch.Tensor,
        actions_seq: torch.Tensor,
        rnn_states_actor: torch.Tensor,
        rnn_states_critic: torch.Tensor,
        masks_seq: torch.Tensor,
        avail_actions_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sequence-aware evaluate_actions for truncated BPTT."""
        log_probs_steps = []
        entropy_steps = []
        values_steps = []
        actor_state = rnn_states_actor
        critic_state = rnn_states_critic
        for t in range(obs_seq.size(0)):
            logits, actor_state = self._actor_forward(obs_seq[t], actor_state, masks_seq[t])
            logits = self._mask_logits(logits, avail_actions_seq[t])
            dist = Categorical(logits=logits)
            log_probs_steps.append(dist.log_prob(actions_seq[t].squeeze(-1)).unsqueeze(-1))
            entropy_steps.append(dist.entropy().unsqueeze(-1))
            values, critic_state = self._critic_forward(state_seq[t], critic_state, masks_seq[t])
            values_steps.append(values)
        return (
            torch.stack(log_probs_steps, dim=0),
            torch.stack(entropy_steps, dim=0),
            torch.stack(values_steps, dim=0),
        )

    def get_values(
        self, state: torch.Tensor, rnn_states_critic: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._critic_forward(state, rnn_states_critic, masks)
