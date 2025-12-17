from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator

import torch


@dataclass
class EpisodeBatch:
    obs: torch.Tensor
    obs_seq: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    masks: torch.Tensor
    active_masks: torch.Tensor
    rewards: torch.Tensor
    neighbor_obs: torch.Tensor
    neighbor_actions: torch.Tensor
    neighbor_obs_seq: torch.Tensor
    neighbor_masks: torch.Tensor
    avail_actions: torch.Tensor


class MAPPOBuffer:
    """Rollout buffer tailored for MAPPO/VITA trainers."""

    def __init__(
        self,
        episode_length: int,
        num_envs: int,
        num_agents: int,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        max_neighbors: int,
        rnn_hidden_dim: int,
        device: torch.device,
        history_length: int = 1,
    ):
        self.episode_length = episode_length
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_neighbors = max_neighbors
        self.rnn_hidden_dim = rnn_hidden_dim
        self.device = device
        self.history_length = max(1, history_length)

        self.obs = torch.zeros(episode_length + 1, num_envs, num_agents, obs_dim, device=device)
        self.states = torch.zeros(episode_length + 1, num_envs, state_dim, device=device)
        self.rnn_states_actor = torch.zeros(
            episode_length + 1, num_envs, num_agents, rnn_hidden_dim, device=device
        )
        self.rnn_states_critic = torch.zeros_like(self.rnn_states_actor)
        self.actions = torch.zeros(episode_length, num_envs, num_agents, 1, device=device, dtype=torch.long)
        self.log_probs = torch.zeros(episode_length, num_envs, num_agents, 1, device=device)
        self.values = torch.zeros(episode_length + 1, num_envs, num_agents, 1, device=device)
        self.rewards = torch.zeros(episode_length, num_envs, num_agents, 1, device=device)
        self.masks = torch.ones(episode_length + 1, num_envs, num_agents, 1, device=device)
        self.active_masks = torch.ones(episode_length + 1, num_envs, num_agents, 1, device=device)
        self.returns = torch.zeros_like(self.values)
        self.advantages = torch.zeros_like(self.values)
        self.neighbor_obs = torch.zeros(
            episode_length, num_envs, num_agents, max_neighbors, obs_dim, device=device
        )
        self.neighbor_actions = torch.zeros(
            episode_length, num_envs, num_agents, max_neighbors, action_dim, device=device
        )
        self.neighbor_masks = torch.ones(
            episode_length, num_envs, num_agents, max_neighbors, 1, device=device
        )
        self.avail_actions = torch.zeros(
            episode_length, num_envs, num_agents, action_dim, device=device
        )
        self.obs_sequences = torch.zeros(
            episode_length,
            num_envs,
            num_agents,
            self.history_length,
            obs_dim,
            device=device,
        )
        self.neighbor_obs_sequences = torch.zeros(
            episode_length,
            num_envs,
            num_agents,
            max_neighbors,
            self.history_length,
            obs_dim,
            device=device,
        )
        self.step = 0

    def reset(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        rnn_states_actor: torch.Tensor,
        rnn_states_critic: torch.Tensor,
        active_masks: torch.Tensor | None = None,
    ) -> None:
        self.obs[0].copy_(obs)
        self.states[0].copy_(state)
        self.rnn_states_actor[0].copy_(rnn_states_actor)
        self.rnn_states_critic[0].copy_(rnn_states_critic)
        if active_masks is not None:
            self.active_masks[0].copy_(active_masks)
        self.step = 0

    def insert(
        self,
        next_obs: torch.Tensor,
        next_state: torch.Tensor,
        rnn_states_actor: torch.Tensor,
        rnn_states_critic: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        active_masks: torch.Tensor | None,
        neighbor_obs: torch.Tensor,
        neighbor_actions: torch.Tensor,
        obs_seq: torch.Tensor,
        neighbor_obs_seq: torch.Tensor,
        avail_actions: torch.Tensor,
        neighbor_masks: torch.Tensor | None = None,
    ) -> None:
        self.obs[self.step + 1].copy_(next_obs)
        self.states[self.step + 1].copy_(next_state)
        self.rnn_states_actor[self.step + 1].copy_(rnn_states_actor)
        self.rnn_states_critic[self.step + 1].copy_(rnn_states_critic)
        self.actions[self.step].copy_(actions)
        self.log_probs[self.step].copy_(log_probs)
        self.values[self.step].copy_(values)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(1.0 - dones)
        if active_masks is not None:
            self.active_masks[self.step + 1].copy_(active_masks)
        if neighbor_masks is None:
            neighbor_masks = torch.ones_like(self.neighbor_masks[self.step])
        self.neighbor_obs[self.step].copy_(neighbor_obs)
        self.neighbor_actions[self.step].copy_(neighbor_actions)
        self.neighbor_masks[self.step].copy_(neighbor_masks)
        self.obs_sequences[self.step].copy_(obs_seq)
        self.neighbor_obs_sequences[self.step].copy_(neighbor_obs_seq)
        self.avail_actions[self.step].copy_(avail_actions)
        self.step = (self.step + 1) % self.episode_length

    def compute_returns(self, last_values: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        self.values[-1] = last_values
        gae = torch.zeros_like(last_values)
        for step in reversed(range(self.episode_length)):
            delta = (
                self.rewards[step]
                + gamma * self.values[step + 1] * self.masks[step + 1]
                - self.values[step]
            )
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.advantages[step] = gae
        self.advantages[-1].zero_()
        self.returns[:-1] = self.advantages[:-1] + self.values[:-1]
        self.returns[-1].copy_(self.values[-1])

    def after_update(self) -> None:
        self.obs[0].copy_(self.obs[-1])
        self.states[0].copy_(self.states[-1])
        self.rnn_states_actor[0].copy_(self.rnn_states_actor[-1])
        self.rnn_states_critic[0].copy_(self.rnn_states_critic[-1])
        self.masks[0].copy_(self.masks[-1])
        self.active_masks[0].copy_(self.active_masks[-1])

    def get_episode_batch(self) -> EpisodeBatch:
        return EpisodeBatch(
            obs=self.obs[:-1].clone(),
            obs_seq=self.obs_sequences.clone(),
            states=self.states[:-1].clone(),
            actions=self.actions.clone(),
            log_probs=self.log_probs.clone(),
            values=self.values[:-1].clone(),
            returns=self.returns[:-1].clone(),
            advantages=self.advantages[:-1].clone(),
            masks=self.masks[:-1].clone(),
            active_masks=self.active_masks[:-1].clone(),
            rewards=self.rewards.clone(),
            neighbor_obs=self.neighbor_obs.clone(),
            neighbor_actions=self.neighbor_actions.clone(),
            neighbor_obs_seq=self.neighbor_obs_sequences.clone(),
            neighbor_masks=self.neighbor_masks.clone(),
            avail_actions=self.avail_actions.clone(),
        )

    def mini_batch_generator(
        self, advantages: torch.Tensor, num_mini_batch: int
    ) -> Iterator[Dict[str, torch.Tensor]]:
        episode_length, num_envs, num_agents, _ = self.actions.shape
        batch_size = episode_length * num_envs * num_agents
        indices = torch.randperm(batch_size, device=self.device)
        mini_batch_size = max(1, batch_size // max(1, num_mini_batch))

        obs = self.obs[:-1].reshape(-1, self.obs_dim)
        states = self.states[:-1].repeat_interleave(self.num_agents, dim=1).reshape(-1, self.state_dim)
        actions = self.actions.reshape(-1, 1)
        old_log_probs = self.log_probs.reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        advantages = advantages.reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        neighbor_obs = self.neighbor_obs.reshape(
            episode_length * num_envs * num_agents, self.max_neighbors, self.obs_dim
        )
        neighbor_actions = self.neighbor_actions.reshape(
            episode_length * num_envs * num_agents, self.max_neighbors, self.action_dim
        )
        neighbor_masks = self.neighbor_masks.reshape(
            episode_length * num_envs * num_agents, self.max_neighbors, 1
        )
        avail_actions = self.avail_actions.reshape(
            episode_length * num_envs * num_agents, self.action_dim
        )
        obs_seq = self.obs_sequences.reshape(
            episode_length * num_envs * num_agents, self.history_length, self.obs_dim
        )
        neighbor_obs_seq = self.neighbor_obs_sequences.reshape(
            episode_length * num_envs * num_agents,
            self.max_neighbors,
            self.history_length,
            self.obs_dim,
        )

        rnn_states_actor = self.rnn_states_actor[:-1].reshape(-1, self.rnn_hidden_dim)
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, self.rnn_hidden_dim)

        for start in range(0, batch_size, mini_batch_size):
            end = min(start + mini_batch_size, batch_size)
            mb_idx = indices[start:end]
            yield {
                "obs": obs[mb_idx],
                "obs_seq": obs_seq[mb_idx],
                "states": states[mb_idx],
                "actions": actions[mb_idx],
                "old_log_probs": old_log_probs[mb_idx],
                "returns": returns[mb_idx],
                "advantages": advantages[mb_idx],
                "masks": masks[mb_idx],
                "active_masks": active_masks[mb_idx],
                "rnn_states_actor": rnn_states_actor[mb_idx],
                "rnn_states_critic": rnn_states_critic[mb_idx],
                "neighbor_obs": neighbor_obs[mb_idx],
                "neighbor_actions": neighbor_actions[mb_idx],
                "neighbor_obs_seq": neighbor_obs_seq[mb_idx],
                "neighbor_masks": neighbor_masks[mb_idx],
                "avail_actions": avail_actions[mb_idx],
            }

    def recurrent_generator(
        self, advantages: torch.Tensor, num_mini_batch: int, data_chunk_length: int
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield contiguous fixed-length sequences for truncated BPTT (on-policy style)."""
        episode_length, num_envs, num_agents, _ = self.actions.shape
        seq_len = max(1, int(data_chunk_length))
        num_mini_batch = max(1, int(num_mini_batch))

        if seq_len <= 1:
            yield from self.mini_batch_generator(advantages, num_mini_batch)
            return

        usable_len = (episode_length // seq_len) * seq_len
        if usable_len < seq_len:
            raise ValueError(
                f"episode_length ({episode_length}) must be >= data_chunk_length ({seq_len})."
            )
        chunks_per_thread = usable_len // seq_len
        threads = num_envs * num_agents
        total_chunks = threads * chunks_per_thread
        if total_chunks < num_mini_batch:
            raise ValueError(
                f"Not enough chunks ({total_chunks}) for num_mini_batch ({num_mini_batch})."
            )
        mini_batch_size = max(1, total_chunks // num_mini_batch)
        perm = torch.randperm(total_chunks, device=self.device)[: mini_batch_size * num_mini_batch]

        obs_steps = self.obs[:-1][:usable_len].reshape(usable_len, threads, self.obs_dim)
        obs_seq_steps = self.obs_sequences[:usable_len].reshape(
            usable_len, threads, self.history_length, self.obs_dim
        )
        state_steps = (
            self.states[:-1][:usable_len]
            .repeat_interleave(num_agents, dim=1)
            .reshape(usable_len, threads, self.state_dim)
        )
        actions_steps = self.actions[:usable_len].reshape(usable_len, threads, 1)
        logp_steps = self.log_probs[:usable_len].reshape(usable_len, threads, 1)
        returns_steps = self.returns[:-1][:usable_len].reshape(usable_len, threads, 1)
        adv_steps = advantages[:usable_len].reshape(usable_len, threads, 1)
        masks_steps = self.masks[:-1][:usable_len].reshape(usable_len, threads, 1)
        active_steps = self.active_masks[:-1][:usable_len].reshape(usable_len, threads, 1)
        avail_steps = self.avail_actions[:usable_len].reshape(usable_len, threads, self.action_dim)
        neighbor_obs_steps = self.neighbor_obs[:usable_len].reshape(
            usable_len, threads, self.max_neighbors, self.obs_dim
        )
        neighbor_act_steps = self.neighbor_actions[:usable_len].reshape(
            usable_len, threads, self.max_neighbors, self.action_dim
        )
        neighbor_mask_steps = self.neighbor_masks[:usable_len].reshape(
            usable_len, threads, self.max_neighbors, 1
        )
        neighbor_obs_seq_steps = self.neighbor_obs_sequences[:usable_len].reshape(
            usable_len, threads, self.max_neighbors, self.history_length, self.obs_dim
        )

        rnn_actor_steps = self.rnn_states_actor[:-1][:usable_len].reshape(
            usable_len, threads, self.rnn_hidden_dim
        )
        rnn_critic_steps = self.rnn_states_critic[:-1][:usable_len].reshape(
            usable_len, threads, self.rnn_hidden_dim
        )

        def _make_chunks(x: torch.Tensor) -> torch.Tensor:
            # x: [T, threads, ...] -> [chunks, seq_len, ...]
            chunked = x.view(chunks_per_thread, seq_len, threads, *x.shape[2:])
            chunked = chunked.permute(2, 0, 1, *range(3, chunked.dim())).contiguous()
            return chunked.view(total_chunks, seq_len, *x.shape[2:])

        obs_chunks = _make_chunks(obs_steps)
        obs_seq_chunks = _make_chunks(obs_seq_steps)
        state_chunks = _make_chunks(state_steps)
        actions_chunks = _make_chunks(actions_steps)
        logp_chunks = _make_chunks(logp_steps)
        returns_chunks = _make_chunks(returns_steps)
        adv_chunks = _make_chunks(adv_steps)
        masks_chunks = _make_chunks(masks_steps)
        active_chunks = _make_chunks(active_steps)
        avail_chunks = _make_chunks(avail_steps)
        neighbor_obs_chunks = _make_chunks(neighbor_obs_steps)
        neighbor_act_chunks = _make_chunks(neighbor_act_steps)
        neighbor_mask_chunks = _make_chunks(neighbor_mask_steps)
        neighbor_obs_seq_chunks = _make_chunks(neighbor_obs_seq_steps)

        rnn_actor_starts = rnn_actor_steps[0:usable_len:seq_len].permute(1, 0, 2).contiguous()
        rnn_actor_starts = rnn_actor_starts.view(total_chunks, self.rnn_hidden_dim)
        rnn_critic_starts = rnn_critic_steps[0:usable_len:seq_len].permute(1, 0, 2).contiguous()
        rnn_critic_starts = rnn_critic_starts.view(total_chunks, self.rnn_hidden_dim)

        for mb in range(num_mini_batch):
            start = mb * mini_batch_size
            end = start + mini_batch_size
            chunk_idx = perm[start:end]
            yield {
                "seq_len": seq_len,
                "obs": obs_chunks[chunk_idx].transpose(0, 1).contiguous(),
                "obs_seq": obs_seq_chunks[chunk_idx].transpose(0, 1).contiguous(),
                "states": state_chunks[chunk_idx].transpose(0, 1).contiguous(),
                "actions": actions_chunks[chunk_idx].transpose(0, 1).contiguous(),
                "old_log_probs": logp_chunks[chunk_idx].transpose(0, 1).contiguous(),
                "returns": returns_chunks[chunk_idx].transpose(0, 1).contiguous(),
                "advantages": adv_chunks[chunk_idx].transpose(0, 1).contiguous(),
                "masks": masks_chunks[chunk_idx].transpose(0, 1).contiguous(),
                "active_masks": active_chunks[chunk_idx].transpose(0, 1).contiguous(),
                "avail_actions": avail_chunks[chunk_idx].transpose(0, 1).contiguous(),
                "neighbor_obs": neighbor_obs_chunks[chunk_idx].transpose(0, 1).contiguous(),
                "neighbor_actions": neighbor_act_chunks[chunk_idx].transpose(0, 1).contiguous(),
                "neighbor_obs_seq": neighbor_obs_seq_chunks[chunk_idx].transpose(0, 1).contiguous(),
                "neighbor_masks": neighbor_mask_chunks[chunk_idx].transpose(0, 1).contiguous(),
                "rnn_states_actor": rnn_actor_starts[chunk_idx].contiguous(),
                "rnn_states_critic": rnn_critic_starts[chunk_idx].contiguous(),
            }
