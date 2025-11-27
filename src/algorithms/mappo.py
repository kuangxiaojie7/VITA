from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn.functional as F

from src.models import PolicyConfig, RecurrentMAPPOPolicy
from src.utils import Logger, MAPPOBuffer


@dataclass
class TrainParams:
    episode_length: int
    updates: int
    lr: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    ppo_epochs: int = 5
    num_mini_batch: int = 4
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 10.0


class MAPPOTrainer:
    def __init__(
        self,
        env,
        policy_cfg: Dict[str, Any],
        train_cfg: Dict[str, Any],
        logger: Logger,
        device: torch.device,
    ):
        self.env = env
        self.logger = logger
        self.device = device
        self.num_envs = env.cfg.num_envs
        self.num_agents = env.n_agents
        self.obs_dim = env.obs_dim
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.max_neighbors = min(policy_cfg.get("max_neighbors", self.num_agents - 1), self.num_agents - 1)
        self.max_neighbors = max(1, self.max_neighbors)
        self.history_length = policy_cfg.get("history_length", 1)
        self.train_cfg = TrainParams(**train_cfg)

        policy_config = PolicyConfig(
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=policy_cfg.get("hidden_dim", 128),
            activation=policy_cfg.get("activation", "relu"),
        )
        self.policy = RecurrentMAPPOPolicy(policy_config).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.train_cfg.lr, eps=1e-8)
        self.buffer = MAPPOBuffer(
            self.train_cfg.episode_length,
            self.num_envs,
            self.num_agents,
            self.obs_dim,
            self.state_dim,
            self.action_dim,
            self.max_neighbors,
            self.policy.rnn_hidden_dim,
            device,
            history_length=self.history_length,
        )

    def train(self) -> None:
        obs_np, state_np, avail_np = self.env.reset()
        obs = torch.from_numpy(obs_np).to(self.device).float()
        state = torch.from_numpy(state_np).to(self.device).float()
        avail_actions = torch.from_numpy(avail_np).to(self.device).float()
        actor_states = torch.zeros(self.num_envs, self.num_agents, self.policy.rnn_hidden_dim, device=self.device)
        critic_states = torch.zeros_like(actor_states)
        masks = torch.ones(self.num_envs, self.num_agents, 1, device=self.device)

        for update in range(1, self.train_cfg.updates + 1):
            self.buffer.reset(obs, state, actor_states, critic_states)
            episode_rewards = 0.0
            for step in range(self.train_cfg.episode_length):
                obs = obs.float()
                state = state.float()
                avail_actions = avail_actions.float()
                neighbor_obs_tensor = self._build_neighbor_tensor(obs).float()
                flat_obs = obs.view(self.num_envs * self.num_agents, self.obs_dim).float()
                flat_state = (
                    state.unsqueeze(1)
                    .repeat(1, self.num_agents, 1)
                    .view(self.num_envs * self.num_agents, self.state_dim)
                )
                flat_state = flat_state.float()
                flat_actor = actor_states.view(self.num_envs * self.num_agents, -1).float()
                flat_critic = critic_states.view(self.num_envs * self.num_agents, -1).float()
                flat_masks = masks.view(self.num_envs * self.num_agents, 1).float()
                flat_avail = avail_actions.view(self.num_envs * self.num_agents, self.action_dim).float()

                raw_actions, log_probs, values, entropy, next_actor, next_critic = self.policy.act(
                    flat_obs, flat_state, flat_actor, flat_critic, flat_masks, flat_avail
                )
                actions = self._ensure_valid_actions(raw_actions, avail_actions)

                env_actions = actions.view(self.num_envs, self.num_agents).cpu().numpy()
                next_obs_np, next_state_np, reward_np, done_np, next_avail_np, _ = self.env.step(env_actions)
                next_obs = torch.from_numpy(next_obs_np).float().to(self.device)
                next_state = torch.from_numpy(next_state_np).float().to(self.device)
                rewards = torch.from_numpy(reward_np).float().unsqueeze(-1).to(self.device)
                dones = torch.from_numpy(done_np.astype(float)).unsqueeze(-1).to(self.device)
                next_avail = torch.from_numpy(next_avail_np).float().to(self.device)

                episode_rewards += rewards.mean().item()

                next_actor_states = next_actor.view(self.num_envs, self.num_agents, -1).detach()
                next_critic_states = next_critic.view(self.num_envs, self.num_agents, -1).detach()
                reshaped_actions = actions.view(self.num_envs, self.num_agents, 1).detach()
                reshaped_log_probs = log_probs.view(self.num_envs, self.num_agents, 1).detach()
                reshaped_values = values.view(self.num_envs, self.num_agents, 1).detach()
                action_one_hot = torch.zeros(
                    self.num_envs, self.num_agents, self.action_dim, device=self.device, dtype=torch.float32
                )
                action_one_hot.scatter_(2, reshaped_actions, 1.0)
                neighbor_action_tensor = self._build_neighbor_tensor(action_one_hot)

                self.buffer.insert(
                    next_obs,
                    next_state,
                    next_actor_states,
                    next_critic_states,
                    reshaped_actions,
                    reshaped_log_probs,
                    reshaped_values,
                    rewards,
                    dones,
                    neighbor_obs_tensor,
                    neighbor_action_tensor,
                    obs.unsqueeze(2),
                    neighbor_obs_tensor.unsqueeze(-2),
                    avail_actions,
                )

                obs = next_obs
                state = next_state
                actor_states = next_actor_states
                critic_states = next_critic_states
                masks = 1.0 - dones
                avail_actions = next_avail

            flat_state = (
                state.unsqueeze(1)
                .repeat(1, self.num_agents, 1)
                .view(self.num_envs * self.num_agents, self.state_dim)
            )
            flat_state = flat_state.float()
            flat_critic = critic_states.view(self.num_envs * self.num_agents, -1).float()
            flat_masks = masks.view(self.num_envs * self.num_agents, 1).float()
            with torch.no_grad():
                next_values, next_critic = self.policy.get_values(flat_state, flat_critic, flat_masks)
            next_values = next_values.view(self.num_envs, self.num_agents, 1)
            self.buffer.compute_returns(next_values, self.train_cfg.gamma, self.train_cfg.gae_lambda)
            loss_dict = self.update_policy()

            log_data = {
                "update": update,
                "episode_reward": episode_rewards / self.train_cfg.episode_length,
            }
            log_data.update(loss_dict)
            self.logger.log(log_data, step=update)
            print(f"[MAPPO] update {update} completed")

    def update_policy(self) -> Dict[str, float]:
        advantages = self.buffer.advantages[:-1]
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-5
        norm_adv = (advantages - adv_mean) / adv_std

        value_loss_epoch = 0.0
        policy_loss_epoch = 0.0
        entropy_epoch = 0.0
        num_updates = 0
        for _ in range(self.train_cfg.ppo_epochs):
            for batch in self.buffer.mini_batch_generator(norm_adv, self.train_cfg.num_mini_batch):
                obs_batch = batch["obs"]
                state_batch = batch["states"]
                actions_batch = batch["actions"]
                old_log_probs_batch = batch["old_log_probs"]
                returns_batch = batch["returns"]
                advantages_batch = batch["advantages"]
                masks_batch = batch["masks"]
                rnn_actor_batch = batch["rnn_states_actor"]
                rnn_critic_batch = batch["rnn_states_critic"]
                avail_batch = batch["avail_actions"]
                log_probs, entropy, values = self.policy.evaluate_actions(
                    obs_batch.float(),
                    state_batch.float(),
                    actions_batch,
                    rnn_actor_batch.float(),
                    rnn_critic_batch.float(),
                    masks_batch.float(),
                    avail_batch.float(),
                )
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - self.train_cfg.clip_param, 1.0 + self.train_cfg.clip_param) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(returns_batch, values)
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.train_cfg.value_loss_coef * value_loss
                    - self.train_cfg.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.train_cfg.max_grad_norm)
                self.optimizer.step()

                policy_loss_epoch += policy_loss.item()
                value_loss_epoch += value_loss.item()
                entropy_epoch += entropy_loss.item()
                num_updates += 1

        self.buffer.after_update()
        return {
            "policy_loss": policy_loss_epoch / max(1, num_updates),
            "value_loss": value_loss_epoch / max(1, num_updates),
            "entropy": entropy_epoch / max(1, num_updates),
        }

    def _build_neighbor_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Create padded neighbor view for each agent."""
        neighbors = []
        num_envs = tensor.size(0)
        feature_dim = tensor.size(-1)
        for agent_idx in range(self.num_agents):
            mask = torch.ones(self.num_agents, dtype=torch.bool, device=tensor.device)
            mask[agent_idx] = False
            neigh = tensor[:, mask, :]
            neigh = neigh[:, : self.max_neighbors, :]
            if neigh.shape[1] < self.max_neighbors:
                pad = torch.zeros(
                    num_envs, self.max_neighbors - neigh.shape[1], feature_dim, device=tensor.device
                )
                neigh = torch.cat([neigh, pad], dim=1)
            neighbors.append(neigh.unsqueeze(1))
        return torch.cat(neighbors, dim=1)

    def _ensure_valid_actions(self, actions: torch.Tensor, avail_actions: torch.Tensor) -> torch.Tensor:
        reshaped = actions.view(self.num_envs, self.num_agents, -1).clone()
        avail = avail_actions.view(self.num_envs, self.num_agents, self.action_dim)
        for env_idx in range(self.num_envs):
            for agent_idx in range(self.num_agents):
                act = reshaped[env_idx, agent_idx, 0].long()
                if avail[env_idx, agent_idx, act] < 0.5:
                    valid = torch.nonzero(avail[env_idx, agent_idx] > 0.5, as_tuple=False)
                    if valid.numel() == 0:
                        act = torch.tensor(0, device=actions.device)
                    else:
                        act = valid[0, 0]
                    reshaped[env_idx, agent_idx, 0] = act
        return reshaped.view(-1, 1)
