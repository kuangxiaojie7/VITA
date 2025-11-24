from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Any

import torch
import torch.nn.functional as F

from src.vita import VITAAgent, VITAAgentConfig
from src.utils import Logger, MAPPOBuffer


@dataclass
class VITATrainParams:
    episode_length: int
    updates: int
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    ppo_epochs: int = 5
    num_mini_batch: int = 4
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 10.0


class VITATrainer:
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
        self.history_length = policy_cfg.get("history_length", 4)
        self.train_cfg = VITATrainParams(**train_cfg)

        agent_cfg = VITAAgentConfig(
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=policy_cfg.get("hidden_dim", 128),
            latent_dim=policy_cfg.get("latent_dim", 64),
            trust_gamma=policy_cfg.get("trust_gamma", 1.0),
            kl_beta=policy_cfg.get("kl_beta", 1e-3),
            trust_lambda=policy_cfg.get("trust_lambda", 0.1),
            max_neighbors=self.max_neighbors,
        )
        self.agent = VITAAgent(agent_cfg).to(device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.train_cfg.lr, eps=1e-8)
        self.buffer = MAPPOBuffer(
            self.train_cfg.episode_length,
            self.num_envs,
            self.num_agents,
            self.obs_dim,
            self.state_dim,
            self.action_dim,
            self.max_neighbors,
            self.agent.rnn_hidden_dim,
            device,
            history_length=self.history_length,
        )
        self.obs_history: Deque[torch.Tensor] | None = None

    def train(self) -> None:
        obs_np, state_np = self.env.reset()
        obs = torch.from_numpy(obs_np).float().to(self.device)
        state = torch.from_numpy(state_np).float().to(self.device)
        actor_states = torch.zeros(self.num_envs, self.num_agents, self.agent.rnn_hidden_dim, device=self.device)
        critic_states = torch.zeros_like(actor_states)
        masks = torch.ones(self.num_envs, self.num_agents, 1, device=self.device)
        self._initialize_history(obs)

        for update in range(1, self.train_cfg.updates + 1):
            self.buffer.reset(obs, state, actor_states, critic_states)
            reward_sum = 0.0
            for step in range(self.train_cfg.episode_length):
                obs_seq = self._stack_history().contiguous()
                neighbor_indices = self._select_topk_neighbors(obs_seq)
                neighbor_seq = self._gather_neighbor_sequences(obs_seq, neighbor_indices).contiguous()
                flat_obs_seq = obs_seq.view(self.num_envs * self.num_agents, self.history_length, self.obs_dim)
                flat_state = (
                    state.unsqueeze(1)
                    .repeat(1, self.num_agents, 1)
                    .view(self.num_envs * self.num_agents, self.state_dim)
                )
                flat_neighbor_seq = neighbor_seq.view(
                    self.num_envs * self.num_agents, self.max_neighbors, self.history_length, self.obs_dim
                )
                flat_actor = actor_states.view(self.num_envs * self.num_agents, -1)
                flat_critic = critic_states.view(self.num_envs * self.num_agents, -1)
                flat_masks = masks.view(self.num_envs * self.num_agents, 1)

                outputs = self.agent.act(
                    flat_obs_seq,
                    flat_state,
                    flat_neighbor_seq,
                    None,
                    flat_actor,
                    flat_critic,
                    flat_masks,
                )
                actions = outputs["actions"]
                log_probs = outputs["log_probs"]
                values = outputs["values"]
                next_actor = outputs["next_actor_state"]
                next_critic = outputs["next_critic_state"]

                env_actions = actions.view(self.num_envs, self.num_agents).cpu().numpy()
                next_obs_np, next_state_np, reward_np, done_np, _ = self.env.step(env_actions)
                next_obs = torch.from_numpy(next_obs_np).float().to(self.device)
                next_state = torch.from_numpy(next_state_np).float().to(self.device)
                rewards = torch.from_numpy(reward_np).float().unsqueeze(-1).to(self.device)
                dones = torch.from_numpy(done_np.astype(float)).unsqueeze(-1).to(self.device)
                reward_sum += rewards.mean().item()

                next_actor_states = next_actor.view(self.num_envs, self.num_agents, -1)
                next_critic_states = next_critic.view(self.num_envs, self.num_agents, -1)
                reshaped_actions = actions.view(self.num_envs, self.num_agents, 1)
                reshaped_log_probs = log_probs.view(self.num_envs, self.num_agents, 1)
                reshaped_values = values.view(self.num_envs, self.num_agents, 1)
                action_one_hot = torch.zeros(
                    self.num_envs, self.num_agents, self.action_dim, device=self.device
                )
                action_one_hot.scatter_(2, reshaped_actions, 1.0)
                neighbor_action_tensor = self._gather_neighbor_actions(action_one_hot, neighbor_indices)

                latest_neighbor_obs = neighbor_seq[:, :, :, -1, :].contiguous()

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
                    latest_neighbor_obs,
                    neighbor_action_tensor,
                    obs_seq,
                    neighbor_seq,
                )

                obs = next_obs
                state = next_state
                actor_states = next_actor_states
                critic_states = next_critic_states
                masks = 1.0 - dones
                self._update_history(next_obs)

            flat_state = (
                state.unsqueeze(1)
                .repeat(1, self.num_agents, 1)
                .view(self.num_envs * self.num_agents, self.state_dim)
            )
            flat_critic = critic_states.view(self.num_envs * self.num_agents, -1)
            flat_masks = masks.view(self.num_envs * self.num_agents, 1)
            with torch.no_grad():
                next_values, _ = self.agent.get_values(flat_state, flat_critic, flat_masks)
            next_values = next_values.view(self.num_envs, self.num_agents, 1)
            self.buffer.compute_returns(next_values, self.train_cfg.gamma, self.train_cfg.gae_lambda)
            loss_dict = self.update_policy()

            log_payload = {
                "update": update,
                "episode_reward": reward_sum / self.train_cfg.episode_length,
                **loss_dict,
            }
            self.logger.log(log_payload, step=update)

    def update_policy(self) -> Dict[str, float]:
        advantages = self.buffer.advantages[:-1]
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-6
        norm_adv = (advantages - adv_mean) / adv_std

        policy_loss_epoch = 0.0
        value_loss_epoch = 0.0
        entropy_epoch = 0.0
        kl_epoch = 0.0
        trust_epoch = 0.0
        updates = 0
        for _ in range(self.train_cfg.ppo_epochs):
            for batch in self.buffer.mini_batch_generator(norm_adv, self.train_cfg.num_mini_batch):
                obs_seq_batch = batch["obs_seq"]
                state_batch = batch["states"]
                actions_batch = batch["actions"]
                old_log_probs_batch = batch["old_log_probs"]
                returns_batch = batch["returns"]
                advantages_batch = batch["advantages"]
                masks_batch = batch["masks"]
                rnn_actor_batch = batch["rnn_states_actor"]
                rnn_critic_batch = batch["rnn_states_critic"]
                neighbor_obs_seq_batch = batch["neighbor_obs_seq"]
                neighbor_actions_batch = batch["neighbor_actions"]
                eval_out = self.agent.evaluate_actions(
                    obs_seq_batch,
                    state_batch,
                    neighbor_obs_seq_batch,
                    neighbor_actions_batch,
                    actions_batch,
                    rnn_actor_batch,
                    rnn_critic_batch,
                    masks_batch,
                )

                log_probs = eval_out["log_probs"]
                entropy = eval_out["entropy"]
                values = eval_out["values"]
                kl_loss = eval_out["kl_loss"]
                trust_loss = eval_out["trust_loss"]

                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(
                    ratio, 1.0 - self.train_cfg.clip_param, 1.0 + self.train_cfg.clip_param
                ) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(returns_batch, values)
                entropy_loss = entropy.mean()

                total_loss = (
                    policy_loss
                    + self.train_cfg.value_loss_coef * value_loss
                    - self.train_cfg.entropy_coef * entropy_loss
                    + kl_loss
                    + self.agent.cfg.trust_lambda * trust_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.train_cfg.max_grad_norm)
                self.optimizer.step()

                policy_loss_epoch += policy_loss.item()
                value_loss_epoch += value_loss.item()
                entropy_epoch += entropy_loss.item()
                kl_epoch += kl_loss.item()
                trust_epoch += trust_loss.item()
                updates += 1

        self.buffer.after_update()
        denom = max(1, updates)
        return {
            "policy_loss": policy_loss_epoch / denom,
            "value_loss": value_loss_epoch / denom,
            "entropy": entropy_epoch / denom,
            "kl": kl_epoch / denom,
            "trust_loss": trust_epoch / denom,
        }

    def _initialize_history(self, obs: torch.Tensor) -> None:
        self.obs_history = deque(
            [obs.clone() for _ in range(self.history_length)], maxlen=self.history_length
        )

    def _update_history(self, obs: torch.Tensor) -> None:
        assert self.obs_history is not None
        self.obs_history.append(obs.clone())

    def _stack_history(self) -> torch.Tensor:
        assert self.obs_history is not None
        history = torch.stack(list(self.obs_history), dim=0)  # [T, envs, agents, obs_dim]
        return history.permute(1, 2, 0, 3).contiguous()

    def _select_topk_neighbors(self, obs_seq: torch.Tensor) -> torch.Tensor:
        # obs_seq: [envs, agents, history, obs_dim]
        latest = obs_seq[:, :, -1, :]
        if self.num_agents == 1:
            return torch.zeros(obs_seq.size(0), 1, 1, dtype=torch.long, device=obs_seq.device)
        dist = torch.cdist(latest, latest, p=2)
        mask = torch.eye(self.num_agents, device=dist.device).unsqueeze(0)
        dist = dist + mask * 1e9
        topk = torch.topk(dist, k=self.max_neighbors, dim=-1, largest=False).indices
        return topk

    def _gather_neighbor_sequences(
        self, obs_seq: torch.Tensor, neighbor_idx: torch.Tensor
    ) -> torch.Tensor:
        # obs_seq: [envs, agents, history, obs_dim]
        envs, agents, history_len, obs_dim = obs_seq.shape
        expanded = obs_seq.unsqueeze(1).expand(envs, agents, agents, history_len, obs_dim)
        idx = (
            neighbor_idx.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(envs, agents, self.max_neighbors, history_len, obs_dim)
        )
        gathered = torch.gather(expanded, 2, idx)
        return gathered.contiguous()

    def _gather_neighbor_actions(
        self, action_tensor: torch.Tensor, neighbor_idx: torch.Tensor
    ) -> torch.Tensor:
        # action_tensor: [envs, agents, action_dim]
        envs, agents, action_dim = action_tensor.shape
        expanded = action_tensor.unsqueeze(1).expand(envs, agents, agents, action_dim)
        idx = neighbor_idx.unsqueeze(-1).expand(envs, agents, self.max_neighbors, action_dim)
        gathered = torch.gather(expanded, 2, idx)
        return gathered.contiguous()
