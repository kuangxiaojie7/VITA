from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from src.models import PolicyConfig, RecurrentMAPPOPolicy
from src.utils import Logger, MAPPOBuffer, RunningMeanStd
from src.envs import make_smac_env


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
    data_chunk_length: int = 10
    use_recurrent_generator: bool = True
    use_policy_active_masks: bool = True
    use_value_active_masks: bool = True
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 10.0
    eval_interval_updates: int = 0
    eval_episodes: int = 0


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
        self.reward_norm = RunningMeanStd()
        self.value_norm = RunningMeanStd()

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
        self._completed_episodes = 0
        self._won_episodes = 0
        self.eval_env = None
        if self.train_cfg.eval_interval_updates > 0 and self.train_cfg.eval_episodes > 0:
            eval_cfg = asdict(self.env.cfg)
            eval_cfg["num_envs"] = 1
            self.eval_env = make_smac_env(eval_cfg)

    def train(self) -> None:
        obs_np, state_np, avail_np = self.env.reset()
        obs = torch.from_numpy(obs_np).to(self.device).float()
        state = torch.from_numpy(state_np).to(self.device).float()
        avail_actions = torch.from_numpy(avail_np).to(self.device).float()
        actor_states = torch.zeros(self.num_envs, self.num_agents, self.policy.rnn_hidden_dim, device=self.device)
        critic_states = torch.zeros_like(actor_states)
        masks = torch.ones(self.num_envs, self.num_agents, 1, device=self.device)
        active_masks = (
            torch.from_numpy(self.env.get_agent_alive_mask()).float().to(self.device).unsqueeze(-1)
        )

        try:
            for update in range(1, self.train_cfg.updates + 1):
                self.buffer.reset(obs, state, actor_states, critic_states, active_masks=active_masks)
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
                    ).float()
                    flat_actor = actor_states.view(self.num_envs * self.num_agents, -1).float()
                    flat_critic = critic_states.view(self.num_envs * self.num_agents, -1).float()
                    flat_masks = masks.view(self.num_envs * self.num_agents, 1).float()
                    flat_avail = avail_actions.view(self.num_envs * self.num_agents, self.action_dim).float()

                    raw_actions, log_probs, values, entropy, next_actor, next_critic = self.policy.act(
                        flat_obs, flat_state, flat_actor, flat_critic, flat_masks, flat_avail
                    )
                    actions = self._ensure_valid_actions(raw_actions, avail_actions)

                    env_actions = actions.view(self.num_envs, self.num_agents).cpu().numpy()
                    next_obs_np, next_state_np, reward_np, done_np, next_avail_np, info_list = self.env.step(env_actions)
                    next_obs = torch.from_numpy(next_obs_np).float().to(self.device)
                    next_state = torch.from_numpy(next_state_np).float().to(self.device)
                    raw_rewards = torch.from_numpy(reward_np).float().unsqueeze(-1).to(self.device)
                    self.reward_norm.update(reward_np)
                    rewards = self.reward_norm.normalize(raw_rewards, use_mean=False)
                    dones = torch.from_numpy(done_np.astype(float)).unsqueeze(-1).to(self.device)
                    next_avail = torch.from_numpy(next_avail_np).float().to(self.device)
                    next_active_masks = (
                        torch.from_numpy(self.env.get_agent_alive_mask()).float().to(self.device).unsqueeze(-1)
                    )

                    episode_rewards += raw_rewards.mean().item()

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
                        next_active_masks,
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
                    active_masks = next_active_masks

                flat_state = (
                    state.unsqueeze(1)
                    .repeat(1, self.num_agents, 1)
                    .view(self.num_envs * self.num_agents, self.state_dim)
                ).float()
                flat_critic = critic_states.view(self.num_envs * self.num_agents, -1).float()
                flat_masks = masks.view(self.num_envs * self.num_agents, 1).float()
                with torch.no_grad():
                    next_values, _ = self.policy.get_values(flat_state, flat_critic, flat_masks)
                next_values = next_values.view(self.num_envs, self.num_agents, 1)
                denorm_next_values = self.value_norm.denormalize(next_values, use_mean=True)
                self.buffer.values.copy_(self.value_norm.denormalize(self.buffer.values, use_mean=True))
                self.buffer.compute_returns(denorm_next_values, self.train_cfg.gamma, self.train_cfg.gae_lambda)
                loss_dict = self.update_policy()

                eval_metrics = {}
                if self._should_run_eval(update):
                    eval_metrics = self._run_evaluation()

                log_data = {
                    "update": update,
                    "episode_reward": episode_rewards / self.train_cfg.episode_length,
                }
                log_data.update(loss_dict)
                log_data.update(eval_metrics)
                self.logger.log(log_data, step=update)
                print(f"[MAPPO] update {update} completed")
        finally:
            self._close_eval_env()

    def update_policy(self) -> Dict[str, float]:
        self.value_norm.update(self.buffer.returns[:-1].detach().cpu().numpy())
        advantages = self.buffer.advantages[:-1]
        active_masks = self.buffer.active_masks[:-1]
        adv_flat = advantages.reshape(-1)
        active_flat = active_masks.reshape(-1)
        valid = active_flat > 0.5
        if valid.any():
            adv_mean = adv_flat[valid].mean()
            adv_std = adv_flat[valid].std(unbiased=False) + 1e-5
        else:
            adv_mean = adv_flat.mean()
            adv_std = adv_flat.std(unbiased=False) + 1e-5
        norm_adv = (advantages - adv_mean) / adv_std

        value_loss_epoch = 0.0
        policy_loss_epoch = 0.0
        entropy_epoch = 0.0
        num_updates = 0
        use_recurrent = bool(self.train_cfg.use_recurrent_generator and self.train_cfg.data_chunk_length > 1)
        use_policy_active = bool(self.train_cfg.use_policy_active_masks)
        use_value_active = bool(self.train_cfg.use_value_active_masks)
        for _ in range(self.train_cfg.ppo_epochs):
            if use_recurrent:
                data_iter = self.buffer.recurrent_generator(
                    norm_adv, self.train_cfg.num_mini_batch, self.train_cfg.data_chunk_length
                )
            else:
                data_iter = self.buffer.mini_batch_generator(norm_adv, self.train_cfg.num_mini_batch)

            for batch in data_iter:
                if use_recurrent:
                    obs_seq = batch["obs"].float()  # [L, N, obs_dim]
                    state_seq = batch["states"].float()  # [L, N, state_dim]
                    actions_seq = batch["actions"]  # [L, N, 1]
                    old_log_probs_seq = batch["old_log_probs"]  # [L, N, 1]
                    returns_seq = batch["returns"]  # [L, N, 1]
                    advantages_seq = batch["advantages"]  # [L, N, 1]
                    masks_seq = batch["masks"].float()  # [L, N, 1]
                    active_seq = batch["active_masks"].float()  # [L, N, 1]
                    avail_seq = batch["avail_actions"].float()  # [L, N, action_dim]
                    rnn_actor = batch["rnn_states_actor"].float()  # [N, hidden]
                    rnn_critic = batch["rnn_states_critic"].float()

                    log_probs_steps = []
                    entropy_steps = []
                    values_steps = []
                    for t in range(obs_seq.size(0)):
                        logits, rnn_actor = self.policy._actor_forward(obs_seq[t], rnn_actor, masks_seq[t])
                        logits = self.policy._mask_logits(logits, avail_seq[t])
                        dist = Categorical(logits=logits)
                        log_probs_steps.append(dist.log_prob(actions_seq[t].squeeze(-1)).unsqueeze(-1))
                        entropy_steps.append(dist.entropy().unsqueeze(-1))
                        values, rnn_critic = self.policy._critic_forward(state_seq[t], rnn_critic, masks_seq[t])
                        values_steps.append(values)

                    log_probs = torch.stack(log_probs_steps, dim=0)
                    entropy = torch.stack(entropy_steps, dim=0)
                    values = torch.stack(values_steps, dim=0)
                    ratio = torch.exp(log_probs - old_log_probs_seq)
                    surr1 = ratio * advantages_seq
                    surr2 = torch.clamp(
                        ratio, 1.0 - self.train_cfg.clip_param, 1.0 + self.train_cfg.clip_param
                    ) * advantages_seq
                    policy_loss_terms = -torch.min(surr1, surr2)
                    active_denom = active_seq.sum().clamp_min(1.0)
                    if use_policy_active:
                        policy_loss = (policy_loss_terms * active_seq).sum() / active_denom
                        entropy_loss = (entropy * active_seq).sum() / active_denom
                    else:
                        policy_loss = policy_loss_terms.mean()
                        entropy_loss = entropy.mean()

                    norm_returns = self.value_norm.normalize(returns_seq, use_mean=True)
                    value_loss_terms = F.huber_loss(values, norm_returns, delta=10.0, reduction="none")
                    if use_value_active:
                        value_loss = (value_loss_terms * active_seq).sum() / active_denom
                    else:
                        value_loss = value_loss_terms.mean()
                else:
                    obs_batch = batch["obs"].float()
                    state_batch = batch["states"].float()
                    actions_batch = batch["actions"]
                    old_log_probs_batch = batch["old_log_probs"]
                    returns_batch = batch["returns"]
                    advantages_batch = batch["advantages"]
                    masks_batch = batch["masks"].float()
                    active_batch = batch["active_masks"].float()
                    rnn_actor_batch = batch["rnn_states_actor"].float()
                    rnn_critic_batch = batch["rnn_states_critic"].float()
                    avail_batch = batch["avail_actions"].float()

                    log_probs, entropy, values = self.policy.evaluate_actions(
                        obs_batch,
                        state_batch,
                        actions_batch,
                        rnn_actor_batch,
                        rnn_critic_batch,
                        masks_batch,
                        avail_batch,
                    )
                    ratio = torch.exp(log_probs - old_log_probs_batch)
                    surr1 = ratio * advantages_batch
                    surr2 = torch.clamp(
                        ratio, 1.0 - self.train_cfg.clip_param, 1.0 + self.train_cfg.clip_param
                    ) * advantages_batch
                    policy_loss_terms = -torch.min(surr1, surr2)
                    active_denom = active_batch.sum().clamp_min(1.0)
                    if use_policy_active:
                        policy_loss = (policy_loss_terms * active_batch).sum() / active_denom
                        entropy_loss = (entropy * active_batch).sum() / active_denom
                    else:
                        policy_loss = policy_loss_terms.mean()
                        entropy_loss = entropy.mean()

                    norm_returns = self.value_norm.normalize(returns_batch, use_mean=True)
                    value_loss_terms = F.huber_loss(values, norm_returns, delta=10.0, reduction="none")
                    if use_value_active:
                        value_loss = (value_loss_terms * active_batch).sum() / active_denom
                    else:
                        value_loss = value_loss_terms.mean()

                loss = policy_loss + self.train_cfg.value_loss_coef * value_loss - self.train_cfg.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.train_cfg.max_grad_norm)
                self.optimizer.step()

                policy_loss_epoch += float(policy_loss.item())
                value_loss_epoch += float(value_loss.item())
                entropy_epoch += float(entropy_loss.item())
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
        # actions: [envs * agents, 1]，avail_actions: [envs, agents, action_dim]
        envs = avail_actions.shape[0]
        reshaped = actions.view(envs, self.num_agents, -1).clone()
        avail = avail_actions.view(envs, self.num_agents, self.action_dim)
        for env_idx in range(envs):
            for agent_idx in range(self.num_agents):
                act = reshaped[env_idx, agent_idx, 0].long()
                if avail[env_idx, agent_idx, act] < 0.5:
                    valid = torch.nonzero(avail[env_idx, agent_idx] > 0.5, as_tuple=False)
                    reshaped[env_idx, agent_idx, 0] = valid[0, 0] if valid.numel() > 0 else 0
        return reshaped.view(-1, 1)


    def _should_run_eval(self, update: int) -> bool:
        return (
            self.eval_env is not None
            and self.train_cfg.eval_interval_updates > 0
            and self.train_cfg.eval_episodes > 0
            and update % self.train_cfg.eval_interval_updates == 0
        )

    def _run_evaluation(self) -> Dict[str, float]:
        if self.eval_env is None:
            return {}
        env = self.eval_env
        wins = 0
        total = 0
        rewards: list[float] = []
        self._won_episodes = 0
        self._completed_episodes = 0
        mode = self.policy.training
        self.policy.eval()
        try:
            for _ in range(self.train_cfg.eval_episodes):
                obs_np, state_np, avail_np = env.reset()
                obs = torch.from_numpy(obs_np).float().to(self.device)
                state = torch.from_numpy(state_np).float().to(self.device)
                avail = torch.from_numpy(avail_np).float().to(self.device)
                actor_states = torch.zeros(env.cfg.num_envs, self.num_agents, self.policy.rnn_hidden_dim, device=self.device)
                critic_states = torch.zeros_like(actor_states)
                masks = torch.ones(env.cfg.num_envs, self.num_agents, 1, device=self.device)
                episode_return = 0.0
                done = False
                while not done:
                    flat_obs = obs.view(env.cfg.num_envs * self.num_agents, self.obs_dim).float()
                    flat_state = (
                        state.unsqueeze(1)
                        .repeat(1, self.num_agents, 1)
                        .view(env.cfg.num_envs * self.num_agents, self.state_dim)
                    ).float()
                    flat_actor = actor_states.view(env.cfg.num_envs * self.num_agents, -1).float()
                    flat_critic = critic_states.view(env.cfg.num_envs * self.num_agents, -1).float()
                    flat_masks = masks.view(env.cfg.num_envs * self.num_agents, 1).float()
                    flat_avail = avail.view(env.cfg.num_envs * self.num_agents, self.action_dim).float()
                    with torch.no_grad():
                        raw_actions, _, _, _, next_actor, next_critic = self.policy.act(
                            flat_obs,
                            flat_state,
                            flat_actor,
                            flat_critic,
                            flat_masks,
                            flat_avail,
                            deterministic=True,
                        )
                    actions = self._ensure_valid_actions(raw_actions, avail)
                    env_actions = actions.view(env.cfg.num_envs, self.num_agents).cpu().numpy()
                    next_obs_np, next_state_np, reward_np, done_np, next_avail_np, info_list = env.step(env_actions)
                    obs = torch.from_numpy(next_obs_np).float().to(self.device)
                    state = torch.from_numpy(next_state_np).float().to(self.device)
                    avail = torch.from_numpy(next_avail_np).float().to(self.device)
                    rewards_tensor = torch.from_numpy(reward_np).float().unsqueeze(-1).to(self.device)
                    actor_states = next_actor.view(env.cfg.num_envs, self.num_agents, -1).detach()
                    critic_states = next_critic.view(env.cfg.num_envs, self.num_agents, -1).detach()
                    masks = 1.0 - torch.from_numpy(done_np.astype(float)).unsqueeze(-1).to(self.device)
                    episode_return += rewards_tensor.mean().item()
                    done = bool(done_np[0, 0] > 0.5)
                    if done:
                        info = info_list[0] if info_list else {}
                        win = info.get("battle_won", 0)
                        if isinstance(win, (list, tuple)):
                            win = win[-1]
                        if float(win) > 0.5:
                            wins += 1
                            self._won_episodes += 1
                        self._completed_episodes += 1
                        total += 1
                        rewards.append(episode_return / self.train_cfg.episode_length)
        finally:
            self.policy.train(mode)
        eval_win_rate = wins / float(total) if total > 0 else 0.0
        self._won_episodes = 0
        self._completed_episodes = 0
        if total == 0:
            return {}
        avg_reward = sum(rewards) / float(len(rewards)) if rewards else 0.0
        return {
            "eval_episode_reward": avg_reward,
            "eval_win_rate": eval_win_rate,
        }

    def _close_eval_env(self) -> None:
        if self.eval_env is not None:
            self.eval_env.close()
            self.eval_env = None
