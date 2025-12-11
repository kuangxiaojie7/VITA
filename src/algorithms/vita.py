from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from typing import Deque, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

from src.vita import VITAAgent, VITAAgentConfig
from src.utils import Logger, MAPPOBuffer
from src.envs import make_smac_env


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
    trust_warmup_updates: int = 0
    kl_warmup_updates: int = 0
    trust_delay_updates: int = 0
    kl_delay_updates: int = 0
    comm_delay_updates: int = 0
    comm_warmup_updates: int = 0
    comm_full_warmup_updates: int = 0
    eval_interval_updates: int = 0
    eval_episodes: int = 0


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
        default_neighbors = self.num_agents - 1 if self.num_agents > 1 else 1
        cfg_neighbors = policy_cfg.get("max_neighbors", default_neighbors)
        if cfg_neighbors is None:
            cfg_neighbors = default_neighbors
        cfg_neighbors = max(1, int(cfg_neighbors))
        self.max_neighbors = min(default_neighbors, cfg_neighbors)
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
            comm_dropout=policy_cfg.get("comm_dropout", 0.1),
            enable_trust=policy_cfg.get("enable_trust", True),
            enable_kl=policy_cfg.get("enable_kl", True),
            trust_threshold=policy_cfg.get("trust_threshold", 0.0),
            attn_bias_coef=policy_cfg.get("attn_bias_coef", 1.0),
            max_neighbors=self.max_neighbors,
        )
        self.agent = VITAAgent(agent_cfg).to(device)
        self.agent.set_comm_enabled(False)
        self.agent.set_comm_strength(0.0)
        sight_ranges = torch.from_numpy(self.env.get_sight_ranges()).float()
        override_range = policy_cfg.get("comm_sight_range")
        if override_range is not None:
            override_range = float(override_range)
            if override_range > 0:
                sight_ranges = torch.full_like(sight_ranges, override_range)
        self.comm_range = sight_ranges.to(device)
        self.current_positions = torch.zeros(self.num_envs, self.num_agents, 2, device=device)
        self.current_alive_mask = torch.ones(self.num_envs, self.num_agents, device=device)
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
        self._current_update: int = 0
        self._completed_episodes: int = 0
        self._won_episodes: int = 0
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
    actor_states = torch.zeros(self.num_envs, self.num_agents, self.agent.rnn_hidden_dim, device=self.device)
    critic_states = torch.zeros_like(actor_states)
    masks = torch.ones(self.num_envs, self.num_agents, 1, device=self.device)
    self._initialize_history(obs)
    self._refresh_env_metadata()

    try:
        for update in range(1, self.train_cfg.updates + 1):
            self._current_update = update
            comm_elapsed = self._current_update - self.train_cfg.comm_delay_updates
            comm_coeff = self._comm_schedule_coeff(
                comm_elapsed, self.train_cfg.comm_warmup_updates, self.train_cfg.comm_full_warmup_updates
            )
            self.agent.set_comm_strength(comm_coeff)
            self.agent.set_comm_enabled(comm_coeff > 0.0)
            trust_elapsed = self._current_update - self.train_cfg.trust_delay_updates
            trust_gate = self._schedule_coeff(trust_elapsed, self.train_cfg.trust_warmup_updates)
            self.agent.set_trust_active(trust_gate > 0.0)
            self.buffer.reset(obs, state, actor_states, critic_states)
            reward_sum = 0.0
            for step in range(self.train_cfg.episode_length):
                obs_seq = self._stack_history().contiguous().float()
                obs = obs.float()
                state = state.float()
                avail_actions = avail_actions.float()
                neighbor_indices, neighbor_mask = self._select_topk_neighbors(self.current_positions, self.current_alive_mask)
                neighbor_seq = self._gather_neighbor_sequences(obs_seq, neighbor_indices).contiguous()
                neighbor_mask = neighbor_mask.to(obs_seq.device).detach()
                flat_obs_seq = obs_seq.view(self.num_envs * self.num_agents, self.history_length, self.obs_dim).float()
                flat_state = (
                    state.unsqueeze(1)
                    .repeat(1, self.num_agents, 1)
                    .view(self.num_envs * self.num_agents, self.state_dim)
                )
                flat_state = flat_state.float()
                flat_neighbor_seq = neighbor_seq.view(
                    self.num_envs * self.num_agents, self.max_neighbors, self.history_length, self.obs_dim
                ).float()
                flat_neighbor_mask = neighbor_mask.view(
                    self.num_envs * self.num_agents, self.max_neighbors, 1
                ).float()
                flat_actor = actor_states.view(self.num_envs * self.num_agents, -1).float()
                flat_critic = critic_states.view(self.num_envs * self.num_agents, -1).float()
                flat_masks = masks.view(self.num_envs * self.num_agents, 1).float()
                flat_avail = avail_actions.view(self.num_envs * self.num_agents, self.action_dim).float()

                outputs = self.agent.act(
                    flat_obs_seq,
                    flat_state,
                    flat_neighbor_seq,
                    flat_neighbor_mask,
                    None,
                    flat_actor,
                    flat_critic,
                    flat_masks,
                    flat_avail,
                )
                actions = self._ensure_valid_actions(outputs["actions"], avail_actions)
                log_probs = outputs["log_probs"]
                values = outputs["values"]
                next_actor = outputs["next_actor_state"]
                next_critic = outputs["next_critic_state"]

                env_actions = actions.view(self.num_envs, self.num_agents).cpu().numpy()
                next_obs_np, next_state_np, reward_np, done_np, next_avail_np, info_list = self.env.step(env_actions)
                next_obs = torch.from_numpy(next_obs_np).float().to(self.device)
                next_state = torch.from_numpy(next_state_np).float().to(self.device)
                next_avail = torch.from_numpy(next_avail_np).float().to(self.device)
                self._refresh_env_metadata()
                rewards = torch.from_numpy(reward_np).float().unsqueeze(-1).to(self.device)
                dones = torch.from_numpy(done_np.astype(float)).unsqueeze(-1).to(self.device)
                reward_sum += rewards.mean().item()

                next_actor_states = next_actor.view(self.num_envs, self.num_agents, -1).detach()
                next_critic_states = next_critic.view(self.num_envs, self.num_agents, -1).detach()
                reshaped_actions = actions.view(self.num_envs, self.num_agents, 1).detach()
                reshaped_log_probs = log_probs.view(self.num_envs, self.num_agents, 1).detach()
                reshaped_values = values.view(self.num_envs, self.num_agents, 1).detach()
                action_one_hot = torch.zeros(
                    self.num_envs, self.num_agents, self.action_dim, device=self.device, dtype=torch.float32
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
                    avail_actions,
                    neighbor_mask,
                )

                obs = next_obs
                state = next_state
                actor_states = next_actor_states
                critic_states = next_critic_states
                masks = 1.0 - dones
                self._update_history(next_obs)
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
                next_values, _ = self.agent.get_values(flat_state, flat_critic, flat_masks)
            next_values = next_values.view(self.num_envs, self.num_agents, 1)
            self.buffer.compute_returns(next_values, self.train_cfg.gamma, self.train_cfg.gae_lambda)
            loss_dict = self.update_policy()

            eval_metrics = {}
            if self._should_run_eval(update):
                eval_metrics = self._run_evaluation()

            log_payload = {
                "update": update,
                "episode_reward": reward_sum / self.train_cfg.episode_length,
                **loss_dict,
                **eval_metrics,
            }
            self.logger.log(log_payload, step=update)
            print(f"[VITA] update {update} completed")
    finally:
        self._close_eval_env()

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
        trust_coeff = self.agent.cfg.trust_lambda * self._schedule_coeff(
            self._current_update - self.train_cfg.trust_delay_updates,
            self.train_cfg.trust_warmup_updates,
        )
        if not self.agent.cfg.enable_trust:
            trust_coeff = 0.0
        kl_coeff = self._schedule_coeff(
            self._current_update - self.train_cfg.kl_delay_updates,
            self.train_cfg.kl_warmup_updates,
        )
        if not self.agent.cfg.enable_kl:
            kl_coeff = 0.0
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
                neighbor_mask_batch = batch["neighbor_masks"]
                avail_batch = batch["avail_actions"]
                eval_out = self.agent.evaluate_actions(
                    obs_seq_batch.float(),
                    state_batch.float(),
                    neighbor_obs_seq_batch.float(),
                    neighbor_mask_batch.float(),
                    neighbor_actions_batch.float(),
                    actions_batch,
                    rnn_actor_batch.float(),
                    rnn_critic_batch.float(),
                    masks_batch.float(),
                    avail_batch.float(),
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
                    + kl_coeff * kl_loss
                    + trust_coeff * trust_loss
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

    @staticmethod
    def _schedule_coeff(elapsed: int, warmup: int) -> float:
        if elapsed <= 0:
            return 0.0
        if warmup <= 0:
            return 1.0
        return min(1.0, elapsed / float(warmup))

    @staticmethod
    def _comm_schedule_coeff(elapsed: int, stage_one: int, stage_two: int) -> float:
        if elapsed <= 0:
            return 0.0
        if stage_one > 0 and elapsed <= stage_one:
            return min(0.5, 0.5 * elapsed / float(stage_one))
        if stage_one <= 0:
            base = 0.0
        else:
            base = 0.5
        if stage_two <= 0:
            return 1.0 if elapsed > stage_one else base
        extra = max(0.0, min(elapsed - max(stage_one, 0), stage_two))
        return min(1.0, max(base, 0.5) + 0.5 * (extra / float(stage_two)))

    def _initialize_history(self, obs: torch.Tensor) -> None:
        self.obs_history = deque(
            [obs.clone() for _ in range(self.history_length)], maxlen=self.history_length
        )

    def _update_history(self, obs: torch.Tensor) -> None:
        assert self.obs_history is not None
        self.obs_history.append(obs.clone())

    def _refresh_env_metadata(self) -> None:
        positions = torch.from_numpy(self.env.get_agent_positions()).float().to(self.device)
        alive = torch.from_numpy(self.env.get_agent_alive_mask()).float().to(self.device)
        self.current_positions = positions
        self.current_alive_mask = alive

    def _stack_history(self) -> torch.Tensor:
        assert self.obs_history is not None
        history = torch.stack(list(self.obs_history), dim=0)  # [T, envs, agents, obs_dim]
        return history.permute(1, 2, 0, 3).contiguous()

    def _stack_history_from(self, history: Deque[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(list(history), dim=0)
        return stacked.permute(1, 2, 0, 3).contiguous()

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
        eval_rewards: list[float] = []
        wins = 0
        total = 0
        self._won_episodes = 0
        self._completed_episodes = 0
        train_mode = self.agent.training
        self.agent.eval()
        saved_history = self.obs_history
        try:
            for _ in range(self.train_cfg.eval_episodes):
                obs_np, state_np, avail_np = env.reset()
                obs = torch.from_numpy(obs_np).float().to(self.device)
                state = torch.from_numpy(state_np).float().to(self.device)
                avail_actions = torch.from_numpy(avail_np).float().to(self.device)
                history = deque([obs.clone() for _ in range(self.history_length)], maxlen=self.history_length)
                actor_states = torch.zeros(env.cfg.num_envs, self.num_agents, self.agent.rnn_hidden_dim, device=self.device)
                critic_states = torch.zeros_like(actor_states)
                masks = torch.ones(env.cfg.num_envs, self.num_agents, 1, device=self.device)
                episode_return = 0.0
                done_flag = False
                while not done_flag:
                    obs_seq = self._stack_history_from(history)
                    positions = torch.from_numpy(env.get_agent_positions()).float().to(self.device)
                    alive = torch.from_numpy(env.get_agent_alive_mask()).float().to(self.device)
                    neighbor_indices, neighbor_mask = self._select_topk_neighbors(positions, alive)
                    neighbor_seq = self._gather_neighbor_sequences(obs_seq, neighbor_indices).contiguous()
                    neighbor_mask = neighbor_mask.to(obs_seq.device)
                    flat_obs_seq = obs_seq.view(env.cfg.num_envs * self.num_agents, self.history_length, self.obs_dim).float()
                    flat_state = (
                        state.unsqueeze(1)
                        .repeat(1, self.num_agents, 1)
                        .view(env.cfg.num_envs * self.num_agents, self.state_dim)
                    ).float()
                    flat_neighbor_seq = neighbor_seq.view(
                        env.cfg.num_envs * self.num_agents,
                        self.max_neighbors,
                        self.history_length,
                        self.obs_dim,
                    ).float()
                    flat_neighbor_mask = neighbor_mask.view(
                        env.cfg.num_envs * self.num_agents, self.max_neighbors, 1
                    ).float()
                    flat_actor = actor_states.view(env.cfg.num_envs * self.num_agents, -1).float()
                    flat_critic = critic_states.view(env.cfg.num_envs * self.num_agents, -1).float()
                    flat_masks = masks.view(env.cfg.num_envs * self.num_agents, 1).float()
                    flat_avail = avail_actions.view(env.cfg.num_envs * self.num_agents, self.action_dim).float()
                    with torch.no_grad():
                        outputs = self.agent.act(
                            flat_obs_seq,
                            flat_state,
                            flat_neighbor_seq,
                            flat_neighbor_mask,
                            None,
                            flat_actor,
                            flat_critic,
                            flat_masks,
                            flat_avail,
                            deterministic=True,
                        )
                    actions = outputs["actions"]
                    next_actor = outputs["next_actor_state"]
                    next_critic = outputs["next_critic_state"]
                    env_actions = actions.view(env.cfg.num_envs, self.num_agents).cpu().numpy()
                    next_obs_np, next_state_np, reward_np, done_np, next_avail_np, info_list = env.step(env_actions)
                    next_obs = torch.from_numpy(next_obs_np).float().to(self.device)
                    state = torch.from_numpy(next_state_np).float().to(self.device)
                    rewards = torch.from_numpy(reward_np).float().unsqueeze(-1).to(self.device)
                    avail_actions = torch.from_numpy(next_avail_np).float().to(self.device)
                    actor_states = next_actor.view(env.cfg.num_envs, self.num_agents, -1).detach()
                    critic_states = next_critic.view(env.cfg.num_envs, self.num_agents, -1).detach()
                    masks = 1.0 - torch.from_numpy(done_np.astype(float)).unsqueeze(-1).to(self.device)
                    history.append(next_obs)
                    episode_return += rewards.mean().item()
                    done_flag = bool(done_np[0, 0] > 0.5)
                    if done_flag:
                        info = info_list[0] if info_list else {}
                        win = info.get("battle_won", 0)
                        if isinstance(win, (list, tuple)):
                            win = win[-1]
                        if float(win) > 0.5:
                            wins += 1
                            self._won_episodes += 1
                        self._completed_episodes += 1
                        total += 1
                        eval_rewards.append(episode_return / self.train_cfg.episode_length)
        finally:
            self.agent.train(train_mode)
            self.obs_history = saved_history
        eval_win_rate = wins / float(total) if total > 0 else 0.0
        self._won_episodes = 0
        self._completed_episodes = 0
        if total == 0:
            return {}
        avg_reward = sum(eval_rewards) / float(len(eval_rewards)) if eval_rewards else 0.0
        return {
            "eval_episode_reward": avg_reward,
            "eval_win_rate": eval_win_rate,
        }

    def _close_eval_env(self) -> None:
        if self.eval_env is not None:
            self.eval_env.close()
            self.eval_env = None

    def _select_topk_neighbors(self, positions: torch.Tensor, alive_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # positions: [envs, agents, 2]
        envs, agents, _ = positions.shape
        if agents == 1:
            idx = torch.zeros(envs, 1, 1, dtype=torch.long, device=positions.device)
            return idx, torch.zeros(envs, 1, 1, 1, device=positions.device)
        dist = torch.cdist(positions, positions, p=2)
        mask = torch.eye(agents, device=positions.device).unsqueeze(0)
        dist = dist + mask * 1e9
        topk = torch.topk(dist, k=self.max_neighbors, dim=-1, largest=False).indices
        neighbor_pos = self._gather_neighbor_positions(positions, topk)
        self_pos = positions.unsqueeze(2).expand_as(neighbor_pos)
        dists = torch.norm(neighbor_pos - self_pos, dim=-1)
        range_tensor = self.comm_range.view(1, agents, 1)
        if range_tensor.device != positions.device:
            range_tensor = range_tensor.to(positions.device)
        sight_mask = (dists <= range_tensor + 1e-6).float()
        neighbor_alive = self._gather_neighbor_scalars(alive_mask, topk)
        self_alive = alive_mask.unsqueeze(-1).expand_as(neighbor_alive)
        combined = sight_mask * neighbor_alive * self_alive
        return topk, combined.unsqueeze(-1)

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

    def _gather_neighbor_positions(
        self, positions: torch.Tensor, neighbor_idx: torch.Tensor
    ) -> torch.Tensor:
        envs, agents, feat_dim = positions.shape
        expanded = positions.unsqueeze(1).expand(envs, agents, agents, feat_dim)
        idx = neighbor_idx.unsqueeze(-1).expand(envs, agents, self.max_neighbors, feat_dim)
        return torch.gather(expanded, 2, idx).contiguous()

    def _gather_neighbor_scalars(
        self, values: torch.Tensor, neighbor_idx: torch.Tensor
    ) -> torch.Tensor:
        envs, agents = values.shape
        expanded = values.unsqueeze(1).expand(envs, agents, agents)
        return torch.gather(expanded, 2, neighbor_idx).contiguous()

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
