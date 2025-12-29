import time
import os
import numpy as np
from functools import reduce
import torch
try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None
from onpolicy.runner.shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)

def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)

def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None else str(value)

class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)
        self._vita_positions = None
        self._vita_alive = None
        self._vita_comm_rng = None
        self._vita_comm_noise_std = 0.0
        self._vita_comm_drop_prob = 0.0
        self._vita_comm_malicious_prob = 0.0
        self._vita_comm_malicious_noise_scale = 3.0
        self._vita_comm_malicious_mode = "bernoulli"
        self._vita_comm_malicious_fixed_agent_id = 0
        self._vita_comm_malicious_senders = None

        if self.algorithm_name == "rvita":
            seed = int(getattr(self.all_args, "seed", 0))
            self._vita_comm_rng = np.random.RandomState(seed)
            self._vita_comm_noise_std = _env_float("ONPOLICY_COMM_NOISE_STD", 0.0)
            self._vita_comm_drop_prob = _env_float("ONPOLICY_COMM_PACKET_DROP_PROB", 0.0)
            self._vita_comm_malicious_prob = _env_float("ONPOLICY_COMM_MALICIOUS_AGENT_PROB", 0.0)
            self._vita_comm_malicious_noise_scale = _env_float("ONPOLICY_COMM_MALICIOUS_NOISE_SCALE", 3.0)
            self._vita_comm_malicious_mode = _env_str("ONPOLICY_COMM_MALICIOUS_MODE", "bernoulli").strip().lower()
            self._vita_comm_malicious_fixed_agent_id = _env_int("ONPOLICY_COMM_MALICIOUS_FIXED_AGENT_ID", 0)

    def _vita_sample_comm_malicious_senders(self, n_envs: int, n_agents: int):
        if self._vita_comm_rng is None:
            return None
        prob = float(self._vita_comm_malicious_prob)
        if prob <= 0.0 or n_envs <= 0 or n_agents <= 0:
            return None
        prob = float(min(1.0, max(0.0, prob)))

        mode = str(getattr(self, "_vita_comm_malicious_mode", "bernoulli") or "bernoulli").strip().lower()
        if mode in {"fixed", "fixed_agent", "fixed_sender"}:
            agent_id = int(getattr(self, "_vita_comm_malicious_fixed_agent_id", 0))
            if agent_id < 0 or agent_id >= n_agents:
                agent_id = 0
            mask = np.zeros((n_envs, n_agents), dtype=bool)
            if prob >= 1.0:
                mask[:, agent_id] = True
            else:
                active = self._vita_comm_rng.rand(n_envs) < prob
                if active.any():
                    mask[active, agent_id] = True
            return mask

        if mode in {"one", "one_per_episode", "episode", "sample_one", "random_one"}:
            mask = np.zeros((n_envs, n_agents), dtype=bool)
            if prob >= 1.0:
                active = np.ones((n_envs,), dtype=bool)
            else:
                active = self._vita_comm_rng.rand(n_envs) < prob
            if active.any():
                env_sel = np.where(active)[0]
                agent_sel = self._vita_comm_rng.randint(0, n_agents, size=env_sel.size)
                mask[env_sel, agent_sel] = True
            return mask

        # Default: independent Bernoulli per sender.
        return (self._vita_comm_rng.rand(n_envs, n_agents) < prob)

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        for episode in range(episodes):
            if self.algorithm_name == "rvita" and hasattr(self.trainer, "set_update"):
                self.trainer.set_update(episode + 1)
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                if self.algorithm_name == "rvita":
                    values, actions, action_log_probs, rnn_states, rnn_states_critic, neighbor_obs, neighbor_masks, neighbor_actions = self.collect(step)
                else:
                    values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                    
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                if self.algorithm_name == "rvita":
                    data = obs, share_obs, rewards, dones, infos, available_actions, \
                        values, actions, action_log_probs, \
                        rnn_states, rnn_states_critic, neighbor_obs, neighbor_masks, neighbor_actions
                else:
                    data = obs, share_obs, rewards, dones, infos, available_actions, \
                           values, actions, action_log_probs, \
                           rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            incre_win_rate = None
            if self.env_name in {"StarCraft2", "SMACv2", "SMAC", "StarCraft2v2"}:
                battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)
                battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
                incre_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)
                incre_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)

                for i, info in enumerate(infos):
                    info0 = info[0] if info else {}
                    if "battles_won" in info0:
                        battles_won[i] = info0["battles_won"]
                        incre_battles_won[i] = battles_won[i] - last_battles_won[i]
                    if "battles_game" in info0:
                        battles_game[i] = info0["battles_game"]
                        incre_battles_game[i] = battles_game[i] - last_battles_game[i]

                denom = float(np.sum(incre_battles_game))
                incre_win_rate = float(np.sum(incre_battles_won) / denom) if denom > 0 else 0.0

                last_battles_game = battles_game
                last_battles_won = battles_won

            json_metrics = {
                "phase": "train",
                "update": episode + 1,
                "total_env_steps": int(total_num_steps),
                "episode_reward": float(np.mean(self.buffer.rewards)),
                **{
                    k: float(v) if np.isscalar(v) else v
                    for k, v in train_infos.items()
                    if k
                    in {
                        "policy_loss",
                        "value_loss",
                        "dist_entropy",
                        "ratio",
                        "actor_grad_norm",
                        "critic_grad_norm",
                        "kl",
                        "kl_raw",
                        "trust_loss",
                        "trust_score_mean",
                        "trust_score_p10",
                        "trust_score_p50",
                        "trust_score_p90",
                        "trust_gate_ratio",
                        "comm_valid_neighbors",
                        "comm_kept_neighbors",
                        "comm_strength",
                        "comm_enabled",
                    }
                },
                "entropy": float(train_infos.get("dist_entropy", 0.0)),
            }
            if incre_win_rate is not None:
                json_metrics["incre_win_rate"] = float(incre_win_rate)

            self.log_json(json_metrics, step=episode + 1)
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.map_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if incre_win_rate is not None:
                    print("incre win rate is {}.".format(incre_win_rate))
                    if self.use_wandb:
                        if wandb is None:
                            raise ImportError("wandb is required when `--use_wandb` is enabled. Please `pip install wandb`.")
                        wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)
                    else:
                        self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)

                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer.active_masks.shape)) 
                
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

        if self.algorithm_name == "rvita":
            self._vita_positions = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.float32)
            self._vita_alive = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32)
            self._vita_comm_malicious_senders = self._vita_sample_comm_malicious_senders(
                self.n_rollout_threads, self.num_agents
            )

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        if self.algorithm_name == "rvita":
            obs_now = self.buffer.obs[step]
            neighbor_obs, neighbor_masks, neighbor_idx = self._vita_compute_neighbors(obs_now)
            act_dim = int(self.envs.action_space[0].n)
            if step > 0:
                neighbor_action_labels = self._vita_gather_neighbor_actions(self.buffer.actions[step - 1], neighbor_idx)
            else:
                neighbor_action_labels = np.zeros(
                    (neighbor_obs.shape[0], neighbor_obs.shape[1], neighbor_obs.shape[2], act_dim),
                    dtype=np.float32,
                )
            self.trainer.policy.set_step_context(
                neighbor_obs.reshape(-1, neighbor_obs.shape[2], neighbor_obs.shape[3]),
                neighbor_masks.reshape(-1, neighbor_masks.shape[2], 1),
                neighbor_action_labels.reshape(-1, neighbor_action_labels.shape[2], neighbor_action_labels.shape[3]),
            )
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            np.concatenate(self.buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        if self.algorithm_name == "rvita":
            return (
                values,
                actions,
                action_log_probs,
                rnn_states,
                rnn_states_critic,
                neighbor_obs,
                neighbor_masks,
                neighbor_action_labels,
            )

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        if self.algorithm_name == "rvita":
            obs, share_obs, rewards, dones, infos, available_actions, \
                values, actions, action_log_probs, rnn_states, rnn_states_critic, neighbor_obs, neighbor_masks, neighbor_actions = data
        else:
            obs, share_obs, rewards, dones, infos, available_actions, \
                values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs

        if self.algorithm_name == "rvita":
            self.buffer.insert(
                share_obs,
                obs,
                rnn_states,
                rnn_states_critic,
                actions,
                action_log_probs,
                values,
                rewards,
                masks,
                bad_masks,
                active_masks,
                available_actions,
                neighbor_obs=neighbor_obs,
                neighbor_actions=neighbor_actions,
                neighbor_masks=neighbor_masks,
            )
            self._vita_update_meta(infos, dones_env=dones_env)
        else:
            self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                               actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions)

    def _vita_comm_range(self) -> float:
        comm_range = float(getattr(self.all_args, "vita_comm_sight_range", 0.0) or 0.0)
        return comm_range if comm_range > 0.0 else 9.0

    def _vita_compute_neighbors(self, obs, *, positions=None, alive=None, comm_malicious_senders=None):
        obs = np.asarray(obs, dtype=np.float32)
        positions = self._vita_positions if positions is None else positions
        alive = self._vita_alive if alive is None else alive

        n_envs, n_agents, obs_dim = obs.shape
        max_neighbors = int(getattr(self.buffer, "vita_max_neighbors", max(1, n_agents - 1)))
        max_neighbors = max(1, min(max_neighbors, n_agents - 1)) if n_agents > 1 else 1

        neighbor_idx = np.zeros((n_envs, n_agents, max_neighbors), dtype=np.int64)
        neighbor_masks = np.zeros((n_envs, n_agents, max_neighbors, 1), dtype=np.float32)
        neighbor_obs = np.zeros((n_envs, n_agents, max_neighbors, obs_dim), dtype=np.float32)

        if n_agents <= 1:
            return neighbor_obs, neighbor_masks, neighbor_idx

        if positions is None or alive is None:
            positions = np.zeros((n_envs, n_agents, 2), dtype=np.float32)
            alive = np.ones((n_envs, n_agents), dtype=np.float32)

        positions = np.asarray(positions, dtype=np.float32)
        alive = np.asarray(alive, dtype=np.float32)
        if alive.ndim == 3 and alive.shape[-1] == 1:
            alive = alive.squeeze(-1)
        if alive.ndim == 1:
            alive = np.broadcast_to(alive[None, :], (n_envs, n_agents))

        diff = positions[:, :, None, :] - positions[:, None, :, :]
        dist = np.linalg.norm(diff, axis=-1).astype(np.float32)  # [envs, agents, agents]
        for agent_id in range(n_agents):
            dist[:, agent_id, agent_id] = 1e9
        neighbor_idx = np.argsort(dist, axis=-1)[..., :max_neighbors].astype(np.int64)
        dist_k = np.take_along_axis(dist, neighbor_idx, axis=-1)

        comm_range = self._vita_comm_range()
        within = dist_k <= (comm_range + 1e-6)
        env_ids = np.arange(n_envs)[:, None, None]
        neighbor_alive = (alive[env_ids, neighbor_idx] > 0.5)
        self_alive = (alive > 0.5)
        combined = within & neighbor_alive & self_alive[..., None]
        neighbor_masks = combined.astype(np.float32)[..., None]

        neighbor_obs = obs[env_ids, neighbor_idx]

        # Optional communication corruption (independent from env-side observation noise).
        # - drop: simulates missing neighbor packets/messages.
        # - malicious: simulates adversarial/garbled neighbor messages from selected senders.
        if self._vita_comm_rng is not None and (self._vita_comm_noise_std > 0.0 or self._vita_comm_drop_prob > 0.0 or self._vita_comm_malicious_prob > 0.0):
            if self._vita_comm_noise_std > 0.0:
                neighbor_obs = neighbor_obs + self._vita_comm_rng.normal(
                    0.0, self._vita_comm_noise_std, size=neighbor_obs.shape
                ).astype(np.float32)

            if self._vita_comm_drop_prob > 0.0:
                drop_mask = (self._vita_comm_rng.rand(n_envs, n_agents, max_neighbors) < self._vita_comm_drop_prob)[..., None]
                drop_mask = drop_mask & (neighbor_masks > 0.5)
                if drop_mask.any():
                    neighbor_masks = neighbor_masks * (1.0 - drop_mask.astype(np.float32))
                    neighbor_obs = neighbor_obs * (1.0 - drop_mask.astype(np.float32))

            if self._vita_comm_malicious_prob > 0.0:
                comm_malicious_senders = self._vita_comm_malicious_senders if comm_malicious_senders is None else comm_malicious_senders
                if comm_malicious_senders is None or comm_malicious_senders.shape[:2] != (n_envs, n_agents):
                    comm_malicious_senders = self._vita_sample_comm_malicious_senders(n_envs, n_agents)
                if comm_malicious_senders is not None:
                    mal_mask = comm_malicious_senders[env_ids, neighbor_idx][..., None]
                    mal_mask = mal_mask & (neighbor_masks > 0.5)
                else:
                    mal_mask = None

                if mal_mask is not None and mal_mask.any():
                    base_std = self._vita_comm_noise_std if self._vita_comm_noise_std > 0.0 else 1.0
                    std = float(max(1e-6, base_std) * max(0.0, self._vita_comm_malicious_noise_scale))
                    noise = self._vita_comm_rng.normal(0.0, std, size=neighbor_obs.shape).astype(np.float32)
                    mal_f = mal_mask.astype(np.float32)
                    neighbor_obs = neighbor_obs * (1.0 - mal_f) + noise * mal_f

        return neighbor_obs, neighbor_masks, neighbor_idx

    def _vita_gather_neighbor_actions(self, actions, neighbor_idx):
        actions = np.asarray(actions)
        act_dim = int(self.envs.action_space[0].n)
        act_idx = actions.squeeze(-1).astype(np.int64)
        one_hot = np.eye(act_dim, dtype=np.float32)[act_idx]  # [envs, agents, act_dim]
        env_ids = np.arange(one_hot.shape[0])[:, None, None]
        return one_hot[env_ids, neighbor_idx]

    def _vita_update_meta(self, infos, *, dones_env=None, positions=None, alive=None) -> None:
        positions = self._vita_positions if positions is None else positions
        alive = self._vita_alive if alive is None else alive
        is_training_meta = (positions is self._vita_positions) and (alive is self._vita_alive)
        if positions is None or alive is None:
            return

        for env_i, info in enumerate(infos):
            try:
                info0 = info[0]
            except Exception:
                info0 = info
            if not isinstance(info0, dict):
                continue
            pos = info0.get("agent_positions")
            am = info0.get("alive_mask")
            if pos is not None:
                positions[env_i] = np.asarray(pos, dtype=np.float32)
            if am is not None:
                alive[env_i] = np.asarray(am, dtype=np.float32)

        if dones_env is not None:
            positions[dones_env == True] = 0.0
            alive[dones_env == True] = 1.0
            if is_training_meta and self._vita_comm_malicious_senders is not None:
                idx = np.asarray(dones_env, dtype=bool)
                if idx.any():
                    sampled = self._vita_sample_comm_malicious_senders(int(idx.sum()), self.num_agents)
                    if sampled is None:
                        self._vita_comm_malicious_senders[idx] = False
                    else:
                        self._vita_comm_malicious_senders[idx] = sampled

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            if self.use_wandb:
                if wandb is None:
                    raise ImportError("wandb is required when `--use_wandb` is enabled. Please `pip install wandb`.")
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        eval_positions = None
        eval_alive = None
        eval_prev_actions = None
        eval_comm_malicious_senders = None
        if self.algorithm_name == "rvita":
            eval_positions = np.zeros((self.n_eval_rollout_threads, self.num_agents, 2), dtype=np.float32)
            eval_alive = np.ones((self.n_eval_rollout_threads, self.num_agents), dtype=np.float32)
            eval_comm_malicious_senders = self._vita_sample_comm_malicious_senders(
                self.n_eval_rollout_threads, self.num_agents
            )

        while True:
            self.trainer.prep_rollout()
            if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
                eval_actions, eval_rnn_states = \
                    self.trainer.policy.act(np.concatenate(eval_share_obs),
                                            np.concatenate(eval_obs),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            np.concatenate(eval_available_actions),
                                            deterministic=True)
            else:
                if self.algorithm_name == "rvita":
                    neighbor_obs, neighbor_masks, neighbor_idx = self._vita_compute_neighbors(
                        eval_obs, positions=eval_positions, alive=eval_alive, comm_malicious_senders=eval_comm_malicious_senders
                    )
                    act_dim = int(self.eval_envs.action_space[0].n)
                    if eval_prev_actions is not None:
                        neighbor_prev_actions = self._vita_gather_neighbor_actions(eval_prev_actions, neighbor_idx)
                    else:
                        neighbor_prev_actions = np.zeros(
                            (neighbor_obs.shape[0], neighbor_obs.shape[1], neighbor_obs.shape[2], act_dim),
                            dtype=np.float32,
                        )
                    self.trainer.policy.set_step_context(
                        neighbor_obs.reshape(-1, neighbor_obs.shape[2], neighbor_obs.shape[3]),
                        neighbor_masks.reshape(-1, neighbor_masks.shape[2], 1),
                        neighbor_prev_actions.reshape(-1, neighbor_prev_actions.shape[2], neighbor_prev_actions.shape[3]),
                    )
                eval_actions, eval_rnn_states = \
                    self.trainer.policy.act(np.concatenate(eval_obs),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            np.concatenate(eval_available_actions),
                                            deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            eval_prev_actions = eval_actions
            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)
            if self.algorithm_name == "rvita":
                self._vita_update_meta(eval_infos, dones_env=eval_dones_env, positions=eval_positions, alive=eval_alive)
                if eval_comm_malicious_senders is not None:
                    idx = np.asarray(eval_dones_env, dtype=bool)
                    if idx.any():
                        sampled = self._vita_sample_comm_malicious_senders(int(idx.sum()), self.num_agents)
                        if sampled is None:
                            eval_comm_malicious_senders[idx] = False
                        else:
                            eval_comm_malicious_senders[idx] = sampled

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                if self.use_wandb:
                    if wandb is None:
                        raise ImportError("wandb is required when `--use_wandb` is enabled. Please `pip install wandb`.")
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)

                update = int(total_num_steps // (self.episode_length * self.n_rollout_threads))
                self.log_json(
                    {
                        "phase": "eval",
                        "update": update,
                        "total_env_steps": int(total_num_steps),
                        "eval_win_rate": float(eval_win_rate),
                        "eval_episode_reward": float(eval_episode_rewards.mean()) if eval_episode_rewards.size else 0.0,
                    },
                    step=update,
                )
                break
