from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


try:
    from smac.env.starcraft2.starcraft2 import StarCraft2Env
except ImportError as exc:  # pragma: no cover - optional dependency
    StarCraft2Env = None
    _SMAC_IMPORT_ERROR = exc
else:
    _SMAC_IMPORT_ERROR = None


@dataclass
class SMACConfig:
    map_name: str
    name: str | None = None
    difficulty: str = "7"
    reward_only_positive: bool = True
    reward_scale: float = 1.0
    obs_last_action: bool = True
    state_last_action: bool = False
    num_envs: int = 1
    obs_noise_std: float = 0.0
    packet_drop_prob: float = 0.0
    malicious_agent_prob: float = 0.0


class SMACWrapper:
    """Thin wrapper to expose batched reset/step API compatible with trainers."""

    def __init__(self, cfg: SMACConfig):
        if StarCraft2Env is None:
            raise ImportError(
                "SMAC environment not available. Install smac and StarCraft II. "
                f"Original import error: {_SMAC_IMPORT_ERROR}"
            )
        self.cfg = cfg
        self.envs: List[StarCraft2Env] = [
            StarCraft2Env(
                map_name=cfg.map_name,
                difficulty=cfg.difficulty,
                reward_only_positive=cfg.reward_only_positive,
                obs_last_action=cfg.obs_last_action,
                state_last_action=cfg.state_last_action,
            )
            for _ in range(cfg.num_envs)
        ]
        self.n_agents = self.envs[0].n_agents
        self.obs_dim = self.envs[0].get_obs_size()
        self.state_dim = self.envs[0].get_state_size()
        self.action_dim = self.envs[0].n_actions
        self._last_positions: np.ndarray | None = None
        self._alive_mask: np.ndarray | None = None
        self._sight_ranges = self._compute_sight_ranges()

    def reset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs, states, avails = [], [], []
        for env in self.envs:
            env.reset()
            obs.append(np.stack(env.get_obs()))  # [n_agents, obs_dim]
            states.append(env.get_state())
            avails.append(np.stack(env.get_avail_actions()))
        obs = np.stack(obs)
        states = np.stack(states)
        avail = np.stack(avails)
        obs = self._process_obs(obs)
        self._update_agent_metadata()
        return obs, states, avail

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Actions shape: [num_envs, n_agents]."""
        next_obs_list, state_list, reward_list, done_list, avail_list, info_list = [], [], [], [], [], []
        for idx, env in enumerate(self.envs):
            env_actions = actions[idx]
            avail = np.stack(env.get_avail_actions())
            for agent_id in range(self.n_agents):
                act = env_actions[agent_id]
                if avail[agent_id, act] < 0.5:
                    valid = np.nonzero(avail[agent_id] > 0.5)[0]
                    env_actions[agent_id] = valid[0] if valid.size > 0 else 0
            if self.cfg.malicious_agent_prob > 0:
                mask = np.random.rand(self.n_agents) < self.cfg.malicious_agent_prob
                env_actions = env_actions.copy()
                for agent_id in range(self.n_agents):
                    if not mask[agent_id]:
                        continue
                    valid = np.nonzero(avail[agent_id] > 0.5)[0]
                    if valid.size > 0:
                        env_actions[agent_id] = np.random.choice(valid)
                    else:
                        env_actions[agent_id] = 0
            reward, terminated, info = env.step(env_actions.tolist())
            reward *= self.cfg.reward_scale
            episode_info = info or {}
            # Capture alive mask before any reset to preserve terminal step masks.
            alive_mask = np.zeros((self.n_agents,), dtype=np.float32)
            for agent_id in range(self.n_agents):
                try:
                    unit = env.get_unit_by_id(agent_id)
                except KeyError:
                    continue
                if unit is None:
                    continue
                alive_mask[agent_id] = 1.0 if getattr(unit, "health", 0.0) > 0 else 0.0
            if terminated:
                terminal_obs = np.stack(env.get_obs())
                terminal_state = env.get_state()
                terminal_avail = np.stack(env.get_avail_actions())
                try:
                    episode_stats = env.get_stats()
                except ZeroDivisionError:
                    episode_stats = {}
                episode_info = {
                    **episode_stats,
                    **episode_info,
                    "alive_mask": alive_mask,
                    "terminal_obs": terminal_obs,
                    "terminal_state": terminal_state,
                    "terminal_avail": terminal_avail,
                }
                # Keep env state in sync with returned observation: reset immediately on termination.
                env.reset()
            else:
                episode_info = {**episode_info, "alive_mask": alive_mask}
            next_obs = np.stack(env.get_obs())
            state = env.get_state()
            avail = np.stack(env.get_avail_actions())
            next_obs_list.append(next_obs)
            state_list.append(state)
            reward_list.append(np.full((self.n_agents,), reward, dtype=np.float32))
            done_list.append(np.array([terminated] * self.n_agents))
            avail_list.append(avail)
            info_list.append(episode_info)
        next_obs = np.stack(next_obs_list)
        next_obs = self._process_obs(next_obs)
        self._update_agent_metadata()
        return (
            next_obs,
            np.stack(state_list),
            np.stack(reward_list),
            np.stack(done_list),
            np.stack(avail_list),
            info_list,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.cfg.obs_noise_std > 0:
            noise = np.random.normal(0, self.cfg.obs_noise_std, size=obs.shape)
            obs = obs + noise
        if self.cfg.packet_drop_prob > 0:
            drop_mask = np.random.rand(*obs.shape[:2]) < self.cfg.packet_drop_prob
            obs[drop_mask] = 0.0
        return obs

    def _compute_sight_ranges(self) -> np.ndarray:
        if not self.envs:
            return np.zeros(0, dtype=np.float32)
        ranges = np.zeros(self.n_agents, dtype=np.float32)
        ref_env = self.envs[0]
        for agent_id in range(self.n_agents):
            ranges[agent_id] = float(ref_env.unit_sight_range(agent_id))
        return ranges

    def _update_agent_metadata(self) -> None:
        positions = np.zeros((self.cfg.num_envs, self.n_agents, 2), dtype=np.float32)
        alive = np.zeros((self.cfg.num_envs, self.n_agents), dtype=np.float32)
        for env_idx, env in enumerate(self.envs):
            for agent_id in range(self.n_agents):
                try:
                    unit = env.get_unit_by_id(agent_id)
                except KeyError:
                    continue
                if unit is None:
                    continue
                positions[env_idx, agent_id, 0] = float(getattr(unit.pos, "x", 0.0))
                positions[env_idx, agent_id, 1] = float(getattr(unit.pos, "y", 0.0))
                alive[env_idx, agent_id] = 1.0 if getattr(unit, "health", 0.0) > 0 else 0.0
        self._last_positions = positions
        self._alive_mask = alive

    def get_agent_positions(self) -> np.ndarray:
        if self._last_positions is None:
            self._update_agent_metadata()
        assert self._last_positions is not None
        return self._last_positions.copy()

    def get_agent_alive_mask(self) -> np.ndarray:
        if self._alive_mask is None:
            self._update_agent_metadata()
        assert self._alive_mask is not None
        return self._alive_mask.copy()

    def get_sight_ranges(self) -> np.ndarray:
        return self._sight_ranges.copy()


def make_smac_env(cfg_dict: Dict[str, Any]) -> SMACWrapper:
    cfg = SMACConfig(**cfg_dict)
    return SMACWrapper(cfg)
