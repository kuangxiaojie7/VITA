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
    difficulty: str = "7"
    reward_only_positive: bool = True
    reward_scale: float = 1.0
    obs_last_action: bool = True
    state_last_action: bool = False
    state_last_enemy_action: bool = False
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
                state_last_enemy_action=cfg.state_last_enemy_action,
            )
            for _ in range(cfg.num_envs)
        ]
        self.n_agents = self.envs[0].n_agents
        self.obs_dim = self.envs[0].get_obs_size()
        self.state_dim = self.envs[0].get_state_size()
        self.action_dim = self.envs[0].n_actions

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        obs, states = [], []
        for env in self.envs:
            env.reset()
            obs.append(np.stack(env.get_obs()))  # [n_agents, obs_dim]
            states.append(env.get_state())
        obs = np.stack(obs)
        states = np.stack(states)
        obs = self._process_obs(obs)
        return obs, states

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Actions shape: [num_envs, n_agents]."""
        next_obs_list, state_list, reward_list, done_list, info_list = [], [], [], [], []
        for idx, env in enumerate(self.envs):
            env_actions = actions[idx]
            if self.cfg.malicious_agent_prob > 0:
                mask = np.random.rand(self.n_agents) < self.cfg.malicious_agent_prob
                random_actions = np.random.randint(0, self.action_dim, size=mask.sum())
                env_actions = env_actions.copy()
                env_actions[mask] = random_actions
            reward, terminated, _ = env.step(env_actions.tolist())
            reward *= self.cfg.reward_scale
            info = env.get_stats()
            next_obs = np.stack(env.get_obs())
            state = env.get_state()
            next_obs_list.append(next_obs)
            state_list.append(state)
            reward_list.append(np.full((self.n_agents,), reward, dtype=np.float32))
            done_list.append(np.array([terminated] * self.n_agents))
            info_list.append(info)
            if terminated:
                env.reset()
        next_obs = np.stack(next_obs_list)
        next_obs = self._process_obs(next_obs)
        return (
            next_obs,
            np.stack(state_list),
            np.stack(reward_list),
            np.stack(done_list),
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


def make_smac_env(cfg_dict: Dict[str, Any]) -> SMACWrapper:
    cfg = SMACConfig(**cfg_dict)
    return SMACWrapper(cfg)
