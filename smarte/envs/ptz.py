"""
PettingZoo to Gymnasium VectorEnv Wrapper

Converts one or more PettingZoo ParallelEnvs into a single Gymnasium VectorEnv,
where each agent in each ParallelEnv becomes a sub-environment.

This enables algorithms like DreamerV3 (which don't natively support multi-agent)
to treat each agent as an independent environment while they share underlying game state.

Example:
    # 2 SC2 game instances, each with 2 marines = 4 sub-environments
    vec_env = PettingZooToVectorEnv([
        lambda: SC2MultiAgentEnv(),  # agents: ["marine_0", "marine_1"]
        lambda: SC2MultiAgentEnv(),  # agents: ["marine_0", "marine_1"]
    ])
    # vec_env.num_envs == 4
    # Positions [0,1] = agents from game 0
    # Positions [2,3] = agents from game 1
"""

from typing import Any, Callable

import gymnasium as gym
from gymnasium.vector.utils import batch_space
import numpy as np
import pettingzoo


class PettingZooToVectorEnv(gym.vector.VectorEnv):
    """
    Wraps multiple PettingZoo ParallelEnvs as a single Gymnasium VectorEnv.

    Each agent in each ParallelEnv becomes a sub-environment in the VectorEnv.
    This enables algorithms like DreamerV3 to treat each agent as an independent
    environment while they share underlying game state.

    Episode handling:
    - When an agent terminates, its slot returns terminated=True
    - Dead agents receive zero observations until entire ParallelEnv resets
    - When ALL agents in a ParallelEnv terminate/truncate, that env auto-resets
    - On auto-reset, returned obs is the first obs of new episode (standard VectorEnv behavior)
    - Terminal observation is available in info["final_observation"]

    Assumptions:
    - All agents across all ParallelEnvs have identical observation and action spaces
    - Agent order is fixed as defined by each ParallelEnv's possible_agents
    """

    def __init__(
        self,
        env_fns: list[Callable[[], pettingzoo.ParallelEnv]],
    ):
        """
        Initialize the wrapper.

        Args:
            env_fns: List of factory functions, each creating a PettingZoo ParallelEnv.
                     Each env can have multiple agents.
        """
        # Create all parallel envs
        self.parallel_envs: list[pettingzoo.ParallelEnv] = [fn() for fn in env_fns]
        self.env_fns = env_fns  # Keep for potential re-creation

        # Build agent mapping: vec_idx -> (env_idx, agent_id)
        # Order is: all agents from env 0, then all from env 1, etc.
        self.agent_mapping: list[tuple[int, str]] = []
        for env_idx, env in enumerate(self.parallel_envs):
            for agent_id in env.possible_agents:
                self.agent_mapping.append((env_idx, agent_id))

        num_envs = len(self.agent_mapping)

        if num_envs == 0:
            raise ValueError("No agents found in any of the provided environments")

        # All agents should have same obs/action space
        sample_env = self.parallel_envs[0]
        sample_agent = sample_env.possible_agents[0]
        single_observation_space = sample_env.observation_space(sample_agent)
        single_action_space = sample_env.action_space(sample_agent)

        # Validate all agents have same spaces
        for env_idx, env in enumerate(self.parallel_envs):
            for agent_id in env.possible_agents:
                obs_space = env.observation_space(agent_id)
                act_space = env.action_space(agent_id)
                if obs_space != single_observation_space:
                    raise ValueError(
                        f"Agent {agent_id} in env {env_idx} has different observation space: "
                        f"{obs_space} vs {single_observation_space}"
                    )
                if act_space != single_action_space:
                    raise ValueError(
                        f"Agent {agent_id} in env {env_idx} has different action space: "
                        f"{act_space} vs {single_action_space}"
                    )

        # Set VectorEnv attributes directly (no super().__init__ call)
        self.num_envs = num_envs
        self.single_observation_space = single_observation_space
        self.single_action_space = single_action_space
        self.observation_space = batch_space(single_observation_space, num_envs)
        self.action_space = batch_space(single_action_space, num_envs)

        # VectorEnv base class attributes
        self.closed = False
        self.metadata: dict[str, Any] = {"autoreset_mode": "next_step"}
        self.spec = None
        self.render_mode = None
        self._np_random = None
        self._np_random_seed = None

        # Track state for each agent slot
        self.agent_terminated: list[bool] = [False] * num_envs
        self.agent_truncated: list[bool] = [False] * num_envs
        self.cached_obs: list[np.ndarray] = [
            np.zeros(single_observation_space.shape, dtype=single_observation_space.dtype)
            for _ in range(num_envs)
        ]

        # Track which parallel envs need reset (all agents done)
        self.env_needs_reset: list[bool] = [False] * len(self.parallel_envs)

        # Build reverse mapping for quick lookup: (env_idx, agent_id) -> vec_idx
        self._reverse_mapping: dict[tuple[int, str], int] = {
            (env_idx, agent_id): vec_idx
            for vec_idx, (env_idx, agent_id) in enumerate(self.agent_mapping)
        }

    def _get_vec_idx(self, env_idx: int, agent_id: str) -> int:
        """Get VectorEnv index for a given env_idx and agent_id."""
        return self._reverse_mapping[(env_idx, agent_id)]

    def _get_agents_for_env(self, env_idx: int) -> list[tuple[int, str]]:
        """Get all (vec_idx, agent_id) pairs for a given parallel env."""
        return [
            (vec_idx, agent_id)
            for vec_idx, (e_idx, agent_id) in enumerate(self.agent_mapping)
            if e_idx == env_idx
        ]

    def _check_env_done(self, env_idx: int) -> bool:
        """Check if all agents in a parallel env are terminated or truncated."""
        for vec_idx, _ in self._get_agents_for_env(env_idx):
            if not self.agent_terminated[vec_idx] and not self.agent_truncated[vec_idx]:
                return False
        return True

    def _reset_parallel_env(self, env_idx: int, seed: int | None = None) -> dict[str, np.ndarray]:
        """Reset a single parallel env and return obs dict."""
        env = self.parallel_envs[env_idx]
        obs_dict, info_dict = env.reset(seed=seed)

        # Update cached obs and reset termination flags for all agents in this env
        for vec_idx, agent_id in self._get_agents_for_env(env_idx):
            self.cached_obs[vec_idx] = obs_dict[agent_id]
            self.agent_terminated[vec_idx] = False
            self.agent_truncated[vec_idx] = False

        self.env_needs_reset[env_idx] = False
        return obs_dict

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset all environments.

        Args:
            seed: Optional random seed
            options: Optional reset options

        Returns:
            observations: np.ndarray of shape (num_envs, *obs_shape)
            infos: dict with per-agent info
        """
        all_obs: list[np.ndarray] = []
        all_infos: list[dict] = []

        for env_idx, env in enumerate(self.parallel_envs):
            # Use different seeds for different envs if seed provided
            env_seed = None if seed is None else seed + env_idx
            obs_dict, info_dict = env.reset(seed=env_seed, options=options)

            # Extract obs for each agent in this env (in order)
            for agent_id in env.possible_agents:
                vec_idx = self._get_vec_idx(env_idx, agent_id)
                obs = obs_dict[agent_id]
                all_obs.append(obs)
                all_infos.append(info_dict.get(agent_id, {}))
                self.cached_obs[vec_idx] = obs

        # Reset termination tracking
        self.agent_terminated = [False] * self.num_envs
        self.agent_truncated = [False] * self.num_envs
        self.env_needs_reset = [False] * len(self.parallel_envs)

        return np.stack(all_obs), {"agent_infos": all_infos}

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Step all environments with the given actions.

        Args:
            actions: np.ndarray of shape (num_envs,) for discrete actions
                     or (num_envs, action_dim) for continuous actions

        Returns:
            observations: np.ndarray of shape (num_envs, *obs_shape)
            rewards: np.ndarray of shape (num_envs,)
            terminations: np.ndarray of shape (num_envs,) - True if agent terminated
            truncations: np.ndarray of shape (num_envs,) - True if agent truncated
            infos: dict with per-agent info and final_observation for terminated agents
        """
        # Group actions by parallel env
        env_actions: list[dict[str, Any]] = [{} for _ in self.parallel_envs]

        for vec_idx, action in enumerate(actions):
            env_idx, agent_id = self.agent_mapping[vec_idx]

            # Only include action if agent is still alive
            if not self.agent_terminated[vec_idx] and not self.agent_truncated[vec_idx]:
                # Convert numpy scalar to python type if needed
                if isinstance(action, np.ndarray) and action.ndim == 0:
                    action = action.item()
                env_actions[env_idx][agent_id] = action

        # Step each parallel env that has active agents
        env_results: list[tuple | None] = [None] * len(self.parallel_envs)

        for env_idx, env in enumerate(self.parallel_envs):
            if env_actions[env_idx]:
                # At least one agent needs stepping
                env_results[env_idx] = env.step(env_actions[env_idx])

        # Collect results for all agent slots
        all_obs: list[np.ndarray] = []
        all_rewards: list[float] = []
        all_terminated: list[bool] = []
        all_truncated: list[bool] = []
        all_infos: list[dict] = []

        # Track which envs need auto-reset after this step
        envs_to_reset: list[int] = []

        for env_idx, env in enumerate(self.parallel_envs):
            result = env_results[env_idx]

            for agent_id in env.possible_agents:
                vec_idx = self._get_vec_idx(env_idx, agent_id)
                info: dict[str, Any] = {}

                if result is not None and agent_id in result[0]:
                    # Agent was stepped
                    obs_dict, rew_dict, term_dict, trunc_dict, info_dict = result

                    obs = obs_dict[agent_id]
                    reward = rew_dict[agent_id]
                    terminated = term_dict[agent_id]
                    truncated = trunc_dict[agent_id]
                    info = info_dict.get(agent_id, {})

                    # Update tracking
                    self.agent_terminated[vec_idx] = terminated
                    self.agent_truncated[vec_idx] = truncated

                    if terminated or truncated:
                        # Store final observation
                        info["final_observation"] = obs
                        # Cache for potential future queries
                        self.cached_obs[vec_idx] = obs
                    else:
                        self.cached_obs[vec_idx] = obs

                else:
                    # Agent was already dead or env wasn't stepped
                    obs = np.zeros(
                        self.single_observation_space.shape,
                        dtype=self.single_observation_space.dtype,
                    )
                    reward = 0.0
                    terminated = self.agent_terminated[vec_idx]
                    truncated = self.agent_truncated[vec_idx]

                all_obs.append(obs)
                all_rewards.append(reward)
                all_terminated.append(terminated)
                all_truncated.append(truncated)
                all_infos.append(info)

        # Check which parallel envs need auto-reset (all agents done)
        for env_idx in range(len(self.parallel_envs)):
            if self._check_env_done(env_idx):
                envs_to_reset.append(env_idx)

        # Auto-reset completed envs and update observations
        for env_idx in envs_to_reset:
            obs_dict = self._reset_parallel_env(env_idx)

            # Replace observations with reset observations for agents in this env
            for vec_idx, agent_id in self._get_agents_for_env(env_idx):
                all_obs[vec_idx] = obs_dict[agent_id]

        return (
            np.stack(all_obs),
            np.array(all_rewards, dtype=np.float32),
            np.array(all_terminated, dtype=bool),
            np.array(all_truncated, dtype=bool),
            {"agent_infos": all_infos},
        )

    def close(self) -> None:
        """Close all parallel environments."""
        if self.closed:
            return
        for env in self.parallel_envs:
            env.close()
        self.closed = True

    def call(self, name: str, *args, **kwargs) -> list[Any]:
        """Call a method on all parallel environments."""
        return [getattr(env, name)(*args, **kwargs) for env in self.parallel_envs]

    def get_attr(self, name: str) -> list[Any]:
        """Get an attribute from all parallel environments."""
        return [getattr(env, name) for env in self.parallel_envs]

    def set_attr(self, name: str, values: list[Any]) -> None:
        """Set an attribute on all parallel environments."""
        for env, value in zip(self.parallel_envs, values):
            setattr(env, name, value)


def make_pettingzoo_vector_env(
    env_fn: Callable[[], pettingzoo.ParallelEnv],
    num_envs: int = 1,
) -> PettingZooToVectorEnv:
    """
    Convenience function to create a PettingZooToVectorEnv.

    Args:
        env_fn: Factory function that creates a PettingZoo ParallelEnv
        num_envs: Number of parallel env instances to create

    Returns:
        PettingZooToVectorEnv wrapping all agents from all envs
    """
    return PettingZooToVectorEnv([env_fn for _ in range(num_envs)])
