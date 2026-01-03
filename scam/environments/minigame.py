import random
from typing import Any

import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import ActionType, AgentID, ObsType

from scam.environments.base import SC2Game

# SC2 Unit Type IDs
UNIT_MARINE = 48
UNIT_ZERGLING = 105


class MinigameEnvironment(ParallelEnv):
    metadata = {"name": "sc2_env_v0"}

    def __init__(self) -> None:
        self.map = "Ladder2019Season1/AutomatonLE.SC2Map"
        self.server = SC2Game()

        self.agents = ["player_0"]
        self.possible_agents = ["player_0"]

    def observation(self) -> np.ndarray:
        """Build stacked numpy array observation from game state."""
        visibility = self.server.get_visibility_grid()
        creep = self.server.get_creep_grid()
        friendly_units = self.server.get_unit_positions(owner=1)
        enemy_units = self.server.get_unit_positions(owner=2)
        return np.stack([visibility, creep, friendly_units, enemy_units], axis=0)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        """Reset the environment by killing all units and spawning new ones."""
        random.seed(seed) if seed else None

        self.server.kill_all_units()

        marines, zerglings = random.randint(3, 7), random.randint(5, 10)
        self.server.spawn_units(unit_type=UNIT_MARINE, owner=1, quantity=marines)
        self.server.spawn_units(unit_type=UNIT_ZERGLING, owner=2, quantity=zerglings)

        observations: dict[AgentID, ObsType] = {"player_0": self.observation()}  # type: ignore[dict-item]
        infos: dict[AgentID, dict[str, Any]] = {"player_0": {}}  # type: ignore[dict-item]

        return observations, infos

    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        # TODO: actions

        self.server.step()
        observations: dict[AgentID, ObsType] = {"player_0": self.observation()}  # type: ignore[dict-item]
        rewards: dict[AgentID, float] = {"player_0": 0.0}  # type: ignore[dict-item]
        terminations: dict[AgentID, bool] = {"player_0": False}  # type: ignore[dict-item]
        truncations: dict[AgentID, bool] = {"player_0": False}  # type: ignore[dict-item]
        infos: dict[AgentID, dict[str, Any]] = {"player_0": {}}  # type: ignore[dict-item]
        return observations, rewards, terminations, truncations, infos
