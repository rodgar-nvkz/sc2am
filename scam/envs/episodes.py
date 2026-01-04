import random

import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import ActionType, AgentID

from scam.infra.client import SC2Client
from scam.infra.multiplayer import Race, SC2MultiplayerGame

UNIT_MARINE = 48
UNIT_ZERGLING = 105


class SC2EpisodeEnvironment(ParallelEnv):
    metadata = {"name": "sc2_env_v1"}

    def __init__(self) -> None:
        self.map = "Ladder2019Season1/AutomatonLE.SC2Map"
        self.game = SC2MultiplayerGame([Race.Terran, Race.Zerg]).launch()
        self.agents = ["player_0", "player_1"]
        self.possible_agents = ["player_0", "player_1"]

    @property
    def cli(self) -> SC2Client:
        return self.game.clients[0]

    def observation(self) -> np.ndarray:
        # obs = self.cli.get_observation()
        return np.stack([[]], axis=0)

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Returns tuple[dict[AgentID, ObsType], dict[AgentID, dict]]"""
        random.seed(seed) if seed else None

        self.cli.kill_all_units()
        self.cli.spawn_units(UNIT_MARINE, (5, 5), owner=1, quantity=1)
        self.cli.spawn_units(UNIT_ZERGLING, (20, 20), owner=2, quantity=1)
        observations, infos = {}, {}
        return observations, infos

    def step(self, actions: dict[AgentID, ActionType]) -> tuple:
        # TODO: actions

        self.game.step()

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        return observations, rewards, terminations, truncations, infos
