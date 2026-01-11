"""
Marine vs Zergling RL Environment

A PettingZoo-compatible environment for training a Marine agent against a scripted Zergling.
Uses vector observations and discrete action space (8 directions + stay + attack).
"""

import math
import random
import sys
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger
from sympy.polys.matrices.linsolve import defaultdict

from scam.infra.game import SC2SingleGame, Terran, Zerg

logger.remove()
logger.add(sys.stderr, level="INFO")


# Unit type IDs
UNIT_MARINE = 48
UNIT_ZERGLING = 105

# Unit stats (for normalization)
MARINE_MAX_HP = 45.0
MARINE_RANGE = 5.0
MARINE_SPEED = 3.15

ZERGLING_MAX_HP = 35.0
ZERGLING_RANGE = 0.1  # Melee
ZERGLING_SPEED = 4.13

# Action space constants
ACTION_STAY = 0
ACTION_MOVE_N = 1
ACTION_MOVE_NE = 2
ACTION_MOVE_E = 3
ACTION_MOVE_SE = 4
ACTION_MOVE_S = 5
ACTION_MOVE_SW = 6
ACTION_MOVE_W = 7
ACTION_MOVE_NW = 8
ACTION_ATTACK = 9

NUM_ACTIONS = 10

# Direction vectors for movement (dx, dy)
DIRECTION_VECTORS = {
    ACTION_STAY: (0.0, 0.0),
    ACTION_MOVE_N: (0.0, 1.0),
    ACTION_MOVE_NE: (0.707, 0.707),
    ACTION_MOVE_E: (1.0, 0.0),
    ACTION_MOVE_SE: (0.707, -0.707),
    ACTION_MOVE_S: (0.0, -1.0),
    ACTION_MOVE_SW: (-0.707, -0.707),
    ACTION_MOVE_W: (-1.0, 0.0),
    ACTION_MOVE_NW: (-0.707, 0.707),
}

# Movement step size (how far to move per action)
MOVE_STEP_SIZE = 2.0

# Environment constants
MAX_EPISODE_STEPS = 22.4 * 10  # 10 realtime seconds

# Spawn configuration
SPAWN_AREA_MIN = 0.0 + 14
SPAWN_AREA_MAX = 32.0 - 14
MIN_SPAWN_DISTANCE = 7.0
MAX_SPAWN_DISTANCE = 8.0


@dataclass
class UnitState:
    """Represents the state of a unit."""

    tag: int
    x: float
    y: float
    health: float
    health_max: float
    weapon_cooldown: float = 0.0
    is_alive: bool = True

    @classmethod
    def from_proto(cls, unit) -> "UnitState":
        return cls(
            tag=unit.tag,
            x=unit.pos.x,
            y=unit.pos.y,
            health=unit.health,
            health_max=unit.health_max,
            weapon_cooldown=getattr(unit, "weapon_cooldown", 0.0),
            is_alive=unit.health > 0,
        )

    def distance_to(self, other: "UnitState") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def angle_to(self, other: "UnitState") -> float:
        """Returns angle in radians from self to other."""
        return math.atan2(other.y - self.y, other.x - self.x)


class SC2GymEnv(gym.Env):
    """1 Marine vs 1 Zergling environment on a Flat map"""

    metadata = {"name": "sc2_mvz_v1", "render_modes": []}
    GAME_STEPS_PER_ENV_STEP = 2

    def __init__(self, env_ctx: dict | None = None) -> None:
        super().__init__()

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        self.game = SC2SingleGame([Terran, Zerg]).launch()
        self.client = self.game.clients[0]
        self.units: dict[int, list] = defaultdict(list)
        self.current_step: int = 0
        self.terminated: bool = False

    def observe_units(self) -> None:
        self.units = defaultdict(list)
        obs = self.client.get_observation()
        for unit in obs.observation.raw_data.units:
            if unit.unit_type not in (UNIT_MARINE, UNIT_ZERGLING) or unit.health <= 0:
                continue
            unit_state = UnitState.from_proto(unit)
            self.units[unit.owner].append(unit_state)

        self.terminated = not all((self.units[1], self.units[2]))
        logger.debug(
            f"Marine alive: {len(self.units[1])}, Zergling alive: {len(self.units[2])}, episode ended: {self.terminated}"
        )

    def clean_battlefield(self) -> None:
        units = sum(self.units.values(), [])
        unit_tags = [u.tag for u in units]
        self.client.kill_units(unit_tags)
        self.game.reset_map()

    def prepare_battlefield(self) -> None:
        logger.debug("Spawning units")
        # Random position for marine
        marine_x = random.uniform(SPAWN_AREA_MIN, SPAWN_AREA_MAX)
        marine_y = random.uniform(SPAWN_AREA_MIN, SPAWN_AREA_MAX)
        self.client.spawn_units(UNIT_MARINE, (marine_x, marine_y), owner=1, quantity=1)

        # Zergling spawns at random distance/angle from marine
        spawn_distance = random.uniform(MIN_SPAWN_DISTANCE, MAX_SPAWN_DISTANCE)
        spawn_angle = random.uniform(0, 2 * math.pi)
        ling_x = marine_x + spawn_distance * math.cos(spawn_angle)
        ling_y = marine_y + spawn_distance * math.sin(spawn_angle)
        self.client.spawn_units(UNIT_ZERGLING, (ling_x, ling_y), owner=2, quantity=1)
        self.game.step(count=2)  # unit spawn takes two frames

        self.observe_units()
        assert len(self.units[1]) == 1, "Marine not spawned correctly"
        assert len(self.units[2]) == 1, "Zergling not spawned correctly"
        logger.debug(
            f"Spawned Marine at ({marine_x:.2f}, {marine_y:.2f}) and Zergling at ({ling_x:.2f}, {ling_y:.2f})"
        )

    def _compute_observation(self) -> np.ndarray:
        """Compute the observation vector for the Marine agent."""
        if not all(self.units.values()):
            return np.zeros(8, dtype=np.float32)

        marine = self.units[1][0]
        zergling = self.units[2][0]
        # Relative position (normalized by ~16 units for local awareness)
        rel_x = (zergling.x - marine.x) / 16.0
        rel_y = (zergling.y - marine.y) / 16.0

        # Clamp to [-1, 1]
        rel_x = max(-1.0, min(1.0, rel_x))
        rel_y = max(-1.0, min(1.0, rel_y))

        # Distance (normalized, max meaningful distance ~20)
        distance = marine.distance_to(zergling)
        distance_norm = min(1.0, distance / 20.0)

        # Health (normalized)
        own_health = marine.health / MARINE_MAX_HP
        enemy_health = zergling.health / ZERGLING_MAX_HP

        # Angle to enemy (normalized to [-1, 1])
        angle_norm = marine.angle_to(zergling) / math.pi

        # In attack range (binary)
        in_range = 1.0 if distance <= MARINE_RANGE else 0.0

        # Weapon ready (binary) - cooldown is in game loops, 0 means ready
        weapon_ready = 1.0 if marine.weapon_cooldown <= 0 else 0.0

        return np.array(
            [
                rel_x,
                rel_y,
                distance_norm,
                own_health,
                enemy_health,
                angle_norm,
                in_range,
                weapon_ready,
            ],
            dtype=np.float32,
        )

    def _compute_reward(self) -> float:
        """Compute reward based on damage dealt/taken and terminal conditions."""
        if self.current_step >= MAX_EPISODE_STEPS:
            return -1.0  # Timeout penalty

        if not self.units[2]:
            win_bonus = 1.0
            hp_left_bonus = (
                0.5 * (self.units[1][0].health / self.units[1][0].health_max)
                if self.units[1]
                else 0.0
            )
            return win_bonus + hp_left_bonus
        return 0.0

    def _agent_action(self, action: int) -> None:
        logger.debug(f"Agent action: {action}")
        marine = self.units[1][0]
        zergling = self.units[2][0]

        if action == ACTION_ATTACK:
            self.client.unit_attack_unit(marine.tag, zergling.tag)
        elif action == ACTION_STAY:
            self.client.unit_stop(marine.tag)
        elif action in DIRECTION_VECTORS:
            dx, dy = DIRECTION_VECTORS[action]
            target_x = marine.x + dx * MOVE_STEP_SIZE
            target_y = marine.y + dy * MOVE_STEP_SIZE
            self.client.unit_move(marine.tag, (target_x, target_y))
        else:
            raise ValueError(f"Invalid action: {action}")

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.current_step = 0
        self.clean_battlefield()
        self.prepare_battlefield()
        return self._compute_observation(), {}

    def step(self, action: np.ndarray) -> tuple:
        logger.debug(f"Environment step {self.current_step} with action {action}")

        self.observe_units()
        if not self.terminated:
            self._agent_action(action.item())
            self.game.step(count=self.GAME_STEPS_PER_ENV_STEP)
            self.current_step += 1

        truncated = self.current_step >= MAX_EPISODE_STEPS
        obs = self._compute_observation()
        reward = self._compute_reward()
        return obs, reward, self.terminated, truncated, {"won": reward > 0}

    def close(self) -> None:
        self.game.close()
