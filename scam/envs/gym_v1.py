"""
Marine vs 2 Zerglings RL Environment

A Gymnasium environment for training a Marine agent against 2 scripted Zerglings.
Uses vector observations and discrete action space (8 directions + stay + attack_z1 + attack_z2).
Zerglings are sorted by tag to ensure consistent ordering across observations.
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
ACTION_ATTACK_Z1 = 9   # Attack first zergling (sorted by tag)
ACTION_ATTACK_Z2 = 10  # Attack second zergling (sorted by tag)

NUM_ACTIONS = 11

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
MAX_EPISODE_STEPS = 22.4 * 30  # 30 realtime seconds

# Spawn configuration
SPAWN_AREA_MIN = 0.0 + 14
SPAWN_AREA_MAX = 32.0 - 14
MIN_SPAWN_DISTANCE = 6.0
MAX_SPAWN_DISTANCE = 9.0

# Number of zerglings
NUM_ZERGLINGS = 2

# Observation space size:
# Marine: own_health (1), weapon_cooldown (1), upgrade level one-hot (3) = 5
# Per zergling (x2): rel_x (1), rel_y (1), distance (1), health (1), angle (1), in_range (1) = 6
# Total: 5 + 6*2 = 17
OBS_SIZE = 17


@dataclass
class UnitState:
    """Represents the state of a unit."""

    tag: int
    x: float
    y: float
    health: float
    health_max: float
    weapon_cooldown: float = 0.0

    @classmethod
    def from_proto(cls, unit) -> "UnitState":
        return cls(
            tag=unit.tag,
            x=unit.pos.x,
            y=unit.pos.y,
            health=unit.health,
            health_max=unit.health_max,
            weapon_cooldown=getattr(unit, "weapon_cooldown", 0.0),
        )

    def distance_to(self, other: "UnitState") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def angle_to(self, other: "UnitState") -> float:
        """Returns angle in radians from self to other."""
        return math.atan2(other.y - self.y, other.x - self.x)


class SC2GymEnv(gym.Env):
    """1 Marine vs 2 Zerglings environment on a Flat map"""

    metadata = {"name": "sc2_mv2z_v1", "render_modes": []}

    def __init__(self, params = None) -> None:
        super().__init__()
        self.params = params or {}

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)

        self.game = SC2SingleGame([Terran, Zerg]).launch()
        self.client = self.game.clients[0]
        self.units: dict[int, list] = defaultdict(list)
        self.current_step: int = 0
        self.terminated: bool = False

        self.upgrade_level = random.choice(self.params.get("upgrade_level", [0, 1, 2]))
        self.game_steps_per_env = random.choice(self.params.get("game_steps_per_env", [2]))
        self.init_game()

    def init_game(self) -> None:
        logger.info(f"Initializing SC2GymEnv with params: {self.params}")
        for _ in range(self.upgrade_level):
            self.client.research_upgrades()
            self.game.step(count=5)  # allow time for upgrade to complete

    def observe_units(self) -> None:
        self.units = defaultdict(list)
        obs = self.client.get_observation()
        for unit in obs.observation.raw_data.units:
            if unit.unit_type not in (UNIT_MARINE, UNIT_ZERGLING) or unit.health <= 0:
                continue
            unit_state = UnitState.from_proto(unit)
            self.units[unit.owner].append(unit_state)

        for units in self.units.values():
            units.sort(key=lambda u: u.tag)

        self.terminated = not all((self.units[1], self.units[2]))
        logger.debug(
            f"Marine alive: {len(self.units[1])}, Zerglings alive: {len(self.units[2])}, episode ended: {self.terminated}"
        )

    def clean_battlefield(self) -> None:
        units = sum(self.units.values(), [])
        unit_tags = [u.tag for u in units]
        self.client.kill_units(unit_tags)
        if self.game.reset_map():
            self.init_game()

    def prepare_battlefield(self) -> None:
        logger.debug("Spawning units")

        # Random position for marine
        marine_x = random.uniform(SPAWN_AREA_MIN, SPAWN_AREA_MAX)
        marine_y = random.uniform(SPAWN_AREA_MIN, SPAWN_AREA_MAX)
        self.client.spawn_units(UNIT_MARINE, (marine_x, marine_y), owner=1, quantity=1)

        # Spawn first zergling at random distance/angle from marine
        spawn_distance = random.uniform(MIN_SPAWN_DISTANCE, MAX_SPAWN_DISTANCE)
        spawn_angle = random.uniform(0, 2 * math.pi)
        ling_x = marine_x + spawn_distance * math.cos(spawn_angle)
        ling_y = marine_y + spawn_distance * math.sin(spawn_angle)
        # Spawn 2 zergling nearby
        self.client.spawn_units(UNIT_ZERGLING, (ling_x, ling_y), owner=2, quantity=1)
        shift_x = random.randint(-2500, 2500) / 1000.0
        shift_y = random.randint(-2500, 2500) / 1000.0
        self.client.spawn_units(UNIT_ZERGLING, (ling_x + shift_x, ling_y + shift_y), owner=2, quantity=1)

        self.game.step(count=2)  # unit spawn takes two frames

        self.observe_units()
        assert len(self.units[1]) == 1, "Marine not spawned correctly"
        assert len(self.units[2]) == 2, f"Expected 2 Zerglings, got {len(self.units[2])}"

        logger.debug("Spawned Marine and 2 Zerglings")

    def _get_zergling_obs(self, marine: UnitState, zergling: UnitState | None) -> list[float]:
        """Get observation features for a single zergling relative to marine."""
        if zergling is None:
            # Dead zergling - return zeros
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

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
        enemy_health = zergling.health / zergling.health_max

        # Angle to enemy (normalized to [-1, 1])
        angle_norm = marine.angle_to(zergling) / math.pi

        # In attack range (binary)
        in_range = 1.0 if distance <= MARINE_RANGE else 0.0

        return [rel_x, rel_y, distance_norm, enemy_health, angle_norm, in_range]

    def _compute_observation(self) -> np.ndarray:
        if not self.units[1]:
            return np.zeros(OBS_SIZE, dtype=np.float32)

        marine = self.units[1][0]
        zerglings = self.units[2]

        # Marine's own state
        own_health = marine.health / marine.health_max
        weapon_cooldown_norm = min(1.0, marine.weapon_cooldown / 15.0)

        # One-hot encode upgrade level (0, 1, or 2)
        upgrade_one_hot = [0.0, 0.0, 0.0]
        upgrade_one_hot[self.upgrade_level] = 1.0

        # Get zergling observations (pad with None if dead)
        z1 = zerglings[0] if len(zerglings) > 0 else None
        z2 = zerglings[1] if len(zerglings) > 1 else None

        z1_obs = self._get_zergling_obs(marine, z1)
        z2_obs = self._get_zergling_obs(marine, z2)

        obs = [own_health, weapon_cooldown_norm, *upgrade_one_hot, *z1_obs, *z2_obs]

        return np.array(obs, dtype=np.float32)

    def _compute_reward(self) -> float:
        """Compute reward based on damage dealt/taken and terminal conditions"""

        if self.current_step >= MAX_EPISODE_STEPS:
            return -2.0

        if not self.units[1] or not self.units[2]:
            ally_health = sum([u.health / u.health_max for u in self.units[1]], 0.0) / 1.0
            enemy_health = sum([u.health / u.health_max for u in self.units[2]], 0.0) / 2.0
            return ally_health  - enemy_health

        return 0.0

    def _agent_action(self, action: int) -> None:
        logger.debug(f"Agent action: {action}")
        marine = self.units[1][0]
        zerglings = self.units[2]

        if action == ACTION_ATTACK_Z1:
            if len(zerglings) > 0:
                self.client.unit_attack_unit(marine.tag, zerglings[0].tag)
            else:  # Z1 is dead, stop instead
                self.client.unit_stop(marine.tag)
        elif action == ACTION_ATTACK_Z2:
            if len(zerglings) > 1:
                self.client.unit_attack_unit(marine.tag, zerglings[1].tag)
            else:  # Both dead, stop
                self.client.unit_stop(marine.tag)
        elif action == ACTION_STAY:
            self.client.unit_stop(marine.tag)
        elif action in DIRECTION_VECTORS:
            dx, dy = DIRECTION_VECTORS[action]
            target_x = marine.x + dx * MOVE_STEP_SIZE
            target_y = marine.y + dy * MOVE_STEP_SIZE
            self.client.unit_move(marine.tag, (target_x, target_y))
        else:
            raise ValueError(f"Invalid action: {action}")

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.current_step = 0
        self.clean_battlefield()
        self.prepare_battlefield()
        return self._compute_observation(), {}

    def step(self, action) -> tuple:
        logger.debug(f"Environment step {self.current_step} with action {action}")

        if not self.terminated:
            self._agent_action(action)
            self.game.step(count=self.game_steps_per_env)
            self.current_step += self.game_steps_per_env

        self.observe_units()
        obs = self._compute_observation()
        reward = self._compute_reward()
        truncated = self.current_step >= MAX_EPISODE_STEPS
        return obs, reward, self.terminated, truncated, {"won": reward > 0}

    def close(self) -> None:
        self.game.close()
