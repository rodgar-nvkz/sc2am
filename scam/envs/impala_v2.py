"""
Marine vs 2 Zerglings RL Environment

A Gymnasium environment for training a Marine agent against 2 scripted Zerglings.
Uses vector observations and hybrid action space:
- Discrete commands: STAY, MOVE, ATTACK_Z1, ATTACK_Z2
- Continuous angle (sin, cos) for MOVE command - relative to reference direction

The observation and action spaces use a relative reference frame based on the
direction to the closest enemy, enabling rotation-invariant learning.
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

# Hybrid action space constants
# Discrete commands
# ACTION_STAY = 0
ACTION_MOVE = 0
ACTION_ATTACK_Z1 = 1
ACTION_ATTACK_Z2 = 2

NUM_COMMANDS = 3

# Movement step size (how far to move per action)
MOVE_STEP_SIZE = 2.0

# Environment constants
MAX_EPISODE_STEPS = 22.4 * 40  # 40 realtime seconds

# Spawn configuration
SPAWN_AREA_MIN = 0.0 + 15
SPAWN_AREA_MAX = 32.0 - 15
MIN_SPAWN_DISTANCE = 6.0
MAX_SPAWN_DISTANCE = 9.0

# Number of zerglings
NUM_ZERGLINGS = 2

# Observation space size:
# Marine: own_health (1), weapon_cooldown (1), upgrade level one-hot (3) = 5
# Per zergling (x2): distance (1), health (1), angle_sin (1), angle_cos (1) = 4
# Total: 5 + 4*2 = 13
OBS_SIZE = 13


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

    def angle_to_sincos(self, other: "UnitState") -> tuple[float, float]:
        """Returns (sin_angle, cos_angle) encoding to avoid discontinuity."""
        angle_rad = math.atan2(other.y - self.y, other.x - self.x)
        return math.sin(angle_rad), math.cos(angle_rad)


class SC2GymEnv(gym.Env):
    """1 Marine vs 2 Zerglings environment with hybrid action space.

    Action space is a Dict with:
    - 'command': Discrete(4) - STAY, MOVE, ATTACK_Z1, ATTACK_Z2
    - 'angle': Box(2) - (sin, cos) of movement direction relative to reference

    The reference direction is the direction to the closest zergling (or world north if none).
    This makes the action space rotation-invariant.
    """

    metadata = {"name": "sc2_mv2z_v2_hybrid", "render_modes": []}

    def __init__(self, params=None) -> None:
        super().__init__()
        self.params = params or {}

        # Hybrid action space: discrete command + continuous angle
        self.action_space = spaces.Dict({
            'command': spaces.Discrete(NUM_COMMANDS),
            'angle': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),  # sin, cos
        })
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)

        self.game = SC2SingleGame([Terran, Zerg]).launch()
        self.client = self.game.clients[0]
        self.units: dict[int, list] = defaultdict(list)
        self.current_step: int = 0
        self.terminated: bool = False

        # Cache the reference angle for consistent use within a step
        self._reference_angle: float = 0.0

        self.upgrade_level = random.choice(self.params.get("upgrade_level", [0, 1, 2]))
        self.game_steps_per_env = random.choice(self.params.get("game_steps_per_env", [1]))
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

    def _compute_reference_angle(self) -> float:
        """Compute reference angle (direction to closest zergling).

        This defines the 'north pole' for relative coordinates.
        Returns 0 if no marine or no zerglings.
        """
        if not self.units[1] or not self.units[2]:
            return 0.0

        marine = self.units[1][0]
        zerglings = self.units[2]

        # Find closest zergling
        closest = min(zerglings, key=lambda z: marine.distance_to(z))
        return marine.angle_to(closest)

    def _get_zergling_obs(self, marine: UnitState, zergling: UnitState | None, ref_angle: float) -> list[float]:
        """Get observation features for a single zergling relative to marine.

        Returns [health, sin(relative_angle), cos(relative_angle), distance]
        All angles are relative to the reference direction.
        """
        if zergling is None:
            # Dead zergling - return zeros
            return [0.0, 0.0, 0.0, 0.0]

        # Health (normalized)
        enemy_health = zergling.health / zergling.health_max

        # Angle relative to reference direction
        absolute_angle = marine.angle_to(zergling)
        relative_angle = absolute_angle - ref_angle

        # Encode as sin/cos to avoid discontinuity
        angle_sin = math.sin(relative_angle)
        angle_cos = math.cos(relative_angle)

        # Distance (normalized, max meaningful distance ~20)
        distance = marine.distance_to(zergling)
        distance_norm = min(1.0, distance / 20.0)

        return [enemy_health, angle_sin, angle_cos, distance_norm]

    def _compute_observation(self) -> np.ndarray:
        if not self.units[1]:
            return np.zeros(OBS_SIZE, dtype=np.float32)

        marine = self.units[1][0]
        zerglings = self.units[2]

        # Compute and cache reference angle
        self._reference_angle = self._compute_reference_angle()

        # Marine's own state
        own_health = marine.health / marine.health_max
        weapon_cooldown_norm = min(1.0, marine.weapon_cooldown / 15.0)

        # One-hot encode upgrade level (0, 1, or 2)
        upgrade_one_hot = [0.0, 0.0, 0.0]
        upgrade_one_hot[self.upgrade_level] = 1.0

        # Get zergling observations (pad with None if dead)
        # Sort by distance so z1 is always the closest
        if zerglings:
            zerglings_sorted = sorted(zerglings, key=lambda z: marine.distance_to(z))
            z1 = zerglings_sorted[0] if len(zerglings_sorted) > 0 else None
            z2 = zerglings_sorted[1] if len(zerglings_sorted) > 1 else None
        else:
            z1, z2 = None, None

        z1_obs = self._get_zergling_obs(marine, z1, self._reference_angle)
        z2_obs = self._get_zergling_obs(marine, z2, self._reference_angle)

        logger.debug(f"Marine health: {own_health}, weapon cooldown: {weapon_cooldown_norm}, upgrade level: {self.upgrade_level}")
        logger.debug(f"Reference angle: {self._reference_angle:.2f} rad")
        logger.debug(f"Zergling 1 obs (closest): {z1_obs}")
        logger.debug(f"Zergling 2 obs: {z2_obs}")

        obs = [own_health, weapon_cooldown_norm, *upgrade_one_hot, *z1_obs, *z2_obs]

        return np.array(obs, dtype=np.float32)

    def _compute_reward(self) -> float:
        """Compute reward based on damage dealt/taken and terminal conditions"""

        if self.current_step >= MAX_EPISODE_STEPS:
            return -1.0

        if not self.units[1] or not self.units[2]:
            ally_health = sum([u.health / u.health_max for u in self.units[1]], 0.0) / 1.0
            enemy_health = sum([u.health / u.health_max for u in self.units[2]], 0.0) / 2.0
            return (1 + ally_health - enemy_health) ** 2

        return -.01  # step penalty to encourage faster resolution

    def _agent_action(self, action: dict) -> None:
        """Execute hybrid action: discrete command + continuous angle."""
        command = action['command']
        angle_sincos = action['angle']

        logger.debug(f"Agent action: command={command}, angle_sincos={angle_sincos}")

        marine = self.units[1][0]
        zerglings = self.units[2]

        # Sort zerglings by distance for consistent attack targeting
        if zerglings:
            zerglings_sorted = sorted(zerglings, key=lambda z: marine.distance_to(z))
        else:
            zerglings_sorted = []

        # if command == ACTION_STAY:
        #     self.client.unit_stop(marine.tag)

        if command == ACTION_MOVE:
            # Convert relative (sin, cos) to absolute angle
            # The angle is relative to reference direction (toward closest enemy)
            relative_angle = math.atan2(angle_sincos[0], angle_sincos[1])
            absolute_angle = self._reference_angle + relative_angle

            dx = math.cos(absolute_angle)
            dy = math.sin(absolute_angle)
            target_x = marine.x + dx * MOVE_STEP_SIZE
            target_y = marine.y + dy * MOVE_STEP_SIZE

            logger.debug(f"Move: ref_angle={self._reference_angle:.2f}, rel_angle={relative_angle:.2f}, abs_angle={absolute_angle:.2f}")
            logger.debug(f"Move target: ({target_x:.2f}, {target_y:.2f})")

            self.client.unit_move(marine.tag, (target_x, target_y))

        elif command == ACTION_ATTACK_Z1:
            # Attack closest zergling
            if len(zerglings_sorted) > 0:
                self.client.unit_attack_unit(marine.tag, zerglings_sorted[0].tag)
            else:
                self.client.unit_stop(marine.tag)

        elif command == ACTION_ATTACK_Z2:
            # Attack second closest zergling
            if len(zerglings_sorted) > 1:
                self.client.unit_attack_unit(marine.tag, zerglings_sorted[1].tag)
            else:
                self.client.unit_stop(marine.tag)
        else:
            raise ValueError(f"Invalid command: {command}")

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.current_step = 0
        self.terminated = False
        self._reference_angle = 0.0
        self.clean_battlefield()
        self.prepare_battlefield()
        return self._compute_observation(), {}

    def step(self, action: dict) -> tuple:
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
