"""
Marine vs 2 Zerglings RL Environment

A Gymnasium environment for training a Marine agent against 2 scripted Zerglings.
Uses vector observations and hybrid action space:
- Discrete commands: STAY, MOVE, ATTACK_Z1, ATTACK_Z2
- Continuous angle (sin, cos) for MOVE command in world-space polar coords

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

from smarte.infra.game import SC2SingleGame, Terran, Zerg

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

# Discrete commands
ACTION_MOVE = 0
ACTION_ATTACK_Z1 = 1
ACTION_ATTACK_Z2 = 2
NUM_COMMANDS = 3

# Movement step size (how far to move per action)
MOVE_STEP_SIZE = 2.0

# Environment constants
MAX_EPISODE_STEPS = 22.4 * 30  # 30 realtime seconds

# Spawn configuration
SPAWN_AREA_MIN = 0.0 + 15
SPAWN_AREA_MAX = 32.0 - 15
MIN_SPAWN_DISTANCE = 12.0
MAX_SPAWN_DISTANCE = 12.0

# Number of zerglings
NUM_ZERGLINGS = 2

# Observation space size:
# Time: time_remaining (1) = 1
# Marine: own_health (1), weapon_cooldown (1), weapon_cooldown_norm (1) = 3
# Per zergling (x2): distance (1), health (1), angle_sin (1), angle_cos (1) = 4
# Total: 1 + 3 + 4*2 = 12
OBS_SIZE = 12


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
        self.action_space = spaces.Dict(
            {
                "command": spaces.Discrete(NUM_COMMANDS),
                "angle": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),  # sin, cos
            }
        )
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)

        self.game = SC2SingleGame([Terran, Zerg]).launch()
        self.client = self.game.clients[0]
        self.units: dict[int, list] = defaultdict(list)
        self.current_step: int = 0
        self.terminated: bool = False

        self.upgrade_level = random.choice(self.params.get("upgrade_level", [0, 1, 2]))
        self.game_steps_per_env = self.params.get("game_steps_per_env", 2)
        self.hp_multiplier = 1
        self.init_game()

    def init_game(self) -> None:
        logger.info(f"Initializing SC2GymEnv with params: {self.params}")
        self.client.enable_enemy_control()
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
        logger.debug(f"Marine alive: {len(self.units[1])}, Zerglings alive: {len(self.units[2])}")

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
        self.client.spawn_units(UNIT_ZERGLING, (ling_x, ling_y), owner=2, quantity=2)
        # shift_x = random.randint(-2500, 2500) / 1000.0
        # shift_y = random.randint(-2500, 2500) / 1000.0
        # self.client.spawn_units(UNIT_ZERGLING, (ling_x + shift_x, ling_y + shift_y), owner=2, quantity=1)

        self.game.step(count=2)  # unit spawn takes two frames

        self.observe_units()
        assert len(self.units[1]) == 1, "Marine not spawned correctly"
        assert len(self.units[2]) == 2, f"Expected 2 Zerglings, got {len(self.units[2])}"

        # Set marine gandicap
        marine = self.units[1][0]
        if self.hp_multiplier != 1.0:
            self.client.set_unit_life(marine.tag, marine.health_max * self.hp_multiplier)

        # Command zerglings to attack marine
        # for zergling in self.units[2]:
        #     self.client.unit_attack_unit(zergling.tag, marine.tag)

        logger.debug(f"Spawned Marine and 2 Zerglings (HP multiplier: {self.hp_multiplier})")

    def _get_zergling_obs(self, marine: UnitState, zergling: UnitState | None) -> list[float]:
        """Get observation features for a single zergling relative to marine.

        Returns [health, sin(relative_angle), cos(relative_angle), distance]
        All angles are relative to the reference direction.
        """
        if zergling is None:
            return [0.0, 0.0, 0.0, 0.0]

        # Health (normalized)
        enemy_health = zergling.health / zergling.health_max

        angle = marine.angle_to(zergling)
        angle_sin, angle_cos = math.sin(angle), math.cos(angle)

        # Distance (normalized, max meaningful distance ~20)
        distance = marine.distance_to(zergling)
        distance_norm = min(1.0, distance / 30.0)

        return [enemy_health, angle_sin, angle_cos, distance_norm]

    def _compute_observation(self) -> np.ndarray:
        if not self.units[1]:
            return np.zeros(OBS_SIZE, dtype=np.float32)

        marine = self.units[1][0]
        zerglings = self.units[2]

        # Marine's own state
        own_health = marine.health / marine.health_max
        weapon_cooldown = int(bool(marine.weapon_cooldown))
        weapon_cooldown_norm = min(1.0, marine.weapon_cooldown / 15.0)

        # Get zergling observations (pad with None if dead)
        if zerglings:
            z1 = zerglings[0] if len(zerglings) > 0 else None
            z2 = zerglings[1] if len(zerglings) > 1 else None
        else:
            z1, z2 = None, None

        z1_obs = self._get_zergling_obs(marine, z1)
        z2_obs = self._get_zergling_obs(marine, z2)

        # Time remaining (1.0 -> 0.0 as episode progresses)
        time_remaining = 1.0 - (self.current_step / MAX_EPISODE_STEPS)

        logger.debug(f"Marine health: {own_health}, cooldown: {weapon_cooldown_norm}")
        logger.debug(f"Zergling 1 obs: {z1_obs}")
        logger.debug(f"Zergling 2 obs: {z2_obs}")
        logger.debug(f"Time remaining: {time_remaining}")

        obs = [time_remaining, own_health, weapon_cooldown, weapon_cooldown_norm, *z1_obs, *z2_obs]

        return np.array(obs, dtype=np.float32)

    def _compute_reward(self) -> float:
        """Compute reward based on damage dealt/taken and terminal conditions"""

        if self.current_step >= MAX_EPISODE_STEPS:
            return -1

        if not self.units[2]:
            return 1

        if not self.units[1]:
            ally_health = sum([u.health / u.health_max for u in self.units[1]], 0.0) / 1.0
            enemy_max_health = ZERGLING_MAX_HP * 2
            enemy_health_left = sum([u.health for u in self.units[2]], 0.0)
            return ally_health - (enemy_health_left / enemy_max_health)

        return 0

    def _agent_action(self, action: dict) -> None:
        """Execute hybrid action: discrete command + continuous angle."""
        command = action["command"]
        angle_sincos = action["angle"]

        logger.debug(f"Agent action: command={command}, angle_sincos={angle_sincos}")

        marine = self.units[1][0]
        zerglings = self.units[2]

        if command == ACTION_MOVE:
            dy, dx = angle_sincos
            target_x = marine.x + dx * MOVE_STEP_SIZE
            target_y = marine.y + dy * MOVE_STEP_SIZE
            self.client.unit_move(marine.tag, (target_x, target_y))
        elif command == ACTION_ATTACK_Z1:
            if len(zerglings) > 0:
                self.client.unit_attack_unit(marine.tag, zerglings[0].tag)
        elif command == ACTION_ATTACK_Z2:
            if len(zerglings) > 1:
                self.client.unit_attack_unit(marine.tag, zerglings[1].tag)
        else:
            raise ValueError(f"Invalid command: {command}")

    def get_action_mask(self) -> np.ndarray:
        """ACTIONS=[MOVE, ATTACK_Z1, ATTACK_Z2]"""
        mask = np.ones(NUM_COMMANDS, dtype=bool)
        marine = self.units[1][0] if self.units[1] else None
        zerglings = self.units[2]
        # alive and in attack range
        mask[ACTION_ATTACK_Z1] = len(zerglings) > 0 and marine and zerglings[0].distance_to(marine) < 6
        mask[ACTION_ATTACK_Z2] = len(zerglings) > 1 and marine and zerglings[1].distance_to(marine) < 6
        return mask

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.current_step = 0
        self.terminated = False
        self.clean_battlefield()
        self.prepare_battlefield()
        return self._compute_observation(), {"action_mask": self.get_action_mask()}

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
        return obs, reward, self.terminated, truncated, {"won": reward > 0, "action_mask": self.get_action_mask()}

    def close(self) -> None:
        self.game.close()
