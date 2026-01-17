"""
Marine vs 2 Zerglings RL Environment

A Gymnasium environment for training a Marine agent against 2 scripted Zerglings.
Uses vector observations and discrete action space:
    - 0 to num_move_directions-1: MOVE in direction (angle = i * 360°/num_move_directions)
    - num_move_directions: ATTACK_Z1
    - num_move_directions+1: ATTACK_Z2
    - num_move_directions+2: STOP
    - num_move_directions+3: SKIP (no-op, continue previous action)

Default: 4 move directions (E, N, W, S at 0°, 90°, 180°, 270°) + 4 other = 8 actions
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

# Default action space (4 move directions)
# Can be overridden via params["num_move_directions"]
DEFAULT_NUM_MOVE_DIRECTIONS = 4

# Movement step size (how far to move per action)
MOVE_STEP_SIZE = 2.0

# Environment constants
MAX_EPISODE_STEPS = 22.4 * 30  # 30 realtime seconds

# Spawn configuration
SPAWN_AREA_MIN = 0.0 + 15
SPAWN_AREA_MAX = 32.0 - 15
MIN_SPAWN_DISTANCE = 5.0
MAX_SPAWN_DISTANCE = 8.0

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
    """1 Marine vs 2 Zerglings environment with discrete action space.

    Action space: Discrete(num_move_directions + 4)
        - 0 to num_move_directions-1: MOVE in direction
        - num_move_directions: ATTACK_Z1
        - num_move_directions+1: ATTACK_Z2
        - num_move_directions+2: STOP (halt unit)
        - num_move_directions+3: SKIP (no-op, continue previous action)

    Default: 4 move directions (E=0, N=1, W=2, S=3) giving 8 total actions.
    """

    metadata = {"name": "sc2_mv2z_v3_discrete", "render_modes": []}

    def __init__(self, params=None) -> None:
        super().__init__()
        self.params = params or {}

        # Configurable action space
        self.num_move_directions = self.params.get("num_move_directions", DEFAULT_NUM_MOVE_DIRECTIONS)
        self.num_actions = self.num_move_directions + 4

        # Compute action indices
        self.action_attack_z1 = self.num_move_directions
        self.action_attack_z2 = self.num_move_directions + 1
        self.action_stop = self.num_move_directions + 2
        self.action_skip = self.num_move_directions + 3

        # Discrete action space
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)

        self.game = SC2SingleGame([Terran, Zerg]).launch()
        self.client = self.game.clients[0]
        self.units: dict[int, list] = defaultdict(list)
        self.current_step: int = 0
        self.terminated: bool = False

        self.upgrade_level = random.choice(self.params.get("upgrade_level", [0, 1, 2]))
        self.game_steps_per_env = 2
        self.hp_multiplier = 1
        self.init_game()

    def _action_to_angle(self, action: int) -> float:
        """Convert move action index to angle in radians.

        Args:
            action: Action index (0 to num_move_directions-1)

        Returns:
            Angle in radians (0 = east/right, counter-clockwise)
        """
        if 0 <= action < self.num_move_directions:
            return action * (2 * math.pi / self.num_move_directions)
        raise ValueError(f"Action {action} is not a move action")

    def _is_move_action(self, action: int) -> bool:
        """Check if action is a move action."""
        return 0 <= action < self.num_move_directions

    def _is_attack_action(self, action: int) -> bool:
        """Check if action is an attack action."""
        return action in (self.action_attack_z1, self.action_attack_z2)

    def init_game(self) -> None:
        logger.info(f"Initializing SC2GymEnv with params: {self.params}")
        logger.info(f"Action space: {self.num_actions} actions ({self.num_move_directions} move + 4 other)")
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
        # Spawn 2 zerglings nearby
        self.client.spawn_units(UNIT_ZERGLING, (ling_x, ling_y), owner=2, quantity=2)

        self.game.step(count=2)  # unit spawn takes two frames

        self.observe_units()
        assert len(self.units[1]) == 1, "Marine not spawned correctly"
        assert len(self.units[2]) == 2, f"Expected 2 Zerglings, got {len(self.units[2])}"

        # Set marine handicap
        marine = self.units[1][0]
        if self.hp_multiplier != 1.0:
            self.client.set_unit_life(marine.tag, marine.health_max * self.hp_multiplier)

        logger.debug(f"Spawned Marine and 2 Zerglings (HP multiplier: {self.hp_multiplier})")

    def _get_zergling_obs(self, marine: UnitState, zergling: UnitState | None) -> list[float]:
        """Get observation features for a single zergling relative to marine.

        Returns [health, sin(angle), cos(angle), distance]
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
        """Compute reward based on terminal conditions.

        Only gives reward at episode end:
        - Win: 1 - (enemy_health_left / enemy_max_health)
        - Timeout: small penalty
        """
        if self.current_step >= MAX_EPISODE_STEPS:
            return -0.01 * self.current_step * self.game_steps_per_env

        if not self.units[1] or not self.units[2]:
            # Terminal state
            enemy_max_health = ZERGLING_MAX_HP * 2
            enemy_health_left = sum([u.health for u in self.units[2]], 0.0)
            return 1 - (enemy_health_left / enemy_max_health)

        return 0

    def _agent_action(self, action: int) -> None:
        """Execute discrete action."""
        logger.debug(f"Agent action: {action}")

        marine = self.units[1][0]
        zerglings = self.units[2]

        if self._is_move_action(action):
            # Move in specified direction
            angle = self._action_to_angle(action)
            dx = math.cos(angle) * MOVE_STEP_SIZE
            dy = math.sin(angle) * MOVE_STEP_SIZE
            target_x = marine.x + dx
            target_y = marine.y + dy
            self.client.unit_move(marine.tag, (target_x, target_y))

        elif action == self.action_attack_z1:
            if len(zerglings) > 0:
                self.client.unit_attack_unit(marine.tag, zerglings[0].tag)

        elif action == self.action_attack_z2:
            if len(zerglings) > 1:
                self.client.unit_attack_unit(marine.tag, zerglings[1].tag)

        elif action == self.action_stop:
            self.client.unit_stop(marine.tag)

        elif action == self.action_skip:
            # Do nothing - continue previous action
            pass

        else:
            raise ValueError(f"Invalid action: {action}")

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.current_step = 0
        self.terminated = False
        self.clean_battlefield()
        self.prepare_battlefield()
        return self._compute_observation(), {}

    def step(self, action: int) -> tuple:
        logger.debug(f"Environment step {self.current_step} with action {action}")

        if not self.terminated:
            self._agent_action(action)
            self.game.step(count=self.game_steps_per_env)
            self.current_step += self.game_steps_per_env

        self.observe_units()
        obs = self._compute_observation()
        reward = self._compute_reward()
        truncated = self.current_step >= MAX_EPISODE_STEPS
        # Win = marine survived AND all zerglings dead
        won = len(self.units[1]) > 0 and len(self.units[2]) == 0
        return obs, reward, self.terminated, truncated, {"won": won}

    def close(self) -> None:
        self.game.close()


# Module-level helpers for backwards compatibility and external use
def get_num_actions(num_move_directions: int = DEFAULT_NUM_MOVE_DIRECTIONS) -> int:
    """Get total number of actions for given move directions."""
    return num_move_directions + 4


def get_action_indices(num_move_directions: int = DEFAULT_NUM_MOVE_DIRECTIONS) -> dict[str, int]:
    """Get action index mapping for given move directions."""
    return {
        "attack_z1": num_move_directions,
        "attack_z2": num_move_directions + 1,
        "stop": num_move_directions + 2,
        "skip": num_move_directions + 3,
    }
