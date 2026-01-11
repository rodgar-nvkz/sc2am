"""
Marine vs Zergling RL Environment

A PettingZoo-compatible environment for training a Marine agent against a scripted Zergling.
Uses vector observations and discrete action space (8 directions + stay + attack).
"""

import math
import random
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pettingzoo
from gymnasium import spaces

from scam.infra.client import SC2Client
from scam.infra.game import SC2SingleGame, Terran, Zerg

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
MAX_EPISODE_STEPS = 22.4 * 20  # 20 realtime seconds

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
    GAME_STEPS_PER_ENV_STEP = 4

    def __init__(self, env_ctx: dict | None = None) -> None:
        super().__init__()

        self.agents = ["marine"]
        self.possible_agents = ["marine"]
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        # # Required for SuperSuit compatibility
        # self.render_mode = None

        self._game = None
        self._client = None
        self.marine: UnitState | None = None
        self.zergling: UnitState | None = None
        self.prev_marine_hp: float = 0.0
        self.prev_zergling_hp: float = 0.0
        self.current_step: int = 0
        self._episode_ended: bool = False

    @property
    def game(self):
        if not self._game:
            self._game = SC2SingleGame([Terran, Zerg]).launch()
        return self._game

    @property
    def client(self):
        if not self._client:
            self._client = self.game.clients[0]
        return self._client

    def _get_units(self) -> tuple[UnitState | None, UnitState | None]:
        """Get current unit states from observation."""
        obs = self.client.get_observation()
        marine = None
        zergling = None

        for unit in obs.observation.raw_data.units:
            if unit.unit_type == UNIT_MARINE and unit.owner == 1:
                marine = UnitState.from_proto(unit)
            elif unit.unit_type == UNIT_ZERGLING and unit.owner == 2:
                zergling = UnitState.from_proto(unit)

        return marine, zergling

    def _compute_observation(self) -> np.ndarray:
        """Compute the observation vector for the Marine agent."""
        if self.marine is None or self.zergling is None:
            return np.zeros(8, dtype=np.float32)

        # Relative position (normalized by ~16 units for local awareness)
        rel_x = (self.zergling.x - self.marine.x) / 16.0
        rel_y = (self.zergling.y - self.marine.y) / 16.0

        # Clamp to [-1, 1]
        rel_x = max(-1.0, min(1.0, rel_x))
        rel_y = max(-1.0, min(1.0, rel_y))

        # Distance (normalized, max meaningful distance ~20)
        distance = self.marine.distance_to(self.zergling)
        distance_norm = min(1.0, distance / 20.0)

        # Health (normalized)
        own_health = self.marine.health / MARINE_MAX_HP
        enemy_health = self.zergling.health / ZERGLING_MAX_HP

        # Angle to enemy (normalized to [-1, 1])
        angle = self.marine.angle_to(self.zergling)
        angle_norm = angle / math.pi

        # In attack range (binary)
        in_range = 1.0 if distance <= MARINE_RANGE else 0.0

        # Weapon ready (binary) - cooldown is in game loops, 0 means ready
        weapon_ready = 1.0 if self.marine.weapon_cooldown <= 0 else 0.0

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
        reward = 0.0

        # Check terminal conditions first (units may be None if dead)
        marine_dead = self.marine is None or self.marine.health <= 0
        zergling_dead = self.zergling is None or self.zergling.health <= 0

        # Terminal rewards - large to dominate the reward signal
        if zergling_dead:
            reward += 10.0
        if marine_dead:
            reward -= 10.0

        # Timeout penalty - discourage stalling/avoiding combat
        timeout = not marine_dead and not zergling_dead and self.current_step >= MAX_EPISODE_STEPS
        if timeout:
            reward -= 15.0

        # Damage-based rewards (only if both units exist)
        if self.marine is not None and self.zergling is not None:
            # Damage dealt to enemy (positive reward)
            damage_dealt = max(0.0, self.prev_zergling_hp - self.zergling.health)
            reward += damage_dealt * 0.05  # Scale factor

            # Damage taken (negative reward) - penalize more heavily
            damage_taken = max(0.0, self.prev_marine_hp - self.marine.health)
            reward -= damage_taken * 0.08

        return reward

    def _agent_action(self, action: int) -> None:
        """Execute the Marine's action."""
        if self.marine is None or not self.marine.is_alive:
            return

        if action == ACTION_ATTACK:
            if self.zergling is not None and self.zergling.is_alive:
                self.client.unit_attack_unit(self.marine.tag, self.zergling.tag)
        elif action in DIRECTION_VECTORS:
            dx, dy = DIRECTION_VECTORS[action]
            if dx == 0.0 and dy == 0.0:
                self.client.unit_stop(self.marine.tag)
            else:
                target_x = self.marine.x + dx * MOVE_STEP_SIZE
                target_y = self.marine.y + dy * MOVE_STEP_SIZE
                self.client.unit_move(self.marine.tag, (target_x, target_y))

    def _spawn_units(self, seed: int | None = None) -> None:
        """Spawn Marine and Zergling at random positions."""

        # Random position for marine (centered in map, away from edges/structures)
        marine_x = random.uniform(SPAWN_AREA_MIN, SPAWN_AREA_MAX)
        marine_y = random.uniform(SPAWN_AREA_MIN, SPAWN_AREA_MAX)
        # logger.debug(f"Marine spawn position: ({marine_x:.2f}, {marine_y:.2f})")

        # Zergling spawns at random distance/angle from marine
        spawn_distance = random.uniform(MIN_SPAWN_DISTANCE, MAX_SPAWN_DISTANCE)
        spawn_angle = random.uniform(0, 2 * math.pi)

        ling_x = marine_x + spawn_distance * math.cos(spawn_angle)
        ling_y = marine_y + spawn_distance * math.sin(spawn_angle)
        # logger.debug(f"Zergling spawn position: ({ling_x:.2f}, {ling_y:.2f})")

        # Spawn units
        self.client.spawn_units(UNIT_MARINE, (marine_x, marine_y), owner=1, quantity=1)
        self.client.spawn_units(UNIT_ZERGLING, (ling_x, ling_y), owner=2, quantity=1)
        self.game.step(count=3)  # require at least 3 steps to spawn units

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment for a new episode."""
        self.current_step = 0
        self._episode_ended = False

        obs = self.client.get_observation()
        units = [u.tag for u in obs.observation.raw_data.units if u.unit_type in (UNIT_MARINE, UNIT_ZERGLING)]
        self.client.kill_units(units)

        self._spawn_units(seed)

        # # Get initial unit states
        self.marine, self.zergling = self._get_units()
        assert self.marine is not None, "Marine unit not found after spawn."
        assert self.zergling is not None, "Zergling unit not found after spawn."

        # Initialize HP tracking for reward computation
        self.prev_marine_hp = self.marine.health if self.marine else MARINE_MAX_HP
        self.prev_zergling_hp = self.zergling.health if self.zergling else ZERGLING_MAX_HP

        # Compute initial observation
        obs = self._compute_observation()
        info = {}
        return obs, info

    def step(self, action: int) -> tuple:
        """Execute one environment step."""
        self.current_step += 1

        # If episode already ended, return terminal state without stepping the game
        if self._episode_ended:
            obs = self._compute_observation()
            return obs, 0.0, True, False, {"won": False}

        # Store previous HP for reward computation
        self.prev_marine_hp = self.marine.health if self.marine else 0.0
        self.prev_zergling_hp = self.zergling.health if self.zergling else 0.0

        self._agent_action(action)
        self.game.step(count=self.GAME_STEPS_PER_ENV_STEP)

        # Get updated unit states
        self.marine, self.zergling = self._get_units()

        # Compute observation
        obs = self._compute_observation()

        # Compute reward
        reward = self._compute_reward()

        # Check termination conditions
        marine_dead = self.marine is None or self.marine.health <= 0
        zergling_dead = self.zergling is None or self.zergling.health <= 0
        terminated = marine_dead or zergling_dead

        # Check truncation (max steps)
        truncated = self.current_step >= MAX_EPISODE_STEPS

        # Mark episode as ended to prevent further game steps
        if terminated or truncated:
            self._episode_ended = True

        return obs, reward, terminated, truncated, {"won": zergling_dead and not marine_dead}

    def close(self) -> None:
        self.game.close()
