"""
Marine vs 2 Zerglings RL Environment

A Gymnasium environment for training a Marine agent against 2 scripted Zerglings.
Uses vector observations and hybrid action space:
- Discrete commands: MOVE, ATTACK_Z1, ATTACK_Z2
- Continuous angle (sin, cos) for MOVE command in world-space polar coords

The observation and action spaces use a relative reference frame based on the
direction to the closest enemy, enabling rotation-invariant learning.

All observation and action space constants are defined as class-level attributes
in SC2GymEnv, allowing easy propagation to model/training code.
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
ZERGLING_MAX_HP = 35.0
MARINGE_SIGHT = 9.0

# Movement step size (how far to move per action)
MOVE_STEP_SIZE = 2.0

# Environment constants
MAX_EPISODE_STEPS = 22.4 * 30  # 30 realtime seconds

# Spawn configuration
SPAWN_AREA_MIN = 32.0
SPAWN_AREA_MAX = 32.0
MIN_SPAWN_DISTANCE = 7.0
MAX_SPAWN_DISTANCE = 14.0

# Number of zerglings
NUM_ZERGLINGS = 2


@dataclass
class DamageTracker:
    """Tracks cumulative and per-step damage dealt/taken from sc2clientprotocol score data."""

    dealt_acc: float = 0.0
    taken_acc: float = 0.0
    dealt_step: float = 0.0
    taken_step: float = 0.0

    def update(self, score_details) -> None:
        dealt = score_details.total_damage_dealt.life + score_details.total_damage_dealt.shields
        taken = score_details.total_damage_taken.life + score_details.total_damage_taken.shields
        self.dealt_step = dealt - self.dealt_acc
        self.taken_step = taken - self.taken_acc
        self.dealt_acc = dealt
        self.taken_acc = taken

    def reset(self) -> None:
        self.dealt_acc = 0.0
        self.taken_acc = 0.0
        self.dealt_step = 0.0
        self.taken_step = 0.0


@dataclass
class UnitState:
    """Represents the state of a unit."""

    tag: int
    x: float
    y: float
    health: float
    health_max: float
    weapon_cooldown: float = 0.0
    facing: float = 0.0  # In radians (0 = East/+X, π/2 = North/+Y)

    @classmethod
    def from_proto(cls, unit) -> "UnitState":
        return cls(
            tag=unit.tag,
            x=unit.pos.x,
            y=unit.pos.y,
            facing=unit.facing,
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
    - 'command': Discrete(3) - MOVE, ATTACK_Z1, ATTACK_Z2
    - 'angle': Box(2) - (sin, cos) of movement direction
    """

    metadata = {"name": "sc2_mv2z_v2_hybrid", "render_modes": []}

    # === Action Space Constants ===
    ACTION_MOVE = 0
    ACTION_ATTACK_Z1 = 1
    ACTION_ATTACK_Z2 = 2
    NUM_COMMANDS = 3
    MOVE_ACTION_ID = ACTION_MOVE

    # === Observation Space Constants ===
    # Time: time_remaining (1) = 1
    # Marine: own_health (1), weapon_cooldown (1), weapon_cooldown_norm (1), facing_sin (1), facing_cos (1) = 5
    # Per zergling (x2): health (1), angle_sin (1), angle_cos (1), distance (1), in_attack_range(1), facing_sin (1), facing_cos (1) = 7
    # Total: 1 + 5 + 7*2 = 24
    OBS_SIZE = 20

    def __init__(self, params=None) -> None:
        super().__init__()
        self.params = params or {}

        # Hybrid action space: discrete command + continuous angle
        self.action_space = spaces.Dict(
            {
                "command": spaces.Discrete(self.NUM_COMMANDS),
                "angle": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),  # sin, cos
            }
        )
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.OBS_SIZE,), dtype=np.float32)

        self.game = SC2SingleGame([Terran, Zerg]).launch()
        self.client = self.game.clients[0]
        self.units: dict[int, list[UnitState]] = defaultdict(list)
        self.current_step: int = 0
        self.terminated: bool = False

        # Damage tracking for per-step rewards
        self.damage = DamageTracker()

        self.upgrade_level = random.choice(self.params.get("upgrade_level", [0, 1, 2]))
        self.game_steps_per_env = self.params.get("game_steps_per_env", 7)
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
        self.damage.update(obs.observation.score.score_details)

        for unit in obs.observation.raw_data.units:
            if unit.unit_type not in (UNIT_MARINE, UNIT_ZERGLING) or unit.health <= 0:
                continue
            unit_state = UnitState.from_proto(unit)
            self.units[unit.owner].append(unit_state)

        for units in self.units.values():
            units.sort(key=lambda u: u.tag)

        for zergling in self.units[2]:  # trigger everyone attack on first hit
            if zergling.health < zergling.health_max and self.units[1]:
                self.client.unit_attack_unit(zergling.tag, self.units[1][0].tag)

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

        # Spawn 2 zergling nearby at random distance/angle from marine
        for _ in range(NUM_ZERGLINGS):
            spawn_distance = random.uniform(MIN_SPAWN_DISTANCE, MAX_SPAWN_DISTANCE)
            spawn_angle = random.uniform(0, 2 * math.pi)
            ling_x = marine_x + spawn_distance * math.cos(spawn_angle)
            ling_y = marine_y + spawn_distance * math.sin(spawn_angle)
            self.client.spawn_units(UNIT_ZERGLING, (ling_x, ling_y), owner=2, quantity=1)

        self.game.step(count=2)  # unit spawn takes two frames

        self.observe_units()
        assert len(self.units[1]) == 1, "Marine not spawned correctly"
        assert len(self.units[2]) == 2, f"Expected 2 Zerglings, got {len(self.units[2])}"

        # Command zerglings to attack marine
        # for zergling in self.units[2]:
        #     self.client.unit_attack_unit(zergling.tag, self.units[1][0].tag)

    def _get_zergling_obs(self, marine: UnitState, zergling: UnitState | None) -> list[float]:
        """Get observation features for a single zergling relative to marine"""
        if zergling is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Health (normalized)
        enemy_health = zergling.health / zergling.health_max

        angle = marine.angle_to(zergling)
        angle_sin, angle_cos = math.sin(angle), math.cos(angle)

        # Distance (normalized, max meaningful distance ~20)
        distance = marine.distance_to(zergling)
        distance_norm = min(1.0, distance / 30.0)
        in_attack_range = distance < MARINGE_SIGHT

        # Zergling facing (sin, cos encoding)
        facing_sin, facing_cos = math.sin(zergling.facing), math.cos(zergling.facing)

        return [
            enemy_health,
            angle_sin,
            angle_cos,
            distance_norm,
            in_attack_range,
            facing_sin,
            facing_cos,
        ]

    def _compute_observation(self) -> np.ndarray:
        if not self.units[1]:
            return np.zeros(self.OBS_SIZE, dtype=np.float32)

        marine = self.units[1][0]
        zerglings = self.units[2]

        # Marine's own state
        own_health = marine.health / marine.health_max
        weapon_cooldown = int(bool(marine.weapon_cooldown))
        weapon_cooldown_norm = min(1.0, marine.weapon_cooldown / 15.0)
        marine_facing_sin, marine_facing_cos = math.sin(marine.facing), math.cos(marine.facing)

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

        obs = [
            time_remaining,
            own_health,
            weapon_cooldown,
            weapon_cooldown_norm,
            marine_facing_sin,
            marine_facing_cos,
            *z1_obs,
            *z2_obs,
        ]

        return np.array(obs, dtype=np.float32)

    def _compute_reward(self) -> float:
        """Compute reward based on damage dealt/taken and terminal conditions"""

        if self.current_step >= MAX_EPISODE_STEPS:
            return -1

        enemy_max_health = ZERGLING_MAX_HP * NUM_ZERGLINGS
        if not self.units[2] or not self.units[1]:
            ally_health = sum([u.health / u.health_max for u in self.units[1]], 0.0) / 1.0
            enemy_health_left = sum([u.health for u in self.units[2]], 0.0)
            difference = ally_health - (enemy_health_left / enemy_max_health)
            return difference if difference <= 0 else (1 + difference) ** 2

        return self.damage.dealt_step / enemy_max_health / 10  # EPISODE_TOTAL = [0;0.1] weak but usefull signal

    def _agent_action(self, action: dict) -> None:
        """Execute hybrid action: discrete command + continuous angle."""
        command = action["command"]
        angle_sincos = action["angle"]

        marine = self.units[1][0]
        zerglings = self.units[2]

        if command == self.ACTION_MOVE:
            # Normalize angle to unit circle (model outputs raw values for correct gradients)
            # Action space is [sin, cos] following standard Cartesian (0=East/+X, π/2=North/+Y)
            raw_sin, raw_cos = angle_sincos
            magnitude = math.sqrt(raw_sin**2 + raw_cos**2)
            dx, dy = raw_cos / magnitude, raw_sin / magnitude
            target_x = marine.x + dx * MOVE_STEP_SIZE
            target_y = marine.y + dy * MOVE_STEP_SIZE

            self.client.unit_move(marine.tag, (target_x, target_y))
        elif command == self.ACTION_ATTACK_Z1:
            if len(zerglings) > 0:
                self.client.unit_attack_unit(marine.tag, zerglings[0].tag)
        elif command == self.ACTION_ATTACK_Z2:
            if len(zerglings) > 1:
                self.client.unit_attack_unit(marine.tag, zerglings[1].tag)
        else:
            raise ValueError(f"Invalid command: {command}")

    def get_action_mask(self) -> np.ndarray:
        """ACTIONS=[MOVE, ATTACK_Z1, ATTACK_Z2]"""
        mask = np.ones(self.NUM_COMMANDS, dtype=bool)
        marine = self.units[1][0] if self.units[1] else None
        zerglings = self.units[2]
        if not marine:
            return mask

        # alive and in attack range
        ready = marine.weapon_cooldown == 0
        mask[self.ACTION_ATTACK_Z1] = len(zerglings) > 0 and ready and zerglings[0].distance_to(marine) < MARINGE_SIGHT
        mask[self.ACTION_ATTACK_Z2] = len(zerglings) > 1 and ready and zerglings[1].distance_to(marine) < MARINGE_SIGHT
        return mask

    def reset(self, *_, **__) -> tuple[np.ndarray, dict[str, Any]]:
        self.current_step = 0
        self.terminated = False
        self.damage.reset()
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
        won = len(self.units[2]) == 0
        obs = self._compute_observation()
        reward = self._compute_reward()
        truncated = self.current_step >= MAX_EPISODE_STEPS
        return obs, reward, self.terminated, truncated, {"won": won, "action_mask": self.get_action_mask()}

    def close(self) -> None:
        self.game.close()
