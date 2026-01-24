"""
Marine vs 2 Zerglings RL Environment

A Gymnasium environment for training a Marine agent against 2 scripted Zerglings.
Uses vector observations and hybrid action space:
- Discrete commands: MOVE, ATTACK_Z1, ATTACK_Z2
- Continuous angle (sin, cos) for MOVE command in world-space polar coords

The observation structure is defined by ObsSpec, which provides:
- Observation layout (sizes, slices)
- Encoding logic (raw state → normalized features)
- Auxiliary prediction targets for the model

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

from .obs import ObsSpec

logger.remove()
logger.add(sys.stderr, level="INFO")


# Unit type IDs (from SC2 API)
UNIT_MARINE = 48
UNIT_ZERGLING = 105

# Unit attack ranges, we are using "game value + 1.0" for stability
UNIT_ATTACK_RANGE: dict[int, float] = {
    UNIT_MARINE: 5.0 + 1.0,
    UNIT_ZERGLING: 0.0 + 1.0,
}

# Movement step size (how far to move per action)
MOVE_STEP_SIZE = 2.0

# Environment constants
MAX_EPISODE_STEPS = 22.4 * 30  # 30 realtime seconds

# Spawn configuration
SPAWN_AREA_MIN = 32.0
SPAWN_AREA_MAX = 32.0
MIN_SPAWN_DISTANCE = 7.0
MAX_SPAWN_DISTANCE = 14.0


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
    type_id: int
    x: float
    y: float
    health: float
    health_max: float
    attack_range: float
    weapon_cooldown: float = 0.0
    facing: float = 0.0  # In radians (0 = East/+X, π/2 = North/+Y)

    @classmethod
    def from_proto(cls, unit) -> "UnitState":
        return cls(
            tag=unit.tag,
            type_id=unit.unit_type,
            x=unit.pos.x,
            y=unit.pos.y,
            facing=unit.facing,
            health=unit.health,
            health_max=unit.health_max,
            attack_range=UNIT_ATTACK_RANGE[unit.unit_type],
            weapon_cooldown=getattr(unit, "weapon_cooldown", 0.0),
        )

    def distance_to(self, other: "UnitState") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def angle_to(self, other: "UnitState") -> float:
        """Returns angle in radians from self to other."""
        return math.atan2(other.y - self.y, other.x - self.x)


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

    # === Observation Spec ===
    obs_spec: ObsSpec = ObsSpec()

    def __init__(self, params=None) -> None:
        super().__init__()
        self.params = params or {}

        # Hybrid action space: discrete command + continuous angle
        command = spaces.Discrete(self.NUM_COMMANDS)
        angle = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # sin, cos
        self.action_space = spaces.Dict({"command": command, "angle": angle})
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_spec.total_size,), dtype=np.float32)

        self.game = SC2SingleGame([Terran, Zerg]).launch()
        self.client = self.game.clients[0]
        self.units: dict[int, list[UnitState]] = defaultdict(list)
        self.current_step: int = 0
        self.terminated: bool = False

        self.damage = DamageTracker()
        self.upgrade_level = random.choice(self.params.get("upgrade_level", [1]))
        self.game_steps_per_env = self.params.get("game_steps_per_env", 4)
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

        # for zergling in self.units[2]:  # trigger everyone attack on first hit
        #     if zergling.health < zergling.health_max and self.units[1]:
        #         self.client.unit_attack_unit(zergling.tag, self.units[1][0].tag)

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

        # Spawn zerglings at random distance/angle from marine
        for _ in range(self.obs_spec.num_enemies):
            spawn_distance = random.uniform(MIN_SPAWN_DISTANCE, MAX_SPAWN_DISTANCE)
            spawn_angle = random.uniform(0, 2 * math.pi)
            ling_x = marine_x + spawn_distance * math.cos(spawn_angle)
            ling_y = marine_y + spawn_distance * math.sin(spawn_angle)
            self.client.spawn_units(UNIT_ZERGLING, (ling_x, ling_y), owner=2, quantity=1)

        self.game.step(count=2)  # unit spawn takes two frames

        self.observe_units()
        self.enemy_max_health = sum(u.health_max for u in self.units[2])  # cached
        assert len(self.units[1]) == 1, "Allies not spawned correctly"
        assert len(self.units[2]) == self.obs_spec.num_enemies, "Enemies not spawned correctly"

    def _compute_observation(self) -> np.ndarray:
        """Build observation using ObsSpec."""
        if not self.units[1]:
            return np.zeros(self.obs_spec.total_size, dtype=np.float32)

        time_remaining = 1.0 - (self.current_step / MAX_EPISODE_STEPS)
        obs = self.obs_spec.build(time_remaining=time_remaining, allies=self.units[1], enemies=self.units[2])
        logger.debug(f"Observation: {obs}")
        return obs

    def _compute_reward(self) -> float:
        """Compute reward based on damage dealt/taken and terminal conditions"""
        if self.current_step >= MAX_EPISODE_STEPS:
            return -1

        # Get max health from actual units (or use cached value from episode start)
        if not self.units[2] or not self.units[1]:
            ally_health = sum([u.health / u.health_max for u in self.units[1]], 0.0) / 1.0
            enemy_health_left = sum([u.health for u in self.units[2]], 0.0)
            difference = ally_health - (enemy_health_left / self.enemy_max_health)
            return difference if difference <= 0 else (1 + difference) ** 2

        return self.damage.dealt_step / self.enemy_max_health / 10  # EPISODE_TOTAL = [0;0.1] weak but useful signal

    def agent_action(self, action: dict) -> None:
        """Execute hybrid action: discrete command + continuous angle."""
        command = action["command"]
        marine = self.units[1][0]
        zerglings = self.units[2]

        if command == self.ACTION_MOVE:
            # Normalize angle to unit circle (model outputs raw values for correct gradients)
            # Action space is [sin, cos] following standard Cartesian (0=East/+X, π/2=North/+Y)
            raw_sin, raw_cos = action["angle"]
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
        mask[self.ACTION_ATTACK_Z1] = (
            len(zerglings) > 0 and ready and zerglings[0].distance_to(marine) < marine.attack_range
        )
        mask[self.ACTION_ATTACK_Z2] = (
            len(zerglings) > 1 and ready and zerglings[1].distance_to(marine) < marine.attack_range
        )
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
            self.agent_action(action)
            self.game.step(count=self.game_steps_per_env)
            self.current_step += self.game_steps_per_env

        self.observe_units()
        won = len(self.units[2]) == 0
        obs = self._compute_observation()
        reward = self._compute_reward()
        truncated = self.current_step >= MAX_EPISODE_STEPS
        action_mask = self.get_action_mask()
        return obs, reward, self.terminated, truncated, {"won": won, "action_mask": action_mask}

    def close(self) -> None:
        self.game.close()
