"""
Marine vs 2 Zerglings RL Environment

Task: Chase zerglings, attack, kite to avoid damage. Win/lose signal only.

Hybrid Action Space:
    command: Discrete(3) - MOVE, ATTACK_Z1, ATTACK_Z2
    angle: Box(2) - [sin, cos] for movement direction (used only with MOVE)

Observations (12-dim float32, normalized to [-1, 1]):
    [0]     time_left       - Episode progress (1.0 -> 0.0)
    [1]     marine_hp       - Marine health fraction
    [2]     cd_binary       - Weapon on cooldown (0 or 1)
    [3]     cd_norm         - Weapon cooldown normalized
    [4:8]   z1_obs          - [hp, sin(angle), cos(angle), dist] for enemy 1
    [8:12]  z2_obs          - [hp, sin(angle), cos(angle), dist] for enemy 2

Enemy Tracking:
    - EnemyTracker assigns fixed slots by tag on first observation
    - Dead enemies return [0, 0, 0, 0] but keep their slot (no shifting)
    - Tags preserved throughout episode for stable credit assignment

Action Masks (returned in info["action_mask"]):
    [MOVE, ATTACK_Z1, ATTACK_Z2] - bool array
    - MOVE: always True
    - ATTACK_Zx: True only if enemy alive AND in range (< MARINE_RANGE)

Reward:
    - Non-terminal steps: 0
    - Terminal: 1 - (enemy_hp_remaining / enemy_max_hp)
    - Timeout: 0

Episode Termination:
    - Marine dies (lose)
    - All zerglings die (win)
    - Timeout at MAX_EPISODE_STEPS

Spawn Configuration (edit SPAWN_DISTANCE for experiments):
    - < 8: In sight range, attack command leads to move-and-attack
    - > 12: Out of sight, must move toward enemy first
    - Default: 14 (out of sight)
"""

import math
import random
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from scam.infra.game import SC2SingleGame, Terran, Zerg

# Unit type IDs
UNIT_MARINE = 48
UNIT_ZERGLING = 105

# Unit stats
MARINE_MAX_HP = 45.0
MARINE_RANGE = 5.0
ZERGLING_MAX_HP = 35.0
NUM_ZERGLINGS = 2

# Actions
ACTION_MOVE = 0
ACTION_ATTACK_Z1 = 1
ACTION_ATTACK_Z2 = 2
ACTION_STOP = 3
NUM_COMMANDS = 4

# Movement
MOVE_STEP_SIZE = 2.0

# Episode limits
MAX_EPISODE_STEPS = int(22.4 * 30)  # 30 seconds realtime
GAME_STEPS_PER_ENV = 2

# Spawn config (edit by hand for experiments)
SPAWN_DISTANCE = 6.0

# Observation: time(1) + marine(hp, cd, cd_norm)(3) + z1(hp, sin, cos, dist)(4) + z2(4) = 12
OBS_SIZE = 12


@dataclass
class UnitState:
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
        return math.atan2(other.y - self.y, other.x - self.x)


class EnemyTracker:
    """Tracks enemies in fixed slots by tag.

    Once a tag is assigned to a slot, it stays there for the episode.
    Dead enemies return None but keep their slot assignment.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._slots: list[UnitState | None] = [None] * capacity
        self._tags: list[int | None] = [None] * capacity

    def reset(self) -> None:
        self._slots = [None] * self.capacity
        self._tags = [None] * self.capacity

    def update(self, units: list[UnitState]) -> None:
        """Update slots from observation.

        First call: assign tags to slots (sorted by tag for determinism).
        Subsequent calls: update by tag match, dead enemies become None.
        """
        if all(t is None for t in self._tags):
            # First observation - assign tags to slots
            sorted_units = sorted(units, key=lambda u: u.tag)
            for i, unit in enumerate(sorted_units[: self.capacity]):
                self._tags[i] = unit.tag
                self._slots[i] = unit
        else:
            # Update by tag match
            tag_to_unit = {u.tag: u for u in units}
            for i, tag in enumerate(self._tags):
                if tag is not None and tag in tag_to_unit:
                    self._slots[i] = tag_to_unit[tag]
                else:
                    self._slots[i] = None

    def __getitem__(self, index: int) -> UnitState | None:
        return self._slots[index]

    def get_tag(self, index: int) -> int | None:
        return self._tags[index]

    def is_alive(self, index: int) -> bool:
        return self._slots[index] is not None

    def all_dead(self) -> bool:
        return all(s is None for s in self._slots)

    def total_health(self) -> float:
        return sum(s.health for s in self._slots if s is not None)


class SC2GymEnv(gym.Env):
    """1 Marine vs 2 Zerglings with hybrid action space."""

    metadata = {"name": "smarte_v1", "render_modes": []}

    def __init__(self, params: dict | None = None) -> None:
        super().__init__()
        self.params = params or {}

        self.action_space = spaces.Dict({
            "command": spaces.Discrete(NUM_COMMANDS),
            "angle": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        })
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)

        self.game = SC2SingleGame([Terran, Zerg]).launch()
        self.client = self.game.clients[0]
        self.client.enable_enemy_control()

        self.marine: UnitState | None = None
        self.enemies = EnemyTracker(NUM_ZERGLINGS)
        self.current_step = 0
        self.terminated = False

    def _observe_units(self) -> None:
        obs = self.client.get_observation()

        marine_list = []
        enemy_list = []

        for unit in obs.observation.raw_data.units:
            if unit.health <= 0:
                continue
            if unit.unit_type == UNIT_MARINE and unit.owner == 1:
                marine_list.append(UnitState.from_proto(unit))
            elif unit.unit_type == UNIT_ZERGLING and unit.owner == 2:
                enemy_list.append(UnitState.from_proto(unit))

        self.marine = marine_list[0] if marine_list else None
        self.enemies.update(enemy_list)
        self.terminated = self.marine is None or self.enemies.all_dead()

    def _clean_battlefield(self) -> None:
        # Query ALL units on the map and kill them, not just tracked ones
        # This prevents leftover units from previous episodes
        obs = self.client.get_observation()
        tags = []
        for unit in obs.observation.raw_data.units:
            if unit.unit_type in (UNIT_MARINE, UNIT_ZERGLING):
                tags.append(unit.tag)
        if tags:
            self.client.kill_units(tags)
        if self.game.reset_map():
            self.client.enable_enemy_control()

    def _prepare_battlefield(self) -> None:
        marine_x, marine_y = random.uniform(15, 17), random.uniform(15, 17)
        self.client.spawn_units(UNIT_MARINE, (marine_x, marine_y), owner=1, quantity=1)

        angle = random.uniform(0, 2 * math.pi)
        ling_x = marine_x + SPAWN_DISTANCE * math.cos(angle)
        ling_y = marine_y + SPAWN_DISTANCE * math.sin(angle)
        self.client.spawn_units(UNIT_ZERGLING, (ling_x, ling_y), owner=2, quantity=NUM_ZERGLINGS)

        self.game.step(count=2)
        self._observe_units()

        assert self.marine is not None, "Marine not spawned"
        assert not self.enemies.all_dead(), "Zerglings not spawned"

    def _get_enemy_obs(self, enemy: UnitState | None) -> list[float]:
        if enemy is None or self.marine is None:
            return [0.0, 0.0, 0.0, 0.0]

        hp = enemy.health / enemy.health_max
        angle = self.marine.angle_to(enemy)
        dist = min(1.0, self.marine.distance_to(enemy) / 30.0)
        return [hp, math.sin(angle), math.cos(angle), dist]

    def _compute_observation(self) -> np.ndarray:
        if self.marine is None:
            return np.zeros(OBS_SIZE, dtype=np.float32)

        time_left = 1.0 - (self.current_step / MAX_EPISODE_STEPS)
        hp = self.marine.health / self.marine.health_max
        cd = float(self.marine.weapon_cooldown > 0)
        cd_norm = min(1.0, self.marine.weapon_cooldown / 15.0)

        obs = [
            time_left, hp, cd, cd_norm,
            *self._get_enemy_obs(self.enemies[0]),
            *self._get_enemy_obs(self.enemies[1]),
        ]
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self) -> float:
        if self.current_step >= MAX_EPISODE_STEPS:
            return 0.0

        if self.marine is None or self.enemies.all_dead():
            enemy_hp_left = self.enemies.total_health()
            enemy_max_hp = ZERGLING_MAX_HP * NUM_ZERGLINGS
            return 1.0 - (enemy_hp_left / enemy_max_hp)

        return 0.0

    def _execute_action(self, action: dict) -> None:
        if self.marine is None:
            return

        command = action["command"]
        mask = self.get_action_mask()

        # If the chosen action is masked, do nothing (or could default to STOP)
        if not mask[command]:
            return

        if command == ACTION_MOVE:
            sin_a, cos_a = action["angle"]
            target_x = self.marine.x + cos_a * MOVE_STEP_SIZE
            target_y = self.marine.y + sin_a * MOVE_STEP_SIZE
            self.client.unit_move(self.marine.tag, (target_x, target_y))

        elif command == ACTION_ATTACK_Z1:
            enemy = self.enemies[0]
            if enemy is not None:
                self.client.unit_attack_unit(self.marine.tag, enemy.tag)

        elif command == ACTION_ATTACK_Z2:
            enemy = self.enemies[1]
            if enemy is not None:
                self.client.unit_attack_unit(self.marine.tag, enemy.tag)

        elif command == ACTION_STOP:
            # Stop current action - unit will hold position
            self.client.unit_stop(self.marine.tag)

    def get_action_mask(self) -> np.ndarray:
        """Get action mask for valid commands.

        Returns:
            bool array [MOVE, ATTACK_Z1, ATTACK_Z2, STOP]
            - MOVE: always True
            - ATTACK_Zx: True only if enemy alive AND in range
            - STOP: always True
        """
        mask = np.ones(NUM_COMMANDS, dtype=bool)

        if self.marine is None or self.marine.weapon_cooldown > 0:
            mask[ACTION_ATTACK_Z1] = False
            mask[ACTION_ATTACK_Z2] = False
            return mask

        for i, action_idx in enumerate([ACTION_ATTACK_Z1, ACTION_ATTACK_Z2]):
            enemy = self.enemies[i]
            if enemy is None:
                mask[action_idx] = False
            else:
                mask[action_idx] = enemy.distance_to(self.marine) < MARINE_RANGE

        return mask

    def reset(self, **__) -> tuple[np.ndarray, dict[str, Any]]:
        self.current_step = 0
        self.terminated = False
        self.marine = None
        self.enemies.reset()
        self._clean_battlefield()
        self._prepare_battlefield()
        return self._compute_observation(), {"action_mask": self.get_action_mask()}

    def step(self, action: dict) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.terminated:
            self._execute_action(action)
            self.game.step(count=GAME_STEPS_PER_ENV)
            self.current_step += GAME_STEPS_PER_ENV

        self._observe_units()

        obs = self._compute_observation()
        reward = self._compute_reward()
        truncated = self.current_step >= MAX_EPISODE_STEPS
        won = self.marine is not None and self.enemies.all_dead()

        return obs, reward, self.terminated, truncated, {"won": won, "action_mask": self.get_action_mask()}

    def close(self) -> None:
        self.game.close()
