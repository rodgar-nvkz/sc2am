"""
Marine vs 2 Zerglings RL Environment (v2)

A Gymnasium environment for training a Marine agent against 2 scripted Zerglings.
Uses Dict observation space with:
- unit_features: vector observations (14 features) for unit states
- terrain: 32x32 grid centered on marine showing walkable terrain and unit positions

Action space: discrete (8 directions + stay + attack_z1 + attack_z2).
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
MAX_EPISODE_STEPS = 22.4 * 10  # 10 realtime seconds

# Spawn configuration
SPAWN_AREA_MIN = 0.0 + 14
SPAWN_AREA_MAX = 32.0 - 14
MIN_SPAWN_DISTANCE = 7.0
MAX_SPAWN_DISTANCE = 8.0

# Number of zerglings
NUM_ZERGLINGS = 2

# Upgrade IDs for Terran Infantry Weapons
UPGRADE_INFANTRY_WEAPONS_1 = 7
UPGRADE_INFANTRY_WEAPONS_2 = 8
UPGRADE_INFANTRY_WEAPONS_3 = 9

# Observation space size for unit features:
# Marine: own_health (1), weapon_cooldown (1) = 2
# Per zergling (x2): rel_x (1), rel_y (1), distance (1), health (1), angle (1), in_range (1) = 6
# Total: 2 + 6*2 = 14
UNIT_FEATURES_SIZE = 14

# Terrain grid configuration
TERRAIN_GRID_SIZE = 32  # 32x32 grid
TERRAIN_CHANNELS = 3    # Channel 0: walkable, Channel 1: allied units, Channel 2: enemy units

# Map configuration
MAP_SIZE = 32.0  # The Flat32s map is 32x32


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
    """1 Marine vs 2 Zerglings environment on a Flat map with Dict observation space"""

    metadata = {"name": "sc2_mv2z_v2", "render_modes": []}

    def __init__(self, env_ctx: dict | None = None) -> None:
        super().__init__()

        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Dict observation space with unit features and terrain grid
        self.observation_space = spaces.Dict({
            "unit_features": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(UNIT_FEATURES_SIZE,),
                dtype=np.float32
            ),
            "terrain": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(TERRAIN_GRID_SIZE, TERRAIN_GRID_SIZE, TERRAIN_CHANNELS),
                dtype=np.float32
            ),
        })

        self.game = SC2SingleGame([Terran, Zerg]).launch()
        self.client = self.game.clients[0]
        self.units: dict[int, list] = defaultdict(list)
        self.current_step: int = 0
        self.terminated: bool = False

        # Cache the terrain walkability grid (doesn't change during game)
        self._terrain_walkable: np.ndarray | None = None

        env_ctx = env_ctx or {}
        self.game_steps_per_env = None

        # Target upgrade level for domain randomization (None = random each episode)
        self.target_upgrade_level: int = env_ctx.get("upgrade_level") or random.randint(0, 2)

    def _initialize_terrain_grid(self) -> None:
        """Initialize the terrain walkability grid from game info."""
        if self._terrain_walkable is not None:
            return

        game_info = self.client.get_game_info()

        # Get pathing grid from game info
        pathing_grid = game_info.start_raw.pathing_grid

        # The pathing grid is a packed bitmap
        width = pathing_grid.size.x
        height = pathing_grid.size.y
        data = pathing_grid.data

        # Unpack the bitmap into a 2D array
        terrain = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                byte_index = (y * width + x) // 8
                bit_index = (y * width + x) % 8
                if byte_index < len(data):
                    # Bit is 1 if pathable
                    pathable = (data[byte_index] >> bit_index) & 1
                    terrain[y, x] = float(pathable)

        # Resize to our grid size if needed
        if width != TERRAIN_GRID_SIZE or height != TERRAIN_GRID_SIZE:
            from scipy.ndimage import zoom
            zoom_y = TERRAIN_GRID_SIZE / height
            zoom_x = TERRAIN_GRID_SIZE / width
            terrain = zoom(terrain, (zoom_y, zoom_x), order=0)  # nearest neighbor

        self._terrain_walkable = terrain
        logger.debug(f"Initialized terrain grid: {terrain.shape}, walkable cells: {terrain.sum():.0f}/{terrain.size}")

    def get_upgrade_level(self) -> int:
        """Get current infantry weapons upgrade level (0-3)."""
        obs = self.client.get_observation()
        upgrade_ids = set(obs.observation.raw_data.player.upgrade_ids)
        logger.debug(f"Current upgrades: {upgrade_ids}")
        if UPGRADE_INFANTRY_WEAPONS_3 in upgrade_ids:
            return 3
        if UPGRADE_INFANTRY_WEAPONS_2 in upgrade_ids:
            return 2
        if UPGRADE_INFANTRY_WEAPONS_1 in upgrade_ids:
            return 1
        return 0

    def observe_units(self) -> None:
        self.units = defaultdict(list)
        obs = self.client.get_observation()
        for unit in obs.observation.raw_data.units:
            if unit.unit_type not in (UNIT_MARINE, UNIT_ZERGLING) or unit.health <= 0:
                continue
            unit_state = UnitState.from_proto(unit)
            self.units[unit.owner].append(unit_state)

        # Sort zerglings by tag to ensure consistent ordering across observations
        self.units[2].sort(key=lambda u: u.tag)

        self.terminated = not all((self.units[1], self.units[2]))
        logger.debug(
            f"Marine alive: {len(self.units[1])}, Zerglings alive: {len(self.units[2])}, episode ended: {self.terminated}"
        )

    def clean_battlefield(self) -> None:
        units = sum(self.units.values(), [])
        unit_tags = [u.tag for u in units]
        self.client.kill_units(unit_tags)
        self.game.reset_map()

    def prepare_battlefield(self) -> None:
        logger.debug("Spawning units")

        # Domain randomization: pick random upgrade level if not specified
        current_upgrade = self.get_upgrade_level()
        for _ in range(max(self.target_upgrade_level - current_upgrade, 0)):
            logger.debug(f"Researching infantry weapons upgrade from level {self.get_upgrade_level()}")
            self.client.research_upgrades()

        # Random position for marine
        marine_x = random.uniform(SPAWN_AREA_MIN, SPAWN_AREA_MAX)
        marine_y = random.uniform(SPAWN_AREA_MIN, SPAWN_AREA_MAX)
        self.client.spawn_units(UNIT_MARINE, (marine_x, marine_y), owner=1, quantity=1)

        # Zerglings spawn at random distance/angle from marine (both at same position)
        spawn_distance = random.uniform(MIN_SPAWN_DISTANCE, MAX_SPAWN_DISTANCE)
        spawn_angle = random.uniform(0, 2 * math.pi)
        ling_x = marine_x + spawn_distance * math.cos(spawn_angle)
        ling_y = marine_y + spawn_distance * math.sin(spawn_angle)
        # Spawn 2 zerglings at the same position
        self.client.spawn_units(UNIT_ZERGLING, (ling_x, ling_y), owner=2, quantity=1)
        shift_x = random.randint(-2500, 2500) / 1000.0
        shift_y = random.randint(-2500, 2500) / 1000.0
        self.client.spawn_units(UNIT_ZERGLING, (ling_x + shift_x, ling_y + shift_y), owner=2, quantity=1)

        self.game.step(count=2)  # unit spawn takes two frames

        self.observe_units()
        assert len(self.units[1]) == 1, "Marine not spawned correctly"
        assert len(self.units[2]) == NUM_ZERGLINGS, f"Expected {NUM_ZERGLINGS} Zerglings, got {len(self.units[2])}"

        logger.debug(
            f"Spawned Marine at ({marine_x:.2f}, {marine_y:.2f}) and {NUM_ZERGLINGS} Zerglings at ({ling_x:.2f}, {ling_y:.2f})"
        )

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

    def _compute_unit_features(self) -> np.ndarray:
        """Compute the unit features observation vector for the Marine agent."""
        if not self.units[1]:
            return np.zeros(UNIT_FEATURES_SIZE, dtype=np.float32)

        marine = self.units[1][0]
        zerglings = self.units[2]  # Already sorted by tag

        # Marine's own state
        own_health = marine.health / marine.health_max
        weapon_cooldown_norm = min(1.0, marine.weapon_cooldown / 15.0)

        # Get zergling observations (pad with None if dead)
        z1 = zerglings[0] if len(zerglings) > 0 else None
        z2 = zerglings[1] if len(zerglings) > 1 else None

        z1_obs = self._get_zergling_obs(marine, z1)
        z2_obs = self._get_zergling_obs(marine, z2)

        obs = [own_health, weapon_cooldown_norm, *z1_obs, *z2_obs]

        return np.array(obs, dtype=np.float32)

    def _compute_terrain_grid(self) -> np.ndarray:
        """Compute the terrain grid observation.

        Returns a (32, 32, 3) grid where:
        - Channel 0: terrain walkability (1.0 = walkable, 0.0 = blocked)
        - Channel 1: allied unit positions (normalized HP at position)
        - Channel 2: enemy unit positions (normalized HP at position)
        """
        # Initialize terrain if needed
        self._initialize_terrain_grid()

        # Create the 3-channel grid
        grid = np.zeros((TERRAIN_GRID_SIZE, TERRAIN_GRID_SIZE, TERRAIN_CHANNELS), dtype=np.float32)

        # Channel 0: terrain walkability
        grid[:, :, 0] = self._terrain_walkable

        # Helper to convert world position to grid position
        def world_to_grid(x: float, y: float) -> tuple[int, int]:
            grid_x = int(x / MAP_SIZE * TERRAIN_GRID_SIZE)
            grid_y = int(y / MAP_SIZE * TERRAIN_GRID_SIZE)
            # Clamp to valid range
            grid_x = max(0, min(TERRAIN_GRID_SIZE - 1, grid_x))
            grid_y = max(0, min(TERRAIN_GRID_SIZE - 1, grid_y))
            return grid_x, grid_y

        # Channel 1: allied units (marines)
        for unit in self.units[1]:
            gx, gy = world_to_grid(unit.x, unit.y)
            grid[gy, gx, 1] = unit.health / unit.health_max

        # Channel 2: enemy units (zerglings)
        for unit in self.units[2]:
            gx, gy = world_to_grid(unit.x, unit.y)
            grid[gy, gx, 2] = unit.health / unit.health_max

        return grid

    def _compute_observation(self) -> dict[str, np.ndarray]:
        """Compute the full Dict observation."""
        return {
            "unit_features": self._compute_unit_features(),
            "terrain": self._compute_terrain_grid(),
        }

    def _compute_reward(self) -> float:
        """Compute reward based on damage dealt/taken and terminal conditions."""
        if self.current_step >= MAX_EPISODE_STEPS:
            return -1.0  # Timeout penalty

        if not self.units[2]:  # All zerglings dead - win!
            win_bonus = 1.0
            hp_left_bonus = (
                self.units[1][0].health / self.units[1][0].health_max
                if self.units[1]
                else 0.0
            )
            return win_bonus + hp_left_bonus * 10.0
        return 0.0

    def _agent_action(self, action: int) -> None:
        logger.debug(f"Agent action: {action}")
        marine = self.units[1][0]
        zerglings = self.units[2]  # Already sorted by tag

        if action == ACTION_ATTACK_Z1:
            if len(zerglings) > 0:
                self.client.unit_attack_unit(marine.tag, zerglings[0].tag)
            else:
                # Z1 is dead, stop instead
                self.client.unit_stop(marine.tag)
        elif action == ACTION_ATTACK_Z2:
            if len(zerglings) > 1:
                self.client.unit_attack_unit(marine.tag, zerglings[1].tag)
            elif len(zerglings) > 0:
                # Z2 is dead, attack Z1 instead
                self.client.unit_attack_unit(marine.tag, zerglings[0].tag)
            else:
                # Both dead, stop
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

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.current_step = 0
        self.clean_battlefield()
        self.prepare_battlefield()
        return self._compute_observation(), {}

    def step(self, action: np.ndarray) -> tuple:
        logger.debug(f"Environment step {self.current_step} with action {action}")

        self.observe_units()
        if not self.terminated:
            self._agent_action(action.item())
            self.game.step(count=self.game_steps_per_env)
            self.current_step += 1

        truncated = self.current_step >= MAX_EPISODE_STEPS
        obs = self._compute_observation()
        reward = self._compute_reward()
        return obs, reward, self.terminated, truncated, {"won": reward > 0}

    def close(self) -> None:
        self.game.close()
