"""Observation specification and builder.

Single source of truth for:
- Observation layout (sizes, slices)
- Encoding logic (raw state → normalized features)
- Auxiliary prediction targets

This module decouples observation structure from both the environment
(which provides raw game state) and the model (which consumes observations).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .env import UnitState


@dataclass
class ObsSpec:
    """Observation structure specification and builder.

    Defines the layout of the observation vector and provides methods to build observations from raw game state.
    This enables models to slice observations by entity type without hardcoding indices.
    """

    # Entity counts
    num_allies: int = 1
    num_enemies: int = 2

    # Feature sizes per entity type
    game_state_size: int = 1  # time_remaining
    ally_feature_size: int = 7  # health, cooldown_norm, facing_sin, facing_cos, x_norm, y_norm, valid
    enemy_feature_size: int = 6  # health, facing_sin, facing_cos, x_norm, y_norm, valid

    # Normalization constants
    max_cooldown: float = 15.0
    map_size: float = 64.0

    # =========================================================================
    # Computed properties
    # =========================================================================

    @property
    def total_size(self) -> int:
        """Total observation vector size."""
        allies = self.num_allies * self.ally_feature_size
        enemies = self.num_enemies * self.enemy_feature_size
        return self.game_state_size + allies + enemies

    @property
    def game_state_slice(self) -> slice:
        return slice(0, self.game_state_size)

    @property
    def ally_slices(self) -> list[slice]:
        """List of slices, one per ally."""
        start = self.game_state_size
        slices = []
        for _ in range(self.num_allies):
            slices.append(slice(start, start + self.ally_feature_size))
            start += self.ally_feature_size
        return slices

    @property
    def enemy_slices(self) -> list[slice]:
        """List of slices, one per enemy."""
        start = self.game_state_size + self.num_allies * self.ally_feature_size
        slices = []
        for _ in range(self.num_enemies):
            slices.append(slice(start, start + self.enemy_feature_size))
            start += self.enemy_feature_size
        return slices

    @property
    def coord_indices(self) -> list[list[int]]:
        """Indices of (x_norm, y_norm, valid) for each unit (ally + enemies).

        Returns list of [x_idx, y_idx, valid_idx] per unit.
        """
        indices = []
        # Ally coords are last 3 of ally features
        for s in self.ally_slices:
            base = s.stop - 3  # x_norm, y_norm, valid are last 3
            indices.append([base, base + 1, base + 2])
        # Enemy coords are last 3 of enemy features
        for s in self.enemy_slices:
            base = s.stop - 3
            indices.append([base, base + 1, base + 2])
        return indices

    @property
    def non_coord_indices(self) -> list[int]:
        """Indices of non-coordinate features in the observation vector."""
        coord_set = set()
        for triple in self.coord_indices:
            coord_set.update(triple)
        return [i for i in range(self.total_size) if i not in coord_set]

    @property
    def num_coord_points(self) -> int:
        """Number of coordinate points (ally + enemies)."""
        return self.num_allies + self.num_enemies

    @property
    def non_coord_size(self) -> int:
        """Number of non-coordinate features."""
        return len(self.non_coord_indices)

    # =========================================================================
    # Observation building
    # =========================================================================

    def build(self, time_remaining: float, allies: list[UnitState], enemies: list[UnitState]) -> np.ndarray:
        """Build observation array from raw game state"""
        obs = np.zeros(self.total_size, dtype=np.float32)

        # Game state
        obs[self.game_state_slice] = [time_remaining]

        # Allies
        for i, ally_slice in enumerate(self.ally_slices):
            if i < len(allies):
                obs[ally_slice] = self._encode_ally(allies[i])
            # else: remains zero (dead/missing ally, valid=0)

        # Enemies
        for i, enemy_slice in enumerate(self.enemy_slices):
            if i < len(enemies):
                obs[enemy_slice] = self._encode_enemy(enemies[i])
            # else: remains zero (dead/missing enemy, valid=0)

        return obs

    def _encode_ally(self, ally: UnitState) -> np.ndarray:
        """Encode ally features.

        Features:
            - health: normalized [0, 1]
            - cooldown_norm: normalized [0, 1]
            - facing_sin, facing_cos: unit facing direction
            - x_norm, y_norm: position normalized by map_size
            - valid: always 1.0 for ally

        Args:
            ally: Ally UnitState

        Returns:
            Array of shape (ally_feature_size,)
        """
        health = ally.health / ally.health_max
        cooldown_norm = min(1.0, ally.weapon_cooldown / self.max_cooldown)
        facing_sin = math.sin(ally.facing)
        facing_cos = math.cos(ally.facing)
        x_norm = ally.x / self.map_size
        y_norm = ally.y / self.map_size
        valid = 1.0

        return np.array([health, cooldown_norm, facing_sin, facing_cos, x_norm, y_norm, valid], dtype=np.float32)

    def _encode_enemy(self, enemy: UnitState) -> np.ndarray:
        """Encode enemy features with absolute coordinates.

        Features:
            - health: normalized [0, 1]
            - facing_sin, facing_cos: enemy facing direction
            - x_norm, y_norm: position normalized by map_size
            - valid: 1.0 if alive, 0.0 if dead/missing

        Args:
            enemy: Enemy UnitState

        Returns:
            Array of shape (enemy_feature_size,)
        """
        health = enemy.health / enemy.health_max
        facing_sin = math.sin(enemy.facing)
        facing_cos = math.cos(enemy.facing)
        x_norm = enemy.x / self.map_size
        y_norm = enemy.y / self.map_size
        valid = 1.0

        return np.array([health, facing_sin, facing_cos, x_norm, y_norm, valid], dtype=np.float32)
