"""Observation specification and builder.

Single source of truth for:
- Observation layout (sizes, slices)
- Encoding logic (raw state → normalized features)

This module decouples observation structure from both the environment
(which provides raw game state) and the model (which consumes observations).

Unit feature layout (coords first for easy slicing):
    [x/64, y/64, health, cooldown_or_-1, range/15, facing_sin, facing_cos, valid]
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

    All units (ally + enemies) share the same 8-feature layout.
    Coordinates are placed first (indices 0,1) for easy slicing in the model.
    """

    # Entity counts
    num_allies: int
    num_enemies: int

    # Feature size per unit (unified)
    unit_feature_size: int = 8

    # Coordinate features are the first 2
    coord_size: int = 2

    # Normalization constants
    max_cooldown: float = 15.0
    max_range: float = 15.0
    map_size: float = 64.0

    # =========================================================================
    # Computed properties
    # =========================================================================

    @property
    def num_units(self) -> int:
        return self.num_allies + self.num_enemies

    @property
    def obs_shape(self) -> tuple[int, int]:
        """Shape of the observation array: (num_units, unit_feature_size)."""
        return (self.num_units, self.unit_feature_size)

    @property
    def non_coord_size(self) -> int:
        """Number of non-coordinate features per unit."""
        return self.unit_feature_size - self.coord_size

    # =========================================================================
    # Observation building
    # =========================================================================

    def build(self, allies: list[UnitState], enemies: list[UnitState]) -> np.ndarray:
        """Build observation array from raw game state.

        Returns:
            Array of shape (num_units, unit_feature_size)
        """
        obs = np.zeros(self.obs_shape, dtype=np.float32)

        # Allies
        assert len(allies) <= self.num_allies
        for i, ally in enumerate(allies):
            obs[i] = self._encode_unit(ally, is_ally=True)

        # Enemies
        assert len(enemies) <= self.num_enemies
        for i, enemy in enumerate(enemies):
            unit_idx = self.num_allies + i
            obs[unit_idx] = self._encode_unit(enemy, is_ally=False)

        return obs

    def _encode_unit(self, unit: UnitState, *, is_ally: bool) -> np.ndarray:
        """Encode unit features with unified layout.

        Layout: [x/64, y/64, health, cooldown_or_-1, range/15, facing_sin, facing_cos, valid]

        Cooldown is normalized for allies, -1 sentinel for enemies.
        """
        x_norm = unit.x / self.map_size
        y_norm = unit.y / self.map_size
        health = unit.health / unit.health_max
        cooldown = min(1.0, unit.weapon_cooldown / self.max_cooldown) if is_ally else -1.0
        attack_range = unit.attack_range / self.max_range
        facing_sin = math.sin(unit.facing)
        facing_cos = math.cos(unit.facing)
        valid = 1.0

        return np.array(
            [x_norm, y_norm, health, cooldown, attack_range, facing_sin, facing_cos, valid], dtype=np.float32
        )
