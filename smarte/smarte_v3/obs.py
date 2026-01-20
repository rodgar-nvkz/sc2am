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
    ally_feature_size: int = 5  # health, cooldown, cooldown_norm, facing_sin, facing_cos
    enemy_feature_size: int = 7  # health, angle_sin, angle_cos, distance, in_range, facing_sin, facing_cos

    # Normalization constants
    max_distance: float = 30.0
    max_cooldown: float = 15.0

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
    def aux_target_slices(self) -> list[slice]:
        """Slices for auxiliary prediction targets"""
        return self.ally_slices + self.enemy_slices

    @property
    def aux_target_size(self) -> int:
        """Number of features predicted by auxiliary task."""
        ally = self.num_allies * self.ally_feature_size
        enemy = self.num_enemies * self.enemy_feature_size
        return ally + enemy

    # =========================================================================
    # Observation building
    # =========================================================================

    def build(self, time_remaining: float, allies: list[UnitState], enemies: list[UnitState]) -> np.ndarray:
        """Build observation array from raw game state"""
        obs = np.zeros(self.total_size, dtype=np.float32)

        # Game state
        obs[self.game_state_slice] = [time_remaining]

        # Allies - use first ally as reference for enemy angles
        reference = allies[0] if allies else None
        for i, ally_slice in enumerate(self.ally_slices):
            if i < len(allies):
                obs[ally_slice] = self._encode_ally(allies[i])
            # else: remains zero (dead/missing ally)

        # Enemies - encoded relative to reference ally
        for i, enemy_slice in enumerate(self.enemy_slices):
            if i < len(enemies) and reference is not None:
                obs[enemy_slice] = self._encode_enemy(reference, enemies[i])
            # else: remains zero (dead/missing enemy)

        return obs

    def _encode_ally(self, ally: UnitState) -> np.ndarray:
        """Encode ally features.

        Features:
            - health: normalized [0, 1]
            - weapon_cooldown: binary (0 or 1)
            - weapon_cooldown_norm: normalized [0, 1]
            - facing_sin, facing_cos: unit facing direction

        Args:
            ally: Ally UnitState

        Returns:
            Array of shape (ally_feature_size,)
        """
        health = ally.health / ally.health_max
        cooldown_binary = float(ally.weapon_cooldown > 0)
        cooldown_norm = min(1.0, ally.weapon_cooldown / self.max_cooldown)
        facing_sin = math.sin(ally.facing)
        facing_cos = math.cos(ally.facing)

        return np.array([health, cooldown_binary, cooldown_norm, facing_sin, facing_cos], dtype=np.float32)

    def _encode_enemy(self, reference: UnitState, enemy: UnitState) -> np.ndarray:
        """Encode enemy features relative to reference ally.

        Features:
            - health: normalized [0, 1]
            - angle_sin, angle_cos: direction from reference to enemy
            - distance: normalized [0, 1]
            - in_attack_range: binary (0 or 1)
            - facing_sin, facing_cos: enemy facing direction

        Args:
            reference: Reference ally UnitState (for relative angle/distance)
            enemy: Enemy UnitState

        Returns:
            Array of shape (enemy_feature_size,)
        """
        health = enemy.health / enemy.health_max

        # Angle from reference to enemy
        angle = reference.angle_to(enemy)
        angle_sin = math.sin(angle)
        angle_cos = math.cos(angle)

        # Distance - in_attack_range uses reference unit's attack range
        distance = reference.distance_to(enemy)
        distance_norm = min(1.0, distance / self.max_distance)
        in_attack_range = float(distance < reference.attack_range)

        # Enemy facing
        facing_sin = math.sin(enemy.facing)
        facing_cos = math.cos(enemy.facing)

        return np.array(
            [health, angle_sin, angle_cos, distance_norm, in_attack_range, facing_sin, facing_cos],
            dtype=np.float32,
        )
