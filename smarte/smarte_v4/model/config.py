"""Model configuration for ActorCritic architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..obs import ObsSpec


@dataclass
class ModelConfig:
    """Configuration for the ActorCritic model architecture.

    AlphaZero-style parallel prediction with masking.
    All action heads predict independently from observations, then
    invalid/unused actions are masked at loss computation time.

    The observation structure is defined by ObsSpec, which provides:
    - Observation layout (sizes, shape)
    - Auxiliary prediction targets
    """

    # Observation spec - single source of truth for observation structure
    obs_spec: ObsSpec

    # Action space
    num_commands: int
    move_action_id: int  # For angle head masking

    # Head settings
    head_hidden_size: int = 16

    # Angle head architecture
    angle_encoder_layers: int = 2
    angle_output_layers: int = 2

    # Continuous action settings (von Mises distribution)
    angle_init_log_concentration: float = 0.0

    # Weight initialization
    init_orthogonal: bool = True
    init_gain: float = 1.41421356  # sqrt(2)
    policy_init_gain: float = 0.05
    value_init_gain: float = 1.0

    # === Unit Embedding Settings ===
    # Shared UnitEncoder embeds full unit features into a learned representation.
    # Heads receive non-coord features + flattened embeddings.
    unit_embed_dim: int = 8
    unit_hidden_size: int = 16

    # === Auxiliary Task Settings ===
    # Pairwise geometry: small shared MLP takes (embed_i || embed_j) and
    # predicts directed (distance, sin, cos). Shuffle-based pairing.
    aux_enabled: bool = True
    aux_hidden_size: int = 16

    # =========================================================================
    # Computed properties
    # =========================================================================

    @property
    def unit_input_size(self) -> int:
        """Input size per unit for UnitEncoder."""
        return self.obs_spec.unit_feature_size

    @property
    def num_units(self) -> int:
        """Number of units (ally + enemies)."""
        return self.obs_spec.num_units

    @property
    def non_coord_size(self) -> int:
        """Total non-coordinate features across all units."""
        return self.obs_spec.non_coord_size * self.num_units

    @property
    def aux_input_size(self) -> int:
        """Input size for pairwise aux head: two embeddings concatenated."""
        return self.unit_embed_dim * 2

    @property
    def head_input_size(self) -> int:
        """Size of input to all heads: unit embeddings + non-coord features.

        Heads see learned unit embeddings concatenated with raw non-coord
        features (everything except x, y).
        """
        return self.num_units * self.unit_embed_dim + self.non_coord_size
