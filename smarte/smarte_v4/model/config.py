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
    - Observation layout (sizes, slices)
    - Auxiliary prediction targets
    """

    # Observation spec - single source of truth for observation structure
    obs_spec: ObsSpec

    # Action space
    num_commands: int
    move_action_id: int  # For angle head masking

    # Head settings
    head_hidden_size: int = 32

    # Angle head architecture
    # Deeper networks can learn more complex angle transformations
    angle_encoder_layers: int = 2  # Number of layers in encoder
    angle_output_layers: int = 2  # Number of layers in output head

    # Continuous action settings (von Mises distribution)
    # The von Mises distribution is the "circular Gaussian" - proper for angles.
    # Concentration parameter κ (kappa):
    #   κ → 0: uniform distribution on circle (maximum exploration)
    #   κ ≈ 1: moderate concentration (std ≈ 65°)
    #   κ ≈ 2: tighter concentration (std ≈ 45°)
    #   κ ≈ 4: fairly concentrated (std ≈ 30°)
    #   κ → ∞: point mass (no exploration)
    #
    # We use log(κ) as the learnable parameter for numerical stability.
    # init_log_concentration = 0.0 → κ = 1.0 (good exploration to start)
    angle_init_log_concentration: float = 0.0

    # Weight initialization
    init_orthogonal: bool = True
    init_gain: float = 1.41421356  # sqrt(2)
    policy_init_gain: float = 0.05
    value_init_gain: float = 1.0

    # === Coordinate Embedding Settings ===
    # Shared PointEncoder embeds (x, y, valid) into a learned representation.
    # Heads receive non-coord features + flattened embeddings.
    coord_embed_dim: int = 32
    coord_hidden_size: int = 64

    # === Auxiliary Task Settings ===
    # Pairwise geometry: small shared MLP takes (embed_i || embed_j) and
    # predicts directed (distance, sin, cos). K random pairs sampled per step.
    aux_enabled: bool = True
    aux_hidden_size: int = 32
    aux_num_samples: int = 6  # pairs sampled per step

    # =========================================================================
    # Computed properties
    # =========================================================================

    @property
    def obs_size(self) -> int:
        """Total observation size (from ObsSpec)."""
        return self.obs_spec.total_size

    @property
    def num_coord_points(self) -> int:
        """Number of coordinate points (ally + enemies)."""
        return self.obs_spec.num_coord_points

    @property
    def coord_flat_size(self) -> int:
        """Size of flattened coordinate embeddings."""
        return self.num_coord_points * self.coord_embed_dim

    @property
    def non_coord_size(self) -> int:
        """Number of non-coordinate features in observation."""
        return self.obs_spec.non_coord_size

    @property
    def aux_input_size(self) -> int:
        """Input size for pairwise aux head: two embeddings concatenated."""
        return self.coord_embed_dim * 2

    @property
    def head_input_size(self) -> int:
        """Size of input to all heads: non-coord features + coord embeddings.

        Heads never see raw (x, y, valid). They see non-coord features
        concatenated with learned coordinate embeddings.
        """
        return self.non_coord_size + self.coord_flat_size
