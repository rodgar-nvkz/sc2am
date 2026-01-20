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

    # === Auxiliary Task Settings ===
    # The auxiliary task forces the encoder to represent observation features
    # that are critical for correct action selection. This prevents encoder
    # collapse where all observations map to similar hidden states, causing
    # policy gradients to cancel across episodes with different optimal actions.
    aux_enabled: bool = True
    aux_hidden_size: int = 16  # Smaller than main head - this should be easy

    # =========================================================================
    # Computed properties
    # =========================================================================

    @property
    def obs_size(self) -> int:
        """Total observation size (from ObsSpec)."""
        return self.obs_spec.total_size

    @property
    def aux_target_slices(self) -> list[slice]:
        """Auxiliary prediction target slices (from ObsSpec)."""
        return self.obs_spec.aux_target_slices

    @property
    def aux_target_size(self) -> int:
        """Number of observation features to predict in auxiliary task."""
        return self.obs_spec.aux_target_size

    @property
    def head_input_size(self) -> int:
        """Size of input to all heads (command, angle, value).

        All heads now use the same input size - just the observation.
        No command conditioning for angle head (AlphaZero-style parallel prediction).
        """
        return self.obs_size
