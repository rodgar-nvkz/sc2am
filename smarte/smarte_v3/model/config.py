"""Model configuration for ActorCritic architecture."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for the ActorCritic model architecture.

    This config controls all architectural decisions, making it easy to:
    - Run ablation experiments (toggle features)
    - Scale up/down network sizes
    - Add new modalities (terrain, etc.)

    Architecture: AlphaZero-style parallel prediction with masking.
    All action heads predict independently from observations, then
    invalid/unused actions are masked at loss computation time.
    """

    # Observation space
    obs_size: int

    # Action space
    num_commands: int
    move_action_id: int  # For angle head masking

    # Head settings
    head_hidden_size: int = 32

    # Continuous action settings
    # -0.5 -> std≈0.6 (too focused, gets stuck in fixed angles)
    # 0.5 -> std≈1.65 (explores more, harder to get stuck)
    # 1.0 -> std≈2.7 (nearly uniform on circle)
    angle_init_log_std: float = 0.75

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

    # Which observation indices to predict as auxiliary targets.
    # Based on env._compute_observation() layout:
    #   [0]: time_remaining
    #   [1]: own_health
    #   [2]: weapon_cooldown (binary)
    #   [3]: weapon_cooldown_norm
    #   [4]: marine_facing_sin
    #   [5]: marine_facing_cos
    #   [6-12]: z1: health, angle_sin, angle_cos, distance_norm, in_attack_range, facing_sin, facing_cos
    #   [13-19]: z2: same as z1
    #
    # Critical features for chase/kite behavior:
    #   - z1_angle_sin (7), z1_angle_cos (8), z1_distance (9)
    #   - z2_angle_sin (14), z2_angle_cos (15), z2_distance (16)
    # These determine "which direction to move" - exactly what the angle head outputs.
    aux_target_indices: list[int] = field(default_factory=lambda: [7, 8, 9, 14, 15, 16])

    @property
    def aux_target_size(self) -> int:
        """Number of observation features to predict in auxiliary task."""
        return len(self.aux_target_indices)

    @property
    def head_input_size(self) -> int:
        """Size of input to all heads (command, angle, value).

        All heads now use the same input size - just the observation.
        No command conditioning for angle head (AlphaZero-style parallel prediction).
        """
        return self.obs_size
