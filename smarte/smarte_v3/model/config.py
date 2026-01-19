"""Model configuration for ActorCritic architecture."""

from dataclasses import dataclass


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

    @property
    def head_input_size(self) -> int:
        """Size of input to all heads (command, angle, value).

        All heads now use the same input size - just the observation.
        No command conditioning for angle head (AlphaZero-style parallel prediction).
        """
        return self.obs_size
