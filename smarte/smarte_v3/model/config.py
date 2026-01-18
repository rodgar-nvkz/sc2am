"""Model configuration for ActorCritic architecture."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the ActorCritic model architecture.

    This config controls all architectural decisions, making it easy to:
    - Run ablation experiments (toggle features)
    - Scale up/down network sizes
    - Add new modalities (terrain, etc.)
    """

    # Observation space
    obs_size: int = 11  # 10 original + 1 step counter

    # Action space
    num_commands: int = 3  # MOVE, ATTACK_Z1, ATTACK_Z2

    # Head settings
    head_hidden_size: int = 32

    # Continuous action settings
    angle_init_log_std: float = -0.5

    # Weight initialization
    init_orthogonal: bool = True
    init_gain: float = 1.41421356  # sqrt(2)
    policy_init_gain: float = 0.01
    value_init_gain: float = 1.0

    @property
    def head_input_size(self) -> int:
        """Size of input to action/value heads."""
        return self.obs_size

    @property
    def angle_head_input_size(self) -> int:
        """Size of input to angle head (includes one-hot command)."""
        return self.head_input_size + self.num_commands
