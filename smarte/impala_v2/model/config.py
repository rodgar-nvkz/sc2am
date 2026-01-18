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

    # Encoder settings
    embed_size: int = 64
    encoder_hidden_size: int = 64
    encoder_num_layers: int = 2
    encoder_activation: str = "tanh"

    # Head settings
    head_hidden_size: int = 32
    cmd_embed_size: int = 16  # Command embedding for angle head conditioning

    # Continuous action settings
    angle_init_log_std: float = -0.5

    # Feature flags for ablation
    use_embedding: bool = True
    use_skip_connections: bool = True

    # Weight initialization
    init_orthogonal: bool = True
    init_gain: float = 1.41421356  # sqrt(2)
    policy_init_gain: float = 0.01
    value_init_gain: float = 1.0

    @property
    def head_input_size(self) -> int:
        """Size of input to action/value heads."""
        size = 0
        if self.use_embedding:
            size += self.embed_size
        if self.use_skip_connections:
            size += self.obs_size
        # If neither, fall back to raw obs
        if size == 0:
            size = self.obs_size
        return size

    @property
    def angle_head_input_size(self) -> int:
        """Size of input to angle head (includes command embedding)."""
        return self.head_input_size + self.cmd_embed_size
