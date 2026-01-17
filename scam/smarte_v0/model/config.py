"""Model configuration for LSTM-based ActorCritic architecture."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the LSTM-based ActorCritic model.

    Architecture:
        obs (obs_size) → VectorEncoder → LSTM → [ActorHead, CriticHead]

    Action space (40 discrete actions):
        - 0-35: MOVE in direction (angle = i * 10°)
        - 36: ATTACK_Z1
        - 37: ATTACK_Z2
        - 38: STOP
        - 39: SKIP (no-op)
    """

    # Observation space
    obs_size: int = 12

    # Action space (40 discrete actions)
    num_actions: int = 40
    num_move_directions: int = 36  # 360° / 10° = 36 directions
    action_attack_z1: int = 36
    action_attack_z2: int = 37
    action_stop: int = 38
    action_skip: int = 39

    # Encoder settings
    embed_size: int = 64
    encoder_hidden_size: int = 64
    encoder_num_layers: int = 2
    encoder_activation: str = "tanh"

    # LSTM settings
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 1

    # Head settings
    head_hidden_size: int = 64

    # Feature flags
    use_skip_connections: bool = True  # Concat raw obs to LSTM output for heads

    # Weight initialization
    init_orthogonal: bool = True
    init_gain: float = 1.41421356  # sqrt(2) for tanh
    policy_init_gain: float = 0.01  # Small init for policy output
    value_init_gain: float = 1.0

    @property
    def head_input_size(self) -> int:
        """Size of input to action/value heads (LSTM output + optional skip)."""
        size = self.lstm_hidden_size
        if self.use_skip_connections:
            size += self.obs_size
        return size

    def get_move_angle(self, action: int) -> float:
        """Convert move action index to angle in radians.

        Args:
            action: Action index (0-35 for move directions)

        Returns:
            Angle in radians (0 = east/right, counter-clockwise)
        """
        import math
        if 0 <= action < self.num_move_directions:
            return action * (2 * math.pi / self.num_move_directions)
        raise ValueError(f"Action {action} is not a move action")

    def is_move_action(self, action: int) -> bool:
        """Check if action is a move action."""
        return 0 <= action < self.num_move_directions

    def is_attack_action(self, action: int) -> bool:
        """Check if action is an attack action."""
        return action in (self.action_attack_z1, self.action_attack_z2)
