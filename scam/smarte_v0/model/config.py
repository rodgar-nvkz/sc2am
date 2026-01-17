"""Model configuration for LSTM-based ActorCritic architecture."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for the LSTM-based ActorCritic model.

    Architecture:
        obs (obs_size) → VectorEncoder → LSTM → [ActorHead, CriticHead]

    Action space (num_move_directions + 4 discrete actions):
        - 0 to num_move_directions-1: MOVE in direction (angle = i * 360°/num_move_directions)
        - num_move_directions: ATTACK_Z1
        - num_move_directions+1: ATTACK_Z2
        - num_move_directions+2: STOP
        - num_move_directions+3: SKIP (no-op)

    Default: 4 move directions (N, E, S, W) + 4 other = 8 actions total
    """

    # Observation space
    obs_size: int = 12

    # Action space - configurable move directions
    # 4 = N, E, S, W (90° each) - simple, good for initial testing
    # 8 = 45° increments
    # 36 = 10° increments (fine-grained)
    num_move_directions: int = 8

    # These are computed from num_move_directions in __post_init__
    num_actions: int = field(init=False)
    action_attack_z1: int = field(init=False)
    action_attack_z2: int = field(init=False)
    action_stop: int = field(init=False)
    action_skip: int = field(init=False)

    def __post_init__(self):
        """Compute action indices based on num_move_directions."""
        self.num_actions = self.num_move_directions + 4
        self.action_attack_z1 = self.num_move_directions
        self.action_attack_z2 = self.num_move_directions + 1
        self.action_stop = self.num_move_directions + 2
        self.action_skip = self.num_move_directions + 3

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
