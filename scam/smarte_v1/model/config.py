"""Model configuration for entity-based attention architecture.

Architecture:
    Marine/Enemy features → Entity Encoders → GRU → Cross-Attention → Heads

Action Space (Hybrid):
    - Action Type: Discrete [MOVE, ATTACK, STOP]
    - Move Direction: Continuous [sin, cos] (only when MOVE)
    - Attack Target: Pointer over N enemies (only when ATTACK)
"""

from dataclasses import dataclass

# Action type indices
ACTION_MOVE = 0
ACTION_ATTACK = 1
ACTION_STOP = 2
NUM_ACTION_TYPES = 3


@dataclass
class ModelConfig:
    """Configuration for entity-based attention ActorCritic model.

    Architecture:
        Marine obs → MarineEncoder → marine_emb
        Enemy obs (N) → EnemyEncoder (shared) → enemy_embs
                            ↓
        GRU (maintains temporal state across steps)
                            ↓
        Cross-Attention (marine queries enemies)
                            ↓
        [ActionTypeHead, MoveHead, AttackHead, ValueHead, AuxHeads]
    """

    # === Observation Space ===
    # Marine features: [hp_norm, cd_norm, can_attack]
    marine_obs_size: int = 3

    # Per-enemy features: [hp_norm, sin(angle), cos(angle), dist_norm]
    enemy_obs_size: int = 4

    # Maximum number of enemies (for padding)
    max_enemies: int = 2

    # Full observation size (for compatibility): time(1) + marine(3) + enemies(N*4)
    @property
    def obs_size(self) -> int:
        return 1 + self.marine_obs_size + self.max_enemies * self.enemy_obs_size

    # === Entity Encoder Settings ===
    entity_embed_size: int = 64  # Output embedding size for entities
    entity_hidden_size: int = 64  # Hidden layer size in entity MLPs
    entity_num_layers: int = 2  # Number of MLP layers
    entity_activation: str = "tanh"

    # === Temporal GRU Settings ===
    use_temporal_encoding: bool = True
    gru_hidden_size: int = 64
    gru_num_layers: int = 1

    # === Cross-Attention Settings ===
    attention_heads: int = 1  # Single head for interpretable attention weights
    attention_dim: int = 64  # Dimension for Q, K, V projections

    # === Head Settings ===
    head_hidden_size: int = 64

    # === Auxiliary Task Settings ===
    use_auxiliary_tasks: bool = True
    aux_damage_horizon: int = 5  # Predict damage in next N steps
    aux_loss_weight: float = 0.1  # Weight for auxiliary losses

    # === Action Space ===
    # Action types are fixed: MOVE=0, ATTACK=1, STOP=2
    num_action_types: int = NUM_ACTION_TYPES

    # Move direction output: continuous [sin, cos] or discretized
    move_direction_continuous: bool = True

    # === Weight Initialization ===
    init_orthogonal: bool = True
    init_gain: float = 1.41421356  # sqrt(2) for tanh
    policy_init_gain: float = 0.01  # Small init for policy outputs
    value_init_gain: float = 1.0

    # === Feature Flags ===
    # Whether to use skip connections (concat raw obs to head inputs)
    use_skip_connections: bool = True

    # Computed properties
    @property
    def backbone_output_size(self) -> int:
        """Size of shared backbone output (GRU hidden + attention context)."""
        return self.gru_hidden_size + self.entity_embed_size

    @property
    def head_input_size(self) -> int:
        """Size of input to heads (backbone + optional skip)."""
        size = self.backbone_output_size
        if self.use_skip_connections:
            # Skip connection adds marine obs (not full obs to avoid redundancy)
            size += self.marine_obs_size
        return size

    def get_activation(self) -> str:
        """Get activation function name."""
        return self.entity_activation

    @staticmethod
    def action_type_name(action_type: int) -> str:
        """Get human-readable name for action type."""
        names = {ACTION_MOVE: "MOVE", ACTION_ATTACK: "ATTACK", ACTION_STOP: "STOP"}
        return names.get(action_type, f"UNKNOWN({action_type})")

    @staticmethod
    def is_move_action(action_type: int) -> bool:
        """Check if action type is MOVE."""
        return action_type == ACTION_MOVE

    @staticmethod
    def is_attack_action(action_type: int) -> bool:
        """Check if action type is ATTACK."""
        return action_type == ACTION_ATTACK
