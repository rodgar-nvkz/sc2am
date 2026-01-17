"""Model package for entity-attention based ActorCritic architecture.

Architecture:
    Observation → EntityEncoder → [marine_emb, enemy_embs]
                        ↓
    TemporalEncoder (GRU) → [h_marine, h_enemies]
                        ↓
    CrossAttention (marine→enemies) → context, attn_weights
                        ↓
    SharedBackbone output: [h_marine; context; time_left]
                        ↓
    ┌───────────────────┼───────────────────┐
    ↓                   ↓                   ↓
ActionTypeHead    MoveDirectionHead    AttackTargetHead
    ↓                   ↓                   ↓
 [MOVE,ATTACK,STOP]  [sin,cos]         enemy_idx
    ↓                   ↓                   ↓
    └─────── HybridAction ─────────────────┘
                        ↓
              ValueHead → V(s)
              AuxiliaryHeads → damage_pred, distance_pred

Action Space (Hybrid):
    - Action Type: Discrete [MOVE=0, ATTACK=1, STOP=2]
    - Move Direction: Continuous [sin, cos] (only when MOVE)
    - Attack Target: Pointer over N enemies (only when ATTACK)

Example:
    from .model import ActorCritic, ModelConfig

    config = ModelConfig(max_enemies=2)
    model = ActorCritic(config)

    hidden = model.get_initial_hidden(batch_size=1)
    output = model(obs, hidden=hidden)

    # Access action components
    action_type = output.action.action_type      # 0=MOVE, 1=ATTACK, 2=STOP
    move_direction = output.action.move_direction  # [sin, cos]
    attack_target = output.action.attack_target    # enemy index

    # Convert to environment action
    env_action = model.to_env_action(output.action)

    # Value and attention
    value = output.value
    attn_weights = output.attn_weights  # Which enemy is being focused on

    # Update hidden for next step
    new_hidden = output.hidden
"""

from .actor_critic import ActorCritic, ActorCriticOutput
from .attention import CrossAttention
from .config import (
    ACTION_ATTACK,
    ACTION_MOVE,
    ACTION_STOP,
    NUM_ACTION_TYPES,
    OBS_ENEMY_COS_OFFSET,
    OBS_ENEMY_DIST_OFFSET,
    OBS_ENEMY_FEATURE_SIZE,
    OBS_ENEMY_HP_OFFSET,
    OBS_ENEMY_SIN_OFFSET,
    OBS_ENEMY_START,
    OBS_MARINE_CD_BINARY_IDX,
    OBS_MARINE_CD_NORM_IDX,
    OBS_MARINE_END,
    OBS_MARINE_HP_IDX,
    OBS_MARINE_SIZE,
    OBS_MARINE_START,
    OBS_TIME_LEFT_IDX,
    OBS_TIME_LEFT_SIZE,
    ModelConfig,
)
from .encoders import (
    EnemyEncoder,
    EntityEncoder,
    EntityMLP,
    MarineEncoder,
    TemporalEncoder,
)
from .heads import (
    ActionHead,
    ActionTypeHead,
    AttackTargetHead,
    AuxiliaryHead,
    CombinedAuxiliaryHead,
    CriticHead,
    DamageAuxHead,
    DistanceAuxHead,
    HeadLoss,
    HeadOutput,
    HybridAction,
    MoveDirectionHead,
    ValueHead,
)

__all__ = [
    # Main model
    "ActorCritic",
    "ActorCriticOutput",
    # Config
    "ModelConfig",
    "ACTION_MOVE",
    "ACTION_ATTACK",
    "ACTION_STOP",
    "NUM_ACTION_TYPES",
    # Observation index constants
    "OBS_TIME_LEFT_IDX",
    "OBS_TIME_LEFT_SIZE",
    "OBS_MARINE_START",
    "OBS_MARINE_HP_IDX",
    "OBS_MARINE_CD_BINARY_IDX",
    "OBS_MARINE_CD_NORM_IDX",
    "OBS_MARINE_END",
    "OBS_MARINE_SIZE",
    "OBS_ENEMY_START",
    "OBS_ENEMY_HP_OFFSET",
    "OBS_ENEMY_SIN_OFFSET",
    "OBS_ENEMY_COS_OFFSET",
    "OBS_ENEMY_DIST_OFFSET",
    "OBS_ENEMY_FEATURE_SIZE",
    # Encoders
    "EntityEncoder",
    "EntityMLP",
    "MarineEncoder",
    "EnemyEncoder",
    "TemporalEncoder",
    # Attention
    "CrossAttention",
    # Heads - Base
    "ActionHead",
    "ValueHead",
    "AuxiliaryHead",
    "HeadOutput",
    "HybridAction",
    "HeadLoss",
    # Heads - Action
    "ActionTypeHead",
    "MoveDirectionHead",
    "AttackTargetHead",
    # Heads - Value
    "CriticHead",
    # Heads - Auxiliary
    "DamageAuxHead",
    "DistanceAuxHead",
    "CombinedAuxiliaryHead",
]
