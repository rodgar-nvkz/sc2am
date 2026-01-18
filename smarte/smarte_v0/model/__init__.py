"""Model package for LSTM-based ActorCritic architecture.

Architecture:
    obs (obs_size) → VectorEncoder (MLP) → LSTM → [ActorHead, CriticHead]

Action space (40 discrete actions):
    - 0-35: MOVE in direction (angle = i * 10°)
    - 36: ATTACK_Z1
    - 37: ATTACK_Z2
    - 38: STOP
    - 39: SKIP (no-op)

Example:
    from .model import ActorCritic, ModelConfig

    config = ModelConfig(obs_size=12, num_actions=40)
    model = ActorCritic(config)

    hidden = model.get_initial_hidden(batch_size=1)
    output = model(obs, hidden=hidden)
    action = output.action.action
    new_hidden = output.hidden
"""

from .actor_critic import ActorCritic, ActorCriticOutput
from .config import ModelConfig
from .encoders import VectorEncoder
from .heads import (
    ActionHead,
    CriticHead,
    DiscreteActionHead,
    HeadLoss,
    HeadOutput,
    ValueHead,
)

__all__ = [
    # Main model
    "ActorCritic",
    "ActorCriticOutput",
    # Config
    "ModelConfig",
    # Encoders
    "VectorEncoder",
    # Heads
    "ActionHead",
    "DiscreteActionHead",
    "CriticHead",
    "HeadLoss",
    "HeadOutput",
    "ValueHead",
]
