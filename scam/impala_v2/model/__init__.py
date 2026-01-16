"""Model package for ActorCritic architecture.

Modular, scalable architecture for RL models:

- config: ModelConfig dataclass for all architectural parameters
- encoders: Observation encoders (VectorEncoder, future TerrainCNN)
- heads: Action and value heads (CommandHead, AngleHead, CriticHead)
- actor_critic: Main ActorCritic model composing all components

Example:
    from scam.impala_v2.model import ActorCritic, ModelConfig

    config = ModelConfig(obs_size=11, num_commands=3)
    model = ActorCritic(config)

    output = model(obs, action_mask=mask)
    losses = model.compute_losses(output, old_log_probs, advantages, ...)
"""

from scam.impala_v2.model.actor_critic import ActorCritic, ActorCriticOutput
from scam.impala_v2.model.config import ModelConfig
from scam.impala_v2.model.encoders import VectorEncoder
from scam.impala_v2.model.heads import (
    ActionHead,
    AngleHead,
    CommandHead,
    CriticHead,
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
    "AngleHead",
    "CommandHead",
    "CriticHead",
    "HeadLoss",
    "HeadOutput",
    "ValueHead",
]
