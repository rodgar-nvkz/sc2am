"""Model package for ActorCritic architecture.

AlphaZero-style parallel prediction architecture for RL models:

- All action heads predict independently from observations
- Masks are REQUIRED (fail-fast design, no backward compatibility for None)
- Command mask: game state constraints (cooldown, range) - required for forward pass
- Angle mask: only train angle when MOVE was selected - required for loss computation

Components:
- config: ModelConfig dataclass for all architectural parameters
- heads: Action and value heads (CommandHead, AngleHead, CriticHead)
- actor_critic: Main ActorCritic model composing all components

Example:
    from smarte.smarte_v3.model import ActorCritic, ModelConfig

    config = ModelConfig(obs_size=11, num_commands=3)
    model = ActorCritic(config)

    # action_mask is REQUIRED (use torch.ones for "all valid" if needed)
    output = model(obs, action_mask=mask)

    # move_mask is REQUIRED for training
    move_mask = (commands == MOVE).float()
    entropy = output.total_entropy(move_mask)

    losses = model.compute_losses(output, old_log_probs, advantages, ...)
"""

from .actor_critic import ActorCritic, ActorCriticOutput
from .config import ModelConfig
from .heads import (
    ActionHead,
    AngleHead,
    CommandHead,
    CriticHead,
    HeadLoss,
    HeadOutput,
)

__all__ = [
    # Main model
    "ActorCritic",
    "ActorCriticOutput",
    # Config
    "ModelConfig",
    # Heads
    "ActionHead",
    "AngleHead",
    "CommandHead",
    "CriticHead",
    "HeadLoss",
    "HeadOutput",
]
