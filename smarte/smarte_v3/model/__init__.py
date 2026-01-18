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
    from smarte.smarte_v3.env import SC2GymEnv
    from smarte.smarte_v3.model import ActorCritic, ModelConfig

    # Initialize config from environment class constants
    config = ModelConfig(
        obs_size=SC2GymEnv.OBS_SIZE,
        num_commands=SC2GymEnv.NUM_COMMANDS,
        move_action_id=SC2GymEnv.MOVE_ACTION_ID,
    )
    model = ActorCritic(config)

    # action_mask is REQUIRED (use torch.ones for "all valid" if needed)
    output = model(obs, action_mask=mask)

    # move_mask is REQUIRED for training (use config.move_action_id)
    move_mask = (commands == config.move_action_id).float()
    entropy = output.total_entropy(move_mask)

    losses = model.compute_losses(output, old_log_probs, advantages, ...)
"""

from .actor_critic import ActorCritic, ActorCriticOutput
from .config import ModelConfig
from .gae import compute_gae_episode
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
    # GAE
    "compute_gae_episode",
    # Heads
    "ActionHead",
    "AngleHead",
    "CommandHead",
    "CriticHead",
    "HeadLoss",
    "HeadOutput",
]
