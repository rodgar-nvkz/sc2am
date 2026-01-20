"""Model package for ActorCritic architecture.

AlphaZero-style parallel prediction architecture for RL models:

- All action heads predict independently from observations
- Command mask: game state constraints (cooldown, range) - required for forward pass
- No action masking for loss/entropy computation (all samples used for all heads)

Components:
- config: ModelConfig dataclass for all architectural parameters
- heads: Action and value heads (CommandHead, AngleHead, CriticHead)
- actor_critic: Main ActorCritic model composing all components

Example:
    from smarte.smarte_v3.env import SC2GymEnv
    from smarte.smarte_v3.model import ActorCritic, ModelConfig

    # Initialize config from environment's ObsSpec (single source of truth)
    config = ModelConfig(
        obs_spec=SC2GymEnv.obs_spec,
        num_commands=SC2GymEnv.NUM_COMMANDS,
        move_action_id=SC2GymEnv.MOVE_ACTION_ID,
    )
    model = ActorCritic(config)

    # action_mask is REQUIRED (use torch.ones for "all valid" if needed)
    output = model(obs, action_mask=mask)

    # Entropy and log_prob computed for all samples (no masking)
    entropy = output.total_entropy()
    log_prob = output.total_log_prob()

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
