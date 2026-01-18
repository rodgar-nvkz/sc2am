"""Main ActorCritic model composing encoders and heads.

This module provides the top-level ActorCritic class that:
1. Encodes observations using VectorEncoder (and future terrain CNN)
2. Produces discrete command actions via CommandHead
3. Produces continuous angle actions via AngleHead (conditioned on command)
4. Estimates state value via CriticHead

The model supports:
- Skip connections (raw obs directly to heads)
- Action masking for invalid commands
- Deterministic evaluation mode
- Encapsulated loss computation in heads
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from smarte.impala_v2.model.config import ModelConfig
from smarte.impala_v2.model.encoders import VectorEncoder
from smarte.impala_v2.model.heads import AngleHead, CommandHead, CriticHead, HeadLoss, HeadOutput


@dataclass
class ActorCriticOutput:
    """Complete output from ActorCritic forward pass.

    Contains outputs from all heads plus shared information.
    """

    # Command head output
    command: HeadOutput

    # Angle head output
    angle: HeadOutput

    # Value estimate
    value: Tensor

    # Convenience properties
    @property
    def total_entropy(self) -> Tensor:
        """Total entropy (command + angle)."""
        return self.command.entropy + self.angle.entropy

    @property
    def total_log_prob(self) -> Tensor:
        """Total log probability (command + angle)."""
        return self.command.log_prob + self.angle.log_prob


class ActorCritic(nn.Module):
    """Autoregressive actor-critic for hybrid discrete-continuous action space.

    Action space:
    - Discrete command: MOVE, ATTACK_Z1, ATTACK_Z2
    - Continuous angle: (sin, cos) for movement direction (used when command=MOVE)

    The angle head is conditioned on the discrete command via embedding.
    This allows the network to learn command-specific angle distributions.

    P(action | obs) = P(command | obs) * P(angle | obs, command)

    Architecture:
        obs -> VectorEncoder -> features
                                   |
                                   +-> CommandHead (features + raw_obs) -> command
                                   |
                                   +-> AngleHead (features + raw_obs + cmd_embed) -> angle
                                   |
                                   +-> CriticHead (features + raw_obs) -> value
    """

    def __init__(self, config: ModelConfig):
        """Initialize ActorCritic model.

        Args:
            config: Model configuration defining architecture.
        """
        super().__init__()
        self.config = config

        # Build components
        self.encoder = VectorEncoder(config) if config.use_embedding else None

        self.command_head = CommandHead(config)
        self.angle_head = AngleHead(config)
        self.value_head = CriticHead(config)

        # Expose heads as ModuleDict for easy iteration
        self.heads = nn.ModuleDict({
            "command": self.command_head,
            "angle": self.angle_head,
            "value": self.value_head,
        })

    def _encode(self, obs: Tensor) -> Tensor:
        """Encode observation to features.

        Args:
            obs: Raw observation (B, obs_size)

        Returns:
            Encoded features (B, embed_size) or zeros if no encoder
        """
        if self.encoder is not None:
            return self.encoder(obs)
        else:
            # Return zeros if no encoder (heads will use raw_obs via skip connection)
            return torch.zeros(obs.shape[0], self.config.embed_size, device=obs.device)

    def forward(
        self,
        obs: Tensor,
        command: Tensor | None = None,
        angle: Tensor | None = None,
        action_mask: Tensor | None = None,
    ) -> ActorCriticOutput:
        """Forward pass through all components.

        Args:
            obs: Observations (B, obs_size)
            command: Optional commands to evaluate (B,). If None, samples new.
            angle: Optional angles to evaluate (B, 2). If None, samples new.
            action_mask: Optional boolean mask where True = valid action (B, num_commands)

        Returns:
            ActorCriticOutput with all head outputs and value
        """
        # Encode observation
        features = self._encode(obs)

        # Command head
        cmd_output = self.command_head(
            features=features,
            raw_obs=obs,
            action=command,
            mask=action_mask,
        )

        # Angle head (conditioned on command)
        angle_output = self.angle_head(
            features=features,
            raw_obs=obs,
            command=cmd_output.action,
            action=angle,
        )

        # Value head
        value = self.value_head(features=features, raw_obs=obs)

        return ActorCriticOutput(command=cmd_output, angle=angle_output, value=value)

    def get_value(self, obs: Tensor) -> Tensor:
        """Get value estimate only (for V-trace computation).

        Args:
            obs: Observations (B, obs_size)

        Returns:
            Value estimates (B,)
        """
        features = self._encode(obs)
        return self.value_head(features=features, raw_obs=obs)

    def get_deterministic_action(
        self,
        obs: Tensor,
        action_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Get deterministic action for evaluation (argmax command, mean angle).

        Args:
            obs: Observations (B, obs_size)
            action_mask: Optional boolean mask where True = valid action (B, num_commands)

        Returns:
            Tuple of:
            - command: (B,) discrete command indices
            - angle: (B, 2) normalized sin/cos angle
        """
        features = self._encode(obs)

        # Deterministic command (argmax)
        command = self.command_head.get_deterministic_action(features=features, raw_obs=obs, mask=action_mask)

        # Deterministic angle (mean of distribution)
        angle = self.angle_head.get_deterministic_action(features=features, raw_obs=obs, command=command)

        return command, angle

    def compute_losses(
        self,
        output: ActorCriticOutput,
        old_cmd_log_prob: Tensor,
        old_angle_log_prob: Tensor,
        advantages: Tensor,
        vtrace_targets: Tensor,
        move_mask: Tensor,
        clip_epsilon: float,
    ) -> dict[str, HeadLoss]:
        """Compute losses for all heads.

        Args:
            output: Forward pass output
            old_cmd_log_prob: Behavior policy command log probs (B,)
            old_angle_log_prob: Behavior policy angle log probs (B,)
            advantages: Advantage estimates (B,)
            vtrace_targets: V-trace value targets (B,)
            move_mask: Float mask where 1.0 = MOVE command (B,)
            clip_epsilon: PPO clipping parameter

        Returns:
            Dictionary of HeadLoss for each head
        """
        cmd_loss = self.command_head.compute_loss(
            new_log_prob=output.command.log_prob,
            old_log_prob=old_cmd_log_prob,
            advantages=advantages,
            clip_epsilon=clip_epsilon,
        )

        angle_loss = self.angle_head.compute_loss(
            new_log_prob=output.angle.log_prob,
            old_log_prob=old_angle_log_prob,
            advantages=advantages,
            clip_epsilon=clip_epsilon,
            mask=move_mask,
        )

        value_loss = self.value_head.compute_loss(
            values=output.value,
            targets=vtrace_targets,
        )

        return {
            "command": cmd_loss,
            "angle": angle_loss,
            "value": value_loss,
        }
