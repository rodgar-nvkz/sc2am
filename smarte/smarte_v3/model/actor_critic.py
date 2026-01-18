"""Main ActorCritic model composing heads for hybrid action space.

AlphaZero-style architecture: all action heads predict independently from
observations, then masking is applied at loss computation time.

This module provides the top-level ActorCritic class that:
1. Produces discrete command actions via CommandHead
2. Produces continuous angle actions via AngleHead (independent, not conditioned on command)
3. Estimates state value via CriticHead

The model supports:
- Action masking for invalid commands
- Proper masked entropy/log_prob for training (mask is REQUIRED)
- Deterministic evaluation mode
- Encapsulated loss computation in heads
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .heads import AngleHead, CommandHead, CriticHead, HeadLoss, HeadOutput


@dataclass
class ActorCriticOutput:
    """Complete output from ActorCritic forward pass"""

    command: HeadOutput
    angle: HeadOutput
    value: Tensor

    def total_entropy(self, move_mask: Tensor) -> Tensor:
        """Compute total entropy with proper masking.

        Args:
            move_mask: Float mask where 1.0 = MOVE command (B,). REQUIRED.

        Returns:
            Total entropy (B,) - angle entropy masked for non-MOVE commands
        """
        return self.command.entropy + self.angle.entropy * move_mask

    def total_log_prob(self, move_mask: Tensor) -> Tensor:
        """Compute total log probability with proper masking.

        For importance sampling, the action taken when command != MOVE
        is just the command itself, not (command, angle). So we should
        only include angle_log_prob when command == MOVE.

        Args:
            move_mask: Float mask where 1.0 = MOVE command (B,).

        Returns:
            Total log prob (B,) - angle log_prob masked for non-MOVE commands
        """
        return self.command.log_prob + self.angle.log_prob * move_mask


class ActorCritic(nn.Module):
    """AlphaZero-style actor-critic for hybrid discrete-continuous action space.

    Action space:
    - Discrete command: MOVE, ATTACK_Z1, ATTACK_Z2
    - Continuous angle: (sin, cos) for movement direction (used when command=MOVE)

    All heads predict independently from observations (no autoregressive conditioning).
    Masking is applied at loss computation time:
    - Command mask: game state constraints (cooldown, range)
    - Angle mask: only train angle when MOVE was selected

    P(action | obs) = P(command | obs) * P(angle | obs)  [independent]

    Architecture:
        obs -> CommandHead -> command
            |
            +-> AngleHead -> angle
            |
            +-> CriticHead -> value
    """

    def __init__(self, config: ModelConfig):
        """Initialize ActorCritic model."""
        super().__init__()
        self.config = config

        self.command_head = CommandHead(config)
        self.angle_head = AngleHead(config)
        self.value_head = CriticHead(config)

        # Expose heads as ModuleDict for easy iteration
        self.heads = nn.ModuleDict(
            {
                "command": self.command_head,
                "angle": self.angle_head,
                "value": self.value_head,
            }
        )

    def forward(
        self, obs: Tensor, command: Tensor | None = None, angle: Tensor | None = None, *, action_mask: Tensor
    ) -> ActorCriticOutput:
        """Forward pass through all components (parallel prediction).

        Args:
            obs: Observations (B, obs_size)
            command: Optional commands to evaluate (B,). If None, samples new.
            angle: Optional angles to evaluate (B, 2). If None, samples new.
            action_mask: Boolean mask where True = valid action (B, num_commands).

        Returns:
            ActorCriticOutput with command, angle, and value outputs
        """
        # All heads predict independently from obs
        command_output = self.command_head(obs=obs, action=command, mask=action_mask)
        angle_output = self.angle_head(obs=obs, action=angle)
        value = self.value_head(obs=obs)

        return ActorCriticOutput(command=command_output, angle=angle_output, value=value)

    def get_value(self, obs: Tensor) -> Tensor:
        """Get value estimate only (for V-trace computation)."""
        return self.value_head(obs=obs)

    def get_deterministic_action(self, obs: Tensor, *, action_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Get deterministic action for evaluation (argmax command, mean angle).

        Args:
            obs: Observations (B, obs_size)
            action_mask: Boolean mask where True = valid action (B, num_commands). REQUIRED.

        Returns:
            Tuple of:
            - command: (B,) discrete command indices
            - angle: (B, 2) normalized sin/cos angle
        """
        # Deterministic command (argmax)
        command = self.command_head.get_deterministic_action(obs=obs, mask=action_mask)

        # Deterministic angle (mean of distribution) - no command conditioning
        angle = self.angle_head.get_deterministic_action(obs=obs)

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
            move_mask: Float mask where 1.0 = MOVE command (B,). REQUIRED.
            clip_epsilon: PPO clipping parameter

        Returns:
            Dictionary of HeadLoss for each head
        """
        # Command head uses all samples (mask is required but ignored internally)
        cmd_mask = torch.ones_like(move_mask)
        cmd_loss = self.command_head.compute_loss(
            new_log_prob=output.command.log_prob,
            old_log_prob=old_cmd_log_prob,
            advantages=advantages,
            clip_epsilon=clip_epsilon,
            mask=cmd_mask,
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

        return {"command": cmd_loss, "angle": angle_loss, "value": value_loss}
