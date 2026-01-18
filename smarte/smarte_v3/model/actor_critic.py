"""Main ActorCritic model composing heads for hybrid action space.

This module provides the top-level ActorCritic class that:
1. Produces discrete command actions via CommandHead
2. Produces continuous angle actions via AngleHead (conditioned on command)
3. Estimates state value via CriticHead

The model supports:
- Action masking for invalid commands
- Deterministic evaluation mode
- Encapsulated loss computation in heads
"""

from dataclasses import dataclass

from torch import Tensor, nn

from .config import ModelConfig
from .heads import AngleHead, CommandHead, CriticHead, HeadLoss, HeadOutput


@dataclass
class ActorCriticOutput:
    """Complete output from ActorCritic forward pass"""

    command: HeadOutput
    angle: HeadOutput
    value: Tensor

    @property
    def total_entropy(self) -> Tensor:
        return self.command.entropy + self.angle.entropy

    @property
    def total_log_prob(self) -> Tensor:
        return self.command.log_prob + self.angle.log_prob


class ActorCritic(nn.Module):
    """Autoregressive actor-critic for hybrid discrete-continuous action space.

    Action space:
    - Discrete command: MOVE, ATTACK_Z1, ATTACK_Z2
    - Continuous angle: (sin, cos) for movement direction (used when command=MOVE)

    The angle head is conditioned on the discrete command via one-hot encoding.
    This allows the network to learn command-specific angle distributions.

    P(action | obs) = P(command | obs) * P(angle | obs, command)

    Architecture:
        obs -> CommandHead -> command
            |
            +-> AngleHead (obs + cmd_onehot) -> angle
            |
            +-> CriticHead -> value
    """

    def __init__(self, config: ModelConfig):
        """Initialize ActorCritic model"""
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
        self,
        obs: Tensor,
        command: Tensor | None = None,
        angle: Tensor | None = None,
        action_mask: Tensor | None = None,
    ) -> ActorCriticOutput:
        """Forward pass through all components.

        Args:
            obs: Observations (B, T)
            command: Optional commands to evaluate (B,). If None, samples new.
            angle: Optional angles to evaluate (B, 2). If None, samples new.
            action_mask: Optional boolean mask where True = valid action (B, num_commands)

        """
        command_output = self.command_head(obs=obs, action=command, mask=action_mask)
        angle_output = self.angle_head(obs=obs, command=command_output.action, action=angle)
        value = self.value_head(obs=obs)
        return ActorCriticOutput(command=command_output, angle=angle_output, value=value)

    def get_value(self, obs: Tensor) -> Tensor:
        """Get value estimate only (for V-trace computation)"""
        return self.value_head(obs=obs)

    def get_deterministic_action(self, obs: Tensor, action_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Get deterministic action for evaluation (argmax command, mean angle).

        Args:
            obs: Observations (B, obs_size)
            action_mask: Optional boolean mask where True = valid action (B, num_commands)

        Returns:
            Tuple of:
            - command: (B,) discrete command indices
            - angle: (B, 2) normalized sin/cos angle
        """
        # Deterministic command (argmax)
        command = self.command_head.get_deterministic_action(obs=obs, mask=action_mask)

        # Deterministic angle (mean of distribution)
        angle = self.angle_head.get_deterministic_action(obs=obs, command=command)

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

        return {"command": cmd_loss, "angle": angle_loss, "value": value_loss}
