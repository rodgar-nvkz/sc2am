"""Main ActorCritic model composing heads for hybrid action space.

AlphaZero-style architecture: all action heads predict independently from
observations, then masking is applied at loss computation time.

This module provides the top-level ActorCritic class that:
1. Produces discrete command actions via CommandHead
2. Produces continuous angle actions via AngleHead (independent, not conditioned on command)
3. Estimates state value via CriticHead
4. Computes auxiliary prediction loss for representation learning

The model supports:
- Action masking for invalid commands
- Proper masked entropy/log_prob for training (mask is REQUIRED)
- Deterministic evaluation mode
- Encapsulated loss computation in heads
- Auxiliary prediction task to prevent encoder collapse
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .heads import AngleHead, CommandHead, CriticHead, HeadLoss, HeadOutput
from .unit_encoder import PairwiseAuxHead, UnitEncoder


@dataclass
class ActorCriticOutput:
    """Complete output from ActorCritic forward pass"""

    command: HeadOutput
    angle: HeadOutput
    value: Tensor

    def total_entropy(self) -> Tensor:
        """Compute total entropy (sum of command and angle entropy).

        Returns:
            Total entropy (B,)
        """
        return self.command.entropy + self.angle.entropy

    def total_log_prob(self) -> Tensor:
        """Compute total log probability (sum of command and angle log probs).

        Returns:
            Total log prob (B,)
        """
        return self.command.log_prob + self.angle.log_prob


class ActorCritic(nn.Module):
    """AlphaZero-style actor-critic for hybrid discrete-continuous action space.

    Action space:
    - Discrete command: MOVE, ATTACK_Z1, ATTACK_Z2
    - Continuous angle: (sin, cos) for movement direction (used when command=MOVE)

    All heads predict independently from observations (no autoregressive conditioning).
    Masking is applied at loss computation time.

    Auxiliary Prediction Task:
    A shared PairwiseAuxHead predicts directed geometry (dist, sin, cos) from
    shuffled pairs of unit embeddings, forcing the UnitEncoder to learn spatial
    representations that support relational reasoning.

    Architecture:
        obs (B, N, 8) -> UnitEncoder -> (B, N, E) -+-> flatten + obs[:,:,2:] -> CommandHead -> command
                                                   +-> AngleHead -> angle
                                                   +-> CriticHead -> value

    Auxiliary (separate pass):
        obs -> UnitEncoder -> PairwiseAuxHead -> (dist, sin, cos)
    """

    def __init__(self, config: ModelConfig):
        """Initialize ActorCritic model."""
        super().__init__()
        self.config = config

        # Unit embedding
        self.unit_encoder = UnitEncoder(config)
        self.aux_head = PairwiseAuxHead(config) if config.aux_enabled else None

        # Action heads
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

    def obs_to_head_input(self, obs: Tensor) -> Tensor:
        """Encode units and concatenate with non-coord features for heads.

        Args:
            obs: (B, N, 8) unit observations

        Returns:
            (B, head_input_size) tensor for action/value heads
        """
        self._cached_embeds = self.unit_encoder(obs)  # (B, N, E)
        non_coord = obs[:, :, self.config.obs_spec.coord_size :]  # (B, N, 6)
        return torch.cat([self._cached_embeds.flatten(1), non_coord.flatten(1)], dim=-1)

    def forward(
        self, obs: Tensor, command: Tensor | None = None, angle: Tensor | None = None, *, action_mask: Tensor
    ) -> ActorCriticOutput:
        """Forward pass through all components (parallel prediction).

        Args:
            obs: Observations (B, N, 8)
            command: Optional commands to evaluate (B,). If None, samples new.
            angle: Optional angles to evaluate (B, 2). If None, samples new.
            action_mask: Boolean mask where True = valid action (B, num_commands).

        Returns:
            ActorCriticOutput with command, angle, and value outputs
        """
        head_input = self.obs_to_head_input(obs)

        command_output = self.command_head(obs=head_input, action=command, mask=action_mask)
        angle_output = self.angle_head(obs=head_input, action=angle)
        value = self.value_head(obs=head_input)

        return ActorCriticOutput(command=command_output, angle=angle_output, value=value)

    def get_value(self, obs: Tensor) -> Tensor:
        """Get value estimate only (for V-trace computation)."""
        head_input = self.obs_to_head_input(obs)
        return self.value_head(obs=head_input)

    def get_deterministic_action(self, obs: Tensor, *, action_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Get deterministic action for evaluation (argmax command, mean angle).

        Args:
            obs: Observations (B, N, 8)
            action_mask: Boolean mask where True = valid action (B, num_commands). REQUIRED.

        Returns:
            Tuple of:
            - command: (B,) discrete command indices
            - angle: (B, 2) normalized sin/cos angle
        """
        head_input = self.obs_to_head_input(obs)

        command = self.command_head.get_deterministic_action(obs=head_input, mask=action_mask)
        angle = self.angle_head.get_deterministic_action(obs=head_input)

        return command, angle

    def compute_aux_loss(self, obs: Tensor) -> Tensor:
        """Compute auxiliary pairwise geometry prediction loss.

        Uses cached embeddings from the most recent forward/_prepare_head_input call.

        Args:
            obs: Observations (B, N, 8) — used for coords and valid mask

        Returns:
            Scalar MSE loss over all valid pairs
        """
        if self.aux_head is None:
            return torch.tensor(0.0, device=obs.device)
        return self.aux_head.compute_loss(obs, self._cached_embeds)

    def compute_losses(
        self,
        output: ActorCriticOutput,
        old_cmd_log_prob: Tensor,
        old_angle_log_prob: Tensor,
        advantages: Tensor,
        vtrace_targets: Tensor,
        clip_epsilon: float,
    ) -> dict[str, HeadLoss]:
        """Compute losses for all heads.

        Args:
            output: Forward pass output
            old_cmd_log_prob: Behavior policy command log probs (B,)
            old_angle_log_prob: Behavior policy angle log probs (B,)
            advantages: Advantage estimates (B,)
            vtrace_targets: V-trace value targets (B,)
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
        )

        value_loss = self.value_head.compute_loss(
            values=output.value,
            targets=vtrace_targets,
        )

        return {"command": cmd_loss, "angle": angle_loss, "value": value_loss}
