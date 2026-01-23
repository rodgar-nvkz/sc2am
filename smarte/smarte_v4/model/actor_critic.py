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


class PointEncoder(nn.Module):
    """Shared encoder: (x, y, valid) -> embedding.

    Same weights applied to all coordinate points (ally and enemies).
    Architecture from coords.py research: 3-layer MLP with SiLU.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, config.coord_hidden_size),
            nn.SiLU(),
            nn.Linear(config.coord_hidden_size, config.coord_hidden_size),
            nn.SiLU(),
            nn.Linear(config.coord_hidden_size, config.coord_embed_dim),
        )
        if config.init_orthogonal:
            for module in self.encoder:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=config.init_gain)
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, points: Tensor) -> Tensor:
        """Encode coordinate points.

        Args:
            points: (B, N, 3) tensor of [x_norm, y_norm, valid]

        Returns:
            (B, N, embed_dim) embeddings
        """
        return self.encoder(points)


class CoordAuxHead(nn.Module):
    """Auxiliary head: predicts pairwise geometry from coordinate embeddings.

    Forces PointEncoder to learn spatial relationships by predicting
    (distance, sin, cos) for all directed pairs of points.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.head = nn.Sequential(
            nn.Linear(config.coord_flat_size, config.aux_hidden_size),
            nn.SiLU(),
            nn.Linear(config.aux_hidden_size, config.aux_hidden_size),
            nn.SiLU(),
            nn.Linear(config.aux_hidden_size, config.aux_output_size),
        )
        if config.init_orthogonal:
            for module in self.head:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=config.init_gain)
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, coord_embeds_flat: Tensor) -> Tensor:
        """Predict pairwise geometry.

        Args:
            coord_embeds_flat: (B, num_points * embed_dim)

        Returns:
            (B, num_pairs, 3) predictions: [distance, sin, cos] per pair
        """
        out = self.head(coord_embeds_flat)
        return out.view(-1, self.config.num_aux_pairs, 3)


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

    Auxiliary Prediction Task:
    The angle head includes an auxiliary prediction head that forces the encoder
    to represent observation features (enemy angles, distances). This prevents
    encoder collapse where all observations map to similar hidden states, causing
    policy gradients to cancel across episodes with different optimal actions.

    Architecture:
        obs -> CommandHead -> command
            |
            +-> AngleHead -> angle
            |       |
            |       +-> AuxHead -> predicted obs features (auxiliary task)
            |
            +-> CriticHead -> value
    """

    def __init__(self, config: ModelConfig):
        """Initialize ActorCritic model."""
        super().__init__()
        self.config = config

        # Coordinate embedding
        self.point_encoder = PointEncoder(config)
        self.aux_head = CoordAuxHead(config) if config.aux_enabled else None

        # Register index tensors as buffers (move with model to device)
        coord_idx = config.obs_spec.coord_indices  # list of [x, y, valid] per point
        self.register_buffer(
            "_coord_indices",
            torch.tensor(coord_idx, dtype=torch.long),  # (num_points, 3)
        )
        self.register_buffer("_non_coord_indices", torch.tensor(config.obs_spec.non_coord_indices, dtype=torch.long))

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

    def _prepare_head_input(self, obs: Tensor) -> Tensor:
        """Extract coords, encode with PointEncoder, concat with non-coord features.

        Embeddings are detached from the policy gradient path — only the aux
        loss trains the PointEncoder. This prevents policy gradients from
        interfering with geometry learning.

        Args:
            obs: (B, obs_size) raw observations

        Returns:
            (B, head_input_size) tensor for action/value heads
        """
        # Extract coordinate points: (B, num_points, 3)
        coord_points = obs[:, self._coord_indices.flatten()].view(-1, self.config.num_coord_points, 3)
        # Extract non-coord features: (B, non_coord_size)
        non_coord = obs[:, self._non_coord_indices]
        # Encode coordinates: (B, num_points, embed_dim)
        coord_embeds = self.point_encoder(coord_points)
        # Flatten embeddings: (B, coord_flat_size)
        coord_flat = coord_embeds.flatten(1)
        # Concatenate: (B, head_input_size)
        return torch.cat([non_coord, coord_flat], dim=-1)

    def _compute_pairwise_targets(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute ground-truth pairwise geometry from raw coordinates.

        Args:
            obs: (B, obs_size) raw observations

        Returns:
            targets: (B, num_pairs, 3) [distance, sin, cos] per directed pair
            mask: (B, num_pairs) validity mask (1.0 if both points valid)
        """
        # Extract coordinate points: (B, num_points, 3)
        coord_points = obs[:, self._coord_indices.flatten()].view(-1, self.config.num_coord_points, 3)
        coords = coord_points[:, :, :2]  # (B, N, 2)
        valid = coord_points[:, :, 2]  # (B, N)

        N = self.config.num_coord_points
        targets = []
        masks = []

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dx = coords[:, j, 0] - coords[:, i, 0]
                dy = coords[:, j, 1] - coords[:, i, 1]
                dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
                angle = torch.atan2(dy, dx)
                pair_valid = valid[:, i] * valid[:, j]
                targets.append(torch.stack([dist, torch.sin(angle), torch.cos(angle)], dim=-1))
                masks.append(pair_valid)

        return torch.stack(targets, dim=1), torch.stack(masks, dim=1)

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
        head_input = self._prepare_head_input(obs)

        command_output = self.command_head(obs=head_input, action=command, mask=action_mask)
        angle_output = self.angle_head(obs=head_input, action=angle)
        value = self.value_head(obs=head_input)

        return ActorCriticOutput(command=command_output, angle=angle_output, value=value)

    def get_value(self, obs: Tensor) -> Tensor:
        """Get value estimate only (for V-trace computation)."""
        head_input = self._prepare_head_input(obs)
        return self.value_head(obs=head_input)

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
        head_input = self._prepare_head_input(obs)

        command = self.command_head.get_deterministic_action(obs=head_input, mask=action_mask)
        angle = self.angle_head.get_deterministic_action(obs=head_input)

        return command, angle

    def compute_aux_loss(self, obs: Tensor) -> Tensor:
        """Compute auxiliary pairwise geometry prediction loss.

        Forces PointEncoder to learn spatial relationships by predicting
        (distance, sin, cos) for all directed pairs of coordinate points.
        Loss is masked: only pairs where both points are valid contribute.

        Args:
            obs: Observations (B, obs_size)

        Returns:
            Scalar masked MSE loss
        """
        if self.aux_head is None:
            return torch.tensor(0.0, device=obs.device)

        # Get coord embeddings
        coord_points = obs[:, self._coord_indices.flatten()].view(-1, self.config.num_coord_points, 3)
        coord_embeds = self.point_encoder(coord_points)
        coord_flat = coord_embeds.flatten(1)

        # Predict pairwise geometry
        pred = self.aux_head(coord_flat)  # (B, num_pairs, 3)

        # Ground-truth targets
        targets, mask = self._compute_pairwise_targets(obs)

        # Masked MSE: average over valid elements (pairs × 3 components)
        mask_expanded = mask.unsqueeze(-1).expand_as(pred)  # (B, num_pairs, 3)
        sq_err = (pred - targets) ** 2 * mask_expanded
        n_valid = mask_expanded.sum().clamp(min=1.0)
        return sq_err.sum() / n_valid

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
