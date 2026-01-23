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

import math
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


class PairwiseAuxHead(nn.Module):
    """Auxiliary head: predicts directed geometry from an embedding pair.

    Small shared MLP takes (embed_i || embed_j) and predicts (dist, sin, cos)
    for the i->j direction. Forces each individual embedding to encode
    position well enough for any pair to recover geometry.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(config.aux_input_size, config.aux_hidden_size),
            nn.SiLU(),
            nn.Linear(config.aux_hidden_size, 3),  # (dist, sin, cos)
        )
        if config.init_orthogonal:
            for module in self.head:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=config.init_gain)
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, embed_pairs: Tensor) -> Tensor:
        """Predict directed geometry for embedding pairs.

        Args:
            embed_pairs: (B*K, embed_dim*2) concatenated [embed_i, embed_j]

        Returns:
            (B*K, 3) predictions: [distance, sin, cos]
        """
        return self.head(embed_pairs)


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
    A shared PairwiseAuxHead predicts directed geometry (dist, sin, cos) from
    pairs of coordinate embeddings, forcing the PointEncoder to learn spatial
    representations that support relational reasoning.

    Architecture:
        obs -> PointEncoder -> coord_embeds -+-> (flatten + non-coord) -> CommandHead -> command
                    |                        +-> AngleHead -> angle
                    |                        +-> CriticHead -> value
                    +-> PairwiseAuxHead(embed_i || embed_j) -> (dist, sin, cos)
    """

    def __init__(self, config: ModelConfig):
        """Initialize ActorCritic model."""
        super().__init__()
        self.config = config

        # Coordinate embedding
        self.point_encoder = PointEncoder(config)
        self.aux_head = PairwiseAuxHead(config) if config.aux_enabled else None

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

    def _extract_coords(self, obs: Tensor) -> Tensor:
        """Extract coordinate points from raw observations.

        Args:
            obs: (B, obs_size) raw observations

        Returns:
            (B, num_points, 3) tensor of [x_norm, y_norm, valid]
        """
        return obs[:, self._coord_indices.flatten()].view(-1, self.config.num_coord_points, 3)

    def _prepare_head_input(self, obs: Tensor) -> Tensor:
        """Extract coords, encode with PointEncoder, concat with non-coord features.

        Args:
            obs: (B, obs_size) raw observations

        Returns:
            (B, head_input_size) tensor for action/value heads
        """
        coord_points = self._extract_coords(obs)
        non_coord = obs[:, self._non_coord_indices]
        coord_embeds = self.point_encoder(coord_points)
        coord_flat = coord_embeds.flatten(1)
        return torch.cat([non_coord, coord_flat], dim=-1)

    def _compute_pair_targets(self, coords: Tensor, idx_i: Tensor, idx_j: Tensor) -> Tensor:
        """Compute directed geometry targets for given pair indices.

        Args:
            coords: (B, N, 2) normalized coordinates
            idx_i: (K,) source point indices
            idx_j: (K,) target point indices

        Returns:
            (B, K, 3) targets: [distance, sin, cos] per directed pair
        """
        # (B, K, 2)
        pi = coords[:, idx_i]
        pj = coords[:, idx_j]
        diff = pj - pi  # (B, K, 2)
        dx = diff[:, :, 0]
        dy = diff[:, :, 1]
        dist = torch.sqrt(dx * dx + dy * dy + 1e-8) * (1.0 / math.sqrt(2))
        angle = torch.atan2(dy, dx)
        return torch.stack([dist, torch.sin(angle), torch.cos(angle)], dim=-1)

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

        Samples K random directed pairs (i, j) where i != j and both valid.
        A small shared MLP predicts (dist, sin, cos) from (embed_i || embed_j).
        Forces each embedding to independently encode position.

        Args:
            obs: Observations (B, obs_size)

        Returns:
            Scalar MSE loss over sampled valid pairs
        """
        if self.aux_head is None:
            return torch.tensor(0.0, device=obs.device)

        coord_points = self._extract_coords(obs)
        coords = coord_points[:, :, :2]  # (B, N, 2)
        valid = coord_points[:, :, 2]  # (B, N)
        coord_embeds = self.point_encoder(coord_points)  # (B, N, embed_dim)

        B, N = coords.shape[0], coords.shape[1]
        K = self.config.aux_num_samples

        # Build all directed pair indices (i, j) where i != j
        all_pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
        num_pairs = len(all_pairs)

        # Sample K pairs (with replacement if K > num_pairs)
        sample_idx = torch.randint(num_pairs, (K,), device=obs.device)
        all_pairs_t = torch.tensor(all_pairs, device=obs.device)  # (num_pairs, 2)
        idx_i = all_pairs_t[sample_idx, 0]  # (K,)
        idx_j = all_pairs_t[sample_idx, 1]  # (K,)

        # Pair validity mask: (B, K)
        pair_valid = valid[:, idx_i] * valid[:, idx_j]

        # Gather embeddings and concatenate: (B, K, embed_dim*2)
        embed_i = coord_embeds[:, idx_i]  # (B, K, embed_dim)
        embed_j = coord_embeds[:, idx_j]  # (B, K, embed_dim)
        embed_pairs = torch.cat([embed_i, embed_j], dim=-1)  # (B, K, embed_dim*2)

        # Predict: (B*K, 3)
        pred = self.aux_head(embed_pairs.view(B * K, -1)).view(B, K, 3)

        # Targets: (B, K, 3)
        targets = self._compute_pair_targets(coords, idx_i, idx_j)

        # Masked MSE
        mask_expanded = pair_valid.unsqueeze(-1).expand_as(pred)  # (B, K, 3)
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
