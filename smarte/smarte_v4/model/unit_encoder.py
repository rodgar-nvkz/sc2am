"""Unit encoder and auxiliary prediction head.

UnitEncoder: shared MLP that encodes full unit features into embeddings.
PairwiseAuxHead: auxiliary head that predicts directed geometry from
embedding pairs, forcing the encoder to learn spatial representations.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import ModelConfig


class UnitEncoder(nn.Module):
    """Shared encoder: full unit features -> embedding.

    Same weights applied to all units (ally and enemies).
    Input: [x, y, health, cooldown, range, facing_sin, facing_cos, valid]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.unit_input_size, config.unit_hidden_size),
            nn.SiLU(),
            nn.Linear(config.unit_hidden_size, config.unit_hidden_size),
            nn.SiLU(),
            nn.Linear(config.unit_hidden_size, config.unit_embed_dim),
        )
        if config.init_orthogonal:
            for module in self.encoder:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=config.init_gain)
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, units: Tensor) -> Tensor:
        """Encode unit observations.

        Args:
            units: (B, N, unit_feature_size) full unit features

        Returns:
            (B, N, embed_dim) embeddings
        """
        return self.encoder(units)


class PairwiseAuxHead(nn.Module):
    """Auxiliary head: predicts directed geometry from embedding pairs.

    Small shared MLP takes (embed_i || embed_j) and predicts (dist, sin, cos)
    for the i->j direction. Trained via shuffle-based pairing of all valid
    unit embeddings across the batch.
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
            embed_pairs: (V, embed_dim*2) concatenated [embed_i, embed_j]

        Returns:
            (V, 3) predictions: [distance, sin, cos]
        """
        return self.head(embed_pairs)

    def compute_loss(self, obs: Tensor, embeds: Tensor) -> Tensor:
        """Compute auxiliary pairwise geometry prediction loss.

        Gathers all valid unit embeddings across the batch, creates a random
        permutation (shuffle), and predicts (dist, sin, cos) for each
        (original, shuffled) pair. Self-pairs naturally produce [0, 0, 1] targets.

        Args:
            obs: Observations (B, N, 8) — used for coords and valid mask
            embeds: Pre-computed embeddings (B, N, E) from UnitEncoder

        Returns:
            Scalar MSE loss over all valid pairs
        """
        coords = obs[:, :, :2]  # (B, N, 2)
        valid = obs[:, :, 7]  # (B, N)

        # Gather all valid embeddings and coords across batch
        valid_mask = valid > 0.5  # (B, N) bool
        valid_embeds = embeds[valid_mask]  # (V, E)
        valid_coords = coords[valid_mask]  # (V, 2)

        if valid_embeds.shape[0] < 2:
            return torch.tensor(0.0, device=obs.device)

        # Shuffle: random permutation of valid embeddings
        perm = torch.randperm(valid_embeds.shape[0], device=obs.device)
        shuffled_embeds = valid_embeds[perm]  # (V, E)
        shuffled_coords = valid_coords[perm]  # (V, 2)

        # Predict geometry for each (orig, shuffled) pair
        pairs = torch.cat([valid_embeds, shuffled_embeds], dim=-1)  # (V, 2E)
        pred = self.head(pairs)  # (V, 3)

        # Targets from coordinates
        targets = self._compute_pair_targets(valid_coords, shuffled_coords)  # (V, 3)

        return F.mse_loss(pred, targets)

    @staticmethod
    def _compute_pair_targets(coords_i: Tensor, coords_j: Tensor) -> Tensor:
        """Compute directed geometry targets for coordinate pairs.

        Args:
            coords_i: (V, 2) source coordinates
            coords_j: (V, 2) target coordinates

        Returns:
            (V, 3) targets: [distance, sin, cos]
        """
        diff = coords_j - coords_i  # (V, 2)
        dx, dy = diff[:, 0], diff[:, 1]
        dist = torch.sqrt(dx * dx + dy * dy + 1e-8) * (1.0 / math.sqrt(2))
        angle = torch.atan2(dy, dx)
        return torch.stack([dist, torch.sin(angle), torch.cos(angle)], dim=-1)
