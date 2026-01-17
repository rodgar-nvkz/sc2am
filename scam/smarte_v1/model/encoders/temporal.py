"""Temporal encoder using GRU for maintaining state across steps.

This module processes entity embeddings through GRUs to capture temporal
context - crucial for learning movement patterns, predicting enemy behavior,
and executing kiting strategies.

Architecture:
    marine_emb (B, d) → Marine GRU → h_marine (B, d)
    enemy_embs (B, N, d) → Enemy GRU (shared) → h_enemies (B, N, d)

The GRU hidden states are maintained across environment steps, allowing
the model to "remember" recent history without explicit frame stacking.
"""

import torch
from torch import Tensor, nn

from ..config import ModelConfig


class TemporalEncoder(nn.Module):
    """GRU-based temporal encoder for entity embeddings.

    Maintains separate GRUs for marine and enemies, with the enemy GRU
    using shared weights across all enemies for generalization.

    Hidden State Format:
        The hidden state is a tuple: (h_marine, h_enemies)
        - h_marine: (num_layers, B, hidden_size)
        - h_enemies: (num_layers, B, N, hidden_size)

    Usage:
        encoder = TemporalEncoder(config)
        hidden = encoder.get_initial_hidden(batch_size=1, num_enemies=2)
        h_marine, h_enemies, new_hidden = encoder(marine_emb, enemy_embs, hidden)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Marine GRU
        self.marine_gru = nn.GRU(
            input_size=config.entity_embed_size,
            hidden_size=config.gru_hidden_size,
            num_layers=config.gru_num_layers,
            batch_first=True,
        )

        # Enemy GRU (shared weights across all enemies)
        self.enemy_gru = nn.GRU(
            input_size=config.entity_embed_size,
            hidden_size=config.gru_hidden_size,
            num_layers=config.gru_num_layers,
            batch_first=True,
        )

        if config.init_orthogonal:
            self._init_weights()

    def _init_weights(self) -> None:
        """Initialize GRU weights with orthogonal initialization."""
        for gru in [self.marine_gru, self.enemy_gru]:
            for name, param in gru.named_parameters():
                if "weight_ih" in name:
                    nn.init.orthogonal_(param, gain=self.config.init_gain)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param, gain=self.config.init_gain)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)

    def get_initial_hidden(
        self,
        batch_size: int,
        num_enemies: int,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Get zero-initialized hidden states for GRUs.

        Args:
            batch_size: Batch size
            num_enemies: Number of enemies
            device: Device to create tensors on

        Returns:
            Tuple of (h_marine, h_enemies):
                - h_marine: (num_layers, B, hidden_size)
                - h_enemies: (num_layers, B, N, hidden_size)
        """
        if device is None:
            device = next(self.parameters()).device

        h_marine = torch.zeros(
            self.config.gru_num_layers,
            batch_size,
            self.config.gru_hidden_size,
            device=device,
        )

        h_enemies = torch.zeros(
            self.config.gru_num_layers,
            batch_size,
            num_enemies,
            self.config.gru_hidden_size,
            device=device,
        )

        return h_marine, h_enemies

    def forward(
        self,
        marine_emb: Tensor,
        enemy_embs: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
        enemy_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        """Forward pass through temporal GRUs.

        Args:
            marine_emb: Marine embedding (B, entity_embed_size)
            enemy_embs: Enemy embeddings (B, N, entity_embed_size)
            hidden: Tuple of (h_marine, h_enemies) from previous step.
                   If None, initializes to zeros.
            enemy_mask: Optional mask where True = valid enemy (B, N).
                       Invalid enemies get zero hidden state.

        Returns:
            Tuple of:
                - h_marine: Temporal marine embedding (B, gru_hidden_size)
                - h_enemies: Temporal enemy embeddings (B, N, gru_hidden_size)
                - new_hidden: Updated hidden state tuple for next step
        """
        batch_size = marine_emb.shape[0]
        num_enemies = enemy_embs.shape[1]

        # Initialize hidden if not provided
        if hidden is None:
            hidden = self.get_initial_hidden(batch_size, num_enemies, marine_emb.device)

        h_marine_prev, h_enemies_prev = hidden

        # === Marine GRU ===
        # Add sequence dimension for GRU (B, 1, d)
        marine_input = marine_emb.unsqueeze(1)
        marine_out, h_marine_new = self.marine_gru(marine_input, h_marine_prev)
        # Remove sequence dimension (B, d)
        h_marine = marine_out.squeeze(1)

        # === Enemy GRU (shared weights) ===
        # Process each enemy through the shared GRU
        # Reshape for batch processing: (B * N, 1, d)
        enemy_input = enemy_embs.view(batch_size * num_enemies, 1, -1)

        # Reshape hidden state: (num_layers, B * N, hidden_size)
        h_enemies_prev_flat = h_enemies_prev.view(
            self.config.gru_num_layers,
            batch_size * num_enemies,
            self.config.gru_hidden_size,
        )

        enemy_out, h_enemies_new_flat = self.enemy_gru(enemy_input, h_enemies_prev_flat)

        # Reshape back: (B, N, hidden_size)
        h_enemies = enemy_out.view(batch_size, num_enemies, self.config.gru_hidden_size)
        h_enemies_new = h_enemies_new_flat.view(
            self.config.gru_num_layers,
            batch_size,
            num_enemies,
            self.config.gru_hidden_size,
        )

        # Zero out hidden states for invalid enemies
        if enemy_mask is not None:
            mask_expanded = enemy_mask.unsqueeze(-1).float()  # (B, N, 1)
            h_enemies = h_enemies * mask_expanded
            # Also mask the new hidden state
            mask_for_hidden = enemy_mask.unsqueeze(0).unsqueeze(-1).float()  # (1, B, N, 1)
            h_enemies_new = h_enemies_new * mask_for_hidden

        return h_marine, h_enemies, (h_marine_new, h_enemies_new)
