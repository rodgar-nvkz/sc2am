"""Cross-attention module for marine-to-enemy targeting.

This module implements the core attention mechanism that allows the marine
to "attend" to enemies. The attention weights directly serve as the
probability distribution for attack target selection.

Architecture:
    Q = W_q(marine_emb)           # Marine as Query
    K = W_k(enemy_embs)           # Enemies as Keys
    V = W_v(enemy_embs)           # Enemies as Values

    attn_weights = softmax(Q @ K.T / sqrt(d_k))  # Attack target probs!
    context = attn_weights @ V                    # Aggregated enemy info

The attention weights are used in two ways:
    1. As the probability distribution for "which enemy to attack"
    2. After pooling with V, as context for value/policy heads
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import ModelConfig


class CrossAttention(nn.Module):
    """Cross-attention from marine to enemies.

    The marine embedding serves as the query, and enemy embeddings serve
    as both keys and values. This design allows:

    1. The attention weights to directly represent "which enemy to target"
    2. The context vector to aggregate enemy information weighted by relevance

    Masking:
        - enemy_mask: True for valid (alive) enemies
        - range_mask: True for enemies in attack range (used for attack action)

    The attention weights can be masked differently depending on use:
        - For attack target selection: mask out-of-range and dead enemies
        - For context aggregation: only mask dead enemies
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.d_k = config.attention_dim
        self.num_heads = config.attention_heads

        # Query projection (from marine embedding)
        self.W_q = nn.Linear(config.gru_hidden_size, config.attention_dim)

        # Key projection (from enemy embeddings)
        self.W_k = nn.Linear(config.gru_hidden_size, config.attention_dim)

        # Value projection (from enemy embeddings)
        self.W_v = nn.Linear(config.gru_hidden_size, config.entity_embed_size)

        # Scaling factor for attention
        self.scale = self.d_k ** 0.5

        if config.init_orthogonal:
            self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with orthogonal initialization."""
        for module in [self.W_q, self.W_k, self.W_v]:
            nn.init.orthogonal_(module.weight, gain=self.config.init_gain)
            nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        marine_emb: Tensor,
        enemy_embs: Tensor,
        enemy_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute cross-attention from marine to enemies.

        Args:
            marine_emb: Marine temporal embedding (B, gru_hidden_size)
            enemy_embs: Enemy temporal embeddings (B, N, gru_hidden_size)
            enemy_mask: Optional mask where True = valid enemy (B, N).
                       Invalid enemies get -inf attention scores.

        Returns:
            Tuple of:
                - context: Aggregated enemy context (B, entity_embed_size)
                - attn_weights: Attention weights over enemies (B, N)
                - attn_logits: Raw attention logits before softmax (B, N)
        """
        # Project to Q, K, V
        Q = self.W_q(marine_emb)  # (B, d_k)
        K = self.W_k(enemy_embs)  # (B, N, d_k)
        V = self.W_v(enemy_embs)  # (B, N, d_v)

        # Compute attention scores: Q @ K.T / sqrt(d_k)
        # Q: (B, d_k) -> (B, 1, d_k)
        # K: (B, N, d_k) -> (B, d_k, N)
        Q = Q.unsqueeze(1)  # (B, 1, d_k)
        K = K.transpose(1, 2)  # (B, d_k, N)

        attn_logits = torch.bmm(Q, K).squeeze(1) / self.scale  # (B, N)

        # Apply mask: set invalid enemies to -inf
        if enemy_mask is not None:
            attn_logits = attn_logits.masked_fill(~enemy_mask, float("-inf"))

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, N)

        # Handle case where all enemies are masked (avoid NaN)
        # If all masked, attn_weights will be NaN, replace with zeros
        if enemy_mask is not None:
            all_masked = ~enemy_mask.any(dim=-1, keepdim=True)  # (B, 1)
            attn_weights = attn_weights.masked_fill(all_masked.expand_as(attn_weights), 0.0)

        # Compute context: weighted sum of values
        # attn_weights: (B, N) -> (B, 1, N)
        # V: (B, N, d_v)
        context = torch.bmm(attn_weights.unsqueeze(1), V).squeeze(1)  # (B, d_v)

        return context, attn_weights, attn_logits

    def get_attack_logits(
        self,
        marine_emb: Tensor,
        enemy_embs: Tensor,
        enemy_mask: Tensor | None = None,
        range_mask: Tensor | None = None,
    ) -> Tensor:
        """Get logits for attack target selection with range masking.

        This is a convenience method that applies both enemy validity mask
        and attack range mask for the attack action head.

        Args:
            marine_emb: Marine temporal embedding (B, gru_hidden_size)
            enemy_embs: Enemy temporal embeddings (B, N, gru_hidden_size)
            enemy_mask: Mask where True = valid (alive) enemy (B, N)
            range_mask: Mask where True = enemy in attack range (B, N)

        Returns:
            Attack target logits with invalid targets masked to -inf (B, N)
        """
        _, _, attn_logits = self.forward(marine_emb, enemy_embs, enemy_mask=None)

        # Combine masks: enemy must be both alive AND in range
        combined_mask = None
        if enemy_mask is not None and range_mask is not None:
            combined_mask = enemy_mask & range_mask
        elif enemy_mask is not None:
            combined_mask = enemy_mask
        elif range_mask is not None:
            combined_mask = range_mask

        if combined_mask is not None:
            attn_logits = attn_logits.masked_fill(~combined_mask, float("-inf"))

        return attn_logits


__all__ = [
    "CrossAttention",
]
