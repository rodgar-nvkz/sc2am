"""Pointer-based attack target head for enemy selection.

This head outputs a categorical distribution over enemy targets. It leverages
the attention logits from the cross-attention module, which already encode
the "relevance" of each enemy to the marine.

Architecture:
    - Input: Attention logits from CrossAttention (B, N)
    - Output: Categorical distribution over N enemies

The attention weights naturally represent "which enemy should I focus on",
making them ideal for attack target selection. Additional masking ensures:
    1. Dead enemies cannot be targeted
    2. Out-of-range enemies cannot be targeted

During training, gradients only flow through this head when action_type == ATTACK.
"""

import torch
from torch import Tensor, distributions, nn

from ..config import ModelConfig
from .base import ActionHead, HeadLoss


class AttackTargetHead(ActionHead):
    """Pointer head for attack target selection.

    Input: Attention logits from cross-attention (B, N)
    Output: Distribution over N enemy targets

    This head is unique in that it doesn't have its own parameters -
    it reuses the attention logits computed by the CrossAttention module.
    This design:
        1. Ensures consistency between "who to pay attention to" and "who to attack"
        2. Reduces parameter count
        3. Provides interpretable attention weights

    Masking:
        - enemy_mask: True for alive enemies
        - range_mask: True for enemies in attack range
        - Combined mask: Only enemies that are BOTH alive AND in range
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Optional: learnable temperature for attention logits
        # Lower temperature = sharper distribution (more deterministic)
        # Higher temperature = softer distribution (more exploration)
        self.log_temperature = nn.Parameter(torch.zeros(1))
        self.min_temperature = 0.1
        self.max_temperature = 2.0

    def forward(
        self,
        attn_logits: Tensor,
        enemy_mask: Tensor | None = None,
        range_mask: Tensor | None = None,
        attack_target: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, distributions.Categorical]:
        """Forward pass: produce attack target distribution.

        Args:
            attn_logits: Raw attention logits from CrossAttention (B, N)
            enemy_mask: Mask where True = enemy is alive (B, N)
            range_mask: Mask where True = enemy is in attack range (B, N)
            attack_target: Optional target to evaluate (B,). If None, samples new.

        Returns:
            Tuple of:
                - attack_target: Sampled or provided target index (B,)
                - log_prob: Log probability of target (B,)
                - entropy: Entropy of the distribution (B,)
                - distribution: Categorical distribution object
        """
        # Apply temperature scaling
        temperature = torch.clamp(
            torch.exp(self.log_temperature),
            self.min_temperature,
            self.max_temperature,
        )
        scaled_logits = attn_logits / temperature

        # Create combined mask: enemy must be alive AND in range
        combined_mask = self._create_combined_mask(enemy_mask, range_mask, attn_logits.device)

        # Apply mask: set invalid targets to -inf
        if combined_mask is not None:
            scaled_logits = scaled_logits.masked_fill(~combined_mask, float("-inf"))

        # Handle edge case: no valid targets
        # This shouldn't happen if action masking is done correctly,
        # but we handle it gracefully by using uniform distribution
        all_invalid = ~combined_mask.any(dim=-1) if combined_mask is not None else None
        if all_invalid is not None and all_invalid.any():
            # For samples with no valid targets, use uniform over all enemies
            batch_size, num_enemies = attn_logits.shape
            uniform_logits = torch.zeros_like(scaled_logits)
            scaled_logits = torch.where(
                all_invalid.unsqueeze(-1).expand_as(scaled_logits),
                uniform_logits,
                scaled_logits,
            )

        # Create distribution
        dist = distributions.Categorical(logits=scaled_logits)

        # Sample or evaluate
        if attack_target is None:
            attack_target = dist.sample()

        log_prob = dist.log_prob(attack_target)
        entropy = dist.entropy()

        return attack_target, log_prob, entropy, dist

    def _create_combined_mask(
        self,
        enemy_mask: Tensor | None,
        range_mask: Tensor | None,
        device: torch.device,
    ) -> Tensor | None:
        """Create combined mask for valid attack targets.

        Args:
            enemy_mask: Alive enemies (B, N)
            range_mask: In-range enemies (B, N)
            device: Device for tensor creation

        Returns:
            Combined mask (B, N) or None if no masks provided
        """
        if enemy_mask is not None and range_mask is not None:
            return enemy_mask & range_mask
        elif enemy_mask is not None:
            return enemy_mask
        elif range_mask is not None:
            return range_mask
        return None

    def get_deterministic_action(
        self,
        attn_logits: Tensor,
        enemy_mask: Tensor | None = None,
        range_mask: Tensor | None = None,
    ) -> Tensor:
        """Get deterministic target (argmax) for evaluation.

        Args:
            attn_logits: Raw attention logits from CrossAttention (B, N)
            enemy_mask: Mask where True = enemy is alive (B, N)
            range_mask: Mask where True = enemy is in attack range (B, N)

        Returns:
            Target indices (B,)
        """
        # Apply masks
        combined_mask = self._create_combined_mask(enemy_mask, range_mask, attn_logits.device)

        logits = attn_logits.clone()
        if combined_mask is not None:
            logits = logits.masked_fill(~combined_mask, float("-inf"))

        return logits.argmax(dim=-1)

    def compute_loss(
        self,
        new_log_prob: Tensor,
        old_log_prob: Tensor,
        advantages: Tensor,
        clip_epsilon: float,
        mask: Tensor | None = None,
    ) -> HeadLoss:
        """Compute PPO-clipped policy loss for attack target.

        This loss is ONLY computed for samples where action_type == ATTACK.
        The mask parameter should be True for ATTACK actions.

        Args:
            new_log_prob: Log prob from current policy (B,)
            old_log_prob: Log prob from behavior policy (B,)
            advantages: Advantage estimates (B,)
            clip_epsilon: PPO clipping parameter
            mask: Mask where True = action was ATTACK (B,)
                  This is CRITICAL for conditional training!

        Returns:
            HeadLoss with loss tensor and metrics dict
        """
        ratio = torch.exp(new_log_prob - old_log_prob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        loss_per_sample = -torch.min(surr1, surr2)

        # Apply mask - only include ATTACK actions in loss
        if mask is not None:
            mask_float = mask.float()
            num_valid = mask_float.sum().clamp(min=1.0)
            loss = (loss_per_sample * mask_float).sum() / num_valid

            # If no ATTACK actions in batch, return zero loss
            if num_valid == 0:
                loss = torch.zeros(1, device=loss_per_sample.device, requires_grad=True).squeeze()
        else:
            loss = loss_per_sample.mean()

        # Compute metrics
        with torch.no_grad():
            if mask is not None and mask.any():
                # Only compute metrics for ATTACK actions
                masked_old = old_log_prob[mask]
                masked_new = new_log_prob[mask]
                masked_ratio = ratio[mask]
                approx_kl = (masked_old - masked_new).mean().item()
                clip_fraction = ((masked_ratio - 1.0).abs() > clip_epsilon).float().mean().item()
            else:
                approx_kl = 0.0
                clip_fraction = 0.0

        return HeadLoss(
            loss=loss,
            metrics={
                "loss": loss.item() if isinstance(loss, Tensor) else loss,
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
                "num_attack_actions": mask.sum().item() if mask is not None else -1,
                "temperature": torch.exp(self.log_temperature).item(),
            },
        )

    def get_target_probs(
        self,
        attn_logits: Tensor,
        enemy_mask: Tensor | None = None,
        range_mask: Tensor | None = None,
    ) -> Tensor:
        """Get probability distribution over targets (for visualization).

        Args:
            attn_logits: Raw attention logits from CrossAttention (B, N)
            enemy_mask: Mask where True = enemy is alive (B, N)
            range_mask: Mask where True = enemy is in attack range (B, N)

        Returns:
            Target probabilities (B, N) - sums to 1 over valid targets
        """
        temperature = torch.clamp(
            torch.exp(self.log_temperature),
            self.min_temperature,
            self.max_temperature,
        )
        scaled_logits = attn_logits / temperature

        combined_mask = self._create_combined_mask(enemy_mask, range_mask, attn_logits.device)
        if combined_mask is not None:
            scaled_logits = scaled_logits.masked_fill(~combined_mask, float("-inf"))

        return torch.softmax(scaled_logits, dim=-1)
