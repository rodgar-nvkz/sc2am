"""Base classes for action and value heads."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class HeadOutput:
    """Standardized output from any action head.

    Attributes:
        action: Sampled or provided action tensor
        log_prob: Log probability of the action
        entropy: Entropy of the distribution
        distribution: Underlying distribution object (for debugging/analysis)
    """

    action: Tensor
    log_prob: Tensor
    entropy: Tensor
    distribution: Any = None


@dataclass
class HeadLoss:
    """Standardized loss output from head.compute_loss().

    Attributes:
        loss: Scalar loss tensor for backprop
        metrics: Dictionary of scalar metrics for logging
    """

    loss: Tensor
    metrics: dict[str, float]


class ActionHead(nn.Module):
    """Base class for action heads (discrete and continuous).

    Action heads are responsible for:
    1. Producing a distribution over actions given features
    2. Sampling from or evaluating actions
    3. Computing policy gradient loss (PPO-style)
    """

    def forward(self, features: Tensor, raw_obs: Tensor, **kwargs) -> HeadOutput:
        """Forward pass: produce action distribution and sample/evaluate.

        Args:
            features: Encoded features from encoder (B, embed_size)
            raw_obs: Raw observation for skip connection (B, obs_size)
            **kwargs: Head-specific arguments (action, mask, etc.)

        Returns:
            HeadOutput with action, log_prob, entropy, distribution
        """
        raise NotImplementedError

    def compute_loss(
        self,
        new_log_prob: Tensor,
        old_log_prob: Tensor,
        advantages: Tensor,
        clip_epsilon: float,
    ) -> HeadLoss:
        """Compute PPO-clipped policy loss.

        This default implementation works for most heads.
        Override for special cases (e.g., masked losses).

        Args:
            new_log_prob: Log prob from current policy (B,)
            old_log_prob: Log prob from behavior policy (B,)
            advantages: Advantage estimates (B,)
            clip_epsilon: PPO clipping parameter

        Returns:
            HeadLoss with loss tensor and metrics dict
        """
        ratio = torch.exp(new_log_prob - old_log_prob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        loss = -torch.min(surr1, surr2).mean()

        # Compute useful metrics
        with torch.no_grad():
            approx_kl = (old_log_prob - new_log_prob).mean().item()
            clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean().item()

        return HeadLoss(
            loss=loss,
            metrics={
                "loss": loss.item(),
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
            },
        )


class ValueHead(nn.Module):
    """Base class for value estimation heads.

    Value heads estimate state value V(s) for the critic.
    """

    def forward(self, features: Tensor, raw_obs: Tensor) -> Tensor:
        """Forward pass: estimate state value.

        Args:
            features: Encoded features from encoder (B, embed_size)
            raw_obs: Raw observation for skip connection (B, obs_size)

        Returns:
            Value estimates (B,)
        """
        raise NotImplementedError

    def compute_loss(self, values: Tensor, targets: Tensor) -> HeadLoss:
        """Compute value loss.

        Args:
            values: Predicted values (B,)
            targets: Target values, e.g., V-trace targets (B,)

        Returns:
            HeadLoss with loss tensor and metrics dict
        """
        raise NotImplementedError
