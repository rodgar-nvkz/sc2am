"""Base classes for action and value heads."""

from abc import ABC, abstractmethod
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


class ActionHead(nn.Module, ABC):
    """Base class for action heads (discrete and continuous).

    Action heads are responsible for:
    1. Producing a distribution over actions given observations
    2. Sampling from or evaluating actions
    3. Computing policy gradient loss (PPO-style)
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> HeadOutput:
        """Forward pass: produce action distribution and sample/evaluate.

        Returns:
            HeadOutput with action, log_prob, entropy, distribution
        """
        ...

    def compute_loss(
        self, new_log_prob: Tensor, old_log_prob: Tensor, advantages: Tensor, clip_epsilon: float
    ) -> HeadLoss:
        """Compute PPO-clipped policy loss.

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

        metrics = {"loss": loss.item(), "approx_kl": approx_kl, "clip_fraction": clip_fraction}
        return HeadLoss(loss=loss, metrics=metrics)
