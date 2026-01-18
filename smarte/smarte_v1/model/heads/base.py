"""Base classes for action and value heads.

This module defines the core data structures and abstract base classes
for the hybrid action space architecture:

    - Action Type: Discrete [MOVE, ATTACK, STOP]
    - Move Direction: Continuous [sin, cos] (conditional on MOVE)
    - Attack Target: Pointer over N enemies (conditional on ATTACK)

HeadOutput is designed to support this conditional/autoregressive structure
where only the relevant action parameters are active for backpropagation.
"""

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class HybridAction:
    """Represents a hybrid action with type and conditional parameters.

    The action space is:
        - action_type: Discrete [0=MOVE, 1=ATTACK, 2=STOP]
        - move_direction: Continuous [sin, cos] (used only if action_type=MOVE)
        - attack_target: Integer index of enemy (used only if action_type=ATTACK)

    During training, gradients only flow through the relevant parameters:
        - If MOVE: gradients through action_type and move_direction
        - If ATTACK: gradients through action_type and attack_target
        - If STOP: gradients only through action_type
    """

    action_type: Tensor  # (B,) discrete action type index
    move_direction: Tensor  # (B, 2) [sin, cos] of move angle
    attack_target: Tensor  # (B,) index of enemy to attack


@dataclass
class HeadOutput:
    """Standardized output from the combined action heads.

    Attributes:
        action: The sampled or provided hybrid action
        log_prob: Combined log probability of the action
                  For MOVE: log P(type=MOVE) + log P(direction|MOVE)
                  For ATTACK: log P(type=ATTACK) + log P(target|ATTACK)
                  For STOP: log P(type=STOP)
        entropy: Combined entropy of the action distribution
        distributions: Dictionary of underlying distributions for debugging

    The log_prob computation respects the conditional structure:
        - Only the active branch contributes to the gradient
        - Masked branches don't affect the loss
    """

    action: HybridAction
    log_prob: Tensor  # (B,) combined log probability
    entropy: Tensor  # (B,) combined entropy
    distributions: dict[str, Any] = field(default_factory=dict)

    # Component log probs for diagnostics
    action_type_log_prob: Tensor | None = None  # (B,)
    move_direction_log_prob: Tensor | None = None  # (B,) or None if not MOVE
    attack_target_log_prob: Tensor | None = None  # (B,) or None if not ATTACK


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
    """Base class for action heads.

    Action heads are responsible for:
    1. Producing a distribution over actions given features
    2. Sampling from or evaluating actions
    3. Computing policy gradient loss (PPO-style)
    """

    def forward(self, *args, **kwargs) -> Any:
        """Forward pass: produce action distribution and sample/evaluate.

        Subclasses define their own signatures. This base class
        uses *args, **kwargs for flexibility.

        Returns:
            Head-specific output
        """
        raise NotImplementedError

    def compute_loss(
        self,
        new_log_prob: Tensor,
        old_log_prob: Tensor,
        advantages: Tensor,
        clip_epsilon: float,
        mask: Tensor | None = None,
    ) -> HeadLoss:
        """Compute PPO-clipped policy loss with optional masking.

        Args:
            new_log_prob: Log prob from current policy (B,)
            old_log_prob: Log prob from behavior policy (B,)
            advantages: Advantage estimates (B,)
            clip_epsilon: PPO clipping parameter
            mask: Optional mask where True = include in loss (B,).
                  Used for conditional heads (e.g., move direction only for MOVE actions)

        Returns:
            HeadLoss with loss tensor and metrics dict
        """
        ratio = torch.exp(new_log_prob - old_log_prob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        loss_per_sample = -torch.min(surr1, surr2)

        # Apply mask if provided
        if mask is not None:
            # Only include masked samples in loss
            mask_float = mask.float()
            num_valid = mask_float.sum().clamp(min=1.0)
            loss = (loss_per_sample * mask_float).sum() / num_valid
        else:
            loss = loss_per_sample.mean()

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

    def forward(self, *args, **kwargs) -> Tensor:
        """Forward pass: estimate state value.

        Subclasses define their own signatures. This base class
        uses *args, **kwargs for flexibility.

        Returns:
            Value estimates (B,)
        """
        raise NotImplementedError

    def compute_loss(
        self,
        values: Tensor,
        targets: Tensor,
        clip_epsilon: float | None = None,
        old_values: Tensor | None = None,
    ) -> HeadLoss:
        """Compute value loss.

        Args:
            values: Predicted values (B,)
            targets: Target values (B,)
            clip_epsilon: Optional clipping parameter
            old_values: Values from behavior policy (B,)

        Returns:
            HeadLoss with loss tensor and metrics dict
        """
        raise NotImplementedError


class AuxiliaryHead(nn.Module):
    """Base class for auxiliary prediction heads.

    Auxiliary heads predict additional targets (e.g., future damage,
    next-step distance) to provide richer training signal and
    prevent "value sinkholes" in sparse reward settings.
    """

    def forward(self, *args, **kwargs) -> Tensor:
        """Forward pass: predict auxiliary target.

        Subclasses define their own signatures. This base class
        uses *args, **kwargs for flexibility.

        Returns:
            Predictions (B,) or (B, output_size)
        """
        raise NotImplementedError

    def compute_loss(
        self,
        predictions: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
    ) -> HeadLoss:
        """Compute auxiliary loss (typically MSE).

        Args:
            predictions: Predicted values (B,) or (B, output_size)
            targets: Target values (B,) or (B, output_size)
            mask: Optional mask where True = include in loss (B,)

        Returns:
            HeadLoss with loss tensor and metrics dict
        """
        raise NotImplementedError
