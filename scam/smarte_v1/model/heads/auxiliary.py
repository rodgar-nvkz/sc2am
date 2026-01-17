"""Auxiliary prediction heads for additional training signal.

These heads predict auxiliary targets to:
    1. Prevent "value sinkholes" in sparse reward settings
    2. Force the backbone to learn meaningful state representations
    3. Provide step-by-step training signal even when V(s) ≈ V(s+1)

Auxiliary Tasks:
    - DamageAuxHead: Predict damage taken in the next N steps
    - DistanceAuxHead: Predict distance to nearest enemy at next step

These tasks have ground truth available from environment observations,
allowing supervised learning alongside RL. The auxiliary loss is weighted
lower than the main policy/value losses (typically 0.1x).
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ModelConfig
from .base import AuxiliaryHead, HeadLoss


class DamageAuxHead(AuxiliaryHead):
    """Auxiliary head for predicting future damage taken.

    Predicts the cumulative damage the marine will take in the next N steps.
    This helps the network learn "danger awareness" - recognizing states
    where enemies are positioned to deal damage.

    Target Computation (in collector):
        For step t, target = sum(damage_taken[t+1:t+N+1])

    This task is particularly useful because:
        1. It's directly related to kiting success
        2. Non-zero targets provide signal even with sparse rewards
        3. Encourages learning enemy attack patterns and timing
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.horizon = config.aux_damage_horizon

        # MLP: backbone_out (+ skip) -> predicted damage
        self.net = nn.Sequential(
            nn.Linear(config.head_input_size, config.head_hidden_size),
            nn.Tanh(),
            nn.Linear(config.head_hidden_size, config.head_hidden_size // 2),
            nn.Tanh(),
            nn.Linear(config.head_hidden_size // 2, 1),
        )

        # Output activation: ReLU ensures non-negative damage prediction
        self.output_activation = nn.ReLU()

        if config.init_orthogonal:
            self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with orthogonal initialization."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=self.config.init_gain)
                nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        features: Tensor,
        marine_obs: Tensor | None = None,
    ) -> Tensor:
        """Predict damage taken in next N steps.

        Args:
            features: Backbone output (B, backbone_output_size)
            marine_obs: Raw marine observation for skip connection (B, marine_obs_size)

        Returns:
            Predicted damage (B,) - non-negative
        """
        if self.config.use_skip_connections and marine_obs is not None:
            x = torch.cat([features, marine_obs], dim=-1)
        else:
            x = features

        pred = self.net(x).squeeze(-1)
        return self.output_activation(pred)

    def compute_loss(
        self,
        predictions: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
    ) -> HeadLoss:
        """Compute auxiliary loss for damage prediction.

        Uses Smooth L1 (Huber) loss which is:
            - MSE-like for small errors (smooth gradients)
            - L1-like for large errors (robust to outliers)

        Args:
            predictions: Predicted damage (B,)
            targets: Actual damage taken in next N steps (B,)
            mask: Optional mask where True = include in loss (B,)
                  Used to exclude terminal states where future is undefined.

        Returns:
            HeadLoss with loss tensor and metrics dict
        """
        # Compute per-sample loss
        loss_per_sample = F.smooth_l1_loss(predictions, targets, reduction="none")

        # Apply mask if provided
        if mask is not None:
            mask_float = mask.float()
            num_valid = mask_float.sum().clamp(min=1.0)
            loss = (loss_per_sample * mask_float).sum() / num_valid
        else:
            loss = loss_per_sample.mean()

        # Compute metrics
        with torch.no_grad():
            mae = (predictions - targets).abs().mean().item()
            pred_mean = predictions.mean().item()
            target_mean = targets.mean().item()

            # Correlation coefficient (how well do predictions track targets)
            if predictions.numel() > 1 and targets.std() > 1e-8:
                pred_centered = predictions - predictions.mean()
                target_centered = targets - targets.mean()
                correlation = (pred_centered * target_centered).mean() / (
                    predictions.std() * targets.std() + 1e-8
                )
                correlation = correlation.item()
            else:
                correlation = 0.0

        return HeadLoss(
            loss=loss,
            metrics={
                "loss": loss.item(),
                "mae": mae,
                "pred_mean": pred_mean,
                "target_mean": target_mean,
                "correlation": correlation,
            },
        )


class DistanceAuxHead(AuxiliaryHead):
    """Auxiliary head for predicting next-step distance to nearest enemy.

    Predicts the distance to the nearest enemy at the next timestep.
    This helps the network learn movement dynamics and enemy behavior.

    Target Computation (in collector):
        For step t, target = min(dist_to_enemy_i for all i) at step t+1

    This task is useful because:
        1. Distance changes are predictable from current state + action
        2. Helps learn the relationship between movement and positioning
        3. Provides signal for chase behavior (decreasing distance = good)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # MLP: backbone_out (+ skip) -> predicted distance
        self.net = nn.Sequential(
            nn.Linear(config.head_input_size, config.head_hidden_size),
            nn.Tanh(),
            nn.Linear(config.head_hidden_size, 1),
        )

        # Output activation: Sigmoid to bound output to [0, 1] (normalized distance)
        self.output_activation = nn.Sigmoid()

        if config.init_orthogonal:
            self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with orthogonal initialization."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=self.config.init_gain)
                nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        features: Tensor,
        marine_obs: Tensor | None = None,
    ) -> Tensor:
        """Predict normalized distance to nearest enemy at next step.

        Args:
            features: Backbone output (B, backbone_output_size)
            marine_obs: Raw marine observation for skip connection (B, marine_obs_size)

        Returns:
            Predicted distance (B,) - normalized to [0, 1]
        """
        if self.config.use_skip_connections and marine_obs is not None:
            x = torch.cat([features, marine_obs], dim=-1)
        else:
            x = features

        pred = self.net(x).squeeze(-1)
        return self.output_activation(pred)

    def compute_loss(
        self,
        predictions: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
    ) -> HeadLoss:
        """Compute auxiliary loss for distance prediction.

        Uses MSE loss since distances are bounded and smooth.

        Args:
            predictions: Predicted distance (B,) - normalized
            targets: Actual distance at next step (B,) - normalized
            mask: Optional mask where True = include in loss (B,)
                  Used to exclude terminal states.

        Returns:
            HeadLoss with loss tensor and metrics dict
        """
        # Compute per-sample loss
        loss_per_sample = F.mse_loss(predictions, targets, reduction="none")

        # Apply mask if provided
        if mask is not None:
            mask_float = mask.float()
            num_valid = mask_float.sum().clamp(min=1.0)
            loss = (loss_per_sample * mask_float).sum() / num_valid
        else:
            loss = loss_per_sample.mean()

        # Compute metrics
        with torch.no_grad():
            mae = (predictions - targets).abs().mean().item()
            pred_mean = predictions.mean().item()
            target_mean = targets.mean().item()
            rmse = torch.sqrt(loss_per_sample.mean()).item()

        return HeadLoss(
            loss=loss,
            metrics={
                "loss": loss.item(),
                "mae": mae,
                "rmse": rmse,
                "pred_mean": pred_mean,
                "target_mean": target_mean,
            },
        )


class CombinedAuxiliaryHead(nn.Module):
    """Combined auxiliary head for all auxiliary predictions.

    This is a convenience wrapper that manages multiple auxiliary heads
    and computes a combined loss.

    Usage:
        aux_head = CombinedAuxiliaryHead(config)
        predictions = aux_head(features, marine_obs)
        loss = aux_head.compute_loss(predictions, targets)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.damage_head = DamageAuxHead(config)
        self.distance_head = DistanceAuxHead(config)

    def forward(
        self,
        features: Tensor,
        marine_obs: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute all auxiliary predictions.

        Args:
            features: Backbone output (B, backbone_output_size)
            marine_obs: Raw marine observation for skip connection (B, marine_obs_size)

        Returns:
            Dictionary with predictions:
                - "damage": Predicted damage in next N steps (B,)
                - "distance": Predicted distance to nearest enemy (B,)
        """
        return {
            "damage": self.damage_head(features, marine_obs),
            "distance": self.distance_head(features, marine_obs),
        }

    def compute_loss(
        self,
        predictions: dict[str, Tensor],
        targets: dict[str, Tensor],
        mask: Tensor | None = None,
    ) -> HeadLoss:
        """Compute combined auxiliary loss.

        Args:
            predictions: Dict with "damage" and "distance" predictions
            targets: Dict with "damage" and "distance" targets
            mask: Optional mask where True = include in loss (B,)

        Returns:
            Combined HeadLoss with total loss and per-task metrics
        """
        damage_loss = self.damage_head.compute_loss(
            predictions["damage"],
            targets["damage"],
            mask=mask,
        )

        distance_loss = self.distance_head.compute_loss(
            predictions["distance"],
            targets["distance"],
            mask=mask,
        )

        # Combine losses (equal weight by default)
        total_loss = damage_loss.loss + distance_loss.loss

        # Merge metrics with prefixes
        metrics = {"loss": total_loss.item()}
        for key, value in damage_loss.metrics.items():
            metrics[f"damage_{key}"] = value
        for key, value in distance_loss.metrics.items():
            metrics[f"distance_{key}"] = value

        return HeadLoss(loss=total_loss, metrics=metrics)


__all__ = [
    "DamageAuxHead",
    "DistanceAuxHead",
    "CombinedAuxiliaryHead",
]
