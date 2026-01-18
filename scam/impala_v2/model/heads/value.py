"""Value head for state value estimation."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scam.impala_v2.model.config import ModelConfig
from scam.impala_v2.model.heads.base import HeadLoss


class CriticHead(nn.Module):
    """Value head for estimating state value V(s).

    Used by the critic to estimate expected returns from a state.
    Receives both encoded features and raw observations (skip connection).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.net = nn.Sequential(
            nn.Linear(config.head_input_size, config.head_hidden_size),
            nn.Tanh(),
            nn.Linear(config.head_hidden_size, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        if not self.config.init_orthogonal:
            return

        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=self.config.init_gain)
                nn.init.constant_(module.bias, 0.0)

        # Value head uses gain=1.0 for output layer
        nn.init.orthogonal_(self.net[-1].weight, gain=self.config.value_init_gain)

    def forward(self, features: Tensor, raw_obs: Tensor) -> Tensor:
        """Forward pass: estimate state value.

        Args:
            features: Encoded features from encoder (B, embed_size)
            raw_obs: Raw observation for skip connection (B, obs_size)

        Returns:
            Value estimates (B,)
        """
        # Build input based on config
        if self.config.use_embedding and self.config.use_skip_connections:
            x = torch.cat([features, raw_obs], dim=-1)
        elif self.config.use_embedding:
            x = features
        else:
            x = raw_obs

        return self.net(x).squeeze(-1)

    def compute_loss(
        self,
        values: Tensor,
        targets: Tensor,
        clip_epsilon: float | None = None,
        old_values: Tensor | None = None,
    ) -> HeadLoss:
        """Compute value loss.

        Supports optional value clipping (PPO-style) to prevent large value updates.

        Args:
            values: Predicted values from current policy (B,)
            targets: Target values, e.g., V-trace targets (B,)
            clip_epsilon: Optional clipping parameter. If provided with old_values,
                         uses clipped value loss.
            old_values: Values from behavior policy (B,). Required if clip_epsilon set.

        Returns:
            HeadLoss with loss tensor and metrics dict
        """
        if clip_epsilon is not None and old_values is not None:
            # Clipped value loss (PPO-style)
            value_clipped = old_values + torch.clamp(
                values - old_values, -clip_epsilon, clip_epsilon
            )
            loss_unclipped = F.smooth_l1_loss(values, targets, reduction="none")
            loss_clipped = F.smooth_l1_loss(value_clipped, targets, reduction="none")
            loss = torch.max(loss_unclipped, loss_clipped).mean()
        else:
            # Standard smooth L1 loss
            loss = F.smooth_l1_loss(values, targets)

        # Compute explained variance for diagnostics
        with torch.no_grad():
            y_var = targets.var()
            if y_var > 0:
                explained_var = 1 - (targets - values).var() / y_var
                explained_var = explained_var.item()
            else:
                explained_var = float("nan")

        return HeadLoss(
            loss=loss,
            metrics={
                "loss": loss.item(),
                "explained_variance": explained_var,
            },
        )
