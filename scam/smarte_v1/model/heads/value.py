"""Value head for state value estimation.

This head estimates V(s), the expected return from the current state.
It uses the shared backbone output (GRU hidden state + attention context)
to make predictions.

Architecture:
    Input: backbone_out (+ optional marine_obs skip connection)
    Output: Scalar value estimate V(s)

The value head is critical for:
    1. Computing advantages for policy gradient
    2. Bootstrapping in TD-learning (solving credit assignment)
    3. Predicting expected outcomes for GAE computation
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ModelConfig
from .base import HeadLoss, ValueHead


class CriticHead(ValueHead):
    """Value head for estimating state value V(s).

    Uses the shared backbone output to estimate expected returns.
    The backbone already encodes:
        - Temporal marine state (via GRU)
        - Attention-weighted enemy context

    This gives the critic all information needed to assess "how likely
    are we to win from this state?"
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # MLP: backbone_out (+ skip) -> value
        self.net = nn.Sequential(
            nn.Linear(config.head_input_size, config.head_hidden_size),
            nn.Tanh(),
            nn.Linear(config.head_hidden_size, 1),
        )

        if config.init_orthogonal:
            self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with orthogonal initialization."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=self.config.init_gain)
                nn.init.constant_(module.bias, 0.0)

        # Value head uses gain=1.0 for output layer
        last_layer = self.net[-1]
        if isinstance(last_layer, nn.Linear):
            nn.init.orthogonal_(last_layer.weight, gain=self.config.value_init_gain)

    def forward(
        self,
        features: Tensor,
        marine_obs: Tensor | None = None,
    ) -> Tensor:
        """Forward pass: estimate state value.

        Args:
            features: Backbone output (B, backbone_output_size)
            marine_obs: Raw marine observation for skip connection (B, marine_obs_size)

        Returns:
            Value estimates (B,)
        """
        # Build input with optional skip connection
        if self.config.use_skip_connections and marine_obs is not None:
            x = torch.cat([features, marine_obs], dim=-1)
        else:
            x = features

        return self.net(x).squeeze(-1)

    def compute_loss(
        self,
        values: Tensor,
        targets: Tensor,
        clip_epsilon: float | None = None,
        old_values: Tensor | None = None,
    ) -> HeadLoss:
        """Compute value loss with optional clipping.

        Supports two modes:
            1. Standard MSE loss (when clip_epsilon is None)
            2. Clipped value loss (PPO-style, when clip_epsilon and old_values provided)

        The clipped loss prevents large value updates that could destabilize
        training when the policy changes significantly.

        Args:
            values: Predicted values from current policy (B,)
            targets: Target values (e.g., V-trace targets or GAE returns) (B,)
            clip_epsilon: Optional clipping parameter. If provided with old_values,
                         uses clipped value loss.
            old_values: Values from behavior policy (B,). Required if clip_epsilon set.

        Returns:
            HeadLoss with loss tensor and metrics dict
        """
        if clip_epsilon is not None and old_values is not None:
            # Clipped value loss (PPO-style)
            # Prevents value function from changing too much
            value_clipped = old_values + torch.clamp(
                values - old_values, -clip_epsilon, clip_epsilon
            )
            loss_unclipped = F.smooth_l1_loss(values, targets, reduction="none")
            loss_clipped = F.smooth_l1_loss(value_clipped, targets, reduction="none")
            loss = torch.max(loss_unclipped, loss_clipped).mean()
        else:
            # Standard smooth L1 loss (Huber loss)
            # More robust to outliers than MSE
            loss = F.smooth_l1_loss(values, targets)

        # Compute explained variance for diagnostics
        # High explained variance = good value predictions
        # Low/negative = value function is not useful for advantage estimation
        with torch.no_grad():
            y_var = targets.var()
            if y_var > 1e-8:
                explained_var = 1 - (targets - values).var() / y_var
                explained_var = explained_var.item()
            else:
                explained_var = float("nan")

            # Additional metrics
            value_mean = values.mean().item()
            value_std = values.std().item()
            target_mean = targets.mean().item()

        return HeadLoss(
            loss=loss,
            metrics={
                "loss": loss.item(),
                "explained_variance": explained_var,
                "value_mean": value_mean,
                "value_std": value_std,
                "target_mean": target_mean,
            },
        )


class DualValueHead(ValueHead):
    """Dual value head with separate networks for positive/negative outcomes.

    This is an experimental architecture that maintains two value estimates:
        - V_win: Expected value conditioned on eventually winning
        - V_lose: Expected value conditioned on eventually losing

    The final value is: V(s) = P(win|s) * V_win + P(lose|s) * V_lose

    This can help in sparse reward settings by separately modeling
    "how good is this state if we win" vs "how bad if we lose".

    Currently not used in main architecture but kept for experimentation.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(config.head_input_size, config.head_hidden_size),
            nn.Tanh(),
        )

        # Separate heads for win/lose value
        self.win_head = nn.Linear(config.head_hidden_size, 1)
        self.lose_head = nn.Linear(config.head_hidden_size, 1)

        # Win probability head
        self.win_prob_head = nn.Linear(config.head_hidden_size, 1)

        if config.init_orthogonal:
            self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in self.shared:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=self.config.init_gain)
                nn.init.constant_(module.bias, 0.0)

        for head in [self.win_head, self.lose_head, self.win_prob_head]:
            nn.init.orthogonal_(head.weight, gain=self.config.value_init_gain)
            nn.init.constant_(head.bias, 0.0)

    def forward(
        self,
        features: Tensor,
        marine_obs: Tensor | None = None,
    ) -> Tensor:
        """Forward pass: estimate state value.

        Args:
            features: Backbone output (B, backbone_output_size)
            marine_obs: Raw marine observation for skip connection (B, marine_obs_size)

        Returns:
            Value estimates (B,)
        """
        if self.config.use_skip_connections and marine_obs is not None:
            x = torch.cat([features, marine_obs], dim=-1)
        else:
            x = features

        hidden = self.shared(x)

        # Get win probability
        win_prob = torch.sigmoid(self.win_prob_head(hidden).squeeze(-1))

        # Get conditional values
        v_win = self.win_head(hidden).squeeze(-1)
        v_lose = self.lose_head(hidden).squeeze(-1)

        # Combined value estimate
        value = win_prob * v_win + (1 - win_prob) * v_lose

        return value

    def forward_detailed(
        self,
        features: Tensor,
        marine_obs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass with detailed outputs for analysis.

        Returns:
            Tuple of (value, win_prob, v_win, v_lose)
        """
        if self.config.use_skip_connections and marine_obs is not None:
            x = torch.cat([features, marine_obs], dim=-1)
        else:
            x = features

        hidden = self.shared(x)

        win_prob = torch.sigmoid(self.win_prob_head(hidden).squeeze(-1))
        v_win = self.win_head(hidden).squeeze(-1)
        v_lose = self.lose_head(hidden).squeeze(-1)
        value = win_prob * v_win + (1 - win_prob) * v_lose

        return value, win_prob, v_win, v_lose

    def compute_loss(
        self,
        values: Tensor,
        targets: Tensor,
        clip_epsilon: float | None = None,
        old_values: Tensor | None = None,
    ) -> HeadLoss:
        """Compute value loss (same as CriticHead for compatibility)."""
        loss = F.smooth_l1_loss(values, targets)

        with torch.no_grad():
            y_var = targets.var()
            explained_var = (1 - (targets - values).var() / y_var).item() if y_var > 1e-8 else float("nan")

        return HeadLoss(
            loss=loss,
            metrics={
                "loss": loss.item(),
                "explained_variance": explained_var,
            },
        )
