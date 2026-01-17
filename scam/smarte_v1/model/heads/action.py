"""Discrete action head for 40-action space.

Actions:
    - 0-35: MOVE in direction (angle = i * 10°)
    - 36: ATTACK_Z1
    - 37: ATTACK_Z2
    - 38: STOP
    - 39: SKIP (no-op)
"""

import torch
from torch import Tensor, distributions, nn

from ..config import ModelConfig
from .base import ActionHead, HeadLoss, HeadOutput


class DiscreteActionHead(ActionHead):
    """Single categorical head over all 40 discrete actions.

    This replaces the previous CommandHead + AngleHead architecture.
    All actions are treated equally for clean backpropagation.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.net = nn.Sequential(
            nn.Linear(config.head_input_size, config.head_hidden_size),
            nn.Tanh(),
            nn.Linear(config.head_hidden_size, config.num_actions),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        if not self.config.init_orthogonal:
            return

        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=self.config.init_gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        # Smaller init for output layer (policy head)
        last_layer = self.net[-1]
        if isinstance(last_layer, nn.Linear):
            nn.init.orthogonal_(last_layer.weight, gain=self.config.policy_init_gain)

    def forward(  # type: ignore[override]
        self,
        features: Tensor,
        raw_obs: Tensor,
        action: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> HeadOutput:
        """Forward pass: produce action distribution and sample/evaluate.

        Args:
            features: LSTM output features (B, lstm_hidden_size)
            raw_obs: Raw observation for skip connection (B, obs_size)
            action: Optional action to evaluate (B,). If None, samples new action.
            mask: Optional boolean mask where True = valid action (B, num_actions)
                  Not used in v0, but kept for future compatibility.

        Returns:
            HeadOutput with discrete action
        """
        # Build input with optional skip connection
        if self.config.use_skip_connections:
            x = torch.cat([features, raw_obs], dim=-1)
        else:
            x = features

        logits = self.net(x)

        # Apply action mask if provided (for future use)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))

        dist = distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return HeadOutput(
            action=action,
            log_prob=dist.log_prob(action),
            entropy=dist.entropy(),
            distribution=dist,
        )

    def get_deterministic_action(
        self,
        features: Tensor,
        raw_obs: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Get deterministic action (argmax) for evaluation.

        Args:
            features: LSTM output features (B, lstm_hidden_size)
            raw_obs: Raw observation for skip connection (B, obs_size)
            mask: Optional boolean mask where True = valid action (B, num_actions)

        Returns:
            Action indices (B,)
        """
        if self.config.use_skip_connections:
            x = torch.cat([features, raw_obs], dim=-1)
        else:
            x = features

        logits = self.net(x)

        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))

        return logits.argmax(dim=-1)

    def compute_loss(
        self,
        new_log_prob: Tensor,
        old_log_prob: Tensor,
        advantages: Tensor,
        clip_epsilon: float,
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

        return HeadLoss(
            loss=loss,
            metrics={
                "loss": loss.item(),
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
            },
        )
