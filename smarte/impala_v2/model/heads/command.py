"""Command head for discrete action selection."""

import torch
from torch import Tensor, distributions, nn

from smarte.impala_v2.model.config import ModelConfig
from smarte.impala_v2.model.heads.base import ActionHead, HeadLoss, HeadOutput


class CommandHead(ActionHead):
    """Discrete action head for command selection.

    Produces a categorical distribution over commands (MOVE, ATTACK_Z1, ATTACK_Z2).
    Supports action masking for invalid actions (e.g., can't attack out-of-range targets).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.net = nn.Sequential(
            nn.Linear(config.head_input_size, config.head_hidden_size),
            nn.Tanh(),
            nn.Linear(config.head_hidden_size, config.num_commands),
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

        # Smaller init for output layer
        nn.init.orthogonal_(self.net[-1].weight, gain=self.config.policy_init_gain)

    def forward(
        self,
        features: Tensor,
        raw_obs: Tensor,
        action: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> HeadOutput:
        """Forward pass: produce command distribution and sample/evaluate.

        Args:
            features: Encoded features from encoder (B, embed_size)
            raw_obs: Raw observation for skip connection (B, obs_size)
            action: Optional command to evaluate (B,). If None, samples new action.
            mask: Optional boolean mask where True = valid action (B, num_commands)

        Returns:
            HeadOutput with discrete command action
        """
        # Build input based on config
        if self.config.use_embedding and self.config.use_skip_connections:
            x = torch.cat([features, raw_obs], dim=-1)
        elif self.config.use_embedding:
            x = features
        else:
            x = raw_obs

        logits = self.net(x)

        # Apply action mask: set invalid actions to -inf
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
            features: Encoded features from encoder (B, embed_size)
            raw_obs: Raw observation for skip connection (B, obs_size)
            mask: Optional boolean mask where True = valid action (B, num_commands)

        Returns:
            Command indices (B,)
        """
        if self.config.use_embedding and self.config.use_skip_connections:
            x = torch.cat([features, raw_obs], dim=-1)
        elif self.config.use_embedding:
            x = features
        else:
            x = raw_obs

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
        """Compute PPO-clipped policy loss for command head.

        Uses the default PPO clipping implementation from base class.
        """
        return super().compute_loss(new_log_prob, old_log_prob, advantages, clip_epsilon)
