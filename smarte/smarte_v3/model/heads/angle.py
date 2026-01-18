"""Angle head for continuous movement direction.

AlphaZero-style: predicts movement angle from observations only,
without conditioning on the selected command. The angle represents
"if we move, where should we go?" - a property of the state, not
of the command choice.

Masking is applied at loss computation time, not during forward pass.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, distributions, nn

from ..config import ModelConfig
from .base import ActionHead, HeadLoss, HeadOutput


class AngleHead(ActionHead):
    """Continuous action head for movement angle.

    Outputs a direction vector (sin, cos) for movement.
    Predicts from observations only (no command conditioning).

    The distribution is a Normal over (sin, cos) with learnable log_std.
    Outputs are normalized to the unit circle.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Network: obs -> angle mean (no command conditioning)
        self.net = nn.Sequential(
            nn.Linear(config.head_input_size, config.head_hidden_size),
            nn.Tanh(),
            nn.Linear(config.head_hidden_size, 2),  # sin, cos
        )

        # Learnable log std (shared across all states)
        self.log_std = nn.Parameter(torch.tensor([config.angle_init_log_std, config.angle_init_log_std]))

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
        output_layer = self.net[-1]
        assert isinstance(output_layer, nn.Linear)
        nn.init.orthogonal_(output_layer.weight, gain=self.config.policy_init_gain)

    def forward(self, obs: Tensor, action: Tensor | None = None) -> HeadOutput:
        """Forward pass: produce angle distribution and sample/evaluate.

        Args:
            obs: Raw observation (B, obs_size)
            action: Optional angle to evaluate (B, 2). If None, samples new action.

        Returns:
            HeadOutput with continuous angle action (normalized sin, cos)
        """
        mean = self.net(obs)

        # Normalize mean to unit circle for stability
        mean = F.normalize(mean, dim=-1)

        std = self.log_std.exp()
        dist = distributions.Normal(mean, std)

        if action is None:
            action = dist.sample()
            # Normalize sampled action to unit circle
            action = F.normalize(action, dim=-1)

        # Log prob: sum over sin/cos dimensions
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return HeadOutput(action=action, log_prob=log_prob, entropy=entropy, distribution=dist)

    def get_deterministic_action(self, obs: Tensor) -> Tensor:
        """Get deterministic action (mean of distribution) for evaluation.

        Args:
            obs: Raw observation (B, obs_size)

        Returns:
            Normalized angle (sin, cos) tensor (B, 2)
        """
        mean = self.net(obs)
        return F.normalize(mean, dim=-1)

    def compute_loss(
        self,
        new_log_prob: Tensor,
        old_log_prob: Tensor,
        advantages: Tensor,
        clip_epsilon: float,
        mask: Tensor,
    ) -> HeadLoss:
        """Compute PPO-clipped policy loss for angle head.

        Only computes loss for steps where MOVE command was selected.

        Args:
            new_log_prob: Log prob from current policy (B,)
            old_log_prob: Log prob from behavior policy (B,)
            advantages: Advantage estimates (B,)
            clip_epsilon: PPO clipping parameter
            mask: Float mask where 1.0 = MOVE command (B,).

        Returns:
            HeadLoss with loss tensor and metrics dict
        """
        ratio = torch.exp(new_log_prob - old_log_prob)

        # Masked loss: only compute for MOVE commands
        surr1 = ratio * advantages * mask
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages * mask
        num_moves = mask.sum().clamp(min=1.0)
        loss = -torch.min(surr1, surr2).sum() / num_moves

        # Metrics only for masked steps
        with torch.no_grad():
            if mask.sum() > 0:
                masked_old = old_log_prob[mask.bool()]
                masked_new = new_log_prob[mask.bool()]
                masked_ratio = ratio[mask.bool()]
                approx_kl = (masked_old - masked_new).mean().item()
                clip_fraction = ((masked_ratio - 1.0).abs() > clip_epsilon).float().mean().item()
            else:
                approx_kl = 0.0
                clip_fraction = 0.0

        metrics = {"loss": loss.item(), "approx_kl": approx_kl, "clip_fraction": clip_fraction}
        return HeadLoss(loss=loss, metrics=metrics)
