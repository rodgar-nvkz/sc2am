"""Continuous move direction head for movement actions.

This head outputs a continuous direction for movement, represented as [sin, cos]
of the movement angle. This is only used when action_type == MOVE.

Architecture:
    - Input: Backbone output (+ optional skip connection)
    - Output: Gaussian distribution over [sin, cos] direction vector

During training, the gradient only flows through this head when the
action type was MOVE. This is achieved through masking in the loss computation.

The direction is normalized to unit length to ensure valid [sin, cos] pairs.
"""

import torch
from torch import Tensor, distributions, nn

from ..config import ModelConfig
from .base import ActionHead, HeadLoss


class MoveDirectionHead(ActionHead):
    """Continuous head for movement direction prediction.

    Input: Backbone output (+ optional skip connection)
    Output: Distribution over 2D direction vector [sin, cos]

    The output is parameterized as a Gaussian with learnable mean and
    log-std. The sampled direction is normalized to unit length.

    Design choices:
        1. Output [sin, cos] directly (matches env action space)
        2. Use Independent Normal distribution for each component
        3. Small initial std for stable exploration
        4. Normalize output to ensure valid unit vector
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # MLP: backbone_out (+ skip) -> direction mean and log_std
        self.net = nn.Sequential(
            nn.Linear(config.head_input_size, config.head_hidden_size),
            nn.Tanh(),
        )

        # Separate outputs for mean and log_std
        self.mean_layer = nn.Linear(config.head_hidden_size, 2)  # [sin, cos]
        self.log_std_layer = nn.Linear(config.head_hidden_size, 2)  # log std for [sin, cos]

        # Clamp log_std to prevent numerical issues
        self.log_std_min = -5.0
        self.log_std_max = 2.0

        if config.init_orthogonal:
            self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with orthogonal initialization."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=self.config.init_gain)
                nn.init.constant_(module.bias, 0.0)

        # Initialize mean layer with small weights for centered output
        nn.init.orthogonal_(self.mean_layer.weight, gain=self.config.policy_init_gain)
        nn.init.constant_(self.mean_layer.bias, 0.0)

        # Initialize log_std layer to produce small initial std
        nn.init.orthogonal_(self.log_std_layer.weight, gain=self.config.policy_init_gain)
        nn.init.constant_(self.log_std_layer.bias, -1.0)  # Initial std ≈ 0.37

    def forward(
        self,
        features: Tensor,
        marine_obs: Tensor | None = None,
        direction: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, distributions.Normal]:
        """Forward pass: produce movement direction distribution.

        Args:
            features: Backbone output (B, backbone_output_size)
            marine_obs: Raw marine observation for skip connection (B, marine_obs_size)
            direction: Optional direction to evaluate (B, 2). If None, samples new.

        Returns:
            Tuple of:
                - direction: Sampled or provided direction [sin, cos] (B, 2)
                - log_prob: Log probability of direction (B,)
                - entropy: Entropy of the distribution (B,)
                - distribution: Normal distribution object
        """
        # Build input with optional skip connection
        if self.config.use_skip_connections and marine_obs is not None:
            x = torch.cat([features, marine_obs], dim=-1)
        else:
            x = features

        # Get hidden representation
        hidden = self.net(x)  # (B, head_hidden_size)

        # Get mean and log_std
        mean = self.mean_layer(hidden)  # (B, 2)
        log_std = self.log_std_layer(hidden)  # (B, 2)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # Create independent normal distribution for each component
        dist = distributions.Normal(mean, std)

        # Sample or evaluate
        if direction is None:
            # Sample and normalize to unit length
            direction = dist.rsample()  # reparameterized sample for gradient flow
            direction = self._normalize(direction)

        # Compute log probability (sum over components)
        # Note: We compute log_prob on the normalized sample directly.
        # Technically, this is an approximation since the Jacobian of the
        # normalization transform should be included. However, this works
        # well in practice because:
        #   1. The std is relatively small, so samples are near unit length
        #   2. PPO uses log prob ratios, so constant biases cancel out
        log_prob = dist.log_prob(direction).sum(dim=-1)  # (B,)

        # Clamp log_prob to prevent extreme values that cause ratio explosion
        # in off-policy training (IMPALA with stale weights)
        log_prob = torch.clamp(log_prob, min=-100.0)

        # Compute entropy (sum over independent components)
        entropy = dist.entropy().sum(dim=-1)  # (B,)

        return direction, log_prob, entropy, dist

    def _normalize(self, direction: Tensor) -> Tensor:
        """Normalize direction to unit length.

        Args:
            direction: Raw direction vector (B, 2)

        Returns:
            Normalized direction (B, 2) with ||direction|| = 1
        """
        norm = torch.norm(direction, dim=-1, keepdim=True).clamp(min=1e-8)
        return direction / norm

    def get_deterministic_action(
        self,
        features: Tensor,
        marine_obs: Tensor | None = None,
    ) -> Tensor:
        """Get deterministic direction (mean) for evaluation.

        Args:
            features: Backbone output (B, backbone_output_size)
            marine_obs: Raw marine observation for skip connection (B, marine_obs_size)

        Returns:
            Normalized mean direction [sin, cos] (B, 2)
        """
        if self.config.use_skip_connections and marine_obs is not None:
            x = torch.cat([features, marine_obs], dim=-1)
        else:
            x = features

        hidden = self.net(x)
        mean = self.mean_layer(hidden)

        return self._normalize(mean)

    def compute_loss(
        self,
        new_log_prob: Tensor,
        old_log_prob: Tensor,
        advantages: Tensor,
        clip_epsilon: float,
        mask: Tensor | None = None,
    ) -> HeadLoss:
        """Compute PPO-clipped policy loss for move direction.

        This loss is ONLY computed for samples where action_type == MOVE.
        The mask parameter should be True for MOVE actions.

        Args:
            new_log_prob: Log prob from current policy (B,)
            old_log_prob: Log prob from behavior policy (B,)
            advantages: Advantage estimates (B,)
            clip_epsilon: PPO clipping parameter
            mask: Mask where True = action was MOVE (B,)
                  This is CRITICAL for conditional training!

        Returns:
            HeadLoss with loss tensor and metrics dict
        """
        ratio = torch.exp(new_log_prob - old_log_prob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        loss_per_sample = -torch.min(surr1, surr2)

        # Apply mask - only include MOVE actions in loss
        if mask is not None:
            mask_float = mask.float()
            num_valid_raw = mask_float.sum()

            # If no MOVE actions in batch, return zero loss
            if num_valid_raw == 0:
                loss = torch.zeros(1, device=loss_per_sample.device, requires_grad=True).squeeze()
            else:
                loss = (loss_per_sample * mask_float).sum() / num_valid_raw
        else:
            loss = loss_per_sample.mean()

        # Compute metrics
        with torch.no_grad():
            num_move = mask.sum().item() if mask is not None else advantages.numel()
            if mask is not None and mask.any():
                # Only compute metrics for MOVE actions
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
                "num_move_actions": int(num_move),
            },
        )

    @staticmethod
    def angle_to_direction(angle: Tensor) -> Tensor:
        """Convert angle in radians to [sin, cos] direction.

        Args:
            angle: Angle in radians (B,)

        Returns:
            Direction vector [sin, cos] (B, 2)
        """
        return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)

    @staticmethod
    def direction_to_angle(direction: Tensor) -> Tensor:
        """Convert [sin, cos] direction to angle in radians.

        Args:
            direction: Direction vector [sin, cos] (B, 2)

        Returns:
            Angle in radians (B,)
        """
        sin_a = direction[:, 0]
        cos_a = direction[:, 1]
        return torch.atan2(sin_a, cos_a)
