"""Angle head for continuous movement direction with von Mises distribution.

Uses von Mises distribution (circular Gaussian) for proper angular exploration.
This solves the exploration problem where Gaussian on (sin, cos) fails to
explore opposite directions effectively.

Key insight: Gaussian noise on (sin, cos) gives poor angular coverage because
the output magnitude affects exploration. With mean=(0, 7) and std=1.8, the
policy can only explore ~20° around its current direction. Von Mises samples
angles directly, giving uniform angular exploration regardless of concentration.

Architecture:
    obs -> encoder -> h -> output_layer -> θ (angle in radians)
                      |
                      +-> aux_head -> predicted observation features

The policy outputs angle θ, samples from von Mises(θ, κ), then converts to
(sin, cos) for the environment. This maintains compatibility while fixing
exploration.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import VonMises

from ..config import ModelConfig
from .base import ActionHead, HeadLoss, HeadOutput


def von_mises_entropy(concentration: Tensor) -> Tensor:
    """Compute entropy of von Mises distribution.

    PyTorch's VonMises doesn't implement entropy(), so we compute it ourselves.

    Entropy of von Mises: H = -log(I_0(κ)) + κ * I_1(κ) / I_0(κ)

    Where I_0 and I_1 are modified Bessel functions of the first kind.
    We use torch.special.i0e and i1e (exponentially scaled versions) for
    numerical stability.

    Args:
        concentration: Concentration parameter κ (can be scalar or tensor)

    Returns:
        Entropy value(s), same shape as concentration
    """
    # I_0(κ) = i0e(κ) * exp(κ)
    # I_1(κ) = i1e(κ) * exp(κ)
    # So I_1(κ) / I_0(κ) = i1e(κ) / i0e(κ)
    i0e = torch.special.i0e(concentration)
    i1e = torch.special.i1e(concentration)

    # log(I_0(κ)) = log(i0e(κ)) + κ
    log_i0 = torch.log(i0e) + concentration

    # Entropy = log(2π) - log(I_0(κ)) + κ * I_1(κ) / I_0(κ)
    #         = log(2π) - log(I_0(κ)) + κ * i1e(κ) / i0e(κ)
    entropy = math.log(2 * math.pi) - log_i0 + concentration * (i1e / i0e)

    return entropy


class AngleHead(ActionHead):
    """Continuous action head using von Mises distribution for angles.

    The von Mises distribution is the circular analog of the Gaussian:
    - Defined on the circle [0, 2π) with natural wrap-around
    - Concentration parameter κ controls spread (κ→0 is uniform, κ→∞ is point mass)
    - Samples angles directly, giving uniform angular exploration

    Output format:
    - Network outputs mean angle θ (scalar per batch element)
    - Action is (sin θ, cos θ) for environment compatibility
    - Log prob computed on the angle using von Mises density

    Auxiliary Task:
    An auxiliary head predicts observation features from the hidden state,
    forcing the encoder to maintain observation-dependent representations.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Encoder: obs -> hidden representation
        # Deeper encoder with SiLU (Swish) for better gradient flow
        encoder_layers = []
        in_size = config.head_input_size
        for i in range(config.angle_encoder_layers):
            encoder_layers.append(nn.Linear(in_size, config.head_hidden_size))
            encoder_layers.append(nn.SiLU())
            in_size = config.head_hidden_size
        self.encoder = nn.Sequential(*encoder_layers)

        # Output head: h -> mean angle θ
        # Deeper head for more capacity to learn angle transformation
        output_layers = []
        for i in range(config.angle_output_layers - 1):
            output_layers.append(nn.Linear(config.head_hidden_size, config.head_hidden_size))
            output_layers.append(nn.SiLU())
        output_layers.append(nn.Linear(config.head_hidden_size, 1))  # Final: scalar angle in radians
        self.output_head = nn.Sequential(*output_layers)

        # Learnable log concentration (κ = exp(log_concentration))
        # Starting with log(1) = 0 gives moderate exploration
        self.log_concentration = nn.Parameter(torch.tensor(config.angle_init_log_concentration))

        # Auxiliary prediction head
        self.aux_enabled = config.aux_enabled
        if self.aux_enabled:
            self.aux_head = nn.Sequential(
                nn.Linear(config.head_hidden_size, config.aux_hidden_size),
                nn.SiLU(),
                nn.Linear(config.aux_hidden_size, config.aux_target_size),
            )
            self.aux_target_indices = config.aux_target_indices

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        if not self.config.init_orthogonal:
            return

        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=self.config.init_gain)
                nn.init.constant_(module.bias, 0.0)

        for module in self.output_head:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=self.config.init_gain)
                nn.init.constant_(module.bias, 0.0)

        if self.aux_enabled:
            for module in self.aux_head:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=self.config.init_gain)
                    nn.init.constant_(module.bias, 0.0)

    def _get_distribution(self, obs: Tensor) -> tuple[VonMises, Tensor]:
        """Get von Mises distribution and mean angle from observation.

        Args:
            obs: Observations (B, obs_size)

        Returns:
            Tuple of (von Mises distribution, mean angle tensor)
        """
        h = self.encoder(obs)
        theta_mean = self.output_head(h).squeeze(-1)  # (B,)

        # Concentration must be positive; use softplus for stability
        # softplus(x) = log(1 + exp(x)), always positive, smooth
        concentration = F.softplus(self.log_concentration)

        # Clamp concentration to avoid numerical issues
        # Very low κ (<0.01) can cause NaN in log_prob
        # Very high κ (>100) provides no exploration
        concentration = concentration.clamp(min=0.01, max=100.0)

        dist = VonMises(theta_mean, concentration)
        return dist, theta_mean

    def forward(self, obs: Tensor, action: Tensor | None = None) -> HeadOutput:
        """Forward pass: produce angle distribution and sample/evaluate.

        Args:
            obs: Observations (B, obs_size)
            action: Optional (sin, cos) action to evaluate (B, 2).
                    If None, samples new action.

        Returns:
            HeadOutput with:
            - action: (sin θ, cos θ) tensor (B, 2)
            - log_prob: log probability of the angle (B,)
            - entropy: entropy of von Mises distribution (B,)
            - distribution: the von Mises distribution object
        """
        dist, theta_mean = self._get_distribution(obs)

        if action is None:
            # Sample angle from von Mises distribution
            theta_sample = dist.sample()
            # Convert to (sin, cos) for environment
            action = torch.stack([torch.sin(theta_sample), torch.cos(theta_sample)], dim=-1)
            log_prob = dist.log_prob(theta_sample)
        else:
            # Action is (sin, cos), convert back to angle for log_prob
            # atan2(sin, cos) gives angle in [-π, π]
            theta_action = torch.atan2(action[:, 0], action[:, 1])
            log_prob = dist.log_prob(theta_action)

        # PyTorch VonMises doesn't implement entropy(), compute it ourselves
        concentration = F.softplus(self.log_concentration).clamp(min=0.01, max=100.0)
        entropy = von_mises_entropy(concentration).expand(obs.shape[0])

        return HeadOutput(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            distribution=dist,
        )

    def compute_aux_loss(self, obs: Tensor) -> Tensor:
        """Compute auxiliary prediction loss.

        Forces the encoder to represent observation features (enemy angles,
        distances) that are critical for correct action selection.

        Args:
            obs: Observations (B, obs_size)

        Returns:
            Scalar MSE loss for auxiliary prediction
        """
        if not self.aux_enabled:
            return torch.tensor(0.0, device=obs.device)

        h = self.encoder(obs)
        aux_pred = self.aux_head(h)
        aux_targets = obs[:, self.aux_target_indices]

        return F.mse_loss(aux_pred, aux_targets)

    def get_deterministic_action(self, obs: Tensor) -> Tensor:
        """Get deterministic action (mean direction) for evaluation.

        Args:
            obs: Observations (B, obs_size)

        Returns:
            (sin θ, cos θ) tensor (B, 2) at the mean angle
        """
        h = self.encoder(obs)
        theta_mean = self.output_head(h).squeeze(-1)  # (B,)

        # Convert mean angle to (sin, cos)
        sin_theta = torch.sin(theta_mean)
        cos_theta = torch.cos(theta_mean)

        return torch.stack([sin_theta, cos_theta], dim=-1)

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
            mask: Float mask where 1.0 = MOVE command (B,)

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

        metrics = {
            "loss": loss.item(),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }
        return HeadLoss(loss=loss, metrics=metrics)

    def get_concentration(self) -> float:
        """Get current concentration parameter (for logging)."""
        with torch.no_grad():
            return F.softplus(self.log_concentration).item()
