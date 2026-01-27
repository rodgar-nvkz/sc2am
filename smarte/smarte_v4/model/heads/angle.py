"""Angle head for continuous movement direction.

Uses Normal approximation of von Mises distribution for fast sampling while
maintaining proper angular exploration behavior.

Architecture:
    head_input -> encoder -> h -> output_layer -> θ (angle in radians)

The policy outputs angle θ, samples using Normal(θ, 1/√κ), then converts to
(sin, cos) for the environment.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ModelConfig
from .base import ActionHead, HeadOutput


class AngleHead(ActionHead):
    """Continuous action head for angles using Normal approximation.

    Uses Normal approximation of von Mises (std = 1/√κ) for fast sampling
    while maintaining proper angular exploration behavior.

    Output format:
    - Network outputs mean angle θ (scalar per batch element)
    - Action is (sin θ, cos θ) for environment compatibility
    - Log prob computed using Normal density

    Input is head_input: non-coord features + coord embeddings.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Encoder: head_input -> hidden representation
        # Deeper encoder with SiLU (Swish) for better gradient flow
        encoder_layers = []
        in_size = config.head_input_size
        for _ in range(config.angle_encoder_layers):
            encoder_layers.append(nn.Linear(in_size, config.head_hidden_size))
            encoder_layers.append(nn.SiLU())
            in_size = config.head_hidden_size
        self.encoder = nn.Sequential(*encoder_layers)

        # Output head: h -> mean angle θ
        # Deeper head for more capacity to learn angle transformation
        output_layers = []
        for _ in range(config.angle_output_layers - 1):
            output_layers.append(nn.Linear(config.head_hidden_size, config.head_hidden_size))
            output_layers.append(nn.SiLU())
        output_layers.append(nn.Linear(config.head_hidden_size, 1))  # Final: scalar angle in radians
        self.output_head = nn.Sequential(*output_layers)

        # Learnable log concentration (κ = exp(log_concentration))
        # Starting with log(1) = 0 gives moderate exploration
        self.log_concentration = nn.Parameter(torch.tensor(config.angle_init_log_concentration))

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

    def forward(
        self, obs: Tensor, action: Tensor | None = None, *, inference: bool = False
    ) -> HeadOutput | tuple[Tensor, Tensor]:
        """Forward pass: produce angle distribution and sample/evaluate.

        Uses Normal approximation of von Mises for fast sampling and consistent
        log_prob computation between collection and training.

        Args:
            obs: Observations (B, obs_size)
            action: Optional (sin, cos) action to evaluate (B, 2).
                    If None, samples new action.
            inference: If True, returns only (action, log_prob) tuple for fast collection.

        Returns:
            If inference=False: HeadOutput with action, log_prob, entropy, distribution
            If inference=True: Tuple of (action, log_prob)
        """
        # Network forward
        h = self.encoder(obs)
        theta_mean = self.output_head(h).squeeze(-1)  # (B,)

        # Get concentration and std for Normal approximation
        concentration = F.softplus(self.log_concentration).clamp(min=0.01, max=100.0)
        std = torch.rsqrt(concentration)  # std ≈ 1/sqrt(κ)

        if action is None:
            # Sample using Normal approximation (faster than von Mises rejection sampling)
            noise = torch.randn_like(theta_mean)
            theta_sample = theta_mean + std * noise
            action = torch.stack([torch.sin(theta_sample), torch.cos(theta_sample)], dim=-1)
            # Normal log_prob: -0.5 * ((x - μ)/σ)² - log(σ) - 0.5*log(2π)
            log_prob = -0.5 * noise * noise - torch.log(std) - 0.9189385332
        else:
            # Action is (sin, cos), convert back to angle for log_prob
            theta_action = torch.atan2(action[:, 0], action[:, 1])
            # Wrap angle difference to [-π, π] to handle circular nature
            delta = theta_action - theta_mean
            delta = delta - 2 * 3.141592653589793 * torch.round(delta / (2 * 3.141592653589793))
            # Normal log_prob for given action
            normalized = delta / std
            log_prob = -0.5 * normalized * normalized - torch.log(std) - 0.9189385332

        if inference:
            return action, log_prob

        # Full output with entropy (Normal entropy: 0.5 * log(2πe * σ²) = log(σ) + 1.4189)
        entropy = (torch.log(std) + 1.4189385332).expand(obs.shape[0])

        return HeadOutput(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            distribution=None,  # No distribution object with Normal approximation
        )

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

    def get_concentration(self) -> float:
        """Get current concentration parameter (for logging)."""
        with torch.no_grad():
            return F.softplus(self.log_concentration).item()
