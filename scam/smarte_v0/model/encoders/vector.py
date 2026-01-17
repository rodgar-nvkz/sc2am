"""Vector encoder for scalar observations (health, distances, angles, etc.)."""

import torch
from torch import Tensor, nn

from ..config import ModelConfig


class VectorEncoder(nn.Module):
    """MLP encoder for vector observations.

    Transforms raw scalar observations into a dense embedding.
    This encoder handles the "simple" observations like health, weapon cooldown,
    distances, and angles.

    Architecture:
        obs (obs_size) -> Linear -> Tanh -> Linear -> Tanh -> features (embed_size)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Build MLP layers
        layers = []
        in_size = config.obs_size

        for i in range(config.encoder_num_layers):
            out_size = config.encoder_hidden_size if i < config.encoder_num_layers - 1 else config.embed_size
            layers.append(nn.Linear(in_size, out_size))
            layers.append(self._get_activation())
            in_size = out_size

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _get_activation(self) -> nn.Module:
        """Get activation function based on config."""
        activation = self.config.encoder_activation.lower()
        if activation == "tanh":
            return nn.Tanh()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "silu":
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        if not self.config.init_orthogonal:
            return

        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=self.config.init_gain)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: Tensor) -> Tensor:
        """Encode vector observation to dense features.

        Args:
            obs: Raw observation tensor (B, obs_size)

        Returns:
            Encoded features (B, embed_size)
        """
        return self.net(obs)

    @property
    def output_size(self) -> int:
        """Size of the output embedding."""
        return self.config.embed_size
