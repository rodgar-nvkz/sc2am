"""Vector encoder for scalar observations (health, distances, angles, etc.).

This is a LEGACY encoder kept for backwards compatibility.
New code should use EntityEncoder from entity.py instead.
"""

from torch import Tensor, nn

from ..config import ModelConfig


class VectorEncoder(nn.Module):
    """MLP encoder for vector observations.

    Transforms raw scalar observations into a dense embedding.
    This encoder handles the "simple" observations like health, weapon cooldown,
    distances, and angles.

    LEGACY: This encoder is kept for backwards compatibility.
    New code should use EntityEncoder instead.

    Architecture:
        obs (obs_size) -> Linear -> Tanh -> Linear -> Tanh -> features (embed_size)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Use new config property names (entity_*) with fallback behavior
        num_layers = config.entity_num_layers
        hidden_size = config.entity_hidden_size
        embed_size = config.entity_embed_size
        activation = config.entity_activation

        # Build MLP layers
        layers = []
        in_size = config.obs_size

        for i in range(num_layers):
            out_size = hidden_size if i < num_layers - 1 else embed_size
            layers.append(nn.Linear(in_size, out_size))
            layers.append(self._get_activation(activation))
            in_size = out_size

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function based on name."""
        activations = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "elu": nn.ELU,
        }
        if activation.lower() not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        return activations[activation.lower()]()

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
        return self.config.entity_embed_size
