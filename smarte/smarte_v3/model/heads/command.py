"""Command head for discrete action selection."""

from torch import Tensor, distributions, nn

from ..config import ModelConfig
from .base import ActionHead, HeadOutput


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
        output_layer = self.net[-1]
        assert isinstance(output_layer, nn.Linear)
        nn.init.orthogonal_(output_layer.weight, gain=self.config.policy_init_gain)

    def forward(self, obs: Tensor, action: Tensor | None = None, *, mask: Tensor) -> HeadOutput:
        """Forward pass: produce command distribution and sample/evaluate.

        Args:
            obs: Observation tensor (B, obs_size)
            action: Optional command to evaluate (B,). If None, samples new action.
            mask: Boolean mask where True = valid action (B, num_commands).

        Returns:
            HeadOutput with discrete command action
        """
        logits = self.net(obs)

        # Apply action mask: set invalid actions to -inf
        logits = logits.masked_fill(~mask, float("-inf"))

        dist = distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return HeadOutput(action=action, log_prob=dist.log_prob(action), entropy=dist.entropy(), distribution=dist)

    def get_deterministic_action(self, obs: Tensor, *, mask: Tensor) -> Tensor:
        """Get deterministic action (argmax) for evaluation.

        Args:
            obs: Observation tensor (B, obs_size)
            mask: Boolean mask where True = valid action (B, num_commands).

        Returns:
            Command indices (B,)
        """
        logits = self.net(obs)

        logits = logits.masked_fill(~mask, float("-inf"))

        return logits.argmax(dim=-1)
